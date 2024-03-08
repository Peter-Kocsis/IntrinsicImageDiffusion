import numpy as np
import torch
from torch import nn
from einops import einsum, rearrange


# ====================== INTENSITY ======================

class Constant(nn.Module):
    def __init__(self,
                 value,
                 exp_val=True):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.exp_val = exp_val

    def forward(self, direction):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return val.unsqueeze(0).expand_as(direction)

    def reg_loss(self):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return torch.sum(val)



class MultipleSphericalGaussians(nn.Module):
    def __init__(self,
                 sg_col=6,
                 sg_row=2,
                 ch=3,
                 single_color=False,
                 w_lamb_reg=0):
        super().__init__()

        self.sg_col = sg_col
        self.sg_row = sg_row
        self.SGNum = self.sg_col * self.sg_row

        self.single_color = single_color
        self.COLORNum = self.SGNum
        if self.single_color:
            self.COLORNum = 1

        self.w_lamb_reg = w_lamb_reg

        self.ch = ch

        self.nearest_dist_sqr = None

        self.weight, self.theta, self.phi, self.lamb = self.init_sg_grid()

        is_enabled = torch.tensor(True)
        self.register_buffer('is_enabled', is_enabled)

    def init_sg_grid(self):
        phiCenter = ((np.arange(self.sg_col) + 0.5) / self.sg_col - 0.5) * np.pi * 2
        thetaCenter = (np.arange(self.sg_row) + 0.5) / self.sg_row * np.pi / 2.0

        phiCenter, thetaCenter = np.meshgrid(phiCenter, thetaCenter)

        thetaCenter = thetaCenter.reshape(self.SGNum, 1).astype(np.float32)
        thetaCenter = torch.from_numpy(thetaCenter).expand([self.SGNum, 1])

        phiCenter = phiCenter.reshape(self.SGNum, 1).astype(np.float32)
        phiCenter = torch.from_numpy(phiCenter).expand([self.SGNum, 1])

        thetaRange = (np.pi / 2 / self.sg_row) * 1.5
        phiRange = (2 * np.pi / self.sg_col) * 1.5

        self.register_buffer('thetaCenter', thetaCenter)
        self.register_buffer('phiCenter', phiCenter)
        self.register_buffer('thetaRange', torch.tensor(thetaRange))
        self.register_buffer('phiRange', torch.tensor(phiRange))

        weight = nn.Parameter(torch.ones((self.COLORNum, self.ch), dtype=torch.float32) * (0))
        theta = nn.Parameter(torch.zeros((self.SGNum, 1), dtype=torch.float32))
        phi = nn.Parameter(torch.zeros((self.SGNum, 1), dtype=torch.float32))
        lamb = nn.Parameter(torch.log(torch.ones(self.SGNum, 1) * np.pi / self.sg_row))

        return weight, theta, phi, lamb

    def deparameterize(self):
        theta = self.thetaRange * torch.tanh(self.theta) + self.thetaCenter

        phi = self.phiRange * torch.tanh(self.phi) + self.phiCenter
        lamb = self.deparameterize_lamb()

        weight = self.deparameterize_weight()

        return weight, theta, phi, lamb

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def deparameterize_lamb(self):
        lamb = torch.exp(self.lamb)
        return lamb

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.cos(phi)
        axisY = torch.sin(theta) * torch.sin(phi)
        axisZ = torch.cos(theta)
        axis = torch.cat([axisX, axisY, axisZ], dim=1)

        return axis

    def forward(self, direction):
        if self.is_enabled:
            weight, theta, phi, lamb = self.deparameterize()

            axis = self.get_axis(theta, phi)

            cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
            cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
            lamb = rearrange(lamb, 'sg 1 -> 1 sg 1')
            weight = rearrange(weight, 'sg c -> 1 sg c')
            sg_val = weight * torch.exp(lamb * (cos_angle - 1))
            sg_val = torch.sum(sg_val, dim=1)

            return sg_val
        else:
            return torch.zeros_like(direction)

    def reg_loss(self):
        if self.is_enabled:
            val = self.deparameterize_weight()
            val = torch.sum(val)

            if self.w_lamb_reg > 0:
                lamb_val = self.deparameterize_lamb()
                lamb_val = torch.sum(lamb_val)
                val += lamb_val * self.w_lamb_reg

            return val
        else:
            return torch.tensor(0, device=self.weight.device, dtype=torch.float32)


# ====================== INCIDENT LIGHTING ======================


class GlobalIncidentLighting(nn.Module):
    def __init__(self,
                 value=Constant((-2, -2, -2), exp_val=True)):
        super().__init__()
        self.value = value

    @property
    def spp(self):
        return 1

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        return normal

    def pdf_direction(self, vpos, direction):
        # (bn, spp, 3, h, w)
        return torch.ones_like(vpos[:, :, :1, ...])

    def forward(self, direction):
        return self.value(direction.squeeze(0)).unsqueeze(0)

    def val_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())

    def pos_reg_loss(self):
        return torch.zeros_like(self.value.reg_loss())


# ====================== EMISSIVE LIGHTING ======================


class FusedSGGridPointLighting(nn.Module):
    def __init__(self,
                 num_lights,
                 position=((-0.1, -0.1, 0), (0.1, 0.1, 0)),
                 vpos_init=False,
                 im_init=False,
                 sg_col=6,
                 sg_row=2,
                 ch=3,
                 single_color=False):
        super().__init__()

        self.num_lights = num_lights

        self.initialized = not vpos_init
        self.im_init = im_init

        self.position = nn.Parameter(self.generate_grid(np.array(position)))

        # Value init
        self.sg_col = sg_col
        self.sg_row = sg_row
        self.SGNum = self.sg_col * self.sg_row

        self.single_color = single_color
        self.COLORNum = self.SGNum
        if self.single_color:
            self.COLORNum = 1

        self.ch = ch

        self.nearest_dist_sqr = None

        self.weight, self.theta, self.phi, self.lamb = self.init_sg_grid()

        is_enabled = torch.ones(self.spp, dtype=torch.bool)
        self.register_buffer('is_enabled', is_enabled)

    def init_sg_grid(self):
        phiCenter = ((np.arange(self.sg_col) + 0.5) / self.sg_col - 0.5) * np.pi * 2
        thetaCenter = (np.arange(self.sg_row) + 0.5) / self.sg_row * np.pi / 2.0

        phiCenter, thetaCenter = np.meshgrid(phiCenter, thetaCenter)

        thetaCenter = thetaCenter.reshape(1, self.SGNum, 1).astype(np.float32)
        thetaCenter = torch.from_numpy(thetaCenter).expand([1, self.SGNum, 1])

        phiCenter = phiCenter.reshape(1, self.SGNum, 1).astype(np.float32)
        phiCenter = torch.from_numpy(phiCenter).expand([1, self.SGNum, 1])

        thetaRange = (np.pi / 2 / self.sg_row) * 1.5
        phiRange = (2 * np.pi / self.sg_col) * 1.5

        self.register_buffer('thetaCenter', thetaCenter)
        self.register_buffer('phiCenter', phiCenter)
        self.register_buffer('thetaRange', torch.tensor(thetaRange))
        self.register_buffer('phiRange', torch.tensor(phiRange))

        weight = nn.Parameter(torch.ones((self.spp, self.COLORNum, self.ch), dtype=torch.float32) * (-4))
        theta = nn.Parameter(torch.zeros((self.spp, self.SGNum, 1), dtype=torch.float32))
        phi = nn.Parameter(torch.zeros((self.spp, self.SGNum, 1), dtype=torch.float32))
        lamb = nn.Parameter(torch.log(torch.ones(self.spp, self.SGNum, 1) * np.pi / self.sg_row))

        return weight, theta, phi, lamb

    def sample_points(self, start, end, num, *args, **kwargs):
        extra_kwargs = {"dtype": np.float32}
        extra_kwargs.update(kwargs)
        if num == 1:
            return np.array([(start + end) / 2], *args, **extra_kwargs)
        else:
            return np.linspace(start, end, num, *args, **extra_kwargs)

    def generate_grid(self, position):
        if isinstance(self.num_lights, int):
            x = self.sample_points(position[0][0], position[1][0], self.num_lights)
            y = self.sample_points(position[0][1], position[1][1], self.num_lights)
            z = self.sample_points(position[0][2], position[1][2], self.num_lights)
            positions = torch.from_numpy(np.stack((x, y, z), axis=-1).reshape(-1, 3))
        elif isinstance(self.num_lights, (tuple, list)):
            x = self.sample_points(position[0][0], position[1][0], self.num_lights[0])
            y = self.sample_points(position[0][1], position[1][1], self.num_lights[1])
            z = self.sample_points(position[0][2], position[1][2], self.num_lights[2])
            positions = torch.from_numpy(np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3))
        else:
            raise NotImplementedError()
        return positions

    def position_init(self, vpos, normal, image):
        if self.initialized:
            return

        if self.im_init and image is not None:
            # Choose N from the brightest pixels
            potential_emitter = image
            potential_emitter[torch.all(potential_emitter < 0.90, dim=1, keepdim=True).expand_as(potential_emitter)] = 0

            intensities = potential_emitter.sum(dim=1)
            potential_emitter = intensities > torch.quantile(intensities, 0.98)

            potential_emitter = potential_emitter[0].nonzero()
            rand_indices = torch.randperm(len(potential_emitter))

            selected_pixels = potential_emitter[rand_indices[:self.spp]]
            u, v = selected_pixels[:, 0], selected_pixels[:, 1]

            offset = 0.01 * normal[0, :, u, v].permute(1, 0).view(-1, 3)
            self.position.data = vpos[0, :, u, v].permute(1, 0).view(-1, 3) + offset
        else:
            # Grid initialization
            u = self.sample_points(0, vpos.shape[-2] - 1, self.num_lights[0], dtype=np.int32)
            v = self.sample_points(0, vpos.shape[-1] - 1, self.num_lights[1], dtype=np.int32)
            u, v = np.meshgrid(u, v)

            offset = 0.01 * normal[0, :, u, v].permute(1, 2, 0).view(-1, 3)
            self.position.data = vpos[0, :, u, v].permute(1, 2, 0).view(-1, 3) + offset

        self.initialized = True

    @property
    def spp(self):
        if isinstance(self.num_lights, int):
            return self.num_lights
        elif isinstance(self.num_lights, (tuple, list)):
            return int(np.prod(self.num_lights))
        else:
            raise NotImplementedError()

    def sample_direction(self, vpos, normal):
        # (bn, spp, 3, h, w)
        return torch.nn.functional.normalize(self.position[None, :, :, None, None] - vpos, dim=2)

    def pdf_direction(self, vpos, direction):
        # (bn, spp, 3, h, w)
        dist_sqr = torch.sum(torch.square(self.position[None, :, :, None, None] - vpos), dim=2, keepdim=True)
        self.nearest_dist_sqr = dist_sqr.permute(1,0,2,3,4).view(self.spp, -1).min(dim=1).values
        return dist_sqr

    def deparameterize(self):
        theta = self.thetaRange * torch.tanh(self.theta) + self.thetaCenter

        phi = self.phiRange * torch.tanh(self.phi) + self.phiCenter
        lamb = self.deparameterize_lamb()

        weight = self.deparameterize_weight()

        return weight, theta, phi, lamb

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def deparameterize_lamb(self):
        lamb = torch.exp(self.lamb)
        return lamb

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.cos(phi)
        axisY = torch.sin(theta) * torch.sin(phi)
        axisZ = torch.cos(theta)
        axis = torch.cat([axisX, axisY, axisZ], dim=2)
        return axis

    def forward(self, direction):
        weight, theta, phi, lamb = self.deparameterize()

        axis = self.get_axis(theta, phi)

        cos_angle = einsum(direction, axis, 'l b c, l sg c -> l b sg')
        cos_angle = rearrange(cos_angle, 'l b sg -> l b sg 1')
        lamb = rearrange(lamb, 'l sg 1 -> l 1 sg 1')
        weight = rearrange(weight, 'l sg c -> l 1 sg c')
        sg_val = weight * torch.exp(lamb * (cos_angle - 1))
        sg_val = torch.sum(sg_val, dim=2)

        # Mask disabled lights
        sg_val = rearrange(self.is_enabled, 'l -> l 1 1') * sg_val

        # # Sum over the lights
        # sg_val = torch.sum(sg_val, dim=0)
        return sg_val

    def val_reg_loss(self):
        val = self.deparameterize_weight()

        # Mask disabled lights
        val = rearrange(self.is_enabled, 'l -> l 1 1') * val

        val = torch.mean(val) * 3
        return val

    def pos_reg_loss(self):
        pos = 1 / self.nearest_dist_sqr.clamp(min=1e-6)

        # Mask disabled lights
        pos = rearrange(self.is_enabled, 'l -> l 1 1') * pos

        return torch.mean(pos)


# ====================== COMPOSITION ======================


class ComposeLighting(nn.Module):
    def __init__(self, lightings):
        super().__init__()
        self.lightings = nn.ModuleDict(lightings)

    @property
    def spp(self):
        return sum((lighting.spp for lighting in self.lighting_values))

    @property
    def sub_spps(self):
        return [lighting.spp for lighting in self.lighting_values]

    @property
    def lighting_values(self):
        return self.lightings.values()

    def position_init(self, vpos, normal, image):
        """
        Initialize the position of the lightings
        :param vpos:
        :param normal:
        :param image:
        :return:
        """
        for lighting in self.lighting_values:
            if hasattr(lighting, "position_init"):
                lighting.position_init(vpos=vpos, normal=normal, image=image)

    def sample_direction(self, vpos, normal):
        """
        Sample directions for each light sources separately
        :param vpos: BS x SPP x 3 x H x W
        :param normal: BS x SPP x 3 x H x W
        :return:
        """
        return torch.cat([lighting.sample_direction(vpos=vpos, normal=normal) for lighting in self.lighting_values], dim=1)

    def pdf_direction(self, vpos, direction):
        """
        Sample directions for each light sources separately
        :param vpos: BS x SPP x 3 x H x W
        :return:
        """
        return torch.cat([lighting.pdf_direction(vpos=vpos, direction=dir) for lighting, dir in zip(self.lighting_values, torch.split(direction, self.sub_spps, dim=1))], dim=1)

    def forward(self, direction):
        """
        Each direction goes for each light sources separately
        :param direction: N x 3
        :return:
        """
        return torch.cat([lighting(direction=dir) for lighting, dir in zip(self.lighting_values, torch.split(direction, self.sub_spps))], dim=0)

    def val_reg_loss(self):
        """
        Calculate the regularization loss for each light sources separately
        :return:
        """
        val_regs = torch.stack([lighting.val_reg_loss() for lighting in self.lighting_values], dim=0)
        return torch.sum(torch.tensor(self.sub_spps, device=val_regs.device) * val_regs) / self.spp

    def pos_reg_loss(self):
        """
        Calculate the regularization loss for each light sources separately
        :return:
        """
        pos_regs = torch.stack([lighting.pos_reg_loss() for lighting in self.lighting_values], dim=0)
        return torch.sum(torch.tensor(self.sub_spps, device=pos_regs.device) * pos_regs) / self.spp

