"""
Adapted from https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/lightnet/models/render/__init__.py
"""
import math
import glm

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from iid.lighting_optimization.brdf import pdf_ggx, eval_ggx, pdf_diffuse, eval_diffuse, pdf_disney, eval_disney
from iid.lighting_optimization.ssrt import ssrt


class IIR_SSRT_RenderLayer(nn.Module):
    def __init__(self,
                 imWidth = 320,
                 imHeight = 240,
                 fov=85,
                 cameraPos = [0, 0, 0],
                 brdf_type = "ggx",
                 spp = 1,
                 double_sided=True,
                 use_ssrt=False,
                 use_specular=False,):
        super().__init__()
        self.imHeight = imHeight
        self.imWidth = imWidth

        self.use_ssrt = use_ssrt
        self.use_specular = use_specular
        self.double_sided = double_sided

        self.fov = fov/180.0 * np.pi
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        v = torch.from_numpy(v)
        pCoord = torch.from_numpy(pCoord)

        up = torch.Tensor([0,1,0])
        # assert(brdf_type in ["disney", "ggx"])
        self.brdf_type = brdf_type
        self.spp = spp

        self.register_buffer('v', v, persistent=False)
        self.register_buffer('pCoord', pCoord, persistent=False)
        self.register_buffer('up', up, persistent=False)

    def forward(
            self,
            lighting_model: nn.Module,
            albedo: torch.Tensor,
            rough: torch.Tensor,
            metal: torch.Tensor,
            normal: torch.Tensor,
            vpos: torch.Tensor,
    ):
        """
        Render according to material, normal and lighting conditions
        Params:
            model: NeRF model to predict lights
            im, albedo, normal, rough, metal, vpos: (bn, c, h, w)
        """
        bn, _, row, col = albedo.shape
        assert (bn == 1)  # Only support bn = 1
        assert (row == self.imHeight and col == self.imWidth), f"row: {row}, col: {col}, imHeight: {self.imHeight}, imWidth: {self.imWidth}"
        dev = albedo.device

        # No clipping
        center_x = col // 2
        center_y = row // 2
        radius = max(center_x, center_y)

        #############################################
        ########## Incident Sampling Start ##########
        #############################################

        cx, cy, cz = create_frame(normal)
        fx, fy, fz = torch.split(torch.cat([cx, cy, cz], dim=0).permute(1,0,2,3), 1, dim=0)

        # ============ W_i - Direction to the camera =================
        wi_world = self.v
        wi = (wi_world[:, 0:1, ...] * fx +
              wi_world[:, 1:2, ...] * fy +
              wi_world[:, 2:3, ...] * fz)
        if self.double_sided:
            wi[:, 2, ...] = torch.abs(wi[:, 2, ...])

        wi_mask = torch.where(wi[:, 2:3, ...] < 1e-6, torch.zeros_like(wi[:, 2:3, ...]),
                              torch.ones_like(wi[:, 2:3, ...]))

        wi[:, 2, ...] = torch.clamp(wi[:, 2, ...], min=1e-3)
        wi = functional.normalize(wi, dim=1, eps=1e-6)
        wi = wi.unsqueeze(1)  # (bn, 1, 3, h, w)

        # Clipping
        left = max(center_x - radius, 0)
        right = center_x + radius
        top = max(center_y - radius, 0)
        bottom = center_y + radius
        wi = wi[:, :, :, top:bottom, left:right]
        wi_mask = wi_mask[:, :, top:bottom, left:right]
        cx = cx[:, :, top:bottom, left:right]
        cy = cy[:, :, top:bottom, left:right]
        cz = cz[:, :, top:bottom, left:right]
        albedo_clip = albedo[:, :, top:bottom, left:right]
        metal_clip = metal[:, :, top:bottom, left:right]
        rough_clip = rough[:, :, top:bottom, left:right]

        irow = wi.size(3)
        icol = wi.size(4)

        # ============ W_o - Direction to the light =================
        wo_emitter = lighting_model.sample_direction(vpos=vpos.unsqueeze(1), normal=normal.unsqueeze(1))
        wo = (wo_emitter[:, :, 0:1, ...] * fx.unsqueeze(1) +
              wo_emitter[:, :, 1:2, ...] * fy.unsqueeze(1) +
              wo_emitter[:, :, 2:3, ...] * fz.unsqueeze(1))
        if self.double_sided:
            wo[:, :, 2, ...] = torch.abs(wo[:, :, 2, ...].clone())

        # Convert to world space
        direction = wo_emitter
        direction = direction.permute(0, 1, 3, 4, 2)  # (bn, spp, h, w, 3)
        direction = direction.permute(1, 0, 2, 3, 4)  # (spp, bn, h, w, 3)
        direction = direction.reshape(lighting_model.spp, -1, 3)

        # W_o BRDF evaluation
        if self.brdf_type == "ggx":
            pdfs = pdf_ggx(albedo_clip, rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_ggx(albedo_clip, rough_clip, metal_clip, wi, wo)
        elif self.brdf_type == "diffuse":
            pdfs = pdf_diffuse(wi, wo)
            eval_diff, eval_spec, mask = eval_diffuse(albedo_clip, wi, wo)
        else:
            pdfs = pdf_disney(rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_disney(albedo_clip, rough_clip, metal_clip, wi, wo)

        # Since we are using emitter sampling, the sampling pdf is 1
        pdfs_brdf = torch.ones_like(pdfs)
        pdfs_brdf = torch.clamp(pdfs_brdf, min=0.001)

        brdfDiffuse = eval_diff.expand([1, lighting_model.spp, 3, irow, icol]) / pdfs_brdf
        brdfSpec = eval_spec.expand([1, lighting_model.spp, 3, irow, icol]) / pdfs_brdf
        # del ndl, pdfs, eval_diff, eval_spec
        #############################################
        ########### Incident Sampling End ###########
        #############################################

        # ##############################
        # ######### SSRT Start #########
        # ##############################
        if self.use_ssrt:
            direction = direction[-1:, ...]
            ssrt_spp = 1
            ssrt_direction = direction.reshape(-1, 3)

            fovy = 2 * math.atan(math.tan(self.fov / 2) / (self.imWidth / self.imHeight))
            depth = -vpos[0, ...].permute(1, 2, 0)  # (h, w, 3)
            depth[:, :, :-1] = 0
            dmin = torch.min(depth[:, :, -1])
            dmax = torch.max(depth[:, :, -1])
            depth /= dmax
            proj = glm.perspective(fovy, self.imWidth / self.imHeight, (dmin.item() / dmax.item()), 1.0)
            proj = torch.from_numpy(np.array(proj)).to(dev)

            depth = -depth
            depth = depth.view(-1, 3, 1)
            pos4 = torch.cat([depth, torch.ones(col * row, 1, 1, device=dev)], dim=1)
            pos4 = (proj.unsqueeze(0) @ pos4)[:, :, 0]
            # TODO: Debug this
            # pos4 = pos4 / pos4[:, -1:]
            depth = (pos4[:, :-1].view(row, col, 3) + 1) * 0.5

            depth = depth[:, :, -1].unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
            depth[torch.where(torch.isinf(depth))] = 1
            depth_start = depth.expand(1, ssrt_spp, row, col)
            depth_start = depth_start[:, :, top:bottom, left:right].flatten()
            Y = torch.arange(0, row, device=dev)
            X = torch.arange(0, col, device=dev)
            Y, X = torch.meshgrid(Y, X)  # (h, w)
            Y = Y[top:bottom, left:right]
            X = X[top:bottom, left:right]
            Y = Y.unsqueeze(0).expand(ssrt_spp, irow, icol).flatten()
            X = X.unsqueeze(0).expand(ssrt_spp, irow, icol).flatten()
            N = ssrt_direction.size(0)
            indices = torch.zeros(N, dtype=torch.long, device=dev)

            ssrt_uv, mask, dz = ssrt(depth, normal, indices, proj, X, Y, ssrt_direction, depth_start)

            ssrt_uv[~mask, ...] = -1
            uncertainty = torch.tanh(10 * dz)

            ssrt_uv = ssrt_uv.flip(1)  # xy->ij: x = j, y = i
            uncertainty[~mask, ...] = 1
            # ##############################
            # ########## SSRT End ##########
            # ##############################

            # ##############################
            # ###### Integrator Start ######
            # ##############################
            vpos = vpos[:, :, top:bottom, left:right]
            positions = vpos.unsqueeze(1).expand(1, self.spp, 3, irow, icol).permute(0, 1, 3, 4, 2).reshape(-1, 3)  # (bn*spp*h*w, 3)
            u = ssrt_uv[:, 0]
            v = ssrt_uv[:, 1]
            normals = normal[indices, :, u, v]
            roughs = rough[indices, :, u, v]
            Kd = albedo * (1 - metal)
            Ks = torch.lerp(torch.empty_like(albedo).fill_(0.04), albedo, metal)
            Kd = Kd[indices, :, u, v]
            Ks = Ks[indices, :, u, v]
            model_kwargs = {
                'uv': ssrt_uv, 'directions': direction,
                'positions': positions, 'index': indices,
                'uncertainty': uncertainty, 'normal': normals,
                'Kd': Kd, 'Ks': Ks, 'rough': roughs
            }

        light = lighting_model(direction=direction)

        pdf_emitter = lighting_model.pdf_direction(vpos=vpos.unsqueeze(1), direction=wo_emitter)
        pdf_emitter = torch.clamp(pdf_emitter, min=0.001)
        ndl = torch.clamp(wo[:, :, 2:, ...], min=0)

        # light = get_light_chunk(model, im, model_kwargs, direction.size(0), self.chunk)
        light = light.view(1, lighting_model.spp, irow, icol, 3)
        light = light.permute(0, 1, 4, 2, 3)  # (bn, spp, 3, h, w)

        light = light * ndl / pdf_emitter

        colorDiffuse = torch.mean(brdfDiffuse * light, dim=1)
        if self.use_specular:
            colorSpec = torch.mean(brdfSpec * light, dim=1)
        else:
            colorSpec = torch.zeros_like(colorDiffuse)

        shading = light
        shading = torch.mean(shading, dim=1)
        ##############################
        ####### Integrator End #######
        ##############################

        return colorDiffuse, colorSpec, wi_mask, shading


def create_frame(n: torch.Tensor, eps:float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    :param: n (bn, 3, ...)
    """
    z = functional.normalize(n, dim=1, eps=eps)
    sgn = torch.where(z[:,2,...] >= 0, 1.0, -1.0)
    a = -1.0 / (sgn + z[:,2,...])
    b = z[:,0,...] * z[:,1,...] * a
    x = torch.stack([1.0 + sgn * z[:,0,...] * z[:,0,...] * a, sgn * b, -sgn * z[:,0,...]], dim=1)
    y = torch.stack([b, sgn + z[:,1,...] * z[:,1,...] * a, -z[:,1,...]], dim=1)
    return x, y, z


def depth_to_vpos(depth: torch.Tensor, fov, permute=False, normalize=True) -> torch.Tensor:
    row, col = depth.shape
    fovx = math.radians(fov)
    fovy = 2 * math.atan(math.tan(fovx / 2) / (col / row))
    vpos = torch.zeros(row, col, 3, device=depth.device)
    if normalize:
        dmax = torch.max(depth)
        depth = depth / dmax
    Y = 1 - (torch.arange(row, device=depth.device) + 0.5) / row
    Y = Y * 2 - 1
    X = (torch.arange(col, device=depth.device) + 0.5) / col
    X = X * 2 - 1
    Y, X = torch.meshgrid(Y, X)
    vpos[:,:,0] = depth * X * math.tan(fovx / 2)
    vpos[:,:,1] = depth * Y * math.tan(fovy / 2)
    vpos[:,:,2] = -depth
    return vpos if not permute else vpos.permute(2,0,1)
