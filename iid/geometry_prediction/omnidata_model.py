import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional


class OmnidataPredictor(nn.Module):
    """
    Class handling the dataset generation and preparation
    """

    def __init__(self,
                 depth_ckpt,
                 normal_ckpt):
        super().__init__()

        self.depth_ckpt = depth_ckpt
        self.normal_ckpt = normal_ckpt

        self.normal_predictor, self.normal_data_transform = self.get_normal_predictor(self.normal_ckpt)
        self.depth_predictor, self.depth_data_transform = self.get_depth_predictor(self.depth_ckpt)

    def get_normal_predictor(self, ckpt_path):
        from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        trans_totensor = transforms.Compose([])

        return model, trans_totensor

    def get_depth_predictor(self, ckpt_path):
        from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        trans_totensor = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])
        return model, trans_totensor

    def forward(self, x):
        """
        Predicts normal and depth of an image
        :param x:
        :return: Cat of depth and normal
        """
        # Resize to 384x512
        original_shape = x.shape[-2:]
        image = functional.resize(x, [384, 512])

        # Predict the normal and depth
        normal = self.predict_normal(image)
        depth = self.predict_depth(image)
        output = torch.cat([depth, normal], dim=1)

        return functional.resize(output, original_shape)

    def predict_normal(self, image):
        im_tensor = self.normal_data_transform(image).cuda()
        with torch.no_grad():
            prediction = self.normal_predictor(im_tensor).clamp(min=0, max=1)
        # Changing to InteriorVerse convention
        prediction = -(prediction * 2 - 1)
        prediction[:, 0, ...] *= -1
        return prediction

    def predict_depth(self, image):
        im_tensor = self.depth_data_transform(image).cuda()
        with torch.no_grad():
            prediction = self.depth_predictor(im_tensor).clamp(min=0, max=1)
        return prediction.unsqueeze(1)
