import torch
from torch.nn.utils import prune


class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "structured"

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        keep = torch.any(tensor > self.threshold, dim=1).any(dim=1)
        return keep

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            # Apply the new pruning method, either from scratch or on top of
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            method._tensor_name = name

            return method

        importance_scores = getattr(module, name)

        method = _get_composite_method(cls, module, name, *args, **kwargs)

        mask = method.compute_mask(importance_scores, default_mask=None) & module.is_enabled
        module.is_enabled = mask & module.is_enabled

        return method
