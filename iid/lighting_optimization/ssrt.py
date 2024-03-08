"""
Adapted from https://github.com/jingsenzhu/IndoorInverseRendering/blob/main/lightnet/models/render/ssrt.py
"""
import torch
import torch.nn as nn


def transform(pos: torch.Tensor, mat: torch.Tensor):
    """
    pos: (bn, 3)
    mat: (4, 4)
    """
    pos4 = torch.cat([pos, torch.ones(pos.size(0), 1, device=pos.device)], dim=1).unsqueeze(-1)  # (bn, 4, 1)
    pos4 = (mat.unsqueeze(0) @ pos4)[:, :, 0]  # (bn, 4)
    pos4 = pos4 / pos4[:, -1:]
    return pos4[:, :-1]  # (bn, 3)


def any_within_screen(pos):
    """
    pos: (bn, 3)
    """
    return torch.any(
        (pos[:, 0] >= 0) & (pos[:, 0] <= 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= 1) & (pos[:, 2] > 0) & (pos[:, 2] < 1))


def march_next(cur_pos: torch.Tensor, ray_dir: torch.Tensor, cur_x: torch.Tensor, cur_y: torch.Tensor, h, w,
               step_x: torch.Tensor, step_y: torch.Tensor):
    """
    cur_pos, ray_dir: (bn, 3)
    cur_x, cur_y, step_x, step_y: (bn)
    """
    next_bx = (cur_x + step_x) / w
    next_by = (cur_y + step_y) / h
    next_cell_boundary = torch.stack([next_bx, 1 - next_by], dim=1)  # (bn, 2)
    step_ratio = (next_cell_boundary - cur_pos[:, :2]) / ray_dir[:, :2]  # (bn, 2)
    dx = torch.where(step_ratio[:, 0] > step_ratio[:, 1], torch.zeros_like(step_x), step_x)  # (bn)
    dy = torch.where(step_ratio[:, 0] > step_ratio[:, 1], step_y, torch.zeros_like(step_y))  # (bn)
    return cur_pos + ray_dir * torch.min(step_ratio, dim=1, keepdim=True).values, dx, dy


def ssrt(depth: torch.Tensor, normal: torch.Tensor, indices: torch.LongTensor, proj: torch.Tensor, x: torch.Tensor,
         y: torch.Tensor, d: torch.Tensor, depth_start: torch.Tensor):
    """
    depth, normal: (sn, ch, h, w)
    proj: (4, 4)
    x, y: (bn)
    d: (bn, 3)
    depth_start: (bn)
    """
    unproj = torch.inverse(proj)
    h, w = depth.shape[2:]
    bn = x.size(0)
    uv = torch.stack([(x + 0.5) / w, 1 - (y + 0.5) / h], dim=1)  # (bn, 2)
    uv = uv * 2 - 1
    depth_start = depth_start * 2 - 1
    proj_pos = torch.stack([uv[:, 0], uv[:, 1], depth_start], dim=1)  # (bn, 3)
    view_pos = transform(proj_pos, unproj)
    # print(view_pos[0])
    next_pos = transform(view_pos + d * 0.001, proj)
    d_proj = next_pos - proj_pos  # (bn, 3)
    cur_x, cur_y = x, y
    cur_pos = (proj_pos + 1) * 0.5
    step_x = torch.where(d_proj[:, 0] >= 0, torch.ones_like(x), -torch.ones_like(x))
    step_y = torch.where(d_proj[:, 1] >= 0, -torch.ones_like(y), torch.ones_like(y))
    cur_pos, dx, dy = march_next(cur_pos, d_proj, cur_x, cur_y, h, w, step_x, step_y)
    cur_x += dx
    cur_y += dy
    mask = torch.zeros_like(x).bool()  # (bn)
    results = torch.zeros(bn, 2, dtype=torch.long, device=x.device)  # (bn, 2)
    dz = torch.zeros(bn, 1, device=x.device)
    while any_within_screen(cur_pos) and not torch.all(mask):
        mask_within_screen = (cur_x >= 0) & (cur_x < w) & (cur_y >= 0) & (cur_y < h)
        i_screen = indices[mask_within_screen]
        x_screen = cur_x[mask_within_screen]
        y_screen = cur_y[mask_within_screen]
        z = depth[i_screen, 0, y_screen, x_screen]
        pz = cur_pos[mask_within_screen, 2]
        mask_step = (pz >= z) & (
                    torch.sum(d[mask_within_screen] * normal[i_screen, :, y_screen, x_screen], dim=1) <= 0) & ~mask[
            mask_within_screen]
        mask[mask_within_screen] |= mask_step
        results[mask_within_screen] = torch.where(mask_step.unsqueeze(-1), torch.stack([x_screen, y_screen], dim=-1),
                                                  results[mask_within_screen])
        dz[mask_within_screen] = torch.where(mask_step.unsqueeze(-1), (pz - z).unsqueeze(-1), dz[mask_within_screen])
        cur_pos, dx, dy = march_next(cur_pos, d_proj, cur_x, cur_y, h, w, step_x, step_y)
        cur_x += dx
        cur_y += dy

    return results, mask, dz


class SSRTEngine(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, depth: torch.Tensor, normal: torch.Tensor, indices: torch.LongTensor, proj: torch.Tensor,
                x: torch.Tensor, y: torch.Tensor, d: torch.Tensor, depth_start: torch.Tensor):
        return ssrt(depth[0, ...], normal[0, ...], indices, proj[0, ...], x, y, d, depth_start)