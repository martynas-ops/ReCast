"""Coordinate-consistent mixed-age BEV feature cache.

The cache is canonicalized into the ego frame of the latest processed frame.
Before scoring or fusion at the next frame, cached features, validity, age, and
source-time maps are transformed together into the new ego frame.  This avoids
the invalid state in which regions refreshed at different times remain in
different historical ego coordinate systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def pose_to_matrix(x: float, y: float, yaw_rad: float, *, device=None,
                   dtype=torch.float32) -> torch.Tensor:
    """Return a homogeneous SE(2) transform from local coordinates to world."""
    c = torch.cos(torch.as_tensor(yaw_rad, device=device, dtype=dtype))
    s = torch.sin(torch.as_tensor(yaw_rad, device=device, dtype=dtype))
    out = torch.eye(3, device=device, dtype=dtype)
    out[0, 0], out[0, 1], out[0, 2] = c, -s, x
    out[1, 0], out[1, 1], out[1, 2] = s, c, y
    return out


def _metric_grid(height: int, width: int, x_range: tuple[float, float],
                 y_range: tuple[float, float], *, device, dtype) -> torch.Tensor:
    """Homogeneous target-cell centers with shape ``[3, H*W]``."""
    xs = torch.linspace(
        x_range[0] + (x_range[1] - x_range[0]) / (2 * width),
        x_range[1] - (x_range[1] - x_range[0]) / (2 * width),
        width,
        device=device,
        dtype=dtype,
    )
    ys = torch.linspace(
        y_range[0] + (y_range[1] - y_range[0]) / (2 * height),
        y_range[1] - (y_range[1] - y_range[0]) / (2 * height),
        height,
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)))


def warp_bev(
    tensor: torch.Tensor,
    source_pose_world: torch.Tensor,
    target_pose_world: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resample a source-ego BEV tensor into a target-ego coordinate frame.

    Args:
        tensor: ``[C,H,W]`` field expressed in ``source_pose_world``.
        source_pose_world: local-source to world homogeneous transform.
        target_pose_world: local-target to world homogeneous transform.
        x_range/y_range: metric extent represented by the tensor.
        mode: interpolation passed to ``grid_sample``.
    """
    if tensor.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {tuple(tensor.shape)}")
    channels, height, width = tensor.shape
    del channels
    target_points = _metric_grid(
        height, width, x_range, y_range, device=tensor.device, dtype=tensor.dtype
    )
    # target-local -> world -> source-local (inverse map for grid_sample)
    target_to_source = torch.linalg.inv(source_pose_world) @ target_pose_world
    source_points = target_to_source @ target_points
    sx = source_points[0].reshape(height, width)
    sy = source_points[1].reshape(height, width)
    nx = 2.0 * (sx - x_range[0]) / (x_range[1] - x_range[0]) - 1.0
    ny = 2.0 * (sy - y_range[0]) / (y_range[1] - y_range[0]) - 1.0
    grid = torch.stack((nx, ny), dim=-1).unsqueeze(0)
    return F.grid_sample(
        tensor.unsqueeze(0),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)


def _expand_regions(mask: torch.Tensor, height: int, width: int,
                    region_h: int, region_w: int) -> torch.Tensor:
    cell_mask = mask.repeat_interleave(region_h, 0).repeat_interleave(region_w, 1)
    return cell_mask[:height, :width]


@dataclass
class CacheOutput:
    feature: torch.Tensor
    selected_regions: torch.Tensor
    score: torch.Tensor
    age: torch.Tensor
    source_time: torch.Tensor
    valid: torch.Tensor


class CoordinateConsistentRegionCache:
    """Stateful cache for one collaborator and one ego sequence.

    Every call first reprojects the complete cache state from the previous ego
    frame into the current ego frame.  The acquisition-time map is transported
    with the features, so mixed-age content remains auditable after reprojection.
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        region_hw: tuple[int, int] = (4, 4),
        age_weight: float = 0.3,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.region_h, self.region_w = region_hw
        self.age_weight = age_weight
        self.feature: Optional[torch.Tensor] = None
        self.valid: Optional[torch.Tensor] = None
        self.age: Optional[torch.Tensor] = None
        self.source_time: Optional[torch.Tensor] = None
        self.pose_world: Optional[torch.Tensor] = None
        self.previous_fresh: Optional[torch.Tensor] = None
        self.previous_fresh_pose_world: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.feature = self.valid = self.age = self.source_time = self.pose_world = None

        self.previous_fresh = self.previous_fresh_pose_world = None
    @staticmethod
    def _region_mean(value: torch.Tensor, region_h: int, region_w: int) -> torch.Tensor:
        # Pad on the far edges so arbitrary feature sizes are supported.
        height, width = value.shape[-2:]
        pad_h = (-height) % region_h
        pad_w = (-width) % region_w
        value = F.pad(value, (0, pad_w, 0, pad_h))
        return F.avg_pool2d(value, kernel_size=(region_h, region_w),
                            stride=(region_h, region_w))

    def step(self, fresh: torch.Tensor, ego_pose_world: torch.Tensor,
             timestamp: int, budget_fraction: float,
             force_refresh: Optional[bool] = None,
             transport: bool = True,
             selection_mode: str = "direct",
             selection_rule: str = "topb",
             selection_threshold: Optional[float] = None,
             score_mode: str = "mean") -> CacheOutput:
        if fresh.ndim != 3:
            raise ValueError("fresh must have shape [C,H,W]")
        if force_refresh is None and not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0,1]")
        if selection_mode not in {"direct", "tx_delta", "random", "oldest"}:
            raise ValueError(f"unknown selection_mode: {selection_mode}")
        if selection_rule not in {"topb", "threshold"}:
            raise ValueError(f"unknown selection_rule: {selection_rule}")
        if selection_rule == "threshold" and selection_threshold is None:
            raise ValueError("selection_threshold is required for threshold selection")
        if score_mode not in {"mean", "max", "saliency", "normalized"}:
            raise ValueError(f"unknown score_mode: {score_mode}")

        _, height, width = fresh.shape
        if self.feature is None:
            self.feature = fresh.detach().clone()
            self.valid = torch.ones((1, height, width), device=fresh.device, dtype=fresh.dtype)
            self.age = torch.zeros_like(self.valid)
            self.source_time = torch.full_like(self.valid, float(timestamp))
            self.pose_world = ego_pose_world.detach().clone()
            self.previous_fresh = fresh.detach().clone()
            self.previous_fresh_pose_world = ego_pose_world.detach().clone()
            rh = (height + self.region_h - 1) // self.region_h
            rw = (width + self.region_w - 1) // self.region_w
            selected = torch.ones((rh, rw), device=fresh.device, dtype=torch.bool)
            score = torch.full((rh, rw), float("inf"), device=fresh.device,
                               dtype=fresh.dtype)
            return CacheOutput(self.feature, selected, score, self.age,
                               self.source_time, self.valid)

        assert self.pose_world is not None
        if transport:
            self.feature = warp_bev(
                self.feature, self.pose_world, ego_pose_world,
                self.x_range, self.y_range, mode="bilinear"
            )
            self.valid = warp_bev(
                self.valid, self.pose_world, ego_pose_world,
                self.x_range, self.y_range, mode="nearest"
            )
            self.age = warp_bev(
                self.age, self.pose_world, ego_pose_world,
                self.x_range, self.y_range, mode="nearest"
            )
            self.source_time = warp_bev(
                self.source_time, self.pose_world, ego_pose_world,
                self.x_range, self.y_range, mode="nearest"
            )
        self.pose_world = ego_pose_world.detach().clone()

        region_age = self._region_mean(self.age, self.region_h, self.region_w).squeeze(0)
        region_valid = self._region_mean(self.valid, self.region_h, self.region_w).squeeze(0)
        if selection_mode == "direct":
            residual = (fresh - self.feature).abs().mean(dim=0, keepdim=True)
            if score_mode == "max":
                score = F.max_pool2d(
                    residual, kernel_size=(self.region_h, self.region_w),
                    stride=(self.region_h, self.region_w), ceil_mode=True
                ).squeeze(0)
            elif score_mode == "saliency":
                saliency = (fresh.abs() + self.feature.abs()).mean(dim=0, keepdim=True)
                score = self._region_mean(
                    residual * (saliency + 1e-3).sqrt(),
                    self.region_h, self.region_w).squeeze(0)
            elif score_mode == "normalized":
                denominator = (fresh.abs() + self.feature.abs()).mean(dim=0, keepdim=True)
                score = self._region_mean(
                    residual / (denominator + 1e-3),
                    self.region_h, self.region_w).squeeze(0)
            else:
                score = self._region_mean(residual, self.region_h, self.region_w).squeeze(0)
            score = score * (1.0 + self.age_weight * region_age)
        elif selection_mode == "tx_delta":
            assert self.previous_fresh is not None
            assert self.previous_fresh_pose_world is not None
            previous = self.previous_fresh
            if transport:
                previous = warp_bev(
                    previous,
                    self.previous_fresh_pose_world,
                    ego_pose_world,
                    self.x_range,
                    self.y_range,
                    mode="bilinear",
                )
            residual = (fresh - previous).abs().mean(dim=0, keepdim=True)
            score = self._region_mean(residual, self.region_h, self.region_w).squeeze(0)
        elif selection_mode == "oldest":
            score = region_age.clone()
        else:
            score = torch.rand_like(region_age)
        # Empty/out-of-view cache must be refilled before valid cached content.
        score = score + (region_valid < 0.5).to(score.dtype) * (score.max() + 1.0)

        n_regions = score.numel()
        if force_refresh is True:
            selected = torch.ones_like(score, dtype=torch.bool)
        elif force_refresh is False:
            selected = torch.zeros_like(score, dtype=torch.bool)
        else:
            if selection_rule == "topb":
                budget = max(1, min(n_regions, round(budget_fraction * n_regions)))
                selected = torch.zeros_like(score, dtype=torch.bool)
                top = torch.topk(score.flatten(), k=budget, largest=True).indices
                selected.flatten()[top] = True
            else:
                assert selection_threshold is not None
                selected = score >= selection_threshold
        cell_selected = _expand_regions(
            selected, height, width, self.region_h, self.region_w
        ).unsqueeze(0)

        self.feature = torch.where(cell_selected, fresh.detach(), self.feature)
        self.valid = torch.where(cell_selected, torch.ones_like(self.valid), self.valid)
        self.age = torch.where(cell_selected, torch.zeros_like(self.age), self.age + 1.0)
        self.source_time = torch.where(
            cell_selected,
            torch.full_like(self.source_time, float(timestamp)),
            self.source_time,
        )
        self.previous_fresh = fresh.detach().clone()
        self.previous_fresh_pose_world = ego_pose_world.detach().clone()
        return CacheOutput(self.feature, selected, score, self.age,
                           self.source_time, self.valid)
