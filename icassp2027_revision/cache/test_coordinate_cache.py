from __future__ import annotations

import math

import torch

from icassp2027_revision.cache.coordinate_cache import (
    CoordinateConsistentRegionCache,
    motion_bounded_lifetime,
    pose_to_matrix,
    warp_bev,
)


def _peak_xy(tensor: torch.Tensor) -> tuple[int, int]:
    flat = int(torch.argmax(tensor))
    width = tensor.shape[-1]
    return flat // width, flat % width


def test_identity_warp() -> None:
    x = torch.randn(2, 20, 30)
    pose = pose_to_matrix(1.0, -2.0, 0.3)
    y = warp_bev(x, pose, pose, (-15, 15), (-10, 10))
    assert torch.allclose(x, y, atol=2e-5), (x - y).abs().max()


def test_translation_direction() -> None:
    x = torch.zeros(1, 21, 21)
    x[0, 10, 10] = 1.0
    source = pose_to_matrix(0.0, 0.0, 0.0)
    # Target ego moved +2 m in world x. A static source feature appears 2 cells
    # behind in target coordinates for a 1 m cell grid.
    target = pose_to_matrix(2.0, 0.0, 0.0)
    y = warp_bev(x, source, target, (-10.5, 10.5), (-10.5, 10.5), mode="nearest")
    assert _peak_xy(y) == (10, 8), _peak_xy(y)


def test_rotation_preserves_center() -> None:
    x = torch.zeros(1, 21, 21)
    x[0, 10, 10] = 1.0
    source = pose_to_matrix(0.0, 0.0, 0.0)
    target = pose_to_matrix(0.0, 0.0, math.pi / 2)
    y = warp_bev(x, source, target, (-10.5, 10.5), (-10.5, 10.5), mode="nearest")
    assert _peak_xy(y) == (10, 10), _peak_xy(y)


def test_mixed_age_state_moves_together() -> None:
    cache = CoordinateConsistentRegionCache(
        (-10.5, 10.5), (-10.5, 10.5), region_hw=(3, 3), age_weight=0.3
    )
    fresh0 = torch.zeros(1, 21, 21)
    fresh0[0, 10, 10] = 1.0
    cache.step(fresh0, pose_to_matrix(0, 0, 0), timestamp=0, budget_fraction=1.0)

    fresh1 = torch.zeros_like(fresh0)
    out = cache.step(
        fresh1, pose_to_matrix(2, 0, 0), timestamp=1, budget_fraction=0.05
    )
    # Feature and source-time metadata are transported by the same transform.
    feature_peak = _peak_xy(out.feature)
    old_time_mask = out.source_time[0] == 0
    assert old_time_mask[feature_peak], (feature_peak, out.source_time[0, feature_peak[0], feature_peak[1]])
    assert out.selected_regions.sum() == max(1, round(0.05 * out.selected_regions.numel()))



def test_controlled_selection_modes() -> None:
    for mode in ("direct", "tx_delta", "random", "oldest"):
        torch.manual_seed(2027)
        cache = CoordinateConsistentRegionCache(
            (-8, 8), (-8, 8), region_hw=(4, 4), age_weight=0.0
        )
        pose0 = pose_to_matrix(0, 0, 0)
        pose1 = pose_to_matrix(1, 0, 0)
        fresh0 = torch.zeros(2, 16, 16)
        fresh1 = fresh0.clone()
        fresh1[:, 4:8, 8:12] = 2.0
        cache.step(fresh0, pose0, timestamp=0, budget_fraction=1.0)
        out = cache.step(
            fresh1, pose1, timestamp=1, budget_fraction=0.25,
            selection_mode=mode,
        )
        assert int(out.selected_regions.sum()) == 4, mode
        assert torch.isfinite(out.score).all(), mode


def test_motion_bounded_lifetime() -> None:
    lifetime = motion_bounded_lifetime(
        relative_speed_mps=1.0,
        yaw_rate_rps=0.0,
        region_radius_m=2.0,
        frame_interval_s=0.1,
        max_accel_mps2=0.0,
        pose_error_m=0.0,
        max_drift_m=0.25,
    )
    assert lifetime == 2


def test_expiration_and_keyframe_controls() -> None:
    cache = CoordinateConsistentRegionCache(
        (-8, 8), (-8, 8), region_hw=(4, 4), age_weight=0.0
    )
    pose = pose_to_matrix(0, 0, 0)
    fresh0 = torch.zeros(2, 16, 16)
    cache.step(fresh0, pose, timestamp=0, budget_fraction=1.0)

    # Age one is still admissible when max_age=1.
    out1 = cache.step(
        fresh0, pose, timestamp=1, budget_fraction=0.25, max_age=1
    )
    assert not out1.expired_regions.any()

    # Age two exceeds the lifetime and all regions become mandatory.
    out2 = cache.step(
        fresh0, pose, timestamp=2, budget_fraction=0.25, max_age=1,
        keyframe_on_expiration_overflow=True,
    )
    assert int(out2.expired_regions.sum()) == 12
    assert out2.keyframe_reason == "expiration_overflow"
    assert out2.selected_regions.all()

    # An abrupt receiver residual triggers a complete keyframe refresh.
    fresh3 = torch.ones_like(fresh0)
    out3 = cache.step(
        fresh3, pose, timestamp=3, budget_fraction=0.25,
        keyframe_score_threshold=0.5,
        keyframe_ratio_threshold=0.5,
    )
    assert out3.keyframe
    assert out3.keyframe_reason == "abrupt_change"
    assert out3.selected_regions.all()


if __name__ == "__main__":
    test_identity_warp()
    test_translation_direction()
    test_rotation_preserves_center()
    test_mixed_age_state_moves_together()
    test_controlled_selection_modes()
    test_motion_bounded_lifetime()
    test_expiration_and_keyframe_controls()
    print("coordinate cache tests: PASS")
