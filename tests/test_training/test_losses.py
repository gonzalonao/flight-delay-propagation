"""Tests for the losses, especially the W2 multi-task combination.

Covers:
    - ``MultiTaskLoss`` with correct shapes, per-component weights,
      gradients and robustness to invalid shapes.
    - Compatibility of ``WeightedHuberLoss`` with multi-channel targets
      (it must slice to channel 0 = ArrDelay).
    - ``compute_node_targets_multi_channel`` produces the 3 channels in the
      expected range.
    - ``compute_bce_classification_metrics`` and its integration in
      ``compute_unified_metrics`` when BCE logits are passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.graph_builder import (
    NUM_TARGET_CHANNELS,
    TARGET_CHANNEL_ARR_DELAY,
    TARGET_CHANNEL_DEP_DELAY,
    TARGET_CHANNEL_PCT_DELAYED,
    compute_node_targets,
    compute_node_targets_multi_channel,
)
from src.evaluation.metrics import (
    compute_bce_classification_metrics,
    compute_unified_metrics,
)
from src.training.losses import MultiTaskLoss, WeightedHuberLoss

# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
def multi_task_batch():
    """Synthetic batch [N=8, H=5, 3] (pred + target) without NaN."""
    torch.manual_seed(42)
    n, h = 8, 5
    pred = torch.randn(n, h, 3)
    arr_target = torch.randn(n, h) * 10
    dep_target = torch.randn(n, h) * 8
    pct_target = torch.rand(n, h)  # already in [0, 1]
    target = torch.stack([arr_target, dep_target, pct_target], dim=-1)
    return pred, target


@pytest.fixture
def airport_map():
    return {"AAA": 0, "BBB": 1, "CCC": 2}


@pytest.fixture
def small_window_df():
    """Three airports, different delays for clearly distinct channels."""
    return pd.DataFrame(
        [
            # AAA: 4 flights, 3 delayed (>= 15 min), arr_mean = 25
            {"Dest": "AAA", "ArrDelay": 30.0, "DepDelay": 20.0},
            {"Dest": "AAA", "ArrDelay": 20.0, "DepDelay": 10.0},
            {"Dest": "AAA", "ArrDelay": 25.0, "DepDelay": 15.0},
            {"Dest": "AAA", "ArrDelay": 25.0, "DepDelay": 12.0},
            # BBB: 2 flights, 0 delayed, arr_mean = 5
            {"Dest": "BBB", "ArrDelay": 5.0, "DepDelay": 2.0},
            {"Dest": "BBB", "ArrDelay": 5.0, "DepDelay": 3.0},
            # CCC: 0 flights in the window → all zeros
        ]
    )


# -- compute_node_targets_multi_channel -------------------------------------


class TestComputeNodeTargetsMultiChannel:
    """The new target builder: [N, 3]."""

    def test_shape_and_channels(self, small_window_df, airport_map):
        out = compute_node_targets_multi_channel(small_window_df, airport_map)
        assert out.shape == (3, NUM_TARGET_CHANNELS)

        # AAA: arr=25, dep=14.25, pct = 4/4 = 1.0 (all >= 15 min)
        assert out[0, TARGET_CHANNEL_ARR_DELAY].item() == pytest.approx(25.0)
        assert out[0, TARGET_CHANNEL_DEP_DELAY].item() == pytest.approx(14.25)
        assert out[0, TARGET_CHANNEL_PCT_DELAYED].item() == pytest.approx(1.0)

        # BBB: arr=5, dep=2.5, pct = 0/2 = 0.0
        assert out[1, TARGET_CHANNEL_ARR_DELAY].item() == pytest.approx(5.0)
        assert out[1, TARGET_CHANNEL_DEP_DELAY].item() == pytest.approx(2.5)
        assert out[1, TARGET_CHANNEL_PCT_DELAYED].item() == pytest.approx(0.0)

        # CCC: no flights, all 0.
        assert torch.allclose(out[2], torch.zeros(3))

    def test_arr_channel_matches_single(self, small_window_df, airport_map):
        """The arr_delay channel must match the 1D version bit-for-bit."""
        single = compute_node_targets(small_window_df, airport_map)
        multi = compute_node_targets_multi_channel(small_window_df, airport_map)
        assert torch.allclose(single, multi[:, TARGET_CHANNEL_ARR_DELAY])

    def test_pct_uses_threshold(self, small_window_df, airport_map):
        """A higher threshold reduces the delayed percentage."""
        out_15 = compute_node_targets_multi_channel(
            small_window_df,
            airport_map,
            delay_threshold=15.0,
        )
        out_30 = compute_node_targets_multi_channel(
            small_window_df,
            airport_map,
            delay_threshold=30.0,
        )
        # AAA with threshold 30: only 1 flight (arr=30) qualifies → 0.25.
        assert out_15[0, TARGET_CHANNEL_PCT_DELAYED].item() == pytest.approx(1.0)
        assert out_30[0, TARGET_CHANNEL_PCT_DELAYED].item() == pytest.approx(0.25)

    def test_empty_df_returns_zeros(self, airport_map):
        empty = pd.DataFrame(columns=["Dest", "ArrDelay", "DepDelay"])
        out = compute_node_targets_multi_channel(empty, airport_map)
        assert out.shape == (3, 3)
        assert torch.allclose(out, torch.zeros(3, 3))


# -- MultiTaskLoss ----------------------------------------------------------


class TestMultiTaskLoss:
    """Combination Huber(arr) + aux·Huber(dep) + bce·BCE(pct)."""

    def test_returns_finite_scalar(self, multi_task_batch):
        pred, target = multi_task_batch
        loss = MultiTaskLoss()(pred, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_components_recorded(self, multi_task_batch):
        pred, target = multi_task_batch
        criterion = MultiTaskLoss()
        loss = criterion(pred, target)
        comps = criterion.last_components
        for k in ("arr_huber", "dep_huber", "pct_bce", "total"):
            assert k in comps
            assert np.isfinite(comps[k])
        # The recorded total matches the backward-able value (to float precision).
        assert comps["total"] == pytest.approx(loss.item(), rel=1e-5)

    def test_zero_aux_weight_ignores_dep(self, multi_task_batch):
        """With aux_weight=0, changing the dep target does not affect the loss."""
        pred, target = multi_task_batch
        criterion = MultiTaskLoss(aux_weight=0.0, bce_weight=0.0)
        loss_a = criterion(pred, target).item()

        target_b = target.clone()
        target_b[..., TARGET_CHANNEL_DEP_DELAY] *= 100.0
        loss_b = criterion(pred, target_b).item()

        assert loss_a == pytest.approx(loss_b, rel=1e-5)

    def test_zero_bce_weight_ignores_pct(self, multi_task_batch):
        pred, target = multi_task_batch
        criterion = MultiTaskLoss(aux_weight=0.0, bce_weight=0.0)
        loss_a = criterion(pred, target).item()

        target_b = target.clone()
        target_b[..., TARGET_CHANNEL_PCT_DELAYED] = (
            1.0 - target_b[..., TARGET_CHANNEL_PCT_DELAYED]
        )
        loss_b = criterion(pred, target_b).item()

        assert loss_a == pytest.approx(loss_b, rel=1e-5)

    def test_gradients_flow_through_all_channels(self, multi_task_batch):
        """The backward must propagate gradient to the 3 channels of pred."""
        pred, target = multi_task_batch
        pred = pred.clone().requires_grad_(True)
        loss = MultiTaskLoss()(pred, target)
        loss.backward()
        # Each channel receives a non-zero gradient (the fixture's
        # randomness guarantees the target differs from pred in each channel).
        for c in range(3):
            assert pred.grad[..., c].abs().sum() > 0, f"No gradient in channel {c}"

    def test_horizon_weights_emphasis(self, multi_task_batch):
        """Raising the last horizon's weight increases its contribution."""
        pred, target = multi_task_batch
        h = pred.shape[1]
        # Same magnitude across all horizons with an extreme weight.
        emphasised = MultiTaskLoss(
            horizon_weights=[1.0] * (h - 1) + [10.0],
            aux_weight=0.0,
            bce_weight=0.0,
        )
        flat = MultiTaskLoss(
            horizon_weights=[1.0] * h,
            aux_weight=0.0,
            bce_weight=0.0,
        )
        loss_e = emphasised(pred, target).item()
        loss_f = flat(pred, target).item()
        # It is not trivial to predict which is larger without knowing the
        # sign of the error, but both must be positive and different.
        assert loss_e != pytest.approx(loss_f, rel=1e-3)
        assert loss_e > 0 and loss_f > 0

    def test_rejects_non_3d_input(self):
        criterion = MultiTaskLoss()
        with pytest.raises(ValueError, match="3D"):
            criterion(torch.randn(8, 5), torch.randn(8, 5, 3))
        with pytest.raises(ValueError, match="3D"):
            criterion(torch.randn(8, 5, 3), torch.randn(8, 5))

    def test_rejects_zero_total_weight(self):
        with pytest.raises(ValueError, match="at least one"):
            MultiTaskLoss(main_weight=0.0, aux_weight=0.0, bce_weight=0.0)


# -- WeightedHuberLoss tolerance to multi-channel target --------------------


class TestSingleTaskLossTolerantToMultiChannelTarget:
    """If target arrives [N, H, 3] and pred [N, H], slice to channel 0."""

    def test_huber_slices_arr_channel(self):
        torch.manual_seed(0)
        pred = torch.randn(8, 5)
        target_arr = torch.randn(8, 5)
        target_dep = torch.randn(8, 5) * 100.0  # noise
        target_pct = torch.rand(8, 5)
        target_3d = torch.stack([target_arr, target_dep, target_pct], dim=-1)

        loss = WeightedHuberLoss()
        # 2D target equivalent to channel 0 of the 3D one.
        loss_2d = loss(pred, target_arr).item()
        loss_3d = loss(pred, target_3d).item()
        assert loss_2d == pytest.approx(loss_3d, rel=1e-5)


# -- compute_bce_classification_metrics ------------------------------------


class TestBCEClassificationMetrics:
    """Metrics derived from the BCE head (sigmoid + threshold)."""

    def test_perfect_predictions(self):
        # Very positive logits → prob >> 0.5; targets already >= 0.5 → all TP.
        logits = np.array([5.0, 5.0, 5.0])
        targets = np.array([0.9, 0.7, 0.51])
        m = compute_bce_classification_metrics(logits, targets)
        assert m["bce_accuracy"] == 1.0
        assert m["bce_precision"] == 1.0
        assert m["bce_recall"] == 1.0
        assert m["bce_f1"] == pytest.approx(1.0)

    def test_perfect_negative_predictions(self):
        # Very negative logits → prob ~ 0; small targets → all TN.
        logits = np.array([-5.0, -5.0, -5.0])
        targets = np.array([0.1, 0.0, 0.49])
        m = compute_bce_classification_metrics(logits, targets)
        assert m["bce_accuracy"] == 1.0
        # precision/recall without positives → 0/1 fallback in the helper.
        assert m["bce_precision"] == 0.0
        assert m["bce_recall"] == 0.0

    def test_threshold_tradeoff(self):
        """Lowering bce_threshold raises recall at the cost of precision."""
        logits = np.array([1.0, -0.5, 2.0, -1.5])  # probs ~ 0.73, 0.38, 0.88, 0.18
        targets = np.array([0.9, 0.6, 0.9, 0.1])
        # high threshold → fewer predicted positives
        high = compute_bce_classification_metrics(logits, targets, bce_threshold=0.7)
        # low threshold → more predicted positives → more recall
        low = compute_bce_classification_metrics(logits, targets, bce_threshold=0.3)
        assert low["bce_recall"] >= high["bce_recall"]


# -- compute_unified_metrics integration -----------------------------------


class TestComputeUnifiedMetricsWithBCE:
    """The bce_* keys are added iff pct_logits and pct_targets are passed."""

    def test_adds_bce_keys_when_provided(self):
        preds = np.array([10.0, 20.0, 5.0])
        targets = np.array([12.0, 18.0, 7.0])
        pct_logits = np.array([1.0, 1.0, -1.0])
        pct_targets = np.array([0.8, 0.6, 0.1])
        m = compute_unified_metrics(
            preds,
            targets,
            delay_threshold=15.0,
            pct_logits=pct_logits,
            pct_targets=pct_targets,
        )
        for k in ("mae", "f1", "bce_f1", "bce_accuracy"):
            assert k in m

    def test_no_bce_keys_when_omitted(self):
        preds = np.array([10.0, 20.0])
        targets = np.array([12.0, 18.0])
        m = compute_unified_metrics(preds, targets)
        for k in ("bce_f1", "bce_precision", "bce_recall", "bce_accuracy"):
            assert k not in m
