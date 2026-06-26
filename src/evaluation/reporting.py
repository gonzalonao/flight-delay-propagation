"""Unified logging of test results.

Before this module, the functions ``_log_test_results`` and
``_log_multi_horizon_results`` were duplicated byte-for-byte in
``scripts/train.py`` and ``scripts/evaluate.py``. Any formatting change
(e.g., adding an uncertainty widget) had to be made twice and silently
masked accidental divergences.

The functions here accept an optional ``logger`` so each script can use its
own logger with a name consistent with the module.
"""

from __future__ import annotations

import logging

from src.utils.logger import setup_logger

_default_logger = setup_logger(__name__)


def log_test_results(
    metrics: dict[str, float],
    model_name: str,
    logger: logging.Logger | None = None,
) -> None:
    """Log the single-horizon test metrics in a unified format.

    Args:
        metrics: Dictionary with keys ``mae``, ``rmse``, ``mape``, ``r2``,
            ``accuracy``, ``precision``, ``recall``, ``f1``.
        model_name: Logical model name (e.g., ``"dense_nn"``).
        logger: Target logger. If ``None`` the module logger is used.
    """
    log = logger or _default_logger
    log.info("=" * 50)
    log.info("TEST RESULTS — %s", model_name)
    log.info("-" * 50)
    log.info("  Regression:")
    log.info("    MAE:  %.4f min", metrics["mae"])
    log.info("    RMSE: %.4f min", metrics["rmse"])
    log.info("    MAPE: %.4f %%", metrics["mape"])
    log.info("    R²:   %.4f", metrics["r2"])
    log.info("  Classification (threshold=15 min):")
    log.info("    ACCURACY:  %.4f", metrics["accuracy"])
    log.info("    PRECISION: %.4f", metrics["precision"])
    log.info("    RECALL:    %.4f", metrics["recall"])
    log.info("    F1:        %.4f", metrics["f1"])
    log.info("=" * 50)


def log_multi_horizon_results(
    metrics: dict[str, dict[str, float]],
    model_name: str,
    horizons: list[int],
    logger: logging.Logger | None = None,
) -> None:
    """Log the multi-horizon metrics in a unified format.

    Args:
        metrics: Dictionary with one ``"horizon_{h}h"`` entry per horizon
            and an ``"average"`` entry with the averages.
        model_name: Logical model name.
        horizons: List of horizons (in hours).
        logger: Target logger. If ``None`` the module logger is used.
    """
    log = logger or _default_logger
    log.info("=" * 60)
    log.info("TEST RESULTS — %s (multi-horizon)", model_name)
    log.info("=" * 60)

    has_bce = "bce_f1" in metrics["average"]

    for h in horizons:
        key = f"horizon_{h}h"
        m = metrics[key]
        log.info("  Horizon +%dh:", h)
        log.info(
            "    MAE: %.4f | RMSE: %.4f | MAPE: %.4f%% | R²: %.4f",
            m["mae"], m["rmse"], m["mape"], m["r2"],
        )
        log.info(
            "    Cls (regr-thr): Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
            m["accuracy"], m["precision"], m["recall"], m["f1"],
        )
        if has_bce:
            log.info(
                "    Cls (BCE head): Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f",
                m["bce_accuracy"], m["bce_precision"],
                m["bce_recall"], m["bce_f1"],
            )

    avg = metrics["average"]
    log.info("-" * 60)
    log.info("  AVERAGE (all horizons):")
    log.info("    Regression:")
    log.info("      MAE:  %.4f min", avg["mae"])
    log.info("      RMSE: %.4f min", avg["rmse"])
    log.info("      MAPE: %.4f %%", avg["mape"])
    log.info("      R²:   %.4f", avg["r2"])
    log.info("    Classification derived from the regressor (threshold=15 min):")
    log.info("      ACCURACY:  %.4f", avg["accuracy"])
    log.info("      PRECISION: %.4f", avg["precision"])
    log.info("      RECALL:    %.4f", avg["recall"])
    log.info("      F1:        %.4f", avg["f1"])
    if has_bce:
        log.info("    BCE-head classification (sigmoid + 0.5):")
        log.info("      ACCURACY:  %.4f", avg["bce_accuracy"])
        log.info("      PRECISION: %.4f", avg["bce_precision"])
        log.info("      RECALL:    %.4f", avg["bce_recall"])
        log.info("      F1:        %.4f", avg["bce_f1"])
    log.info("=" * 60)
