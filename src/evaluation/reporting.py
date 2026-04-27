"""Logging unificado de resultados de test.

Antes de este módulo las funciones ``_log_test_results`` y
``_log_multi_horizon_results`` estaban duplicadas byte-a-byte en
``scripts/train.py`` y ``scripts/evaluate.py``. Cualquier cambio
de formato (e.g., añadir un widget de incertidumbre) había que
hacerlo dos veces y silenciaba divergencias accidentales.

Las funciones aquí aceptan un ``logger`` opcional para que cada
script use su propio logger con nombre coherente con el módulo.
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
    """Loguea las métricas de test single-horizon en formato unificado.

    Args:
        metrics: Diccionario con claves ``mae``, ``rmse``, ``mape``, ``r2``,
            ``accuracy``, ``precision``, ``recall``, ``f1``.
        model_name: Nombre lógico del modelo (e.g., ``"dense_nn"``).
        logger: Logger destino. Si ``None`` se usa el logger del módulo.
    """
    log = logger or _default_logger
    log.info("=" * 50)
    log.info("RESULTADOS EN TEST — %s", model_name)
    log.info("-" * 50)
    log.info("  Regresión:")
    log.info("    MAE:  %.4f min", metrics["mae"])
    log.info("    RMSE: %.4f min", metrics["rmse"])
    log.info("    MAPE: %.4f %%", metrics["mape"])
    log.info("    R²:   %.4f", metrics["r2"])
    log.info("  Clasificación (umbral=15 min):")
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
    """Loguea las métricas multi-horizonte en formato unificado.

    Args:
        metrics: Diccionario con una entrada ``"horizon_{h}h"`` por cada
            horizonte y una entrada ``"average"`` con los promedios.
        model_name: Nombre lógico del modelo.
        horizons: Lista de horizontes (en horas).
        logger: Logger destino. Si ``None`` se usa el logger del módulo.
    """
    log = logger or _default_logger
    log.info("=" * 60)
    log.info("RESULTADOS EN TEST — %s (multi-horizonte)", model_name)
    log.info("=" * 60)

    has_bce = "bce_f1" in metrics["average"]

    for h in horizons:
        key = f"horizon_{h}h"
        m = metrics[key]
        log.info("  Horizonte +%dh:", h)
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
    log.info("  PROMEDIO (todos los horizontes):")
    log.info("    Regresión:")
    log.info("      MAE:  %.4f min", avg["mae"])
    log.info("      RMSE: %.4f min", avg["rmse"])
    log.info("      MAPE: %.4f %%", avg["mape"])
    log.info("      R²:   %.4f", avg["r2"])
    log.info("    Clasificación derivada del regresor (umbral=15 min):")
    log.info("      ACCURACY:  %.4f", avg["accuracy"])
    log.info("      PRECISION: %.4f", avg["precision"])
    log.info("      RECALL:    %.4f", avg["recall"])
    log.info("      F1:        %.4f", avg["f1"])
    if has_bce:
        log.info("    Clasificación del head BCE (sigmoid + 0.5):")
        log.info("      ACCURACY:  %.4f", avg["bce_accuracy"])
        log.info("      PRECISION: %.4f", avg["bce_precision"])
        log.info("      RECALL:    %.4f", avg["bce_recall"])
        log.info("      F1:        %.4f", avg["bce_f1"])
    log.info("=" * 60)
