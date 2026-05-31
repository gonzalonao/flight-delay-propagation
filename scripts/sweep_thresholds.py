"""Barrido de umbrales de decisión (Phase 1, sin reentrenar).

Carga un checkpoint de modelo de secuencias (seq2seq_gnn / spatiotemporal_gnn),
hace **un único forward** sobre los splits de validación y test, y barre puntos
de operación en memoria para subir recall sin tocar el modelo:

  - E1a: umbral del regresor ``pred_threshold`` (label fijo en 15 min).
  - E1b: umbral de probabilidad del head BCE ``bce_threshold``.
  - E1c: ensemble OR — positivo si ``regr >= t*`` OR ``sigmoid(bce) >= b*``,
         evaluado contra la verdad derivada del regresor (ArrDelay >= 15).

El punto de operación se elige en **validación** (máximo recall sujeto a un piso
de precisión) y se reporta en **test** — elegirlo directamente en test infla el
número. Reutiliza ``collect_sequence_predictions`` de ``src.evaluation.metrics``
para no re-ejecutar el modelo por cada umbral.

Uso:
    python scripts/sweep_thresholds.py \
        --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt \
        --config configs/weekend/seq2seq_gnn_large.yaml \
        --min-precision 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.graph_builder import (
    build_graph_dataset,
    create_temporal_sequences,
    split_graphs_temporal,
)
from src.data.loader import load_multiple_years
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import (
    collect_sequence_predictions,
    compute_bce_classification_metrics,
    compute_classification_metrics,
)
from src.models.factory import SEQUENCE_MODELS, build_model
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.io import get_data_dir, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


# --- Helpers de clasificación ---------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prf_from_labels(
    pred_labels: np.ndarray, target_labels: np.ndarray
) -> dict[str, float]:
    """Precision/recall/F1/accuracy a partir de etiquetas binarias."""
    tp = int(np.sum((pred_labels == 1) & (target_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (target_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (target_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (target_labels == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _avg(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Media por clave (replica el bucket 'average' de los evaluadores)."""
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def _per_horizon_regr(
    arr_pred: list[np.ndarray],
    arr_target: list[np.ndarray],
    pred_threshold: float,
    label_threshold: float,
) -> dict[str, float]:
    """Media de las métricas derivadas del regresor sobre los horizontes."""
    rows = [
        compute_classification_metrics(
            p, t, pred_threshold=pred_threshold, label_threshold=label_threshold,
        )
        for p, t in zip(arr_pred, arr_target)
    ]
    return _avg(rows)


def _per_horizon_bce(
    pct_logit: list[np.ndarray],
    pct_target: list[np.ndarray],
    bce_threshold: float,
) -> dict[str, float]:
    rows = [
        compute_bce_classification_metrics(lg, tg, bce_threshold=bce_threshold)
        for lg, tg in zip(pct_logit, pct_target)
    ]
    # Renombra bce_* → nombres simples para una tabla homogénea.
    return {
        "precision": float(np.mean([r["bce_precision"] for r in rows])),
        "recall": float(np.mean([r["bce_recall"] for r in rows])),
        "f1": float(np.mean([r["bce_f1"] for r in rows])),
        "accuracy": float(np.mean([r["bce_accuracy"] for r in rows])),
    }


def _per_horizon_or(
    arr_pred: list[np.ndarray],
    arr_target: list[np.ndarray],
    pct_logit: list[np.ndarray],
    t: float,
    b: float,
    label_threshold: float,
) -> dict[str, float]:
    """OR-ensemble contra la verdad del regresor (arr_target >= label)."""
    rows = []
    for p, tgt, lg in zip(arr_pred, arr_target, pct_logit):
        pred_labels = ((p >= t) | (_sigmoid(lg) >= b)).astype(int)
        target_labels = (tgt >= label_threshold).astype(int)
        rows.append(_prf_from_labels(pred_labels, target_labels))
    return _avg(rows)


# --- Pickers (sobre arrays agrupados de validación) ------------------------


def _pick_regr(
    pred, target, grid, min_precision, label_threshold,
) -> tuple[float, dict[str, float]]:
    best = None
    feasible = None
    for t in grid:
        m = _prf_from_labels(
            (pred >= t).astype(int), (target >= label_threshold).astype(int)
        )
        cand = (t, m)
        if m["precision"] >= min_precision:
            if feasible is None or m["recall"] > feasible[1]["recall"]:
                feasible = cand
        if best is None or m["f1"] > best[1]["f1"]:
            best = cand
    chosen = feasible if feasible is not None else best
    return float(chosen[0]), chosen[1]


def _pick_bce_for_truth(
    pct_logit, truth_labels, grid, min_precision,
) -> tuple[float, dict[str, float]]:
    """Elige b* maximizando recall (piso de precisión) contra ``truth_labels``."""
    prob = _sigmoid(pct_logit)
    best = None
    feasible = None
    for b in grid:
        m = _prf_from_labels((prob >= b).astype(int), truth_labels)
        cand = (b, m)
        if m["precision"] >= min_precision:
            if feasible is None or m["recall"] > feasible[1]["recall"]:
                feasible = cand
        if best is None or m["f1"] > best[1]["f1"]:
            best = cand
    chosen = feasible if feasible is not None else best
    return float(chosen[0]), chosen[1]


def _pick_or(
    pred, target, pct_logit, pred_grid, bce_grid, min_precision, label_threshold,
) -> tuple[float, float, dict[str, float]]:
    truth = (target >= label_threshold).astype(int)
    prob = _sigmoid(pct_logit)
    best = None
    feasible = None
    for t in pred_grid:
        regr_pos = pred >= t
        for b in bce_grid:
            pred_labels = (regr_pos | (prob >= b)).astype(int)
            m = _prf_from_labels(pred_labels, truth)
            cand = (t, b, m)
            if m["precision"] >= min_precision:
                if feasible is None or m["recall"] > feasible[2]["recall"]:
                    feasible = cand
            if best is None or m["f1"] > best[2]["f1"]:
                best = cand
    chosen = feasible if feasible is not None else best
    return float(chosen[0]), float(chosen[1]), chosen[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep de umbrales (Phase 1)")
    p.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pt")
    p.add_argument(
        "--config", default="configs/weekend/seq2seq_gnn_large.yaml",
        help="Config que reproduce la arquitectura del checkpoint",
    )
    p.add_argument(
        "--min-precision", type=float, default=0.50,
        help="Piso de precisión al elegir el punto de operación en val",
    )
    p.add_argument(
        "--pred-grid", default="5,20,0.5",
        help="Rejilla del regresor 'start,stop,step' (min). Stop inclusivo.",
    )
    p.add_argument(
        "--bce-grid", default="0.05,0.95,0.05",
        help="Rejilla BCE 'start,stop,step' (prob). Stop inclusivo.",
    )
    p.add_argument(
        "--label-threshold", type=float, default=15.0,
        help="Umbral que define el positivo real (ArrDelay min). No se mueve.",
    )
    p.add_argument("--out", default=None, help="Ruta opcional para volcar JSON")
    return p.parse_args()


def _grid(spec: str) -> np.ndarray:
    start, stop, step = (float(x) for x in spec.split(","))
    return np.arange(start, stop + step / 2, step)


def _fmt(tag: str, m: dict[str, float]) -> str:
    return (
        f"{tag:<28} P={m['precision']:.4f}  R={m['recall']:.4f}  "
        f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("reproducibility", {}).get("seed", 42))

    model_name = config["model"]["name"]
    if model_name not in SEQUENCE_MODELS:
        logger.error(
            "sweep_thresholds soporta modelos de secuencia %s; recibido '%s'.",
            sorted(SEQUENCE_MODELS), model_name,
        )
        return

    label_threshold = args.label_threshold
    pred_grid = _grid(args.pred_grid)
    bce_grid = _grid(args.bce_grid)

    logger.info("=" * 60)
    logger.info("SWEEP DE UMBRALES (Phase 1) - %s", model_name)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info(
        "label=%.0f min | min_precision=%.2f | pred_grid=[%.1f..%.1f] | "
        "bce_grid=[%.2f..%.2f]",
        label_threshold, args.min_precision,
        pred_grid[0], pred_grid[-1], bce_grid[0], bce_grid[-1],
    )
    logger.info("=" * 60)

    # --- Datos (idéntico a evaluate.py) ---
    data_dir = get_data_dir("raw", config)
    df = load_multiple_years(
        data_dir, config["data"].get("years", [2018]),
        columns=config["data"].get("columns"),
        sample_frac=None, skip_missing=True,
    )
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)
    device = select_device(config)

    graphs, _airport_map, _norm = build_graph_dataset(df, airports, config)
    split_cfg = config.get("split", {})
    splits = split_graphs_temporal(
        graphs,
        train_end=split_cfg.get("train_end"),
        val_end=split_cfg.get("val_end"),
    )
    input_window = config.get("graph", {}).get("input_window", 6)
    val_seq = create_temporal_sequences(splits["val"], input_window)
    test_seq = create_temporal_sequences(splits["test"], input_window)
    if not val_seq or not test_seq:
        logger.error(
            "Splits insuficientes (val=%d, test=%d secuencias).",
            len(val_seq), len(test_seq),
        )
        return

    horizons = config.get("graph", {}).get("prediction_horizons", [1, 2, 3, 4, 5])
    n_h = len(horizons)

    # --- Modelo + checkpoint ---
    sample_ea = splits["test"][0].edge_attr
    edge_dim = (
        sample_ea.shape[1]
        if sample_ea is not None and sample_ea.dim() == 2 else None
    )
    model = build_model(config, splits["test"][0].x.shape[1], edge_dim=edge_dim)
    info = load_checkpoint(args.checkpoint, model, expected_model_name=model_name)
    logger.info("Checkpoint cargado: época %s", info.get("epoch"))
    model = model.to(device)

    # --- Forward único por split ---
    logger.info("Forward sobre val (%d secuencias)...", len(val_seq))
    val = collect_sequence_predictions(model, val_seq, device, n_h)
    logger.info("Forward sobre test (%d secuencias)...", len(test_seq))
    test = collect_sequence_predictions(model, test_seq, device, n_h)

    if not val["has_pct"]:
        logger.error(
            "El modelo no emite el head pct (multi-task). E1b/E1c no aplican; "
            "sólo E1a (regresor) tiene sentido."
        )

    # Arrays agrupados (todos los horizontes) para elegir el punto de operación.
    v_pred = np.concatenate(val["arr_pred"])
    v_tgt = np.concatenate(val["arr_target"])
    v_logit = np.concatenate(val["pct_logit"]) if val["has_pct"] else None

    # ---------- E1a: umbral del regresor ----------
    t_star, v_regr = _pick_regr(
        v_pred, v_tgt, pred_grid, args.min_precision, label_threshold,
    )
    base_regr_test = _per_horizon_regr(
        test["arr_pred"], test["arr_target"], label_threshold, label_threshold,
    )
    tuned_regr_test = _per_horizon_regr(
        test["arr_pred"], test["arr_target"], t_star, label_threshold,
    )

    results: dict[str, object] = {
        "min_precision": args.min_precision,
        "label_threshold": label_threshold,
        "E1a_pred_threshold_star": t_star,
        "E1a_val": v_regr,
        "E1a_test_baseline": base_regr_test,
        "E1a_test_tuned": tuned_regr_test,
    }

    logger.info("")
    logger.info("-- E1a: regresor (label=%.0f) --", label_threshold)
    logger.info("  val pick: pred_threshold* = %.2f min", t_star)
    logger.info("  %s", _fmt("TEST baseline (pred=15)", base_regr_test))
    logger.info("  %s", _fmt(f"TEST tuned (pred={t_star:.1f})", tuned_regr_test))

    # ---------- E1b + E1c (requieren head pct) ----------
    if val["has_pct"]:
        # E1b: contra la verdad propia del head BCE (pct >= 0.5), como en los logs.
        v_pct_truth = (np.concatenate(val["pct_target"]) >= 0.5).astype(int)
        b_star, v_bce = _pick_bce_for_truth(
            v_logit, v_pct_truth, bce_grid, args.min_precision,
        )
        base_bce_test = _per_horizon_bce(
            test["pct_logit"], test["pct_target"], 0.5,
        )
        tuned_bce_test = _per_horizon_bce(
            test["pct_logit"], test["pct_target"], b_star,
        )
        results.update({
            "E1b_bce_threshold_star": b_star,
            "E1b_val": v_bce,
            "E1b_test_baseline": base_bce_test,
            "E1b_test_tuned": tuned_bce_test,
        })
        logger.info("")
        logger.info("-- E1b: head BCE (verdad pct>=0.5) --")
        logger.info("  val pick: bce_threshold* = %.2f", b_star)
        logger.info("  %s", _fmt("TEST baseline (bce=0.5)", base_bce_test))
        logger.info("  %s", _fmt(f"TEST tuned (bce={b_star:.2f})", tuned_bce_test))

        # E1c: OR-ensemble contra la verdad del regresor (arr >= label).
        or_t, or_b, v_or = _pick_or(
            v_pred, v_tgt, v_logit, pred_grid, bce_grid,
            args.min_precision, label_threshold,
        )
        or_test = _per_horizon_or(
            test["arr_pred"], test["arr_target"], test["pct_logit"],
            or_t, or_b, label_threshold,
        )
        results.update({
            "E1c_or_pred_threshold": or_t,
            "E1c_or_bce_threshold": or_b,
            "E1c_val": v_or,
            "E1c_test": or_test,
        })
        logger.info("")
        logger.info("-- E1c: OR-ensemble (verdad arr>=%.0f) --", label_threshold)
        logger.info("  val pick: regr>=%.2f OR sigmoid(bce)>=%.2f", or_t, or_b)
        logger.info("  %s", _fmt("TEST baseline (pred=15)", base_regr_test))
        logger.info("  %s", _fmt("TEST OR-ensemble", or_test))

    # --- Resumen vs baseline headline (recall derivado del regresor) ---
    logger.info("")
    logger.info("== RESUMEN (recall derivado del regresor, label=%.0f) ==", label_threshold)
    logger.info("  baseline recall = %.4f", base_regr_test["recall"])
    logger.info(
        "  E1a tuned     = %.4f  (delta %+.4f, prec %.4f)",
        tuned_regr_test["recall"],
        tuned_regr_test["recall"] - base_regr_test["recall"],
        tuned_regr_test["precision"],
    )
    if val["has_pct"]:
        logger.info(
            "  E1c OR-ens    = %.4f  (delta %+.4f, prec %.4f)",
            or_test["recall"],
            or_test["recall"] - base_regr_test["recall"],
            or_test["precision"],
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Resultados volcados en %s", args.out)


if __name__ == "__main__":
    main()
