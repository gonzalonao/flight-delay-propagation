"""Compara estrategias de fusión GNN ↔ per-vuelo, por horizonte (test 2019-Q4).

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

No modifica ni el GNN ni el modelo per-vuelo: entrena el GBM base per-vuelo,
consume las predicciones del GNN (scripts/predict.py --split all) y evalúa, a
nivel per-vuelo y point-in-time:

    gbm_only · gnn_only · blend(w*) · switch · stack

Ajustes (w de blend, elección de switch, meta-GBM de stack) en VAL (out-of-sample),
métricas en TEST. Requiere el frame del GNN con splits val+test.

Uso:
    set PYTHONIOENCODING=utf-8
    uv run python scripts/compare_fusion.py \
        --gnn-preds outputs/flight_level/gnn_preds_all.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.flight_level.features import (
    FEATURE_SETS, TARGET, CATEGORICAL_FEATURES,
    add_datetimes, build_airport_hourly_stats, build_work_frame,
    clean, filter_top_airports, load_flights,
)
from src.flight_level.fusion import (
    attach_gnn_feature, best_blend_weight, f1_at, load_gnn_predictions,
)
from src.flight_level.model import FlightDelayModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/flight_level.yaml")
    ap.add_argument("--gnn-preds", default="outputs/flight_level/gnn_preds_all.parquet")
    ap.add_argument("--feature-set", default="E_all")
    ap.add_argument("--sample-train", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs/flight_level_fusion")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg, mcfg = cfg["data"], cfg["model"]
    seed = cfg.get("seed", 42)
    train_end = pd.Timestamp(dcfg["train_end"])
    val_end = pd.Timestamp(dcfg["test_start"])          # val = [train_end, test_start)
    test_start = pd.Timestamp(dcfg["test_start"])
    horizons = mcfg["horizons"]
    thresholds = mcfg["thresholds"]
    feats = FEATURE_SETS[args.feature_set]
    cat = [c for c in feats if c in CATEGORICAL_FEATURES]
    sample_train = args.sample_train if args.sample_train is not None else dcfg.get("sample_train")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] vuelos ...", flush=True)
    df = add_datetimes(clean(filter_top_airports(load_flights(dcfg["years"]), dcfg["top_n_airports"])))
    hourly = build_airport_hourly_stats(df)
    gnn = load_gnn_predictions(args.gnn_preds)

    train_df = df[df["dep_dt"] < train_end]
    val_df = df[(df["dep_dt"] >= train_end) & (df["dep_dt"] < val_end)]
    test_df = df[df["dep_dt"] >= test_start]
    if sample_train and len(train_df) > sample_train:
        train_df = train_df.sample(sample_train, random_state=seed)
    print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}", flush=True)

    report: dict = {"config": cfg, "feature_set": args.feature_set, "by_horizon": {}}
    rows = []
    for h in horizons:
        t0 = time.time()
        w_tr = build_work_frame(train_df, h, hourly)
        w_va = attach_gnn_feature(build_work_frame(val_df, h, hourly), gnn, h)
        w_te = attach_gnn_feature(build_work_frame(test_df, h, hourly), gnn, h)

        # GBM base per-vuelo (entrenado solo en train; sin feature GNN).
        base = FlightDelayModel(categorical_features=cat, random_state=seed, **mcfg["hgb"])
        base.fit(w_tr[feats], w_tr[TARGET])
        gbm_va, gbm_te = base.predict(w_va[feats]), base.predict(w_te[feats])

        y_va, y_te = w_va[TARGET].to_numpy(), w_te[TARGET].to_numpy()
        fill = float(w_tr[TARGET].mean())
        gnn_va = np.where(np.isnan(w_va["gnn_pred_arr_delay"]), fill, w_va["gnn_pred_arr_delay"])
        gnn_te = np.where(np.isnan(w_te["gnn_pred_arr_delay"]), fill, w_te["gnn_pred_arr_delay"])
        coverage = float(np.mean(~np.isnan(w_te["gnn_pred_arr_delay"].to_numpy())))

        # --- estrategias ---
        preds_te: dict[str, np.ndarray] = {"gbm_only": gbm_te, "gnn_only": gnn_te}

        # blend: w* en val.
        w = best_blend_weight(y_va, gbm_va, gnn_va)
        preds_te["blend"] = w * gbm_te + (1 - w) * gnn_te

        # switch: el mejor de los dos en val (constante por horizonte).
        use_gbm = f1_at(y_va, gbm_va) >= f1_at(y_va, gnn_va)
        preds_te["switch"] = gbm_te if use_gbm else gnn_te

        # stack: meta-GBM sobre [gbm, gnn, lead, dep_hour], ajustado en val.
        meta_cols = ["m_gbm", "m_gnn", "m_lead", "m_hour"]
        Xva = pd.DataFrame({"m_gbm": gbm_va, "m_gnn": gnn_va,
                            "m_lead": w_va["gnn_lead_h"].to_numpy(),
                            "m_hour": w_va["dep_hour"].to_numpy()}, columns=meta_cols)
        Xte = pd.DataFrame({"m_gbm": gbm_te, "m_gnn": gnn_te,
                            "m_lead": w_te["gnn_lead_h"].to_numpy(),
                            "m_hour": w_te["dep_hour"].to_numpy()}, columns=meta_cols)
        meta = FlightDelayModel(random_state=seed, max_iter=200, learning_rate=0.05,
                                max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0)
        meta.fit(Xva, pd.Series(y_va))
        preds_te["stack"] = meta.predict(Xte)

        # --- evaluación en test ---
        h_report = {"blend_w": float(w), "switch_used": "gbm" if use_gbm else "gnn",
                    "gnn_coverage": coverage, "strategies": {}}
        for name, pred in preds_te.items():
            m = FlightDelayModel.evaluate(y_te, pred, thresholds)
            h_report["strategies"][name] = m
            bt = m["by_threshold"]
            rows.append((h, name, m["mae"], bt["15"]["recall"], bt["15"]["precision"],
                         bt["30"]["recall"]))
        report["by_horizon"][str(h)] = h_report
        print(f"[h={h}] blend_w={w:.1f} switch={'gbm' if use_gbm else 'gnn'} "
              f"gnn_cov={coverage:.2f} ({time.time()-t0:.0f}s)", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- resumen ---
    print(f"\n=== FUSIÓN per-vuelo (test) — feature set {args.feature_set} ===")
    print(f"{'H':>3s} {'strategy':10s} {'MAE':>7s} {'rec@15':>7s} {'pre@15':>7s} {'rec@30':>7s}")
    best_by_h = {}
    for h, name, mae, r15, p15, r30 in rows:
        print(f"{h:3d} {name:10s} {mae:7.2f} {r15:7.3f} {p15:7.3f} {r30:7.3f}")
        if h not in best_by_h or r15 > best_by_h[h][1]:
            best_by_h[h] = (name, r15)
    print("\nMejor estrategia por horizonte (recall@15):")
    for h in sorted(best_by_h):
        print(f"  H={h}: {best_by_h[h][0]} (recall@15={best_by_h[h][1]:.3f})")
    print(f"\n[done] informe -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
