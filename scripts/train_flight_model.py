"""Entrena el modelo per-vuelo por horizonte (track SEPARADO del GNN baseline).

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

Para cada horizonte H entrena un GBM sobre features **point-in-time** (solo datos
observables en ``t_pred = salida_programada − H``) y reporta MAE/RMSE + recall y
precision derivados a 15/30/45/60 min. Esto da números honestos por horizonte:
a mayor H, el estado del avión entrante es menos observable y la señal decae.

Uso:
    set PYTHONIOENCODING=utf-8
    uv run python scripts/train_flight_model.py                       # E_all, todos los H
    uv run python scripts/train_flight_model.py --ablation            # todos los feature sets
    uv run python scripts/train_flight_model.py --sample-train 200000 # rápido
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.flight_level.features import (
    FEATURE_SETS,
    add_datetimes,
    build_airport_hourly_stats,
    clean,
    filter_top_airports,
    load_flights,
    CATEGORICAL_FEATURES,
    assemble,
)
from src.flight_level.model import FlightDelayModel


def _temporal_masks(df: pd.DataFrame, train_end: str, test_start: str):
    return (df["dep_dt"] < pd.Timestamp(train_end),
            df["dep_dt"] >= pd.Timestamp(test_start))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/flight_level.yaml")
    ap.add_argument("--ablation", action="store_true",
                    help="Entrena todos los feature sets (si no, solo el del config).")
    ap.add_argument("--sample-train", type=int, default=None,
                    help="Override del submuestreo de train del config.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg, mcfg = cfg["data"], cfg["model"]
    seed = cfg.get("seed", 42)
    out_dir = Path(args.out_dir or cfg.get("out_dir", "outputs/flight_level"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_train = args.sample_train if args.sample_train is not None else dcfg.get("sample_train")
    horizons = mcfg["horizons"]
    thresholds = mcfg["thresholds"]
    feature_sets = list(FEATURE_SETS) if args.ablation else [mcfg["feature_set"]]

    print(f"[load] years={dcfg['years']} top_n={dcfg['top_n_airports']}", flush=True)
    df = load_flights(dcfg["years"])
    df = filter_top_airports(df, dcfg["top_n_airports"])
    df = clean(df)
    df = add_datetimes(df)
    print(f"[load] vuelos limpios: {len(df):,}", flush=True)

    print("[airport-state] construyendo estadísticos horarios realizados ...", flush=True)
    hourly = build_airport_hourly_stats(df)

    is_train, is_test = _temporal_masks(df, dcfg["train_end"], dcfg["test_start"])
    train_df, test_df = df[is_train], df[is_test]
    if sample_train and len(train_df) > sample_train:
        train_df = train_df.sample(sample_train, random_state=seed)
    print(f"[split] train={len(train_df):,}  test={len(test_df):,}  "
          f"(sample_train={sample_train})", flush=True)

    report: dict = {"config": cfg, "results": {}}
    print(f"\n{'set':22s} {'H':>3s} {'MAE':>7s} {'rec@15':>7s} {'pre@15':>7s} "
          f"{'rec@30':>7s} {'rec@60':>7s}")
    for fs_name in feature_sets:
        feats = FEATURE_SETS[fs_name]
        cat = [c for c in feats if c in CATEGORICAL_FEATURES]
        report["results"][fs_name] = {}
        for h in horizons:
            X_tr, y_tr = assemble(train_df, h, hourly, feats)
            X_te, y_te = assemble(test_df, h, hourly, feats)
            t0 = time.time()
            model = FlightDelayModel(
                categorical_features=cat, random_state=seed, **mcfg["hgb"]
            ).fit(X_tr, y_tr)
            metrics = model.evaluate(y_te.to_numpy(), model.predict(X_te), thresholds)
            metrics["fit_seconds"] = round(time.time() - t0, 1)
            report["results"][fs_name][str(h)] = metrics

            # Persistir el modelo del feature set elegido (no toda la ablación).
            if not args.ablation or fs_name == mcfg["feature_set"]:
                model.save(out_dir / f"model_{fs_name}_h{h}.joblib")

            bt = metrics["by_threshold"]
            print(f"{fs_name:22s} {h:3d} {metrics['mae']:7.2f} "
                  f"{bt['15']['recall']:7.3f} {bt['15']['precision']:7.3f} "
                  f"{bt['30']['recall']:7.3f} {bt['60']['recall']:7.3f}", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] informe -> {out_dir / 'report.json'}")
    print("Nota: rec@H usa el estado entrante observable en t_pred = salida - H "
          "(honesto por horizonte).")


if __name__ == "__main__":
    main()
