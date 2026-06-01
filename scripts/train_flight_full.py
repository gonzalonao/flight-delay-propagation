"""Modelo per-vuelo mejorado: datos completos + LightGBM (+tune) + meteo.

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

Entrena por horizonte sobre TODO el train (no submuestreo) con LightGBM y early
stopping en validación, comparando dos juegos de features point-in-time para
aislar la aportación de la meteorología:

    E_all          -> schedule + route + airline + rotation + airport-state
    F_all_weather  -> E_all + meteo origen(as-of t_pred) / destino(forecast llegada)

Con ``--tune`` hace una búsqueda aleatoria compacta de hiperparámetros (en una
submuestra, seleccionando por F1@15 en val) y reutiliza los mejores en todos los
horizontes. Métricas finales en test (2019-Q4).

Uso:
    set PYTHONIOENCODING=utf-8
    uv run python scripts/train_flight_full.py --tune
    uv run python scripts/train_flight_full.py --sample-train 300000   # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.flight_level.features import (
    CATEGORICAL_FEATURES, FEATURE_SETS, F_WEATHER, TARGET,
    add_datetimes, build_airport_hourly_stats, build_work_frame,
    clean, filter_top_airports, load_flights,
)
from src.flight_level.fusion import f1_at
from src.flight_level.model import FlightDelayModel
from src.flight_level.weather import add_weather_asof, load_weather_long

CONFIGS = [("E_all_lgbm", "E_all"), ("F_weather_lgbm", "F_all_weather")]
TUNE_SPACE = {
    "num_leaves": [127, 255, 511],
    "learning_rate": [0.02, 0.03, 0.05],
    "min_child_samples": [100, 200, 500],
    "colsample_bytree": [0.7, 0.9],
    "subsample": [0.7, 0.9],
}


def _needs_weather(feats: list[str]) -> bool:
    return any(c in feats for c in F_WEATHER)


def assemble_X(split_df, h, hourly, weather, feats):
    work = build_work_frame(split_df, h, hourly)
    if _needs_weather(feats):
        work = add_weather_asof(work, weather, h)
    return work[feats].copy(), work[TARGET].to_numpy(dtype="float32")


def align_categories(X_tr, others, cat_cols):
    """Codifica las categóricas con categorías de TRAIN (códigos consistentes)."""
    cats = {c: X_tr[c].astype("category").cat.categories for c in cat_cols}
    X_tr = X_tr.copy()
    for c in cat_cols:
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats[c])
    aligned = []
    for X in others:
        X = X.copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=cats[c])
        aligned.append(X)
    return X_tr, aligned


def tune(X_tr, y_tr, X_va, y_va, cat_cols, seed, n_iter=8):
    rng = random.Random(seed)
    best, best_f1 = None, -1.0
    for _ in range(n_iter):
        params = {k: rng.choice(v) for k, v in TUNE_SPACE.items()}
        m = FlightDelayModel(categorical_features=cat_cols, backend="lightgbm",
                             random_state=seed, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        f1 = f1_at(y_va, m.predict(X_va))
        if f1 > best_f1:
            best, best_f1 = params, f1
    print(f"[tune] best F1@15(val)={best_f1:.3f} params={best}", flush=True)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/flight_level.yaml")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sample-train", type=int, default=None,
                    help="Submuestreo de train (por defecto: TODO).")
    ap.add_argument("--out-dir", default="outputs/flight_level_full")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg, mcfg = cfg["data"], cfg["model"]
    seed = cfg.get("seed", 42)
    train_end = pd.Timestamp(dcfg["train_end"])
    val_end = pd.Timestamp(dcfg["test_start"])
    horizons, thresholds = mcfg["horizons"], mcfg["thresholds"]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] vuelos ...", flush=True)
    df = add_datetimes(clean(filter_top_airports(load_flights(dcfg["years"]), dcfg["top_n_airports"])))
    hourly = build_airport_hourly_stats(df)
    airports = sorted(set(df["Origin"].unique()) | set(df["Dest"].unique()))
    weather = load_weather_long(airports)
    print(f"[load] vuelos={len(df):,}  meteo={weather['airport'].nunique()} aeropuertos", flush=True)

    train_df = df[df["dep_dt"] < train_end]
    val_df = df[(df["dep_dt"] >= train_end) & (df["dep_dt"] < val_end)]
    test_df = df[df["dep_dt"] >= val_end]
    if args.sample_train and len(train_df) > args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=seed)
    print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}", flush=True)

    # Tuning (una vez, sobre el set más rico; reutilizado en todo).
    tuned = {}
    if args.tune:
        feats = FEATURE_SETS["F_all_weather"]
        cat = [c for c in feats if c in CATEGORICAL_FEATURES]
        sub = train_df.sample(min(800000, len(train_df)), random_state=seed)
        Xs, ys = assemble_X(sub, horizons[0], hourly, weather, feats)
        Xv, yv = assemble_X(val_df, horizons[0], hourly, weather, feats)
        Xs, (Xv,) = align_categories(Xs, [Xv], cat)
        tuned = tune(Xs, ys, Xv, yv, cat, seed)

    report = {"config": cfg, "tuned_params": tuned, "results": {}}
    rows = []
    for cfg_name, fs_name in CONFIGS:
        feats = FEATURE_SETS[fs_name]
        cat = [c for c in feats if c in CATEGORICAL_FEATURES]
        report["results"][cfg_name] = {}
        for h in horizons:
            t0 = time.time()
            X_tr, y_tr = assemble_X(train_df, h, hourly, weather, feats)
            X_va, y_va = assemble_X(val_df, h, hourly, weather, feats)
            X_te, y_te = assemble_X(test_df, h, hourly, weather, feats)
            X_tr, (X_va, X_te) = align_categories(X_tr, [X_va, X_te], cat)

            model = FlightDelayModel(categorical_features=cat, backend="lightgbm",
                                     random_state=seed, **tuned)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            metrics = model.evaluate(y_te, model.predict(X_te), thresholds)
            metrics["fit_seconds"] = round(time.time() - t0, 1)
            report["results"][cfg_name][str(h)] = metrics
            model.save(out_dir / f"model_{cfg_name}_h{h}.joblib")

            bt = metrics["by_threshold"]
            rows.append((cfg_name, h, metrics["mae"], bt["15"]["recall"],
                         bt["15"]["precision"], bt["30"]["recall"]))
            print(f"[{cfg_name} h={h}] MAE={metrics['mae']:.2f} "
                  f"rec@15={bt['15']['recall']:.3f} pre@15={bt['15']['precision']:.3f} "
                  f"({metrics['fit_seconds']:.0f}s)", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== per-vuelo FULL+LightGBM (test) ===")
    print(f"{'config':16s} {'H':>3s} {'MAE':>7s} {'rec@15':>7s} {'pre@15':>7s} {'rec@30':>7s}")
    for cfg_name, h, mae, r15, p15, r30 in rows:
        print(f"{cfg_name:16s} {h:3d} {mae:7.2f} {r15:7.3f} {p15:7.3f} {r30:7.3f}")
    print("\nReferencia previa (E_all HistGBM, 600k): H1 rec@15=0.389 MAE=18.18")
    print(f"[done] informe -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
