"""Recall-targeted per-flight model: operating-point tuning + class weighting.

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

Parte del mejor juego de features (``F_all_weather``, datos completos, LightGBM
con los hiperparámetros ya afinados) y ataca específicamente el *recall* por dos
vías, ambas point-in-time y sin tocar el pipeline base:

1. **Operating-point tuning sobre la regresión.** En lugar de decidir "retrasado"
   con el corte natural ``ŷ ≥ T`` (T=15/30), se elige en *validación* el corte
   ``c`` que maximiza F-beta (β=2, favorece recall) y se aplica en test. Bajar el
   corte sube recall a costa de precisión: expone el frente de Pareto.
2. **Clasificador con coste asimétrico.** Un ``LGBMClassifier`` con
   ``is_unbalance`` entrenado directamente sobre la etiqueta binaria (ArrDelay≥15),
   con el umbral de probabilidad afinado en val por F-beta.

Para cada horizonte compara, en test: regresión@corte natural (línea base de la
Parte 4), regresión@corte afinado, y clasificador@umbral afinado.

Uso:
    set PYTHONIOENCODING=utf-8
    uv run python scripts/train_flight_recall.py --beta 2.0
    uv run python scripts/train_flight_recall.py --sample-train 300000   # smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score

from src.flight_level.features import (
    CATEGORICAL_FEATURES, FEATURE_SETS, TARGET,
    add_datetimes, build_airport_hourly_stats, build_work_frame, clean,
    filter_top_airports, load_flights,
)
from src.flight_level.model import FlightDelayModel
from src.flight_level.weather import add_weather_asof, load_weather_long

# Hiperparámetros ya afinados en la corrida completa (val F1@15=0.521).
TUNED = dict(num_leaves=127, learning_rate=0.03, min_child_samples=100,
             colsample_bytree=0.7, subsample=0.9)
FEATS = FEATURE_SETS["F_all_weather"]


def assemble_X(split_df, h, hourly, weather, feats):
    """(X, y) point-in-time con meteo para un horizonte."""
    work = build_work_frame(split_df, h, hourly)
    work = add_weather_asof(work, weather, h)
    return work[feats].copy(), work[TARGET].to_numpy(dtype="float32")


def align_categories(X_tr, others, cat_cols):
    """Codifica categóricas con las categorías de TRAIN (códigos consistentes)."""
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


def _scores(y_true_bin: np.ndarray, pred_bin: np.ndarray, beta: float) -> dict:
    rec = recall_score(y_true_bin, pred_bin, zero_division=0)
    prec = precision_score(y_true_bin, pred_bin, zero_division=0)
    b2 = beta * beta
    denom = b2 * prec + rec
    fb = 0.0 if denom == 0 else (1 + b2) * prec * rec / denom
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"recall": float(rec), "precision": float(prec),
            "f1": float(f1), "fbeta": float(fb)}


def best_cutoff(y_true_bin: np.ndarray, score: np.ndarray, beta: float,
                grid: np.ndarray) -> tuple[float, dict]:
    """Corte que maximiza F-beta sobre ``score`` (val). Devuelve (corte, métricas)."""
    best_c, best = float(grid[0]), {"fbeta": -1.0}
    for c in grid:
        m = _scores(y_true_bin, score >= c, beta)
        if m["fbeta"] > best["fbeta"]:
            best_c, best = float(c), m
    return best_c, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/flight_level.yaml")
    ap.add_argument("--beta", type=float, default=2.0, help="Peso de recall en F-beta.")
    ap.add_argument("--sample-train", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs/flight_level_recall")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg, mcfg = cfg["data"], cfg["model"]
    seed = cfg.get("seed", 42)
    train_end = pd.Timestamp(dcfg["train_end"])
    val_end = pd.Timestamp(dcfg["test_start"])
    horizons, thresholds = mcfg["horizons"], mcfg["thresholds"]
    beta = args.beta
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] vuelos ...", flush=True)
    df = add_datetimes(clean(filter_top_airports(load_flights(dcfg["years"]), dcfg["top_n_airports"])))
    hourly = build_airport_hourly_stats(df)
    airports = sorted(set(df["Origin"].unique()) | set(df["Dest"].unique()))
    weather = load_weather_long(airports)

    train_df = df[df["dep_dt"] < train_end]
    val_df = df[(df["dep_dt"] >= train_end) & (df["dep_dt"] < val_end)]
    test_df = df[df["dep_dt"] >= val_end]
    if args.sample_train and len(train_df) > args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=seed)
    print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} beta={beta}", flush=True)

    cat = [c for c in FEATS if c in CATEGORICAL_FEATURES]
    report = {"config": cfg, "beta": beta, "tuned_params": TUNED, "results": {}}
    rows = []
    for h in horizons:
        try:
            t0 = time.time()
            X_tr, y_tr = assemble_X(train_df, h, hourly, weather, FEATS)
            X_va, y_va = assemble_X(val_df, h, hourly, weather, FEATS)
            X_te, y_te = assemble_X(test_df, h, hourly, weather, FEATS)
            X_tr, (X_va, X_te) = align_categories(X_tr, [X_va, X_te], cat)

            # 1) Regresor (idéntico a la Parte 4) -> scores en val/test.
            reg = FlightDelayModel(categorical_features=cat, backend="lightgbm",
                                   random_state=seed, **TUNED)
            reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            s_va, s_te = reg.predict(X_va), reg.predict(X_te)

            # 2) Clasificador con coste asimétrico para la etiqueta de 15 min.
            from lightgbm import LGBMClassifier
            import lightgbm as lgb
            yb_tr, yb_va = (y_tr >= 15).astype(int), (y_va >= 15).astype(int)
            clf = LGBMClassifier(verbose=-1, n_estimators=2000, is_unbalance=True,
                                 random_state=seed, n_jobs=-1, reg_lambda=1.0, **TUNED)
            clf.fit(X_tr, yb_tr, eval_set=[(X_va, yb_va)],
                    eval_metric="auc", categorical_feature=cat or "auto",
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            p_va = clf.predict_proba(X_va)[:, 1]
            p_te = clf.predict_proba(X_te)[:, 1]

            h_res = {"by_threshold": {}}
            for T in thresholds:
                yb_va_T, yb_te_T = (y_va >= T).astype(int), (y_te >= T).astype(int)
                # Regresión @ corte natural (línea base).
                nat = _scores(yb_te_T, s_te >= T, beta)
                # Regresión @ corte afinado en val (F-beta).
                grid = np.arange(-10.0, T + 10.0, 0.5)
                c_reg, _ = best_cutoff(yb_va_T, s_va, beta, grid)
                tuned_reg = _scores(yb_te_T, s_te >= c_reg, beta)
                entry = {"natural": nat, "tuned_reg": tuned_reg, "reg_cutoff": c_reg}
                # Clasificador (entrenado a 15) @ umbral afinado: solo T=15.
                if T == 15:
                    c_clf, _ = best_cutoff(yb_va_T, p_va, beta, np.linspace(0.05, 0.95, 37))
                    entry["tuned_clf"] = _scores(yb_te_T, p_te >= c_clf, beta)
                    entry["clf_prob_cutoff"] = c_clf
                h_res["by_threshold"][str(T)] = entry

            h_res["fit_seconds"] = round(time.time() - t0, 1)
            report["results"][str(h)] = h_res
            reg.save(out_dir / f"reg_h{h}.joblib")

            b15 = h_res["by_threshold"]["15"]
            rows.append((h, b15["natural"]["recall"], b15["natural"]["precision"],
                         b15["tuned_reg"]["recall"], b15["tuned_reg"]["precision"],
                         b15["tuned_clf"]["recall"], b15["tuned_clf"]["precision"]))
            print(f"[h={h}] @15  natural rec={b15['natural']['recall']:.3f}/pre={b15['natural']['precision']:.3f}"
                  f" | tuned_reg rec={b15['tuned_reg']['recall']:.3f}/pre={b15['tuned_reg']['precision']:.3f}"
                  f" | clf rec={b15['tuned_clf']['recall']:.3f}/pre={b15['tuned_clf']['precision']:.3f}"
                  f" ({h_res['fit_seconds']:.0f}s)", flush=True)
        except Exception as e:  # robustez overnight: no abortar toda la corrida
            print(f"[h={h}] ERROR: {e!r}", flush=True)
            report["results"][str(h)] = {"error": repr(e)}

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== recall-targeted (test, @15; beta={beta}) ===")
    print(f"{'H':>3s} {'nat_rec':>7s} {'nat_pre':>7s} {'tun_rec':>7s} {'tun_pre':>7s} {'clf_rec':>7s} {'clf_pre':>7s}")
    for h, nr, npre, tr, tp, cr, cp in rows:
        print(f"{h:3d} {nr:7.3f} {npre:7.3f} {tr:7.3f} {tp:7.3f} {cr:7.3f} {cp:7.3f}")
    print(f"[done] informe -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
