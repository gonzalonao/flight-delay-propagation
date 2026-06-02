"""Inbound-delay chaining for long horizons (recover the rotation signal).

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

Más allá de ~H=2 la etapa entrante del mismo avión aún no ha despegado en
``t_pred``, así que ``inbound_delay_estimate`` cae a NaN y el modelo se apoya en
el "suelo" del horario. Aquí **predecimos** el retraso de esa etapa entrante con
un modelo base que usa solo su *programación* (ruta, aerolínea, hora, calendario
— conocidos con antelación arbitraria, por tanto point-in-time), y lo inyectamos
como estimación entrante cuando aún no es observable.

Cadena de un paso:

    M_in : (schedule+route+airline) -> ArrDelay        (entrenado una vez)
    para cada vuelo con inbound 'scheduled' (cadena válida, sin despegar):
        inbound_delay_filled = M_in(programación de la etapa entrante)
    resto: inbound_delay_filled = inbound_delay_estimate observado (o NaN).

Se entrena el modelo outbound con dos features extra
(``inbound_delay_filled``, ``inbound_was_predicted``) sobre ``F_all_weather`` y
se compara, por horizonte, contra la línea base sin cadena (Parte 4).

Uso:
    set PYTHONIOENCODING=utf-8
    uv run python scripts/train_flight_chain.py
    uv run python scripts/train_flight_chain.py --sample-train 300000   # smoke
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
    CATEGORICAL_FEATURES, FEATURE_SETS, F_AIRLINE, F_ROUTE, F_SCHEDULE, TARGET,
    add_datetimes, build_airport_hourly_stats, build_work_frame, clean,
    filter_top_airports, load_flights,
)
from src.flight_level.model import FlightDelayModel
from src.flight_level.weather import add_weather_asof, load_weather_long


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


TUNED = dict(num_leaves=127, learning_rate=0.03, min_child_samples=100,
             colsample_bytree=0.7, subsample=0.9)
FEATS = FEATURE_SETS["F_all_weather"]
SCHED_FEATS = F_SCHEDULE + F_ROUTE + F_AIRLINE          # features de M_in
AUG = FEATS + ["inbound_delay_filled", "inbound_was_predicted"]


def inbound_schedule_X(work: pd.DataFrame) -> pd.DataFrame:
    """Programación de la etapa ENTRANTE (fila previa del mismo Tail), con los
    mismos nombres de columna que espera M_in. ``work`` viene tail-ordenado de
    ``build_work_frame``."""
    g = work.groupby("Tail_Number", sort=False)
    inb = pd.DataFrame(index=work.index)
    for c in SCHED_FEATS:
        inb[c] = g[c].shift(1)
    return inb


def assemble_chain(split_df, h, hourly, weather, m_in, sched_cats):
    """Construye (X_aug, y) con la estimación entrante predicha rellenando los
    casos 'scheduled' (no observables aún en t_pred)."""
    work = build_work_frame(split_df, h, hourly)
    work = add_weather_asof(work, weather, h)

    # Predicción de la etapa entrante a partir de su programación.
    inb_X = inbound_schedule_X(work)
    valid = inb_X[SCHED_FEATS].notna().all(axis=1)
    inb_pred = np.full(len(work), np.nan, dtype="float64")
    if valid.any():
        Xp = inb_X.loc[valid, SCHED_FEATS].copy()
        for c in sched_cats:                       # alinear categorías con M_in
            Xp[c] = pd.Categorical(Xp[c], categories=m_in.feature_categories_[c])
        inb_pred[valid.to_numpy()] = m_in.predict(Xp)

    status = work["inbound_status"].astype(str).to_numpy()
    filled = work["inbound_delay_estimate"].astype("float64").to_numpy().copy()
    mask_sched = status == "scheduled"
    filled[mask_sched] = inb_pred[mask_sched]
    work["inbound_delay_filled"] = filled.astype("float32")
    work["inbound_was_predicted"] = mask_sched.astype("int8")

    return work[AUG].copy(), work[TARGET].to_numpy(dtype="float32"), float(mask_sched.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/flight_level.yaml")
    ap.add_argument("--sample-train", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs/flight_level_chain")
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

    train_df = df[df["dep_dt"] < train_end]
    val_df = df[(df["dep_dt"] >= train_end) & (df["dep_dt"] < val_end)]
    test_df = df[df["dep_dt"] >= val_end]
    if args.sample_train and len(train_df) > args.sample_train:
        train_df = train_df.sample(args.sample_train, random_state=seed)
    print(f"[split] train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}", flush=True)

    # --- M_in: predictor del retraso de la etapa entrante (solo programación) ---
    sched_cats = [c for c in SCHED_FEATS if c in CATEGORICAL_FEATURES]
    Xs = train_df[SCHED_FEATS].copy()
    ys = train_df[TARGET].to_numpy(dtype="float32")
    Xs_v = val_df[SCHED_FEATS].copy()
    yv = val_df[TARGET].to_numpy(dtype="float32")
    Xs, (Xs_v,) = align_categories(Xs, [Xs_v], sched_cats)
    m_in = FlightDelayModel(categorical_features=sched_cats, backend="lightgbm",
                            random_state=seed, **TUNED)
    m_in.fit(Xs, ys, eval_set=[(Xs_v, yv)])
    m_in.feature_categories_ = {c: Xs[c].cat.categories for c in sched_cats}
    print(f"[M_in] entrenado (schedule->ArrDelay), MAE(val)="
          f"{np.abs(m_in.predict(Xs_v) - yv).mean():.2f}", flush=True)

    cat = [c for c in AUG if c in CATEGORICAL_FEATURES]
    report = {"config": cfg, "tuned_params": TUNED, "results": {}}
    rows = []
    for h in horizons:
        try:
            t0 = time.time()
            X_tr, y_tr, frac_tr = assemble_chain(train_df, h, hourly, weather, m_in, sched_cats)
            X_va, y_va, _ = assemble_chain(val_df, h, hourly, weather, m_in, sched_cats)
            X_te, y_te, frac_te = assemble_chain(test_df, h, hourly, weather, m_in, sched_cats)
            X_tr, (X_va, X_te) = align_categories(X_tr, [X_va, X_te], cat)

            model = FlightDelayModel(categorical_features=cat, backend="lightgbm",
                                     random_state=seed, **TUNED)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            metrics = model.evaluate(y_te, model.predict(X_te), thresholds)
            metrics["fit_seconds"] = round(time.time() - t0, 1)
            metrics["frac_chained_test"] = round(frac_te, 4)
            report["results"][str(h)] = metrics
            model.save(out_dir / f"chain_h{h}.joblib")

            bt = metrics["by_threshold"]
            rows.append((h, metrics["mae"], bt["15"]["recall"], bt["15"]["precision"],
                         bt["30"]["recall"], frac_te))
            print(f"[chain h={h}] MAE={metrics['mae']:.2f} rec@15={bt['15']['recall']:.3f} "
                  f"pre@15={bt['15']['precision']:.3f} rec@30={bt['30']['recall']:.3f} "
                  f"(chained={frac_te:.1%}, {metrics['fit_seconds']:.0f}s)", flush=True)
        except Exception as e:
            print(f"[chain h={h}] ERROR: {e!r}", flush=True)
            report["results"][str(h)] = {"error": repr(e)}

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== inbound-chaining (test) ===")
    print(f"{'H':>3s} {'MAE':>7s} {'rec@15':>7s} {'pre@15':>7s} {'rec@30':>7s} {'chained':>8s}")
    for h, mae, r15, p15, r30, fr in rows:
        print(f"{h:3d} {mae:7.2f} {r15:7.3f} {p15:7.3f} {r30:7.3f} {fr:7.1%}")
    print("\nLínea base sin cadena (F_all_weather, Parte 4):")
    print("  H1 rec@15=0.480  H4 rec@15=0.357  H6 rec@15=0.333  H8 rec@15=0.321")
    print(f"[done] informe -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
