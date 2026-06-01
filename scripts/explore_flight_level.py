"""Estudio exploratorio: ¿cuánta señal se pierde al agregar a nivel aeropuerto-hora?

Motivación
----------
El modelo GNN predice ``ArrDelay`` **agregado por (aeropuerto, hora)**. Al
promediar todos los vuelos de una franja horaria se descarta toda la
información que varía *dentro* de la hora:

* **Aerolínea operadora** (perfiles de puntualidad muy distintos).
* **Estado de rotación del avión**: el avión asignado a un vuelo llega tarde de
  su etapa anterior -> sale tarde. Es el mecanismo de propagación dominante
  (BTS atribuye ~30-40 % de los minutos de retraso a "Late Aircraft").
* Holgura de turnaround, etc.

Este script **cuantifica** ese valor antes de comprometer arquitectura: entrena
un GBM (``HistGradientBoostingRegressor``, nativo en sklearn) sobre conjuntos de
features anidados y mide la mejora marginal de cada bloque de señal.

Disciplina anti-fuga
--------------------
Para predecir ``ArrDelay`` de un vuelo SOLO se usan campos conocidos *antes* de
la salida:

* Programación: ruta, distancia, hora programada, duración programada, calendario.
* Identidad: aerolínea operadora (la matrícula NO se usa como feature; solo se
  emplea para reconstruir la rotación).
* Rotación: resultado de la **etapa anterior** del mismo avión (su ``ArrDelay``,
  el turnaround programado y la holgura). Supuesto: al predecir cerca de la
  salida, la llegada del avión entrante ya se observa (feature estándar de
  "late aircraft" en la literatura). En producción se sustituiría por la
  predicción del tramo entrante para horizontes largos.

Se EXCLUYEN explícitamente los campos posteriores a la salida del propio vuelo
(``DepDelay``, ``DepTime``, ``WheelsOff``, ``TaxiOut``, ``AirTime``,
``ActualElapsedTime``, ``ArrTime``...), que filtrarían el target.

Uso
---
    set PYTHONIOENCODING=utf-8
    uv run python scripts/explore_flight_level.py            # 2018-2019, top-70
    uv run python scripts/explore_flight_level.py --sample-train 500000  # rapido
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

# --- Constantes del estudio (deben coincidir con el GNN para comparar) -------
DEFAULT_YEARS = [2018, 2019]
DEFAULT_TOP_N = 70
DEFAULT_TRAIN_END = "2019-07-01"
DEFAULT_TEST_START = "2019-10-01"
THRESHOLDS = [15, 30, 45, 60]
RAW_DIR = Path("data/raw")

# Columnas crudas necesarias (solo pre-salida + identidad + target).
RAW_COLUMNS = [
    "FlightDate", "Operating_Airline", "Origin", "Dest", "Tail_Number",
    "CRSDepTime", "CRSArrTime", "Distance", "CRSElapsedTime",
    "DayOfWeek", "Month", "ArrDelay", "Cancelled", "Diverted",
]

# Bloques de features para la ablación.
F_SCHEDULE = ["Distance", "CRSElapsedTime", "dep_hour", "Month", "DayOfWeek"]
F_ROUTE = ["Origin", "Dest"]            # categóricas
F_AIRLINE = ["Operating_Airline"]       # categórica
F_ROTATION = [
    "prev_arr_delay", "sched_turnaround_min", "turnaround_slack",
    "is_first_leg", "prev_delayed15", "rotation_continuity",
]
CATEGORICALS = set(F_ROUTE + F_AIRLINE)


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------
def load_years(years: list[int]) -> pd.DataFrame:
    """Carga y concatena los parquet anuales con poda de columnas."""
    frames = []
    for y in years:
        path = RAW_DIR / f"Combined_Flights_{y}.parquet"
        print(f"[load] {path} ...", flush=True)
        frames.append(pd.read_parquet(path, columns=RAW_COLUMNS, engine="pyarrow"))
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] total filas crudas: {len(df):,}", flush=True)
    return df


def filter_top_airports(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Mismo criterio que el preprocesado del GNN: ambos extremos en el top-N."""
    activity = (
        df["Origin"].value_counts()
        .add(df["Dest"].value_counts(), fill_value=0)
        .sort_values(ascending=False)
    )
    top = set(activity.head(top_n).index)
    mask = df["Origin"].isin(top) & df["Dest"].isin(top)
    out = df[mask].copy()
    print(f"[filter] top-{top_n} aeropuertos: {len(df):,} -> {len(out):,} filas", flush=True)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina cancelados/desviados y filas sin target; tipa la fecha."""
    df = df[~df["Cancelled"].fillna(False) & ~df["Diverted"].fillna(False)].copy()
    df = df.dropna(subset=["ArrDelay"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ingeniería de features
# ---------------------------------------------------------------------------
def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """Convierte enteros HHMM (1430 -> 870) a minutos desde medianoche."""
    s = s.fillna(0).astype(int)
    return (s // 100).clip(0, 23) * 60 + (s % 100).clip(0, 59)


def add_schedule_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Construye dep_dt / arr_dt programados (con rollover nocturno) y dep_hour."""
    dep_min = _hhmm_to_minutes(df["CRSDepTime"])
    arr_min = _hhmm_to_minutes(df["CRSArrTime"])
    df["dep_hour"] = (dep_min // 60).astype("int16")
    df["dep_dt"] = df["FlightDate"] + pd.to_timedelta(dep_min, unit="m")
    arr_dt = df["FlightDate"] + pd.to_timedelta(arr_min, unit="m")
    # Llegada programada antes que salida -> vuelo nocturno, cae al día siguiente.
    overnight = arr_min < dep_min
    arr_dt = arr_dt + pd.to_timedelta(overnight.astype(int), unit="D")
    df["arr_dt"] = arr_dt
    return df


def add_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la cadena de rotación por matrícula y deriva la señal de
    'avión entrante tarde' (late aircraft).

    Para cada Tail_Number se ordenan los vuelos por hora de salida programada y
    se toma la **etapa anterior** del mismo avión:

    * ``prev_arr_delay``        retraso de llegada de esa etapa (min).
    * ``sched_turnaround_min``  minutos programados entre llegada previa y salida.
    * ``turnaround_slack``      turnaround - retraso entrante (negativo = el
                                retraso se come la holgura -> probable salida tarde).
    * ``is_first_leg``          1 si no hay etapa previa (primer vuelo del avión).
    * ``prev_delayed15``        1 si la etapa previa llegó con >=15 min.
    * ``rotation_continuity``   1 si el destino previo == origen actual (cadena
                                física real; 0 = hueco de datos/reposicionamiento).
    """
    df = df.sort_values(["Tail_Number", "dep_dt"], kind="stable")
    g = df.groupby("Tail_Number", sort=False)

    prev_arr_delay = g["ArrDelay"].shift(1)
    prev_arr_dt = g["arr_dt"].shift(1)
    prev_dest = g["Dest"].shift(1)

    turnaround = (df["dep_dt"] - prev_arr_dt).dt.total_seconds() / 60.0
    continuity = (prev_dest.values == df["Origin"].values)
    is_first = prev_arr_delay.isna()

    # Turnaround implausible (>12 h) o sin continuidad => no es la rotación real:
    # invalidamos las features de rotación (NaN -> el GBM lo maneja nativamente).
    bad_chain = (~continuity) | (turnaround > 12 * 60) | (turnaround < 0)
    prev_arr_delay = prev_arr_delay.where(~bad_chain)
    turnaround = turnaround.where(~bad_chain)

    df["prev_arr_delay"] = prev_arr_delay.astype("float32")
    df["sched_turnaround_min"] = turnaround.astype("float32")
    df["turnaround_slack"] = (turnaround - prev_arr_delay).astype("float32")
    df["is_first_leg"] = is_first.astype("int8")
    df["prev_delayed15"] = (prev_arr_delay >= 15).astype("float32")  # NaN-safe
    df["rotation_continuity"] = pd.Series(continuity, index=df.index).astype("int8")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_schedule_datetimes(df)
    df = add_rotation_features(df)
    # Tipar categóricas para el soporte nativo de HistGBR.
    for c in CATEGORICALS:
        df[c] = df[c].astype("category")
    return df


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, train_end: str, test_start: str):
    train = df[df["dep_dt"] < pd.Timestamp(train_end)]
    test = df[df["dep_dt"] >= pd.Timestamp(test_start)]
    return train, test


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Recall/precision/F1 derivados de la regresión a varios umbrales."""
    out = {}
    for thr in THRESHOLDS:
        t = y_true >= thr
        p = y_pred >= thr
        rec = recall_score(t, p, zero_division=0)
        prec = precision_score(t, p, zero_division=0)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        out[str(thr)] = {
            "positive_rate": float(t.mean()),
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
        }
    return out


def fit_eval(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int
) -> tuple[HistGradientBoostingRegressor, dict, np.ndarray]:
    cat = [f for f in features if f in CATEGORICALS]
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0,
        categorical_features=cat if cat else None,
        random_state=seed,
    )
    t0 = time.time()
    model.fit(train[features], train["ArrDelay"])
    y_pred = model.predict(test[features])
    y_true = test["ArrDelay"].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "fit_seconds": round(time.time() - t0, 1),
        "by_threshold": classification_metrics(y_true, y_pred),
    }
    return model, metrics, y_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    ap.add_argument("--test-start", default=DEFAULT_TEST_START)
    ap.add_argument("--sample-train", type=int, default=None,
                    help="Submuestrea N filas de train para acelerar (None = todo).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/research/flight_level")
    args = ap.parse_args()

    df = load_years(args.years)
    df = filter_top_airports(df, args.top_n)
    df = clean(df)
    df = build_features(df)

    train, test = temporal_split(df, args.train_end, args.test_start)
    print(f"[split] train={len(train):,}  test={len(test):,}", flush=True)
    if args.sample_train and len(train) > args.sample_train:
        train = train.sample(args.sample_train, random_state=args.seed)
        print(f"[split] train submuestreado -> {len(train):,}", flush=True)

    # Conjuntos de features anidados (ablación).
    feature_sets = {
        "A_schedule":            F_SCHEDULE + F_ROUTE,
        "B_schedule_airline":    F_SCHEDULE + F_ROUTE + F_AIRLINE,
        "C_schedule_rotation":   F_SCHEDULE + F_ROUTE + F_ROTATION,
        "D_all":                 F_SCHEDULE + F_ROUTE + F_AIRLINE + F_ROTATION,
    }

    results = {}
    full_model = None
    for name, feats in feature_sets.items():
        print(f"\n[fit] {name}  ({len(feats)} features) ...", flush=True)
        model, metrics, _ = fit_eval(train, test, feats, args.seed)
        results[name] = {"features": feats, **metrics}
        r15 = metrics["by_threshold"]["15"]
        print(f"  MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
              f"recall@15={r15['recall']:.3f}  prec@15={r15['precision']:.3f}",
              flush=True)
        if name == "D_all":
            full_model = (model, feats)

    # Importancia por permutación sobre el modelo completo (submuestra de test).
    print("\n[perm-importance] modelo D_all sobre 40k de test ...", flush=True)
    model, feats = full_model
    test_s = test.sample(min(40000, len(test)), random_state=args.seed)
    pi = permutation_importance(
        model, test_s[feats], test_s["ArrDelay"],
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=args.seed,
    )
    importances = sorted(
        ({"feature": f, "mae_increase": float(m), "std": float(s)}
         for f, m, s in zip(feats, pi.importances_mean, pi.importances_std)),
        key=lambda d: d["mae_increase"], reverse=True,
    )

    # Persistir.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "years": args.years, "top_n": args.top_n,
            "train_end": args.train_end, "test_start": args.test_start,
            "n_train": int(len(train)), "n_test": int(len(test)),
            "gnn_baseline": {"mae": 10.0, "recall_at_15": 0.352, "precision_at_15": 0.523},
        },
        "ablation": results,
        "permutation_importance_D_all": importances,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] informe -> {out_dir / 'report.json'}")

    # Resumen legible en consola.
    print("\n=== RESUMEN ABLACIÓN (test) ===")
    print(f"{'set':24s} {'MAE':>7s} {'RMSE':>7s} {'rec@15':>7s} {'rec@30':>7s} {'rec@60':>7s}")
    for name, r in results.items():
        bt = r["by_threshold"]
        print(f"{name:24s} {r['mae']:7.3f} {r['rmse']:7.3f} "
              f"{bt['15']['recall']:7.3f} {bt['30']['recall']:7.3f} {bt['60']['recall']:7.3f}")
    print(f"\nGNN baseline (agg aeropuerto-hora): MAE~10.0  recall@15~0.352\n")
    print("=== TOP FEATURES (aumento de MAE al permutar) ===")
    for d in importances[:10]:
        print(f"  {d['feature']:24s} +{d['mae_increase']:.3f}")


if __name__ == "__main__":
    main()
