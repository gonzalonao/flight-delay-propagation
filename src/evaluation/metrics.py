"""Métricas de evaluación unificadas para modelos de predicción de retrasos.

Todos los modelos (tabulares y GNN) producen predicciones de regresión
(retraso en minutos) y se evalúan con el mismo conjunto de métricas:
- Regresión: MAE, RMSE, MAPE, R²
- Clasificación derivada: Accuracy, Precision, Recall, F1
  (aplicando un umbral sobre la predicción continua de ArrDelay)

A partir de W2 los modelos multi-tarea (``output_channels=3``) producen
también un canal de **logits** para ``pct_arr_delayed_15``. Cuando ese
canal está presente, se computan métricas de clasificación adicionales
con el prefijo ``bce_`` (``bce_accuracy``, ``bce_precision``, etc.) que
salen directamente del head BCE (sigmoid + threshold 0.5). Las dos
familias de métricas coexisten para permitir comparación apples-to-
apples entre la decisión derivada del regresor y la del clasificador
dedicado.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.graph_builder import (
    NUM_TARGET_CHANNELS,
    TARGET_CHANNEL_ARR_DELAY,
    TARGET_CHANNEL_PCT_DELAYED,
)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        MAE en las mismas unidades que los datos (minutos).
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        RMSE en las mismas unidades que los datos (minutos).
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mape(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1.0
) -> float:
    """Mean Absolute Percentage Error.

    Usa epsilon para evitar división por cero en targets cercanos a 0.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.
        epsilon: Valor mínimo del denominador.

    Returns:
        MAPE como porcentaje (0-100+).
    """
    denominator = np.maximum(np.abs(targets), epsilon)
    return float(np.mean(np.abs(predictions - targets) / denominator) * 100)


def compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Coeficiente de determinación R².

    R² = 1 significa predicción perfecta.
    R² = 0 significa que el modelo es igual de bueno que predecir la media.
    R² < 0 significa que el modelo es peor que predecir la media.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        Valor R².
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Calcula todas las métricas de regresión.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        Diccionario con mae, rmse, mape, r2.
    """
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "mape": compute_mape(predictions, targets),
        "r2": compute_r2(predictions, targets),
    }


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    pred_threshold: float = 15.0,
    label_threshold: float | None = None,
) -> dict[str, float]:
    """Calcula métricas de clasificación binaria derivadas de regresión.

    Convierte predicciones y targets continuos (minutos de retraso) a clases
    binarias. El **label** se binariza con ``label_threshold`` (la verdad de
    negocio: ArrDelay ≥ 15 min) y la **predicción** con ``pred_threshold`` (el
    *punto de operación* del regresor, ajustable). Desacoplarlos permite subir
    recall bajando ``pred_threshold`` sin redefinir qué cuenta como retraso
    real — un ajuste de operating-point estándar, no un cambio de problema.

    Si ``label_threshold`` es ``None`` se iguala a ``pred_threshold``, lo que
    reproduce el comportamiento histórico (un único umbral para ambos lados).

    Args:
        predictions: Predicciones continuas (minutos de retraso).
        targets: Valores reales continuos (minutos de retraso).
        pred_threshold: Umbral aplicado a la predicción (operating-point).
        label_threshold: Umbral que define el positivo real. ``None`` ⇒
            igual a ``pred_threshold`` (compatibilidad hacia atrás).

    Returns:
        Diccionario con accuracy, precision, recall, f1.
    """
    if label_threshold is None:
        label_threshold = pred_threshold

    pred_labels = (predictions >= pred_threshold).astype(int)
    target_labels = (targets >= label_threshold).astype(int)

    tp = np.sum((pred_labels == 1) & (target_labels == 1))
    fp = np.sum((pred_labels == 1) & (target_labels == 0))
    fn = np.sum((pred_labels == 0) & (target_labels == 1))
    tn = np.sum((pred_labels == 0) & (target_labels == 0))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_bce_classification_metrics(
    pct_logits: np.ndarray,
    pct_targets: np.ndarray,
    bce_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> dict[str, float]:
    """Métricas de clasificación binaria derivadas del head BCE de W2.

    El modelo emite ``logits`` para ``pct_arr_delayed_15`` (logit ∈ ℝ);
    aplicamos sigmoid y comparamos con ``bce_threshold`` (0.5 por
    defecto) para obtener la etiqueta predicha. El target es la
    fracción real de vuelos delayed en la ventana ∈ [0, 1]; se binariza
    contra ``target_threshold`` (por defecto 0.5 — "más de la mitad
    delayed" cuenta como ventana retrasada). Bajar ``bce_threshold``
    es la palanca natural para subir recall a cambio de precision.

    Args:
        pct_logits: Logits ``[N]`` del canal pct (pre-sigmoid).
        pct_targets: Targets ``[N]`` ∈ [0, 1] del mismo canal.
        bce_threshold: Probabilidad mínima para predecir "delayed".
        target_threshold: Fracción mínima del target que cuenta como
            etiqueta positiva.

    Returns:
        Dict con ``bce_accuracy``, ``bce_precision``, ``bce_recall``,
        ``bce_f1``.
    """
    probs = 1.0 / (1.0 + np.exp(-pct_logits))
    pred_labels = (probs >= bce_threshold).astype(int)
    target_labels = (pct_targets >= target_threshold).astype(int)

    tp = int(np.sum((pred_labels == 1) & (target_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (target_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (target_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (target_labels == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "bce_accuracy": float(accuracy),
        "bce_precision": float(precision),
        "bce_recall": float(recall),
        "bce_f1": float(f1),
    }


def compute_classification_sweep(
    predictions: np.ndarray,
    targets: np.ndarray,
    pred_thresholds: np.ndarray,
    label_threshold: float = 15.0,
) -> list[dict[str, float]]:
    """Barre umbrales de predicción y devuelve P/R/F1 por umbral.

    El label se mantiene fijo en ``label_threshold`` (verdad de negocio); sólo
    se mueve el umbral aplicado a la predicción. Útil para elegir el punto de
    operación que maximiza recall a una precisión mínima (ver
    ``pick_pred_threshold``).

    Args:
        predictions: Predicciones continuas (minutos).
        targets: Valores reales continuos (minutos).
        pred_thresholds: Umbrales de predicción a evaluar.
        label_threshold: Umbral fijo que define el positivo real.

    Returns:
        Lista de dicts ``{"pred_threshold", "precision", "recall", "f1",
        "accuracy"}`` en el mismo orden que ``pred_thresholds``.
    """
    rows: list[dict[str, float]] = []
    for t in pred_thresholds:
        m = compute_classification_metrics(
            predictions, targets,
            pred_threshold=float(t), label_threshold=label_threshold,
        )
        rows.append({"pred_threshold": float(t), **m})
    return rows


def pick_pred_threshold(
    predictions: np.ndarray,
    targets: np.ndarray,
    candidate_thresholds: np.ndarray,
    min_precision: float = 0.55,
    label_threshold: float = 15.0,
) -> tuple[float, dict[str, float]]:
    """Elige el umbral de predicción que maximiza recall con un piso de precisión.

    Pensado para ajustar el punto de operación **en validación** y luego
    reportarlo en test. Entre los umbrales cuya precisión ≥ ``min_precision``
    devuelve el de mayor recall (desempata por F1). Si ninguno alcanza el piso,
    cae al umbral de mayor F1 (para no devolver un operating-point degenerado).

    Args:
        predictions: Predicciones continuas de validación.
        targets: Valores reales de validación.
        candidate_thresholds: Rejilla de umbrales candidatos.
        min_precision: Precisión mínima aceptable.
        label_threshold: Umbral fijo que define el positivo real.

    Returns:
        ``(best_threshold, metrics_at_best)``.
    """
    sweep = compute_classification_sweep(
        predictions, targets, candidate_thresholds, label_threshold,
    )
    feasible = [r for r in sweep if r["precision"] >= min_precision]
    pool = feasible if feasible else sweep
    best = max(pool, key=lambda r: (r["recall"], r["f1"]))
    return best["pred_threshold"], best


def compute_bce_sweep(
    pct_logits: np.ndarray,
    pct_targets: np.ndarray,
    bce_thresholds: np.ndarray,
    target_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Barre umbrales de probabilidad del head BCE y devuelve P/R/F1 por umbral."""
    rows: list[dict[str, float]] = []
    for b in bce_thresholds:
        m = compute_bce_classification_metrics(
            pct_logits, pct_targets,
            bce_threshold=float(b), target_threshold=target_threshold,
        )
        rows.append({"bce_threshold": float(b), **m})
    return rows


def pick_bce_threshold(
    pct_logits: np.ndarray,
    pct_targets: np.ndarray,
    candidate_thresholds: np.ndarray,
    min_precision: float = 0.55,
    target_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Análogo de ``pick_pred_threshold`` para el head BCE (sobre probabilidad)."""
    sweep = compute_bce_sweep(
        pct_logits, pct_targets, candidate_thresholds, target_threshold,
    )
    feasible = [r for r in sweep if r["bce_precision"] >= min_precision]
    pool = feasible if feasible else sweep
    best = max(pool, key=lambda r: (r["bce_recall"], r["bce_f1"]))
    return best["bce_threshold"], best


def compute_unified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    delay_threshold: float = 15.0,
    pct_logits: np.ndarray | None = None,
    pct_targets: np.ndarray | None = None,
    pred_threshold: float | None = None,
    bce_threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula métricas unificadas: regresión + clasificación derivada.

    Función estándar para evaluar todos los modelos del proyecto.
    Combina métricas de regresión (MAE, RMSE, MAPE, R²) con métricas
    de clasificación (Accuracy, Precision, Recall, F1) derivadas
    de aplicar un umbral de retraso a las predicciones continuas.

    Si se pasan ``pct_logits`` y ``pct_targets`` (ruta multi-tarea de
    W2), se añaden además las métricas ``bce_*`` del head BCE
    dedicado, permitiendo comparar la decisión thresholded-from-
    regression con la del clasificador entrenado explícitamente.

    Args:
        predictions: Predicciones continuas de ArrDelay (minutos).
        targets: Valores reales de ArrDelay (minutos).
        delay_threshold: Umbral que define el positivo real (label, min).
        pct_logits: Logits del canal pct_arr_delayed (opcional).
        pct_targets: Targets ∈ [0, 1] del canal pct (opcional).
        pred_threshold: Punto de operación del regresor (min). ``None`` ⇒
            igual a ``delay_threshold`` (comportamiento histórico). Bajarlo
            sube recall sin redefinir el label.
        bce_threshold: Umbral de probabilidad del head BCE (default 0.5).

    Returns:
        Diccionario con todas las métricas. Si los args ``pct_*`` están
        presentes incluye también ``bce_accuracy/precision/recall/f1``.
    """
    if pred_threshold is None:
        pred_threshold = delay_threshold

    regression = compute_all_metrics(predictions, targets)
    classification = compute_classification_metrics(
        predictions, targets,
        pred_threshold=pred_threshold, label_threshold=delay_threshold,
    )
    out = {**regression, **classification}

    if pct_logits is not None and pct_targets is not None:
        out.update(
            compute_bce_classification_metrics(
                pct_logits, pct_targets, bce_threshold=bce_threshold,
            )
        )

    return out


@torch.no_grad()
def evaluate_multi_horizon_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    prediction_horizons: list[int],
    delay_threshold: float = 15.0,
    pred_threshold: float | None = None,
    bce_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Evalúa un modelo GNN multi-horizonte sobre grafos temporales.

    Calcula métricas unificadas para cada horizonte de predicción
    y un promedio global.

    Args:
        model: Modelo GNN multi-horizonte en modo eval.
        graphs: Lista de grafos PyG con targets [num_nodes, num_horizons].
        device: Dispositivo (cpu/cuda).
        prediction_horizons: Lista de horizontes (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Umbral que define el positivo real (label, min).
        pred_threshold: Operating-point del regresor (min); ``None`` ⇒
            ``delay_threshold``.
        bce_threshold: Umbral de probabilidad del head BCE (default 0.5).

    Returns:
        Diccionario con métricas por horizonte y promedio:
        {"horizon_1h": {...}, "horizon_2h": {...}, ..., "average": {...}}
    """
    model.eval()
    num_horizons = len(prediction_horizons)

    # Acumular predicciones y targets por horizonte. ``pct_*`` sólo se
    # popula si el modelo emite el tercer canal (multi-tarea W2).
    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_logits: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    has_pct_head = False

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = (
            graph.edge_attr.to(device) if graph.edge_attr is not None else None
        )
        mask = graph.active_mask

        # Salida [num_nodes, num_horizons] o [num_nodes, num_horizons, C]
        preds = model(x, edge_index, edge_attr=edge_attr).cpu().numpy()
        targets_np = graph.y.numpy()
        mask_np = mask.numpy()

        # Detecta si el modelo emite el head pct (canal 2). Asumimos que
        # si la última dim coincide con NUM_TARGET_CHANNELS estamos en
        # multi-task; las dimensiones se ramifican aquí, no en el modelo.
        pred_has_channels = preds.ndim == 3
        target_has_channels = targets_np.ndim == 3

        for h_idx in range(num_horizons):
            if pred_has_channels:
                all_preds[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_preds[h_idx].append(preds[mask_np, h_idx])

            if target_has_channels:
                all_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_targets[h_idx].append(targets_np[mask_np, h_idx])

            if pred_has_channels and target_has_channels and preds.shape[-1] >= NUM_TARGET_CHANNELS:
                has_pct_head = True
                all_pct_logits[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )
                all_pct_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )

    if not all_preds[0]:
        empty = {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
        result = {}
        for h in prediction_horizons:
            result[f"horizon_{h}h"] = empty.copy()
        result["average"] = empty.copy()
        return result

    # Calcular métricas por horizonte
    result: dict[str, dict[str, float]] = {}
    all_metrics_for_avg: list[dict[str, float]] = []

    for h_idx, h in enumerate(prediction_horizons):
        preds_h = np.concatenate(all_preds[h_idx])
        targets_h = np.concatenate(all_targets[h_idx])
        if has_pct_head:
            pct_logits_h = np.concatenate(all_pct_logits[h_idx])
            pct_targets_h = np.concatenate(all_pct_targets[h_idx])
            metrics_h = compute_unified_metrics(
                preds_h, targets_h, delay_threshold,
                pct_logits=pct_logits_h, pct_targets=pct_targets_h,
                pred_threshold=pred_threshold, bce_threshold=bce_threshold,
            )
        else:
            metrics_h = compute_unified_metrics(
                preds_h, targets_h, delay_threshold,
                pred_threshold=pred_threshold, bce_threshold=bce_threshold,
            )
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    # Promedio de métricas
    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(
            np.mean([m[key] for m in all_metrics_for_avg])
        )
    result["average"] = avg_metrics

    return result


@torch.no_grad()
def collect_sequence_predictions(
    model: torch.nn.Module,
    sequences: list[list[Data]],
    device: torch.device,
    num_horizons: int,
) -> dict[str, object]:
    """Forward único sobre cada secuencia → arrays crudos por horizonte.

    Recolecta predicciones y targets (enmascarados con ``active_mask`` del
    último grafo de cada secuencia) **sin** aplicar ningún umbral. Esto permite
    barrer puntos de operación (``pred_threshold`` / ``bce_threshold``) en
    memoria sin re-ejecutar el modelo — el caso de uso del sweep de Phase 1.

    Args:
        model: Modelo que acepta ``list[Data]``.
        sequences: Lista de secuencias (cada una ``list[Data]``).
        device: Dispositivo.
        num_horizons: Número de horizontes que emite el modelo.

    Returns:
        Dict con:
          - ``arr_pred`` / ``arr_target``: ``list[np.ndarray]`` (uno por
            horizonte, concatenado sobre todas las secuencias) del canal
            ArrDelay.
          - ``pct_logit`` / ``pct_target``: idem para el head BCE, o ``None``
            si el modelo no emite el canal pct.
          - ``has_pct``: bool.
          - ``n_items``: nº de secuencias procesadas.
    """
    model.eval()
    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_logits: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    has_pct_head = False

    for sequence in sequences:
        seq_on_device = []
        for g in sequence:
            g_dev = g.clone()
            g_dev.x = g.x.to(device)
            g_dev.edge_index = g.edge_index.to(device)
            if g.edge_attr is not None:
                g_dev.edge_attr = g.edge_attr.to(device)
            if g.y is not None:
                g_dev.y = g.y.to(device)
            if g.active_mask is not None:
                g_dev.active_mask = g.active_mask.to(device)
            seq_on_device.append(g_dev)

        preds = model(seq_on_device).cpu().numpy()

        last_graph = sequence[-1]
        targets_np = last_graph.y.numpy()
        mask_np = last_graph.active_mask.numpy()

        pred_has_channels = preds.ndim == 3
        target_has_channels = targets_np.ndim == 3

        for h_idx in range(num_horizons):
            if pred_has_channels:
                all_preds[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_preds[h_idx].append(preds[mask_np, h_idx])

            if target_has_channels:
                all_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_targets[h_idx].append(targets_np[mask_np, h_idx])

            if pred_has_channels and target_has_channels and preds.shape[-1] >= NUM_TARGET_CHANNELS:
                has_pct_head = True
                all_pct_logits[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )
                all_pct_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )

    def _cat(rows: list[list[np.ndarray]]) -> list[np.ndarray]:
        return [
            np.concatenate(r) if r else np.array([], dtype=float) for r in rows
        ]

    return {
        "arr_pred": _cat(all_preds),
        "arr_target": _cat(all_targets),
        "pct_logit": _cat(all_pct_logits) if has_pct_head else None,
        "pct_target": _cat(all_pct_targets) if has_pct_head else None,
        "has_pct": has_pct_head,
        "n_items": len(sequences),
    }


@torch.no_grad()
def evaluate_multi_horizon_sequence_model(
    model: torch.nn.Module,
    sequences: list[list[Data]],
    device: torch.device,
    prediction_horizons: list[int],
    delay_threshold: float = 15.0,
    pred_threshold: float | None = None,
    bce_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Evalúa un modelo de secuencias multi-horizonte (SpatioTemporalGNN).

    Igual que ``evaluate_multi_horizon_graph_model`` pero opera sobre
    secuencias de grafos en vez de grafos individuales. Los targets y la
    máscara se toman del último grafo de cada secuencia. Reutiliza
    ``collect_sequence_predictions`` para el forward.

    Args:
        model: Modelo que acepta ``list[Data]`` (ya en device).
        sequences: Lista de secuencias, cada una es ``list[Data]``.
        device: Dispositivo (cpu/cuda).
        prediction_horizons: Lista de horizontes (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Umbral que define el positivo real (label, min).
        pred_threshold: Operating-point del regresor (min); ``None`` ⇒
            ``delay_threshold``.
        bce_threshold: Umbral de probabilidad del head BCE (default 0.5).

    Returns:
        Diccionario con métricas por horizonte y promedio:
        {"horizon_1h": {...}, ..., "average": {...}}
    """
    num_horizons = len(prediction_horizons)

    if not sequences:
        empty = {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
        result = {f"horizon_{h}h": empty.copy() for h in prediction_horizons}
        result["average"] = empty.copy()
        return result

    collected = collect_sequence_predictions(
        model, sequences, device, num_horizons,
    )
    has_pct_head = collected["has_pct"]

    result: dict[str, dict[str, float]] = {}
    all_metrics_for_avg: list[dict[str, float]] = []

    for h_idx, h in enumerate(prediction_horizons):
        preds_h = collected["arr_pred"][h_idx]
        targets_h = collected["arr_target"][h_idx]
        if has_pct_head:
            metrics_h = compute_unified_metrics(
                preds_h, targets_h, delay_threshold,
                pct_logits=collected["pct_logit"][h_idx],
                pct_targets=collected["pct_target"][h_idx],
                pred_threshold=pred_threshold, bce_threshold=bce_threshold,
            )
        else:
            metrics_h = compute_unified_metrics(
                preds_h, targets_h, delay_threshold,
                pred_threshold=pred_threshold, bce_threshold=bce_threshold,
            )
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(
            np.mean([m[key] for m in all_metrics_for_avg])
        )
    result["average"] = avg_metrics

    return result


@torch.no_grad()
def evaluate_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    delay_threshold: float = 15.0,
    pred_threshold: float | None = None,
    bce_threshold: float = 0.5,
) -> dict[str, float]:
    """Evalúa un modelo GNN sobre una lista de grafos temporales.

    El modelo produce predicciones continuas (minutos de retraso)
    y se evalúa con métricas unificadas de regresión + clasificación.

    Args:
        model: Modelo GNN en modo eval.
        graphs: Lista de grafos PyG con active_mask.
        device: Dispositivo (cpu/cuda).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas unificadas (regresión + clasificación).
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = (
            graph.edge_attr.to(device) if graph.edge_attr is not None else None
        )
        mask = graph.active_mask

        # Salida directa del modelo (regresión, sin sigmoid)
        preds = model(x, edge_index, edge_attr=edge_attr).squeeze(-1)
        preds = preds.cpu().numpy()

        all_predictions.append(preds[mask.numpy()])
        all_targets.append(graph.y[mask].numpy())

    if not all_predictions:
        return {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return compute_unified_metrics(
        predictions, targets, delay_threshold,
        pred_threshold=pred_threshold, bce_threshold=bce_threshold,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    delay_threshold: float = 15.0,
    pred_threshold: float | None = None,
    bce_threshold: float = 0.5,
) -> dict[str, float]:
    """Evalúa un modelo tabular sobre un dataloader completo.

    Calcula métricas unificadas de regresión + clasificación derivada.

    Args:
        model: Modelo de PyTorch en modo eval.
        dataloader: DataLoader con datos de test.
        device: Dispositivo (cpu/cuda).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas unificadas (regresión + clasificación).
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for features, targets in dataloader:
        features = features.to(device)
        predictions = model(features).squeeze(-1).cpu().numpy()
        all_predictions.append(predictions)
        all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return compute_unified_metrics(
        predictions, targets, delay_threshold,
        pred_threshold=pred_threshold, bce_threshold=bce_threshold,
    )
