"""Script de evaluación de modelos entrenados.

Carga un checkpoint y ejecuta métricas sobre el conjunto de test.

Uso:
    python scripts/evaluate.py --checkpoint outputs/best_dense_nn.pt --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FlightDelayDataset, create_splits, get_feature_columns
from src.data.features import build_feature_matrix
from src.data.graph_builder import build_graph_dataset, split_graphs_temporal
from src.data.loader import load_flight_data
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import evaluate_graph_model, evaluate_model
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_predictions_vs_actual,
)
from src.models.basic_gcn import BasicGCN
from src.models.dense_nn import DenseNN
from src.utils.config import load_config
from src.utils.io import get_data_dir, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

GRAPH_MODELS = {"basic_gcn"}

logger = setup_logger(__name__)


def _log_test_results(metrics: dict[str, float], model_name: str) -> None:
    """Muestra los resultados de test en formato unificado.

    Args:
        metrics: Diccionario con métricas unificadas.
        model_name: Nombre del modelo evaluado.
    """
    logger.info("=" * 50)
    logger.info("RESULTADOS EN TEST — %s", model_name)
    logger.info("-" * 50)
    logger.info("  Regresión:")
    logger.info("    MAE:  %.4f min", metrics["mae"])
    logger.info("    RMSE: %.4f min", metrics["rmse"])
    logger.info("    MAPE: %.4f %%", metrics["mape"])
    logger.info("    R²:   %.4f", metrics["r2"])
    logger.info("  Clasificación (umbral=15 min):")
    logger.info("    ACCURACY:  %.4f", metrics["accuracy"])
    logger.info("    PRECISION: %.4f", metrics["precision"])
    logger.info("    RECALL:    %.4f", metrics["recall"])
    logger.info("    F1:        %.4f", metrics["f1"])
    logger.info("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluar modelo entrenado")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint del modelo (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generar gráficos de evaluación",
    )
    return parser.parse_args()


def _build_model(model_name: str, model_config: dict, input_dim: int) -> torch.nn.Module:
    """Instancia el modelo según su nombre y configuración.

    Args:
        model_name: Nombre del modelo.
        model_config: Configuración específica del modelo.
        input_dim: Dimensión de entrada.

    Returns:
        Modelo de PyTorch.
    """
    if model_name == "dense_nn":
        return DenseNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            dropout=model_config.get("dropout", 0.3),
        )

    if model_name == "basic_gcn":
        return BasicGCN(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
        )

    raise ValueError(f"Modelo no reconocido: {model_name}")


def main() -> None:
    """Punto de entrada principal de la evaluación."""
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    model_name = config["model"]["name"]
    logger.info("=" * 60)
    logger.info("EVALUACIÓN - %s", model_name)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("=" * 60)

    # --- Carga de datos ---
    data_dir = get_data_dir("raw", config)
    year = config["data"].get("years", [2018])[0]
    columns = config["data"].get("columns")

    # Para evaluación NO se usa muestreo (evaluar sobre datos completos)
    df = load_flight_data(data_dir, year, columns=columns, sample_frac=None)

    # --- Preprocesamiento ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"].get(model_name, {})

    delay_threshold = config.get("evaluation", {}).get(
        "delay_threshold_minutes", 15
    )

    if model_name in GRAPH_MODELS:
        # --- Pipeline de grafos ---
        graphs, airport_map = build_graph_dataset(df, airports, config)
        graph_splits = split_graphs_temporal(graphs)
        test_graphs = graph_splits["test"]

        if not test_graphs:
            logger.error("No hay grafos de test para evaluar.")
            return

        input_dim = test_graphs[0].x.shape[1]
        model = _build_model(model_name, model_config, input_dim)

        checkpoint_info = load_checkpoint(args.checkpoint, model)
        logger.info("Checkpoint cargado: época %d, métricas=%s",
                    checkpoint_info["epoch"], checkpoint_info["metrics"])

        model = model.to(device)
        metrics = evaluate_graph_model(
            model, test_graphs, device, delay_threshold
        )

    else:
        # --- Pipeline tabular ---
        target_col = config.get("features", {}).get("target", "ArrDelay")
        df, encoders, scaler = build_feature_matrix(df, config)

        splits = create_splits(df, config, target_col=target_col)
        test_features, test_targets = splits["test"]

        input_dim = test_features.shape[1]
        model = _build_model(model_name, model_config, input_dim)

        checkpoint_info = load_checkpoint(args.checkpoint, model)
        logger.info("Checkpoint cargado: época %d, métricas=%s",
                    checkpoint_info["epoch"], checkpoint_info["metrics"])

        model = model.to(device)

        test_dataset = FlightDelayDataset(test_features, test_targets)
        batch_size = config["training"].get("batch_size", 512)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metrics = evaluate_model(model, test_loader, device, delay_threshold)

    # --- Resultados unificados ---
    _log_test_results(metrics, model_name)

    # --- Gráficos opcionales (solo modelos tabulares por ahora) ---
    if args.plots and model_name not in GRAPH_MODELS:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for features, targets in test_loader:
                preds = model(features.to(device)).squeeze(-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        plot_predictions_vs_actual(predictions, targets)
        plot_error_distribution(predictions, targets)


if __name__ == "__main__":
    main()
