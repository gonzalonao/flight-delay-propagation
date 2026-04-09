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
from src.data.loader import load_flight_data
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import compute_all_metrics, evaluate_model
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_predictions_vs_actual,
)
from src.models.dense_nn import DenseNN
from src.utils.config import load_config
from src.utils.io import get_data_dir, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


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


def main() -> None:
    """Punto de entrada principal de la evaluación."""
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("EVALUACIÓN - %s", config["model"]["name"])
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("=" * 60)

    # --- Carga de datos (solo test) ---
    data_dir = get_data_dir("raw", config)
    year = config["data"].get("years", [2018])[0]
    columns = config["data"].get("columns")

    # Para evaluación NO se usa muestreo (evaluar sobre datos completos)
    df = load_flight_data(data_dir, year, columns=columns, sample_frac=None)

    # --- Preprocesamiento ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    target_col = config.get("features", {}).get("target", "ArrDelay")
    df, encoders, scaler = build_feature_matrix(df, config)

    splits = create_splits(df, config, target_col=target_col)
    test_features, test_targets = splits["test"]

    # --- Modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_features.shape[1]

    model_name = config["model"]["name"]
    model_config = config["model"].get(model_name, {})

    if model_name == "dense_nn":
        model = DenseNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            dropout=model_config.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    # Cargar checkpoint
    checkpoint_info = load_checkpoint(args.checkpoint, model)
    logger.info("Checkpoint cargado: época %d, métricas=%s",
                checkpoint_info["epoch"], checkpoint_info["metrics"])

    model = model.to(device)

    # --- Evaluación ---
    test_dataset = FlightDelayDataset(test_features, test_targets)
    batch_size = config["training"].get("batch_size", 512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metrics = evaluate_model(model, test_loader, device)

    logger.info("=" * 40)
    logger.info("RESULTADOS EN TEST:")
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name.upper(), value)
    logger.info("=" * 40)

    # --- Gráficos opcionales ---
    if args.plots:
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
