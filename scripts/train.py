"""Script principal de entrenamiento.

Carga la configuración, prepara los datos, instancia el modelo
y ejecuta el bucle de entrenamiento.

Uso:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FlightDelayDataset, create_splits, get_feature_columns
from src.data.features import build_feature_matrix
from src.data.loader import load_flight_data
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import evaluate_model
from src.models.dense_nn import DenseNN
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import get_data_dir, get_output_dir
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


MODEL_REGISTRY = {
    "dense_nn": DenseNN,
}


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo de predicción de retrasos"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    return parser.parse_args()


def build_model(config: dict, input_dim: int) -> torch.nn.Module:
    """Instancia el modelo según la configuración.

    Args:
        config: Configuración completa.
        input_dim: Número de features de entrada.

    Returns:
        Modelo de PyTorch.
    """
    model_name = config["model"]["name"]
    model_config = config["model"].get(model_name, {})

    if model_name == "dense_nn":
        return DenseNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "relu"),
        )

    raise ValueError(f"Modelo no reconocido: {model_name}")


def main() -> None:
    """Punto de entrada principal del entrenamiento."""
    args = parse_args()
    config = load_config(args.config)

    # Fijar semilla para reproducibilidad
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO - %s", config["model"]["name"])
    logger.info("=" * 60)

    # --- Carga de datos ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    sample_frac = config["data"].get("sample_frac")

    logger.info("Cargando datos: años=%s, muestreo=%.1f%%",
                years, (sample_frac or 1.0) * 100)

    # Cargar un año a la vez para gestionar memoria
    year = years[0]
    df = load_flight_data(data_dir, year, columns=columns,
                          sample_frac=sample_frac, random_seed=seed)

    # --- Preprocesamiento ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    # --- Feature engineering ---
    target_col = config.get("features", {}).get("target", "ArrDelay")
    df, encoders, scaler = build_feature_matrix(df, config)

    # --- Train/Val/Test split ---
    splits = create_splits(df, config, target_col=target_col)
    train_features, train_targets = splits["train"]
    val_features, val_targets = splits["val"]

    # --- Datasets y DataLoaders ---
    train_dataset = FlightDelayDataset(train_features, train_targets)
    val_dataset = FlightDelayDataset(val_features, val_targets)

    batch_size = config["training"].get("batch_size", 512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Modelo ---
    input_dim = train_features.shape[1]
    model = build_model(config, input_dim)
    logger.info("Modelo: %s | Parámetros: %d",
                config["model"]["name"],
                sum(p.numel() for p in model.parameters()))

    # --- Optimizador y scheduler ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    training_config = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
    )

    scheduler_name = training_config.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.get("epochs", 100)
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
    else:
        scheduler = None

    criterion = torch.nn.MSELoss()

    # --- Entrenamiento ---
    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{config['model']['name']}.pt")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
    )

    # --- Evaluación final ---
    if "test" in splits:
        test_features, test_targets = splits["test"]
        test_dataset = FlightDelayDataset(test_features, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metrics = evaluate_model(model, test_loader, device)
        logger.info("=" * 40)
        logger.info("RESULTADOS EN TEST:")
        for name, value in metrics.items():
            logger.info("  %s: %.4f", name.upper(), value)
        logger.info("=" * 40)

    logger.info("Entrenamiento completado. Checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
