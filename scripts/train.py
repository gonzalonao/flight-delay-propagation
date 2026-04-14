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
from src.data.graph_builder import (
    build_graph_dataset,
    create_temporal_sequences,
    split_graphs_temporal,
)
from src.data.loader import load_flight_data
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import (
    evaluate_graph_model,
    evaluate_model,
    evaluate_multi_horizon_graph_model,
    evaluate_multi_horizon_sequence_model,
)
from src.models.basic_gcn import BasicGCN
from src.models.dense_nn import DenseNN
from src.models.multi_horizon_gat import MultiHorizonGAT
from src.models.seq2seq_gnn import Seq2SeqGNN
from src.models.spatiotemporal_gnn import SpatioTemporalGNN
from src.training.graph_trainer import GraphTrainer, SequenceGraphTrainer
from src.training.losses import WeightedMSELoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import get_data_dir, get_output_dir
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


MODEL_REGISTRY = {
    "dense_nn": DenseNN,
    "basic_gcn": BasicGCN,
    "multi_horizon_gat": MultiHorizonGAT,
    "spatiotemporal_gnn": SpatioTemporalGNN,
    "seq2seq_gnn": Seq2SeqGNN,
}

# Modelos que usan grafos PyG en vez de datos tabulares
GRAPH_MODELS = {
    "basic_gcn", "multi_horizon_gat", "spatiotemporal_gnn", "seq2seq_gnn",
}

# Modelos que producen targets multi-horizonte
MULTI_HORIZON_MODELS = {
    "multi_horizon_gat", "spatiotemporal_gnn", "seq2seq_gnn",
}

# Modelos que procesan secuencias de grafos temporales
SEQUENCE_MODELS = {"spatiotemporal_gnn", "seq2seq_gnn"}


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

    if model_name == "basic_gcn":
        return BasicGCN(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
        )

    if model_name == "multi_horizon_gat":
        graph_config = config.get("graph", {})
        horizons = graph_config.get("prediction_horizons", [1, 2, 3, 4, 5])
        return MultiHorizonGAT(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 3),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
        )

    if model_name == "spatiotemporal_gnn":
        graph_config = config.get("graph", {})
        horizons = graph_config.get("prediction_horizons", [1, 2, 3, 4, 5])
        return SpatioTemporalGNN(
            input_dim=input_dim,
            gnn_hidden=model_config.get("gnn_hidden", 64),
            lstm_hidden=model_config.get("lstm_hidden", 128),
            num_heads=model_config.get("num_heads", 4),
            num_gnn_layers=model_config.get("num_gnn_layers", 2),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
        )

    if model_name == "seq2seq_gnn":
        graph_config = config.get("graph", {})
        horizons = graph_config.get("prediction_horizons", [1, 2, 3, 4, 5])
        return Seq2SeqGNN(
            input_dim=input_dim,
            gnn_hidden=model_config.get("gnn_hidden", 64),
            lstm_hidden=model_config.get("lstm_hidden", 128),
            num_heads=model_config.get("num_heads", 4),
            num_gnn_layers=model_config.get("num_gnn_layers", 2),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
        )

    raise ValueError(f"Modelo no reconocido: {model_name}")


def _build_optimizer_and_scheduler(
    model: torch.nn.Module, config: dict
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Crea optimizador y scheduler según la configuración.

    Args:
        model: Modelo de PyTorch.
        config: Configuración completa.

    Returns:
        Tupla de (optimizador, scheduler o None).
    """
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

    return optimizer, scheduler


def _log_test_results(metrics: dict[str, float], model_name: str) -> None:
    """Muestra los resultados de test en formato unificado.

    Agrupa las métricas en regresión y clasificación derivada para
    facilitar la comparación entre modelos.

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


def _log_multi_horizon_results(
    metrics: dict[str, dict[str, float]],
    model_name: str,
    horizons: list[int],
) -> None:
    """Muestra los resultados multi-horizonte en formato unificado.

    Args:
        metrics: Diccionario con métricas por horizonte y promedio.
        model_name: Nombre del modelo evaluado.
        horizons: Lista de horizontes evaluados.
    """
    logger.info("=" * 60)
    logger.info("RESULTADOS EN TEST — %s (multi-horizonte)", model_name)
    logger.info("=" * 60)

    for h in horizons:
        key = f"horizon_{h}h"
        m = metrics[key]
        logger.info("  Horizonte +%dh:", h)
        logger.info(
            "    MAE: %.4f | RMSE: %.4f | MAPE: %.4f%% | R²: %.4f",
            m["mae"], m["rmse"], m["mape"], m["r2"],
        )
        logger.info(
            "    Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f",
            m["accuracy"], m["precision"], m["recall"], m["f1"],
        )

    avg = metrics["average"]
    logger.info("-" * 60)
    logger.info("  PROMEDIO (todos los horizontes):")
    logger.info("    Regresión:")
    logger.info("      MAE:  %.4f min", avg["mae"])
    logger.info("      RMSE: %.4f min", avg["rmse"])
    logger.info("      MAPE: %.4f %%", avg["mape"])
    logger.info("      R²:   %.4f", avg["r2"])
    logger.info("    Clasificación (umbral=15 min):")
    logger.info("      ACCURACY:  %.4f", avg["accuracy"])
    logger.info("      PRECISION: %.4f", avg["precision"])
    logger.info("      RECALL:    %.4f", avg["recall"])
    logger.info("      F1:        %.4f", avg["f1"])
    logger.info("=" * 60)


def _train_tabular(config: dict, df, airports: list[str]) -> None:
    """Pipeline de entrenamiento para modelos tabulares (DenseNN, LSTM)."""
    target_col = config.get("features", {}).get("target", "ArrDelay")
    df, encoders, scaler = build_feature_matrix(df, config)

    splits = create_splits(df, config, target_col=target_col)
    train_features, train_targets = splits["train"]
    val_features, val_targets = splits["val"]

    train_dataset = FlightDelayDataset(train_features, train_targets)
    val_dataset = FlightDelayDataset(val_features, val_targets)

    batch_size = config["training"].get("batch_size", 512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    model = build_model(config, input_dim)
    logger.info("Modelo: %s | Parámetros: %d",
                config["model"]["name"],
                sum(p.numel() for p in model.parameters()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)
    criterion = torch.nn.MSELoss()

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

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
    )

    # Evaluación final
    if "test" in splits:
        test_features, test_targets = splits["test"]
        test_dataset = FlightDelayDataset(test_features, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        metrics = evaluate_model(model, test_loader, device, delay_threshold)
        _log_test_results(metrics, config["model"]["name"])

    logger.info("Entrenamiento completado. Checkpoint: %s", checkpoint_path)


def _train_graph(config: dict, df, airports: list[str]) -> None:
    """Pipeline de entrenamiento para modelos basados en grafos (GCN, GAT)."""
    model_name = config["model"]["name"]
    is_multi_horizon = model_name in MULTI_HORIZON_MODELS

    # Para modelos single-horizon, no pasar prediction_horizons al builder
    graph_config = config.copy()
    if not is_multi_horizon:
        graph_config = {**config, "graph": {**config.get("graph", {})}}
        graph_config["graph"].pop("prediction_horizons", None)

    # Construir grafos temporales
    graphs, airport_map = build_graph_dataset(df, airports, graph_config)
    graph_splits = split_graphs_temporal(graphs)

    train_graphs = graph_splits["train"]
    val_graphs = graph_splits["val"]

    if not train_graphs:
        logger.error(
            "No hay grafos de entrenamiento. Revisa los datos y la configuración."
        )
        return

    # El input_dim viene de las node features del primer grafo
    input_dim = train_graphs[0].x.shape[1]
    model = build_model(config, input_dim)
    logger.info(
        "Modelo: %s | Parámetros: %d | Nodos: %d",
        model_name,
        sum(p.numel() for p in model.parameters()),
        len(airport_map),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)

    # Seleccionar función de pérdida
    loss_config = training_config.get("loss", "mse")
    if loss_config == "weighted_mse":
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        delay_weight = training_config.get("delay_weight", 2.0)
        horizon_weights = training_config.get("horizon_weights")
        criterion = WeightedMSELoss(
            high_delay_threshold=delay_threshold,
            high_delay_weight=delay_weight,
            horizon_weights=horizon_weights,
        )
        hw_str = f", horizon_weights={horizon_weights}" if horizon_weights else ""
        logger.info(
            "Pérdida: WeightedMSE (umbral=%.0f min, peso=%.1f%s)",
            delay_threshold, delay_weight, hw_str,
        )
    else:
        criterion = torch.nn.MSELoss()

    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{model_name}.pt")

    trainer = GraphTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    trainer.fit(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
    )

    # Evaluación final sobre grafos de test
    test_graphs = graph_splits["test"]
    if test_graphs:
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )

        if is_multi_horizon:
            horizons = config.get("graph", {}).get(
                "prediction_horizons", [1, 2, 3, 4, 5]
            )
            metrics = evaluate_multi_horizon_graph_model(
                model, test_graphs, device, horizons, delay_threshold
            )
            _log_multi_horizon_results(metrics, model_name, horizons)
        else:
            metrics = evaluate_graph_model(
                model, test_graphs, device, delay_threshold
            )
            _log_test_results(metrics, model_name)

    logger.info("Entrenamiento completado. Checkpoint: %s", checkpoint_path)


def _train_sequence_graph(config: dict, df, airports: list[str]) -> None:
    """Pipeline de entrenamiento para modelos de secuencias de grafos.

    Para modelos como SpatioTemporalGNN que procesan secuencias de
    snapshots temporales consecutivos (e.g., 6 horas de historia).
    """
    model_name = config["model"]["name"]

    # Construir grafos temporales (multi-horizonte)
    graphs, airport_map = build_graph_dataset(df, airports, config)
    graph_splits = split_graphs_temporal(graphs)

    # Crear secuencias por split (evita fuga entre conjuntos)
    input_window = config.get("graph", {}).get("input_window", 6)
    train_sequences = create_temporal_sequences(
        graph_splits["train"], input_window
    )
    val_sequences = create_temporal_sequences(
        graph_splits["val"], input_window
    )

    if not train_sequences:
        logger.error(
            "No hay secuencias de entrenamiento. Revisa los datos y "
            "la configuración (input_window=%d, grafos_train=%d).",
            input_window, len(graph_splits["train"]),
        )
        return

    logger.info(
        "Secuencias: train=%d, val=%d (input_window=%d)",
        len(train_sequences), len(val_sequences), input_window,
    )

    # El input_dim viene de las node features del primer grafo
    input_dim = train_sequences[0][0].x.shape[1]
    model = build_model(config, input_dim)
    logger.info(
        "Modelo: %s | Parámetros: %d | Nodos: %d | Secuencia: %d grafos",
        model_name,
        sum(p.numel() for p in model.parameters()),
        len(airport_map),
        input_window,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)

    # Seleccionar función de pérdida (misma lógica que _train_graph)
    loss_config = training_config.get("loss", "mse")
    if loss_config == "weighted_mse":
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        delay_weight = training_config.get("delay_weight", 2.0)
        horizon_weights = training_config.get("horizon_weights")
        criterion = WeightedMSELoss(
            high_delay_threshold=delay_threshold,
            high_delay_weight=delay_weight,
            horizon_weights=horizon_weights,
        )
        hw_str = f", horizon_weights={horizon_weights}" if horizon_weights else ""
        logger.info(
            "Pérdida: WeightedMSE (umbral=%.0f min, peso=%.1f%s)",
            delay_threshold, delay_weight, hw_str,
        )
    else:
        criterion = torch.nn.MSELoss()

    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{model_name}.pt")

    trainer = SequenceGraphTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    trainer.fit(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
    )

    # Evaluación final sobre secuencias de test
    test_sequences = create_temporal_sequences(
        graph_splits["test"], input_window
    )
    if test_sequences:
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        horizons = config.get("graph", {}).get(
            "prediction_horizons", [1, 2, 3, 4, 5]
        )
        metrics = evaluate_multi_horizon_sequence_model(
            model, test_sequences, device, horizons, delay_threshold
        )
        _log_multi_horizon_results(metrics, model_name, horizons)

    logger.info("Entrenamiento completado. Checkpoint: %s", checkpoint_path)


def main() -> None:
    """Punto de entrada principal del entrenamiento."""
    args = parse_args()
    config = load_config(args.config)

    # Fijar semilla para reproducibilidad
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    model_name = config["model"]["name"]
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO - %s", model_name)
    logger.info("=" * 60)

    # --- Carga de datos ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    sample_frac = config["data"].get("sample_frac")

    logger.info("Cargando datos: años=%s, muestreo=%.1f%%",
                years, (sample_frac or 1.0) * 100)

    year = years[0]
    df = load_flight_data(data_dir, year, columns=columns,
                          sample_frac=sample_frac, random_seed=seed)

    # --- Preprocesamiento ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    # --- Entrenar según tipo de modelo ---
    if model_name in SEQUENCE_MODELS:
        _train_sequence_graph(config, df, airports)
    elif model_name in GRAPH_MODELS:
        _train_graph(config, df, airports)
    else:
        _train_tabular(config, df, airports)


if __name__ == "__main__":
    main()
