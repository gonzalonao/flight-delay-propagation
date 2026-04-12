"""Construcción de grafos de aeropuertos para modelos GNN.

Transforma datos tabulares de vuelos en grafos donde los nodos son
aeropuertos y las aristas son rutas aéreas. Genera snapshots temporales
con features agregadas por aeropuerto y targets continuos de retraso
(minutos de retraso promedio en la siguiente ventana temporal).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_airport_mapping(airports: list[str]) -> dict[str, int]:
    """Crea un mapeo de código IATA a índice numérico.

    Args:
        airports: Lista ordenada de códigos de aeropuerto.

    Returns:
        Diccionario {código_IATA: índice}.
    """
    return {code: idx for idx, code in enumerate(sorted(airports))}


def build_edge_index(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    min_flights: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construye la matriz de adyacencia (edge_index) desde las rutas.

    Crea aristas bidireccionales entre aeropuertos que tienen al menos
    `min_flights` vuelos en el dataset. Los pesos de las aristas son
    el número normalizado de vuelos en la ruta.

    Args:
        df: DataFrame con columnas Origin y Dest.
        airport_map: Mapeo de código IATA a índice.
        min_flights: Mínimo de vuelos para crear una arista.

    Returns:
        Tupla de (edge_index [2, num_edges], edge_weight [num_edges]).
    """
    # Contar vuelos por ruta
    route_counts = (
        df.groupby(["Origin", "Dest"])
        .size()
        .reset_index(name="count")
    )

    # Filtrar rutas con pocos vuelos
    route_counts = route_counts[route_counts["count"] >= min_flights]

    # Filtrar rutas cuyos aeropuertos están en el mapeo
    valid = set(airport_map.keys())
    route_counts = route_counts[
        route_counts["Origin"].isin(valid) & route_counts["Dest"].isin(valid)
    ]

    # Construir edge_index (bidireccional)
    sources = []
    targets = []
    weights = []

    for _, row in route_counts.iterrows():
        src = airport_map[row["Origin"]]
        dst = airport_map[row["Dest"]]
        w = row["count"]
        # Arista en ambas direcciones
        sources.extend([src, dst])
        targets.extend([dst, src])
        weights.extend([w, w])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # Normalizar pesos al rango [0, 1]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if weights_tensor.numel() > 0 and weights_tensor.max() > 0:
        weights_tensor = weights_tensor / weights_tensor.max()

    logger.info(
        "Grafo construido: %d nodos, %d aristas (min_flights=%d)",
        len(airport_map), edge_index.shape[1], min_flights,
    )

    return edge_index, weights_tensor


def compute_node_features(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    delay_threshold: float = 15.0,
) -> torch.Tensor:
    """Calcula features por aeropuerto para una ventana temporal.

    Features por nodo (aeropuerto como origen):
    - avg_dep_delay: Retraso de salida promedio
    - std_dep_delay: Desviación estándar del retraso
    - pct_delayed: Porcentaje de vuelos con retraso > umbral
    - num_flights: Número de vuelos (normalizado)
    - avg_arr_delay_incoming: Retraso de llegada promedio (como destino)
    - std_arr_delay_incoming: Desviación estándar del retraso de llegada

    Args:
        window_df: DataFrame filtrado a una ventana temporal.
        airport_map: Mapeo de código IATA a índice.
        delay_threshold: Umbral de minutos para considerar un vuelo retrasado.

    Returns:
        Tensor de node features [num_nodes, num_features].
    """
    num_nodes = len(airport_map)
    num_features = 6
    features = torch.zeros(num_nodes, num_features, dtype=torch.float32)

    if window_df.empty:
        return features

    # Features como aeropuerto de ORIGEN
    origin_groups = window_df.groupby("Origin")
    for airport, group in origin_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        dep_delays = group["DepDelay"].values
        features[idx, 0] = np.nanmean(dep_delays)
        features[idx, 1] = np.nanstd(dep_delays) if len(dep_delays) > 1 else 0.0
        features[idx, 2] = np.mean(dep_delays > delay_threshold)
        features[idx, 3] = len(dep_delays)

    # Features como aeropuerto de DESTINO (retrasos de llegada)
    dest_groups = window_df.groupby("Dest")
    for airport, group in dest_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        arr_delays = group["ArrDelay"].values
        features[idx, 4] = np.nanmean(arr_delays)
        features[idx, 5] = np.nanstd(arr_delays) if len(arr_delays) > 1 else 0.0

    # Normalizar num_flights
    max_flights = features[:, 3].max()
    if max_flights > 0:
        features[:, 3] = features[:, 3] / max_flights

    # Reemplazar NaN por 0
    features = torch.nan_to_num(features, nan=0.0)

    return features


def compute_node_targets(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
) -> torch.Tensor:
    """Genera targets continuos: retraso promedio de salida por aeropuerto.

    Calcula el retraso promedio (en minutos) de los vuelos que salen de
    cada aeropuerto en la ventana temporal futura. Esto permite evaluar
    con métricas de regresión y derivar clasificación binaria usando un
    umbral externo.

    Args:
        window_df: DataFrame de la ventana FUTURA (target).
        airport_map: Mapeo de código IATA a índice.

    Returns:
        Tensor continuo [num_nodes] con retraso promedio en minutos.
    """
    num_nodes = len(airport_map)
    targets = torch.zeros(num_nodes, dtype=torch.float32)

    if window_df.empty:
        return targets

    origin_groups = window_df.groupby("Origin")
    for airport, group in origin_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        targets[idx] = group["DepDelay"].mean()

    return torch.nan_to_num(targets, nan=0.0)


def create_temporal_graphs(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    window_hours: int = 1,
    delay_threshold: float = 15.0,
) -> list[Data]:
    """Crea snapshots de grafos temporales a partir de datos de vuelos.

    Divide los datos en ventanas temporales y genera un grafo por ventana.
    Las features de cada nodo son las estadísticas de retraso del aeropuerto
    en esa ventana. El target es si el aeropuerto estará retrasado en la
    ventana siguiente.

    Args:
        df: DataFrame preprocesado con FlightDate y Hour.
        airport_map: Mapeo de código IATA a índice.
        edge_index: Matriz de adyacencia [2, num_edges].
        edge_weight: Pesos de aristas [num_edges].
        window_hours: Horas por ventana temporal.
        delay_threshold: Umbral de retraso en minutos.

    Returns:
        Lista de objetos Data de PyG (un grafo por ventana).
    """
    # Crear timestamp combinando FlightDate + Hour
    df = df.copy()
    if "Hour" not in df.columns:
        df["Hour"] = (df["CRSDepTime"] // 100).clip(0, 23)

    df["timestamp"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
        df["Hour"], unit="h"
    )

    # Ordenar por timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Crear ventanas temporales
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    window_delta = pd.Timedelta(hours=window_hours)

    graphs = []
    current_time = min_time

    while current_time + 2 * window_delta <= max_time:
        # Ventana actual (features)
        window_mask = (
            (df["timestamp"] >= current_time)
            & (df["timestamp"] < current_time + window_delta)
        )
        window_df = df[window_mask]

        # Ventana siguiente (targets)
        next_mask = (
            (df["timestamp"] >= current_time + window_delta)
            & (df["timestamp"] < current_time + 2 * window_delta)
        )
        next_df = df[next_mask]

        # Solo crear grafo si ambas ventanas tienen datos suficientes
        if len(window_df) >= 10 and len(next_df) >= 10:
            node_features = compute_node_features(
                window_df, airport_map, delay_threshold
            )
            node_targets = compute_node_targets(next_df, airport_map)

            # Máscara de nodos con actividad (para ignorar aeropuertos sin datos)
            active_mask = node_features.abs().sum(dim=1) > 0

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weight.unsqueeze(-1),
                y=node_targets,
                active_mask=active_mask,
            )
            graphs.append(graph)

        current_time += window_delta

    logger.info(
        "Snapshots temporales creados: %d grafos (ventana=%dh, umbral=%d min)",
        len(graphs), window_hours, delay_threshold,
    )

    return graphs


def build_graph_dataset(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
) -> tuple[list[Data], dict[str, int]]:
    """Pipeline completo: de DataFrame a lista de grafos PyG.

    Args:
        df: DataFrame preprocesado (con FlightDate, Hour, Origin, Dest, etc.).
        airports: Lista de códigos de aeropuerto filtrados.
        config: Configuración con secciones 'graph' y 'evaluation'.

    Returns:
        Tupla de (lista de grafos Data, mapeo de aeropuertos).
    """
    graph_config = config.get("graph", {})
    eval_config = config.get("evaluation", {})

    airport_map = build_airport_mapping(airports)

    min_flights = graph_config.get("min_route_flights", 50)
    edge_index, edge_weight = build_edge_index(df, airport_map, min_flights)

    window_hours = graph_config.get("temporal_window_hours", 1)
    delay_threshold = eval_config.get("delay_threshold_minutes", 15)

    graphs = create_temporal_graphs(
        df=df,
        airport_map=airport_map,
        edge_index=edge_index,
        edge_weight=edge_weight,
        window_hours=window_hours,
        delay_threshold=delay_threshold,
    )

    return graphs, airport_map


def split_graphs_temporal(
    graphs: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[Data]]:
    """Divide los grafos temporales en train/val/test.

    Los grafos ya están ordenados temporalmente, así que se usa una
    separación secuencial para evitar fuga de información temporal.

    Args:
        graphs: Lista de grafos PyG ordenados temporalmente.
        train_ratio: Proporción de grafos para entrenamiento.
        val_ratio: Proporción de grafos para validación.

    Returns:
        Diccionario con claves 'train', 'val', 'test'.
    """
    n = len(graphs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": graphs[:train_end],
        "val": graphs[train_end:val_end],
        "test": graphs[val_end:],
    }

    for name, split in splits.items():
        logger.info("Graph split %s: %d grafos", name, len(split))

    return splits
