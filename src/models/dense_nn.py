"""Red neuronal densa (fully connected) como modelo baseline.

Arquitectura simple de capas lineales con dropout y activación configurable.
Sirve como referencia para comparar con modelos más complejos (LSTM, GNN).
"""

import torch
import torch.nn as nn


class DenseNN(nn.Module):
    """Red neuronal fully connected para regresión de retrasos.

    Arquitectura configurable con N capas ocultas, dropout y activación.
    Predice el retraso de llegada (ArrDelay) en minutos.

    Args:
        input_dim: Número de features de entrada.
        hidden_dims: Lista con dimensiones de capas ocultas (e.g., [256, 128, 64]).
        dropout: Probabilidad de dropout entre capas.
        activation: Función de activación ("relu" o "elu").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Seleccionar función de activación
        act_fn = nn.ReLU() if activation == "relu" else nn.ELU()

        # Construir capas secuencialmente
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Capa de salida: un valor (regresión)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor de entrada [batch_size, input_dim].

        Returns:
            Predicciones [batch_size, 1].
        """
        return self.network(x)
