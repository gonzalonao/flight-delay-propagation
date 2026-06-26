"""Dense (fully connected) neural network as a baseline model.

A simple architecture of linear layers with dropout and a configurable
activation. Serves as a reference for comparing against more complex models
(LSTM, GNN).
"""

import torch
import torch.nn as nn


class DenseNN(nn.Module):
    """Fully connected neural network for delay regression.

    Configurable architecture with N hidden layers, dropout and activation.
    Predicts the arrival delay (ArrDelay) in minutes.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden-layer dimensions (e.g., [256, 128, 64]).
        dropout: Dropout probability between layers.
        activation: Activation function ("relu" or "elu").
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

        # Select activation function
        act_fn = nn.ReLU() if activation == "relu" else nn.ELU()

        # Build layers sequentially
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer: a single value (regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim].

        Returns:
            Predictions [batch_size, 1].
        """
        if x.size(0) == 1 and self.training:
            self.eval()
            out = self.network(x)
            self.train()
            return out
        return self.network(x)
