"""Tests para el modelo DenseNN."""

import torch
import pytest

from src.models.dense_nn import DenseNN


class TestDenseNN:
    """Tests para la clase DenseNN."""

    def test_output_shape(self):
        """Verifica que la salida tiene la forma correcta."""
        model = DenseNN(input_dim=10, hidden_dims=[32, 16])
        x = torch.randn(8, 10)  # batch=8, features=10
        out = model(x)
        assert out.shape == (8, 1)

    def test_default_hidden_dims(self):
        """Verifica que las dimensiones por defecto funcionan."""
        model = DenseNN(input_dim=20)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 1)

    def test_single_sample(self):
        """Verifica que funciona con un solo ejemplo."""
        model = DenseNN(input_dim=5, hidden_dims=[8])
        x = torch.randn(1, 5)
        out = model(x)
        assert out.shape == (1, 1)

    def test_gradient_flow(self):
        """Verifica que los gradientes fluyen correctamente."""
        model = DenseNN(input_dim=10, hidden_dims=[16, 8])
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)

        out = model(x)
        loss = torch.nn.MSELoss()(out, target)
        loss.backward()

        # Verificar que todos los parámetros tienen gradientes
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"
