"""Tests para BaseTrainer y los wrappers subclase (W4.3).

Verifican que tras el colapso de la lógica común:

- :class:`BaseTrainer` ejecuta el ciclo train/val/early-stop/checkpoint.
- :class:`Trainer` (tabular), :class:`GraphTrainer` y
  :class:`SequenceGraphTrainer` preservan su API pública histórica
  (``train_loader``, ``train_graphs``, ``train_sequences``).
- La máscara de nodos activos se respeta y los batches sin nodos
  activos se ignoran sin romper el promedio de pérdida.
- ``_align_for_val`` extrae el canal de ArrDelay cuando el modelo
  emite salidas multi-tarea ``[N, H, 3]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
pyg = pytest.importorskip("torch_geometric")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from src.training.base_trainer import BaseTrainer  # noqa: E402
from src.training.graph_trainer import (  # noqa: E402
    GraphTrainer,
    SequenceGraphTrainer,
    _align_for_val_channel0,
)
from src.training.trainer import Trainer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures sintéticos
# ---------------------------------------------------------------------------


def _make_tabular_loader(n: int = 32, in_dim: int = 4, batch_size: int = 8) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, in_dim)
    y = (x.sum(dim=1) * 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _make_graph(num_nodes: int = 6, in_dim: int = 3, multi_horizon: bool = False) -> Data:
    torch.manual_seed(0)
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor(
        [[i, (i + 1) % num_nodes] for i in range(num_nodes)],
        dtype=torch.long,
    ).t().contiguous()
    if multi_horizon:
        y = torch.randn(num_nodes, 3)
    else:
        y = torch.randn(num_nodes)
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    return Data(x=x, edge_index=edge_index, y=y, active_mask=active_mask)


class _TinyDense(nn.Module):
    def __init__(self, in_dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _TinyGNN(nn.Module):
    """Modelo de grafos minimal sin GCN: ignora aristas, mapea features."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self._out_dim = out_dim

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        out = self.fc(x)
        return out  # [N, out_dim]


class _TinySeqGNN(nn.Module):
    """Modelo secuencial minimal: promedia features del último grafo."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self._out_dim = out_dim

    def forward(self, sequence: list[Data]) -> torch.Tensor:
        last = sequence[-1]
        out = self.fc(last.x)
        if self._out_dim == 1:
            return out.squeeze(-1)
        return out


# ---------------------------------------------------------------------------
# BaseTrainer
# ---------------------------------------------------------------------------


class TestBaseTrainerCoreLoop:
    """Comportamiento del bucle común a través de las tres subclases."""

    def test_tabular_trainer_runs_and_history_shapes(self, tmp_path: Path):
        loader = _make_tabular_loader()
        model = _TinyDense()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        history = trainer.fit(
            train_loader=loader, val_loader=loader,
            epochs=2, patience=10,
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            model_name="tiny_dense",
        )
        assert set(history.keys()) == {"train_loss", "val_loss"}
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_graph_trainer_runs(self):
        graphs = [_make_graph() for _ in range(3)]
        model = _TinyGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = GraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        loss = trainer.train_epoch(graphs)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_sequence_trainer_runs(self):
        sequences = [[_make_graph() for _ in range(2)] for _ in range(3)]
        model = _TinySeqGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        loss = trainer.train_epoch(sequences)
        assert isinstance(loss, float)
        assert loss >= 0


class TestActiveMaskHandling:
    """La máscara de nodos activos se aplica y los batches vacíos se omiten."""

    def test_graphs_with_zero_active_mask_are_skipped(self):
        good = _make_graph()
        bad = _make_graph()
        bad.active_mask = torch.zeros_like(bad.active_mask)
        graphs = [good, bad]

        model = _TinyGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = GraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        loss = trainer.train_epoch(graphs)
        # Solo el grafo "good" contribuye; el promedio se calcula sobre n=1.
        assert isinstance(loss, float)
        assert loss >= 0

    def test_all_empty_masks_returns_zero(self):
        graphs = []
        for _ in range(2):
            g = _make_graph()
            g.active_mask = torch.zeros_like(g.active_mask)
            graphs.append(g)

        model = _TinyGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = GraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        # max(n_batches, 1) evita la división por cero; debe devolver 0.0.
        assert trainer.train_epoch(graphs) == 0.0


class TestValAlignment:
    """``_align_for_val`` extrae canal 0 cuando hay multi-tarea."""

    def test_align_for_val_slices_channel_0_when_3d(self):
        pred = torch.randn(6, 5, 3)
        target = torch.randn(6, 5, 3)
        pred_v, target_v = _align_for_val_channel0(pred, target)
        assert pred_v.shape == (6, 5)
        assert target_v.shape == (6, 5)
        assert torch.equal(pred_v, pred[..., 0])
        assert torch.equal(target_v, target[..., 0])

    def test_align_for_val_is_identity_when_2d(self):
        pred = torch.randn(6, 5)
        target = torch.randn(6, 5)
        pred_v, target_v = _align_for_val_channel0(pred, target)
        assert torch.equal(pred_v, pred)
        assert torch.equal(target_v, target)


class TestEarlyStoppingPath:
    """``fit`` corta antes del max de épocas si val_loss no mejora."""

    def test_early_stops_with_short_patience(self, tmp_path: Path):
        loader = _make_tabular_loader()
        model = _TinyDense()
        # LR=0 ⇒ ningún parámetro cambia ⇒ val_loss constante ⇒ early stop.
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        history = trainer.fit(
            train_loader=loader, val_loader=loader,
            epochs=20, patience=2,
            checkpoint_path=None,
        )
        assert len(history["train_loss"]) < 20


class TestSubclassApiBackcompat:
    """Las subclases mantienen los nombres de parámetro históricos."""

    def test_trainer_fit_accepts_train_loader_kwarg(self):
        loader = _make_tabular_loader(n=8, batch_size=4)
        model = _TinyDense()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        trainer.fit(train_loader=loader, val_loader=loader, epochs=1, patience=5)

    def test_graph_trainer_fit_accepts_train_graphs_kwarg(self):
        graphs = [_make_graph() for _ in range(2)]
        model = _TinyGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = GraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        trainer.fit(train_graphs=graphs, val_graphs=graphs, epochs=1, patience=5)

    def test_sequence_trainer_fit_accepts_train_sequences_kwarg(self):
        sequences = [[_make_graph() for _ in range(2)] for _ in range(2)]
        model = _TinySeqGNN()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        trainer.fit(
            train_sequences=sequences, val_sequences=sequences,
            epochs=1, patience=5,
        )


class TestBaseTrainerDirectInstantiation:
    """``BaseTrainer`` requiere implementar ``_forward_batch``."""

    def test_raises_when_forward_batch_not_overridden(self):
        model = _TinyDense()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        base = BaseTrainer(
            model=model, optimizer=opt,
            criterion=nn.MSELoss(),
            device=torch.device("cpu"),
        )
        loader = _make_tabular_loader(n=4, batch_size=4)
        with pytest.raises(NotImplementedError):
            base.train_epoch(loader)
