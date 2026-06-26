"""Visualization functions for training and evaluation results.

Generates plots of training curves, error distributions and predictions vs
actual values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot the training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists.
        save_path: Path where the figure is saved (optional).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path | None = None,
    max_points: int = 5000,
) -> None:
    """Scatter plot of predictions vs actual values.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.
        save_path: Path where the figure is saved (optional).
        max_points: Maximum number of points to show (sampled if more).
    """
    # Sample if there are too many points
    if len(predictions) > max_points:
        idx = np.random.choice(len(predictions), max_points, replace=False)
        predictions = predictions[idx]
        targets = targets[idx]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(targets, predictions, alpha=0.3, s=10, color="steelblue")

    # Perfect-prediction line
    lims = [
        min(targets.min(), predictions.min()),
        max(targets.max(), predictions.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Delay (min)", fontsize=12)
    ax.set_ylabel("Predicted Delay (min)", fontsize=12)
    ax.set_title("Predictions vs Actual", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Histogram of the error distribution (prediction - actual).

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.
        save_path: Path where the figure is saved (optional).
    """
    errors = predictions - targets

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(
        x=np.mean(errors),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean error: {np.mean(errors):.1f} min",
    )

    ax.set_xlabel("Prediction Error (min)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Error Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
