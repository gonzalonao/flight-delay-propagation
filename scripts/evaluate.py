"""Script de evaluación de modelos entrenados.

Carga un checkpoint y ejecuta métricas sobre el conjunto de test.

Uso:
    python scripts/evaluate.py --checkpoint outputs/best_model.pt --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger

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
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal de la evaluación."""
    args = parse_args()
    config = load_config(args.config)

    logger.info("Evaluando checkpoint: %s", args.checkpoint)
    logger.info("Modelo: %s", config["model"]["name"])

    # TODO: Cargar datos de test
    # TODO: Cargar modelo y checkpoint
    # TODO: Ejecutar métricas
    logger.info("Pipeline de evaluación pendiente de implementación")


if __name__ == "__main__":
    main()
