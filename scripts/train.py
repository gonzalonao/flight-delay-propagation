"""Script principal de entrenamiento.

Carga la configuración, prepara los datos, instancia el modelo
y ejecuta el bucle de entrenamiento.

Uso:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenar modelo de predicción de retrasos")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del entrenamiento."""
    args = parse_args()
    config = load_config(args.config)

    # Fijar semilla para reproducibilidad
    set_seed(config.get("reproducibility", {}).get("seed", 42))

    logger.info("Configuración cargada desde: %s", args.config)
    logger.info("Modelo seleccionado: %s", config["model"]["name"])

    # TODO: Implementar pipeline de datos (Phase 1)
    # TODO: Implementar entrenamiento (Phase 2+)
    logger.info("Pipeline de entrenamiento pendiente de implementación")


if __name__ == "__main__":
    main()
