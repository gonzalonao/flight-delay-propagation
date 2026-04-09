"""Configuración del sistema de logging del proyecto.

Configura un logger con formato consistente para usar en todos los módulos.
"""

import logging
import sys


def setup_logger(
    name: str = "flight_delay",
    level: int = logging.INFO,
) -> logging.Logger:
    """Crea y configura un logger con salida a consola.

    Args:
        name: Nombre del logger.
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si se llama múltiples veces
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
