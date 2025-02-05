"""Package level variables and initialises the package logger"""

__version__: str = "0.1.0"

from typing import Optional
import logging

from phaethon.celestial_objects import (
    CircularOrbitFromSemiMajorAxis,
    CircularOrbitFromPeriod,
    Planet,
    PlanetarySystem,
    Star,
)
from phaethon.outgassing import VapourEngine
from phaethon.gas_mixture import IdealGasMixture
from phaethon.pipeline import PhaethonPipeline
from phaethon.fastchem_coupling import FastChemCoupler

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter."""
    fmt: str = (
        "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()]"
    )
    fmt += " - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Set up the logging for debugging: debug to the console."""
    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger


def debug_file_logger(logfile: str = f"{__package__}.log") -> logging.Logger:
    """Set up the logging to a file (debug) and to the console (info)."""

    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    package_logger.addHandler(console_handler)
    
    # File logger
    file_handler: logging.Handler = logging.FileHandler(logfile)
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)

    return package_logger
