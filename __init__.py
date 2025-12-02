# ecoeval/__init__.py
from .config import EcoEvalConfig
from .core import run_benchmark
from .datasets import load_dataset_by_name
from .energy import run_with_energy

__all__ = [
    "EcoEvalConfig",
    "run_benchmark",
    "load_dataset_by_name",
    "run_with_energy",
]
