"""Top-level package for lazyqml."""

__author__ = """Diego García Vega, Fernando Álvaro Plou Llorente, Alejandro Leal Castaño"""
__email__ = "garciavdiego@uniovi.es, ploufernando@uniovi.es, lealcalejandro@uniovi.es"
__version__ = "0.0.20"

from .lazyqml import QuantumClassifier

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

__all__ = ['QuantumClassifier']