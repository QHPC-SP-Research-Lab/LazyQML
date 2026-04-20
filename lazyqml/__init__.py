"""Top-level package for lazyqml"""

__author__ = """QHPC Group (University of Oviedo)"""
__email__ = "https://qhpc.uniovi.es"
__version__ = "0.1.10"

from .lazyqml import QuantumClassifier

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

__all__ = ['QuantumClassifier']