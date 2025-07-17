from lazyqml.Models import QNNTorch, QNNBag, QSVM, QKNN
from lazyqml.Global.globalEnums import *
from lazyqml.Utils import get_simulation_type, get_max_bond_dim

import pennylane as qml
import numpy as np

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, nqubits, embedding, ansatz,
                 n_class, layers=5, shots=1,
                 n_samples=1.0, n_features=1.0,
                 lr=0.01, batch_size=8, epochs=50,
                 seed=1234, backend=Backend.lightningQubit, numPredictors=10, K=20):
        
        if model == Model.QSVM:
            return QSVM(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend)
        
        elif model == Model.QKNN:
            return QKNN(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend, k=K)
        
        elif model == Model.QNN:

            # Create device
            if get_simulation_type() == "tensor":
                if backend != Backend.lightningTensor:
                    device_kwargs = {
                        "max_bond_dim": get_max_bond_dim(),
                        "cutoff": np.finfo(np.complex128).eps,
                        "contract": "auto-mps",
                    }
                else:
                    device_kwargs = {
                        "max_bond_dim": get_max_bond_dim(),
                        "cutoff": 1e-10,
                        "cutoff_mode": "abs",
                    }
                    
                qdevice = qml.device(backend.value, wires=nqubits, method='mps', **device_kwargs)
                diff_method = 'best'
            else:
                qdevice = qml.device(backend.value, wires=nqubits)
                diff_method = 'adjoint'

            return QNNTorch(nqubits=nqubits, ansatz=ansatz, 
                        embedding=embedding, n_class=n_class, 
                        layers=layers, epochs=epochs, shots=shots, 
                        lr=lr, batch_size=batch_size, seed=seed, device=qdevice, backend=backend, diff_method=diff_method)
        
        elif model == Model.QNN_BAG:
            return QNNBag(nqubits=nqubits, ansatz=ansatz, embedding=embedding, 
                          n_class=n_class, layers=layers, epochs=epochs, 
                          n_samples=n_samples, n_features=n_features,
                          shots=shots, lr=lr, batch_size=batch_size,
                          seed=seed, backend=backend,
                          n_estimators=numPredictors)