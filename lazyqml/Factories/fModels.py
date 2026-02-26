from lazyqml.Models import QNN, QNNBag, QSVM, FastQSVM, QKNN, BaseHybridQNNModel, BasicHybridModel
from lazyqml.Global.globalEnums import *
from lazyqml.Utils import get_simulation_type, get_max_bond_dim

import pennylane as qml
import numpy as np
import GPUtil

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, nqubits, embedding, ansatz,
                 n_class, layers=5, shots=1,
                 n_samples=1.0, n_features=1.0,
                 lr=0.01, batch_size=8, epochs=50,
                 seed=1234, backend=Backend.lightningQubit, numPredictors=10, K=20,
                 mem_budget_mb=None):
        
        if model == Model.FastQSVM:
            device = qml.device(backend.value, wires=nqubits)
            qnode  = qml.qnode(device, diff_method=None)

            return FastQSVM(nqubits=nqubits, embedding=embedding, qnode=qnode, mem_budget_mb=mem_budget_mb)

        if model == Model.QSVM:
            return QSVM(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend)

        elif model == Model.QKNN:
            return QKNN(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend, k=K)
        
        elif model == Model.QNN:
            #print(f"N = {nqubits} -> backend {backend.value}")

            params = {
                'nqubits': nqubits,
                'ansatz': ansatz,
                'embedding': embedding,
                'n_class': n_class,
                'layers': layers,
                'epochs': epochs,
                'shots': shots, 
                'lr': lr,
                'batch_size': batch_size,
                'seed': seed,
                'torch_device': "cpu",
                'backend': backend.value,
                'diff_method': "best"
            }

            # print(params)
            return QNN(**params)
        
        elif model == Model.QNN_BAG:
            return QNNBag(nqubits=nqubits, ansatz=ansatz, embedding=embedding, 
                          n_class=n_class, layers=layers, epochs=epochs, 
                          n_samples=n_samples, n_features=n_features,
                          shots=shots, lr=lr, batch_size=batch_size,
                          seed=seed, backend=backend,
                          n_estimators=numPredictors)