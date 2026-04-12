from lazyqml.Models import HybridCNNQNN, QKNN, FastQKNN, MPSQKNN, QNN, QNNBag, MPSQNN, QSVM, FastQSVM, MPSQSVM
from lazyqml.Global.globalEnums import *
from lazyqml.Utils import get_simulation_type

import pennylane as qml

class ModelFactory:
    def __init__(self) -> None:
        pass

    def getModel(self, model, nqubits, embedding, ansatz, n_class, input_shape=None, layers=5, shots=1, n_samples=1.0, n_features=1.0,
                lr=0.01, batch_size=8, epochs=50, seed=1234, backend=Backend.lightningQubit, numPredictors=10, K=5, mem_budget_mb=None):

        simulation_type    = get_simulation_type()
        statevector_models = {Model.QKNN, Model.FastQKNN, Model.QSVM, Model.FastQSVM, Model.QNN, Model.QNN_BAG}
        tensor_models      = {Model.MPSQKNN, Model.MPSQSVM, Model.MPSQNN}

        if model in statevector_models and simulation_type != "statevector":
            raise ValueError(f"For {model.name}, the simulation type must be 'statevector'")

        if model in tensor_models and simulation_type != "tensor":
            raise ValueError(f"For {model.name}, the simulation type must be 'tensor'")

        if model == Model.HybridCNNQNN:
            if input_shape is None:
                raise ValueError("HybridCNNQNN requires input_shape.")

            params = {"input_shape": input_shape, "nqubits": nqubits, "ansatz": ansatz, "embedding": embedding, "n_class": n_class, "layers": layers,
                    "epochs": epochs, "lr": lr, "shots": shots, "batch_size": batch_size, "torch_device": "cpu", "backend": backend, "diff_method": "best", "seed": seed}
            return HybridCNNQNN(**params)


        if model == Model.QKNN:
            return QKNN(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend, k=K)

        if model == Model.FastQKNN:
            backend_name = backend.value if isinstance(backend, Backend) else backend
            device = qml.device(backend_name, wires=nqubits)
            qnode  = qml.qnode(device, diff_method=None)
            return FastQKNN(nqubits=nqubits, embedding=embedding, qnode=qnode, k=K, mem_budget_mb=mem_budget_mb)

        if model == Model.MPSQKNN:
            return MPSQKNN(nqubits=nqubits, embedding=embedding, k=K)


        if model == Model.QSVM:
            return QSVM(nqubits=nqubits, embedding=embedding, shots=shots, seed=seed, backend=backend)

        if model == Model.FastQSVM:
            backend_name = backend.value if isinstance(backend, Backend) else backend
            device = qml.device(backend_name, wires=nqubits)
            qnode  = qml.qnode(device, diff_method=None)
            return FastQSVM(nqubits=nqubits, embedding=embedding, qnode=qnode, mem_budget_mb=mem_budget_mb)

        if model == Model.MPSQSVM:
            return MPSQSVM(nqubits=nqubits, embedding=embedding)


        if model == Model.QNN:
            params = {"nqubits": nqubits, "ansatz": ansatz, "embedding": embedding, "n_class": n_class, "layers": layers, "epochs": epochs, "shots": shots,
                    "lr": lr, "batch_size": batch_size, "seed": seed, "torch_device": "cpu", "backend": backend, "diff_method": "best"}
            return QNN(**params)

        if model == Model.QNN_BAG:
            params = {"nqubits": nqubits, "ansatz": ansatz, "embedding": embedding, "n_class": n_class, "layers": layers, "epochs": epochs, "n_samples": n_samples,
                    "n_features": n_features, "n_estimators": numPredictors, "shots": shots, "diff_method": "best", "lr": lr, "batch_size": batch_size, "seed": seed, "backend": backend}
            return QNNBag(**params)

        if model == Model.MPSQNN:
            params = {"nqubits": nqubits, "ansatz": ansatz, "embedding": embedding, "n_class": n_class, "layers": layers, "epochs": epochs, "shots": shots, "lr": lr,
                    "batch_size": batch_size, "torch_device": "cpu", "backend": "default.tensor", "seed": seed, "diff_method": None}
            return MPSQNN(**params)


        raise ValueError(f"Unknown model:  {model}")