import pennylane as qml
from pennylane.operation import Operation, AnyWires

import torch.nn.functional as F

class DenseAngleEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, id=None):
        
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > 2*len(wires):
            raise ValueError(
                f"Features must be of length {2*len(wires)} or less; got length {n_features}."
            )

        self._hyperparameters = {}

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(features, wires):

        op_list = []

        n_qubits = len(wires)

        batched = qml.math.ndim(features) > 1
        shape = tuple(features.shape)
        n_features = shape[0] if not batched else shape[1]

        # Padding if necessary
        if n_features < 2*n_qubits:
            padding = [0] * (2*n_qubits - n_features)
            if len(shape) > 1:
                padding = [padding] * shape[0]
            padding = qml.math.convert_like(padding, features)
            features = qml.math.hstack([features, padding])

        features = qml.math.T(features) if batched else features

        # qml.AngleEmbedding(x[..., :N], wires=wires, rotation='Y')
        for i in wires:
            op_list.append(qml.RY(features[i], wires=i))
            op_list.append(qml.PhaseShift(features[n_qubits + i], wires=i))

        return op_list