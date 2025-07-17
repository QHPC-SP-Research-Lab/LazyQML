import pennylane as qml
from pennylane.operation import Operation, AnyWires

from itertools import combinations
import numpy as np

class ZZEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, id=None):

        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._hyperparameters = {}

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)


    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(features, wires):

        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features

        op_list = []

        nload = min(len(features), len(wires))
        
        for i in range(nload):
            op_list.append(qml.Hadamard(i))
            op_list.append(qml.RZ(2.0 * features[i], wires=i))

        for q0, q1 in list(combinations(range(nload), 2)):
            op_list.append(qml.CZ(wires=[q0, q1]))
            op_list.append(qml.RZ(2.0 * (np.pi - features[q0]) * (np.pi - features[q1]), wires=q1))
            op_list.append(qml.CZ(wires=[q0, q1]))

        return op_list