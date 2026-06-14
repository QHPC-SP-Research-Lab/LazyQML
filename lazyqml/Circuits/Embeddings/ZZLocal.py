import pennylane as qml
from pennylane.operation import Operation

import numpy as np


class ZZLocalEmbedding(Operation):
    num_wires   = None
    grad_method = None

    def __init__(self, features, wires, id=None):
        shape      = qml.math.shape(features)[-1:]
        n_features = shape[0]
        if n_features > len(wires):
            raise ValueError(f"Features must be of length {len(wires)} or less; got length {n_features}.")

        self._hyperparameters = {}

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(features, wires):
        batched  = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features
        op_list  = []
        nload    = min(len(features), len(wires))

        active_wires = list(wires[:nload])

        for k, w in enumerate(active_wires):
            op_list.append(qml.Hadamard(wires=w))
            op_list.append(qml.RZ(2.0 * features[k], wires=w))

        for k0, w0 in enumerate(active_wires[:-1]):
            k1 = k0 + 1
            w1 = active_wires[k1]
            op_list.append(qml.CZ(wires=[w0, w1]))
            op_list.append(qml.RZ(2.0 * (np.pi - features[k0]) * (np.pi - features[k1]), wires=w1))
            op_list.append(qml.CZ(wires=[w0, w1]))

        return op_list
