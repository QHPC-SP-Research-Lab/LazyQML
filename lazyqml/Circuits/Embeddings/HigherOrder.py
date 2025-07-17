import pennylane as qml
from pennylane.operation import Operation, AnyWires

class HigherOrderEmbedding(Operation):
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

        op_list.extend([qml.RY(features[i], i) for i in wires])

        for i in wires[1:]:
            op_list.append(qml.CNOT(wires = [i - 1, i]))
            op_list.append(qml.RY(features[i - 1]*features[i], wires=i))

        return op_list