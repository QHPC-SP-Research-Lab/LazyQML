import numpy as np
from itertools import combinations

class HigherOrderEmbeddingMPS():
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def __call__(self, circuit, features):
        """
        Apply higher-order embedding to an existing quimb CircuitMPS. 

        Args:
            circuit (qtn.CircuitMPS): Existing circuit object.
            features (array-like): Feature vector.
        """
        features = np.asarray(features, dtype=np.float64)

        if len(features) > self.nqubits:
            raise ValueError(
                f"Features must be of length {self.nqubits} or less; "
                f"got length {len(features)}."
            )

        # ---- First-order embedding ----
        for i in range(len(features)):
            circuit.apply_gate("RY", features[i], i)

        # ---- Higher-order interactions ----
        for i in range(1, len(features)):
            circuit.apply_gate("CNOT", i - 1, i)
            interaction_angle = features[i - 1] * features[i]
            circuit.apply_gate("RY", interaction_angle, i)
