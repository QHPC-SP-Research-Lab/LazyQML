import numpy as np
from itertools import combinations


class ZZEmbeddingMPS():
    def __init__(self, nqubits):
        self.nqubits = nqubits

    def __call__(self, circuit, features):
        """
        Apply ZZ feature embedding to an existing quimb CircuitMPS.

        Args:
            circuit (qtn.CircuitMPS): Existing circuit
            features (array-like): Feature vector
        """
        features = np.asarray(features, dtype=np.float64)

        if len(features) > self.nqubits:
            raise ValueError(
                f"Features must be of length {self.nqubits} or less; "
                f"got length {len(features)}."
            )

        nload = min(len(features), self.nqubits)

        # ---- First layer ----
        for i in range(nload):
            circuit.apply_gate("H", i)
            circuit.apply_gate("RZ", 2.0 * features[i], i)

        # ---- ZZ interactions ----
        #for q0, q1 in combinations(range(nload), 2): -> this is all to all, slow 
        for q0 in range(nload - 1): # -> this all to next
            q1 = q0 + 1     # only for all to next. For all to all must be erased
            circuit.apply_gate("CZ", q0, q1)
            angle = 2.0 * (np.pi - features[q0]) * (np.pi - features[q1])
            circuit.apply_gate("RZ", angle, q1)
            circuit.apply_gate("CZ", q0, q1)
