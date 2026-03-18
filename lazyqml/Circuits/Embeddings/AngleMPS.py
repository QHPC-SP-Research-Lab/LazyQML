import numpy as np

class AngleEmbeddingMPS:

    def __init__(self, nqubits, rot):
        """
        Args:
            nqubits (int): number of qubits
            rot (str): rotation gate name ("RX", "RY", "RZ", etc.)
        """
        self.nqubits = nqubits
        self.rot = rot.upper()  # normalize

    def __call__(self, circuit, features):
        """
        Apply angle embedding with zero padding.

        If rot == "RZ", apply a Hadamard before the rotation.
        """

        features = np.asarray(features, dtype=np.float64)

        required_dim = self.nqubits

        # ---- Zero padding ----
        if len(features) < required_dim:
            features = np.pad(
                features,
                (0, required_dim - len(features)),
                mode="constant"
            )

        elif len(features) > required_dim:
            raise ValueError(
                f"Features must be of length <= {required_dim}; "
                f"got length {len(features)}."
            )

        # ---- Apply embedding ----
        for i in range(self.nqubits):

            # If RZ embedding → rotate in X-basis via Hadamard
            if self.rot == "RZ":
                circuit.apply_gate("H", i)

            circuit.apply_gate(self.rot, features[i], i)