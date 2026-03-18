import numpy as np


class DenseAngleEmbeddingMPS():

    def __init__(self, nqubits):
        self.nqubits = nqubits

    def __call__(self, circuit, features):
        """
        Apply dense angle embedding with zero padding.

        Expected feature layout:
            first nqubits     -> RY angles
            next  nqubits     -> PHASE angles
        """

        features = np.asarray(features, dtype=np.float64)

        # We need 2 * nqubits parameters total
        required_dim = 2 * self.nqubits

        # Zero pad if necessary
        if len(features) < required_dim:
            padded = np.zeros(required_dim, dtype=np.float64)
            padded[:len(features)] = features
            features = padded

        elif len(features) > required_dim:
            raise ValueError(
                f"Features must be of length <= {required_dim}; "
                f"got length {len(features)}."
            )

        # Split features
        ry_angles = features[:self.nqubits]
        phase_angles = features[self.nqubits:required_dim]

        # ---- Dense Angle Embedding ----
        for i in range(self.nqubits):
            circuit.apply_gate("RY", ry_angles[i], i)
            circuit.apply_gate("PHASE", phase_angles[i], i)


        