import numpy as np

from lazyqml.Global.globalEnums import Embedding


def _prepare_features(X, expected_dim):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] > expected_dim:
        raise ValueError(f"Features must be of length <= {expected_dim}; got length {X.shape[1]}.")
    if X.shape[1] < expected_dim:
        X = np.pad(X, ((0, 0), (0, expected_dim - X.shape[1])), mode="constant")
    return X


def _finalize_kernel_matrix(K, is_symmetric: bool = False):
    if is_symmetric:
        K = 0.5 * (K + K.T)
    np.clip(K, 0.0, 1.0, out=K)
    if is_symmetric:
        np.fill_diagonal(K, 1.0)
    return K


def _supports_analytic_kernel(embedding):
    return embedding in {Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.DENSE_ANGLE}


def _analytic_angle_kernel(X1, X2, nqubits, kernel_dtype):
    A = _prepare_features(X1, nqubits)
    B = _prepare_features(X2, nqubits)
    delta = A[:, None, :] - B[None, :, :]
    K = np.prod(np.cos(0.5 * delta) ** 2, axis=2, dtype=np.float64)
    return K.astype(kernel_dtype, copy=False)


def _analytic_dense_angle_kernel(X1, X2, nqubits, kernel_dtype):
    A = _prepare_features(X1, 2 * nqubits)
    B = _prepare_features(X2, 2 * nqubits)

    theta_a = A[:, None, :nqubits]
    theta_b = B[None, :, :nqubits]
    phi_delta = B[None, :, nqubits:] - A[:, None, nqubits:]

    c_term = np.cos(0.5 * theta_a) * np.cos(0.5 * theta_b)
    s_term = np.sin(0.5 * theta_a) * np.sin(0.5 * theta_b)

    per_qubit = (c_term * c_term) + (s_term * s_term) + (2.0 * c_term * s_term * np.cos(phi_delta))
    K = np.prod(per_qubit, axis=2, dtype=np.float64)
    return K.astype(kernel_dtype, copy=False)


def _analytic_kernel(embedding, X1, X2, nqubits, kernel_dtype):
    if embedding in {Embedding.RX, Embedding.RY, Embedding.RZ}:
        return _analytic_angle_kernel(X1, X2, nqubits, kernel_dtype)
    if embedding == Embedding.DENSE_ANGLE:
        return _analytic_dense_angle_kernel(X1, X2, nqubits, kernel_dtype)
    raise ValueError(f"Embedding {embedding} does not admit an analytic kernel.")
