import sys
import gc
import numpy     as np
import pennylane as qml
import psutil
import warnings

from contextlib              import nullcontext
from sklearn.svm             import SVC
from threadpoolctl           import threadpool_limits

import numpy as np
from sklearn.svm import SVC
import pennylane as qml
from lazyqml.Factories import CircuitFactory
from lazyqml.Utils import printer, _numpy_math_api
from lazyqml.Interfaces.iModel import Model

from functools import partial

from time import time

# import torch
# from itertools import product
# from itertools import combinations_with_replacement

from threadpoolctl import threadpool_limits

# from sys import getsizeof

# -----------------------------------------------------------------------------
# Parámetros globales de memoria
# -----------------------------------------------------------------------------

##### Esto va en Global._config
# Faltaria añadir funciones para obtener estos valores en Utils

MEM_Threshold = 0.75
MEM_Safety    = 1.60
MEM_GBytes    = float(psutil.virtual_memory().available) / (1024**3)

STATE_DTYPE   = np.complex128   # dtype de los statevectors
KERNEL_DTYPE  = np.float64      # dtype de la matriz kernel (real)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# QSVM Fast_Block
# -----------------------------------------------------------------------------

class QSVM(Model):
    def __init__(self,
            nqubits,
            embedding,
            qnode,
            cores=1,
            max_cached_states=None,
            ram_gb=MEM_GBytes,
            fraction=MEM_Threshold,
            safety=MEM_Safety,
            state_dtype=STATE_DTYPE,
            kernel_dtype=KERNEL_DTYPE
        ):
        super().__init__()

        # self.nqubits = nqubits
        # self.embedding = embedding
        # self.shots = shots
        # self.device = qml.device(backend.value, wires=nqubits)
        # self.device = qml.device('default.qubit', wires=nqubits)
        self.circuit_factory = CircuitFactory(nqubits, nlayers=0)
        # self.kernel_circ = self._build_kernel()
        # self.qkernel = None
        # self.X_train = None

        self.nqubits           = nqubits
        self.qnode             = qnode
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(embedding)
        self.cores             = cores
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None

        self.state_dtype       = state_dtype
        self.kernel_dtype      = kernel_dtype
        self.ram_gb            = ram_gb
        self.ram_fraction      = fraction
        self.safety            = safety

        self.numpy_api = _numpy_math_api()

        if max_cached_states is None:
            self.max_cached_states = self._estimate_max_cached_states()
        else:
            self.max_cached_states = max_cached_states

        # Slower
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=0)(self.kernel_circ)
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.batch_kernel_circ)

        # Faster batching
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.kernel_circ)

    # -------------------------------------------------------------------------
    # max number of statevectors we can put in RAM
    # -------------------------------------------------------------------------
    def _estimate_max_cached_states(self):
        bytes_per_complex = np.dtype(self.state_dtype).itemsize
        bytes_vstate      = bytes_per_complex * (1 << self.nqubits)
        max_bytes         = self.ram_gb * (1024**3) * self.ram_fraction
        bytes_effect      = self.safety * bytes_vstate
        return max(1, int(max_bytes // bytes_effect))

    # def _build_kernel(self):
    #     """Build the quantum kernel using a given embedding and ansatz."""
    #     # Get the embedding circuit from the circuit factory
    #     # embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
    #     # adj_embedding_circuit = qml.adjoint(embedding_circuit)
        
    #     # # Define the kernel circuit with adjoint embedding for the quantum kernel
    #     # @qml.qnode(self.device, diff_method=None)
    #     # def kernel(x1, x2):

    #     #     embedding_circuit(x1, wires=range(self.nqubits))
    #     #     adj_embedding_circuit(x2, wires=range(self.nqubits))

    #     #     return qml.probs(wires = range(self.nqubits))
        
    #     # return kernel

    # -------------------------------------------------------------------------
    # QNode returns statevector
    # -------------------------------------------------------------------------
    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        @self.qnode
        def kernel(x):
            self.embedding_circuit(x, wires=range(self.nqubits))
            return qml.state()
        return kernel

    # -------------------------------------------------------------------------
    # Context manager for BLAS threads
    # -------------------------------------------------------------------------
    def _threadpool_ctx(self):
        api = _numpy_math_api()
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=self.cores, user_api=api)

    # -------------------------------------------------------------------------
    # block sizes for X1 (test) y X2 (train)
    # -------------------------------------------------------------------------
    def _compute_block_sizes(self, n1, n2):
        max_states = max(2, self.max_cached_states)
        ratio = n1 / (n1 + n2)

        if ratio < 0.2:
            frac = 0.80   # X1 very small
        elif ratio < 0.4:
            frac = 0.60   # X1 moderate
        else:
            frac = 0.50   # X1 an X2 equals

        bs1 = int(max_states * frac)
        bs2 = max_states - bs1

        bs1 = max(1, min(n1, bs1))
        bs2 = max(1, min(n2, bs2))
        return bs1, bs2
    
    # -------------------------------------------------------------------------
    # MODO C CPU: bloques en RAM
    # -------------------------------------------------------------------------
    def _quantum_kernel_block(self, X1, X2, is_symmetric=False):
        n1       = X1.shape[0]  # nº muestras X1 (test)
        n2       = X2.shape[0]  # nº muestras X2 (train)
        bs1, bs2 = self._compute_block_sizes(n1, n2)
        dim      = 1 << self.nqubits

        kernel_matrix = np.empty((n1, n2),   dtype=self.kernel_dtype)
        x_buf         = np.empty((bs1, dim), dtype=self.state_dtype)
        y_buf         = np.empty((bs2, dim), dtype=self.state_dtype)
        k_buf         = np.empty((bs1, bs2), dtype=self.kernel_dtype)

        if is_symmetric:
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                m1    = i_end - i_start

                with threadpool_limits(limits=1, user_api=self.numpy_api):
                    for k in range(m1):
                        x_buf[k, :] = self.kernel_circ(X1[i_start + k])
                x_view = x_buf[:m1, :]

                for j_start in range(i_start, n2, bs2):
                    j_end = min(j_start + bs2, n2)
                    m2    = j_end - j_start

                    with threadpool_limits(limits=1, user_api=self.numpy_api):
                        for k in range(m2):
                            y_buf[k, :] = self.kernel_circ(X1[j_start + k])
                    y_view = y_buf[:m2, :]

                    with self._threadpool_ctx():
                        gram = x_view @ y_view.conj().T
                    k_view = k_buf[:m1, :m2]
                    np.square(gram.real, out=k_view)
                    k_view += gram.imag * gram.imag

                    kernel_matrix[i_start:i_end, j_start:j_end] = k_view
                    if j_start != i_start:
                        kernel_matrix[j_start:j_end, i_start:i_end] = k_view.T
        else:
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                m1    = i_end - i_start

                with threadpool_limits(limits=1, user_api=self.numpy_api):
                    for k in range(m1):
                        x_buf[k, :] = self.kernel_circ(X1[i_start + k])
                x_view = x_buf[:m1, :]

                for j_start in range(0, n2, bs2):
                    j_end = min(j_start + bs2, n2)
                    m2    = j_end - j_start

                    with threadpool_limits(limits=1, user_api=self.numpy_api):
                        for k in range(m2):
                            y_buf[k, :] = self.kernel_circ(X2[j_start + k])
                    y_view = y_buf[:m2, :]

                    with self._threadpool_ctx():
                        gram = x_view @ y_view.conj().T
                    k_view = k_buf[:m1, :m2]
                    np.square(gram.real, out=k_view)
                    k_view += gram.imag * gram.imag

                    kernel_matrix[i_start:i_end, j_start:j_end] = k_view
        return kernel_matrix

    # -------------------------------------------------------------------------
    # Dispatcher: for now only A and C modes in CPU
    # -------------------------------------------------------------------------
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""

        are_equal = np.array_equal(X1, X2)

        dim = 1 << self.nqubits
        n1  = X1.shape[0] # if (X1 != X2) -> X1 is X_test
        n2  = X2.shape[0] # if (X1 != X2) -> X2 is X_train

        if are_equal:
            A_mode = (n1 <= self.max_cached_states)
        else:
            A_mode = (n1 + n2 <= self.max_cached_states)

        if A_mode:
            x_sv = np.empty((n1, dim), dtype=self.state_dtype)
            if not are_equal:
                y_sv = np.empty((n2, dim), dtype=self.state_dtype)

            # We can use batching (x_sv = self.kernel_circ(X1)), but not all backend works with it
            with threadpool_limits(limits=1, user_api=self.numpy_api):
                for k in range(n1):
                    x_sv[k, :] = self.kernel_circ(X1[k])

            if are_equal:
                y_sv = x_sv
            else:
                with threadpool_limits(limits=1, user_api=self.numpy_api):
                    for k in range(n2):
                        y_sv[k, :] = self.kernel_circ(X2[k])

            with self._threadpool_ctx():
                gram = x_sv @ y_sv.conj().T

            R = np.empty(gram.shape, dtype=self.kernel_dtype)
            np.square(gram.real, out=R)
            R += gram.imag * gram.imag

            return R
        else:
            # C Mode in CPU
            return self._quantum_kernel_block(X1, X2, is_symmetric=are_equal)

    # -------------------------------------------------------------------------
    # API fit / predict
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X
        printer.print("\t\tTraining the SVM...")

        self.qkernel = self._quantum_kernel(X, X)
        self.svm     = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tSVM training complete.")


    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        
        printer.print(f"\t\t\tComputing kernel between test and training data...")

        kernel_test = self._quantum_kernel(X, self.X_train)

        if kernel_test.shape[1] == 0:
            raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")

        preds = self.svm.predict(kernel_test)
        return preds

    @property
    def n_params(self):
        return None
