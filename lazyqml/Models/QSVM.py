#import warnings 

from contextlib import nullcontext

import numpy     as np
import quimb.tensor as qtn
import pennylane as qml
from sklearn.svm   import SVC
from threadpoolctl import threadpool_limits

from lazyqml.Factories         import CircuitFactory
from lazyqml.Global            import config
from lazyqml.Global.globalEnums import Embedding
from lazyqml.Interfaces.iModel import Model
from lazyqml.Utils             import printer, _numpy_math_api, get_max_bond_dim

#warnings.filterwarnings("ignore")


from lazyqml.Models._kernel_utils import _analytic_kernel, _finalize_kernel_matrix, _supports_analytic_kernel




def _build_precomputed_svc(
    *,
    C=1.0,
    class_weight=None,
    tol=1e-3,
    cache_size=200,
    max_iter=-1,
    shrinking=True,
    probability=False,
    random_state=None,
    decision_function_shape="ovr",
    break_ties=False,
    verbose=False,
):
    return SVC(
        kernel="precomputed",
        C=C,
        class_weight=class_weight,
        tol=tol,
        cache_size=cache_size,
        max_iter=max_iter,
        shrinking=shrinking,
        probability=probability,
        random_state=random_state,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        verbose=verbose,
    )

# -----------------------------------------------------------------------------
# QSVM : QSVM baseline
# -----------------------------------------------------------------------------
class QSVM(Model):
    def __init__(
        self,
        nqubits,
        embedding,
        backend,
        shots,
        seed=1234,
        *,
        C=1.0,
        class_weight=None,
        tol=1e-3,
        cache_size=200,
        max_iter=-1,
        shrinking=True,
        probability=False,
        random_state=None,
        decision_function_shape="ovr",
        break_ties=False,
        verbose=False,
    ):
        super().__init__()

        self.nqubits           = nqubits
        self.embedding         = embedding
        self.shots             = shots
        self.svc_params        = {
            "C": C,
            "class_weight": class_weight,
            "tol": tol,
            "cache_size": cache_size,
            "max_iter": max_iter,
            "shrinking": shrinking,
            "probability": probability,
            "random_state": seed if random_state is None else random_state,
            "decision_function_shape": decision_function_shape,
            "break_ties": break_ties,
            "verbose": verbose,
        }
        self.device            = qml.device(backend.value, wires=nqubits, seed=seed, shots=self.shots)
        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.svm               = None

    def _build_kernel(self):
        """Build the quantum kernel using a given embedding."""
        embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        adj_embedding_circuit = qml.adjoint(embedding_circuit)

        @qml.qnode(self.device, diff_method=None)
        def kernel_probs(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            adj_embedding_circuit(x2, wires=range(self.nqubits))
            return qml.probs(wires=range(self.nqubits))

        def kernel(x1, x2):
            return kernel_probs(x1, x2)[0]

        return kernel

    def fit(self, X, y):
        self.X_train = X

        printer.print("	Training QSVM...")

        assume_norm = (self.shots is None)

        self.qkernel = qml.kernels.square_kernel_matrix(X, self.kernel_circ, assume_normalized_kernel=assume_norm)
        self.qkernel = _finalize_kernel_matrix(self.qkernel, is_symmetric=True)

        self.svm = _build_precomputed_svc(**self.svc_params)
        self.svm.fit(self.qkernel, y)
        printer.print("	QSVM training complete.")

    def predict(self, X):
        try:
            if self.X_train is None or self.svm is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")

            if len(X) == 0:
                raise ValueError("X must contain at least one sample.")

            printer.print("	Testing QSVM...")
            kernel_test = qml.kernels.kernel_matrix(X, self.X_train, self.kernel_circ)
            np.clip(kernel_test, 0.0, 1.0, out=kernel_test)
            printer.print("	QSVM testing complete.")

            return self.svm.predict(kernel_test)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None


# -----------------------------------------------------------------------------
# FastQSVM : Analytic kernel, statevector mode A, or exact QSVM fallback
# -----------------------------------------------------------------------------
class FastQSVM(Model):
    def __init__(
            self,
            nqubits,
            embedding,
            qnode,
            *,
            mem_budget_mb = None,
            cores: int = 1,
            state_dtype=config.state_dtype,
            kernel_dtype=config.kernel_dtype,
            C=1.0,
            class_weight=None,
            tol=1e-3,
            cache_size=200,
            max_iter=-1,
            shrinking=True,
            probability=False,
            random_state=None,
            decision_function_shape="ovr",
            break_ties=False,
            verbose=False,
        ):
        super().__init__()

        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.embedding         = embedding
        self.qnode             = qnode
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(embedding)
        self.cores             = cores
        self.kernel_circ       = self._build_state_kernel()
        self.overlap_circ      = self._build_overlap_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.svm               = None

        self.mem_budget_mb = mem_budget_mb
        self.state_dtype   = state_dtype
        self.kernel_dtype  = kernel_dtype
        self.numpy_api     = _numpy_math_api()
        self.svc_params    = {
            "C": C,
            "class_weight": class_weight,
            "tol": tol,
            "cache_size": cache_size,
            "max_iter": max_iter,
            "shrinking": shrinking,
            "probability": probability,
            "random_state": random_state,
            "decision_function_shape": decision_function_shape,
            "break_ties": break_ties,
            "verbose": verbose,
        }

    def _build_state_kernel(self):
        @self.qnode
        def kernel(x):
            self.embedding_circuit(x, wires=range(self.nqubits))
            return qml.state()
        return kernel

    def _build_overlap_kernel(self):
        adj_embedding_circuit = qml.adjoint(self.embedding_circuit)

        @self.qnode
        def kernel_probs(x1, x2):
            self.embedding_circuit(x1, wires=range(self.nqubits))
            adj_embedding_circuit(x2, wires=range(self.nqubits))
            return qml.probs(wires=range(self.nqubits))

        def kernel(x1, x2):
            return kernel_probs(x1, x2)[0]

        return kernel

    # -------------------------------------------------------------------------
    # Context manager for BLAS threads
    # -------------------------------------------------------------------------
    def _threadpool_ctx(self):
        api = self.numpy_api
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=self.cores, user_api=api)

    def _single_thread_ctx(self):
        api = self.numpy_api
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=1, user_api=api)

    def _is_analytic_embedding(self):
        return _supports_analytic_kernel(self.embedding)

    def _analytic_kernel(self, X1, X2):
        return _analytic_kernel(self.embedding, X1, X2, self.nqubits, self.kernel_dtype)

    def _fits_mode_a(self, n1: int, n2: int, is_symmetric: bool = False):
        if self.mem_budget_mb is None:
            return True

        dim = 1 << int(self.nqubits)
        overhead = config.fast_overhead

        number_states = n1 if is_symmetric else (n1 + n2)
        bytes_state = np.dtype(self.state_dtype).itemsize
        bytes_state_need = bytes_state * dim * number_states

        n_elems = (n1 * n1) if is_symmetric else (n1 * n2)
        bytes_kernel = np.dtype(self.kernel_dtype).itemsize
        bytes_work = n_elems * (bytes_kernel + bytes_state)

        mib_total_need = ((bytes_state_need + bytes_work) * overhead) / (1024 * 1024)
        return mib_total_need <= float(self.mem_budget_mb)

    def _statevector_kernel(self, X1, X2, is_symmetric: bool = False):
        dim = 1 << int(self.nqubits)
        n1 = int(X1.shape[0])
        n2 = int(X2.shape[0])

        x_sv = np.empty((n1, dim), dtype=self.state_dtype)
        y_sv = x_sv if is_symmetric else np.empty((n2, dim), dtype=self.state_dtype)

        with self._single_thread_ctx():
            for k in range(n1):
                x_sv[k, :] = self.kernel_circ(X1[k])

        if not is_symmetric:
            with self._single_thread_ctx():
                for k in range(n2):
                    y_sv[k, :] = self.kernel_circ(X2[k])

        with self._threadpool_ctx():
            gram = x_sv @ y_sv.conj().T

        R = np.empty(gram.shape, dtype=self.kernel_dtype)
        np.square(gram.real, out=R)
        R += gram.imag * gram.imag
        return _finalize_kernel_matrix(R, is_symmetric=is_symmetric)

    def _exact_kernel(self, X1, X2, is_symmetric: bool = False):
        if is_symmetric:
            K = qml.kernels.square_kernel_matrix(
                X1,
                self.overlap_circ,
                assume_normalized_kernel=True,
            )
        else:
            K = qml.kernels.kernel_matrix(X1, X2, self.overlap_circ)
        K = K.astype(self.kernel_dtype, copy=False)
        return _finalize_kernel_matrix(K, is_symmetric=is_symmetric)

    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        """Calculate the quantum kernel matrix for SVM."""
        if self._is_analytic_embedding():
            return _finalize_kernel_matrix(self._analytic_kernel(X1, X2), is_symmetric=is_symmetric)

        n1 = int(X1.shape[0])
        n2 = int(X2.shape[0])

        if not self._fits_mode_a(n1, n2, is_symmetric=is_symmetric):
            return self._exact_kernel(X1, X2, is_symmetric=is_symmetric)

        try:
            return self._statevector_kernel(X1, X2, is_symmetric=is_symmetric)
        except MemoryError:
            return self._exact_kernel(X1, X2, is_symmetric=is_symmetric)

    # -------------------------------------------------------------------------
    # API fit / predict
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X
        printer.print("	Training FastQSVM...")

        self.qkernel = self._quantum_kernel(X, X, True)
        self.svm     = _build_precomputed_svc(**self.svc_params)
        self.svm.fit(self.qkernel, y)
        printer.print("	FastQSVM training complete.")


    def predict(self, X):
        if self.X_train is None or self.svm is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")        

        printer.print("	Testing FastQSVM...")
        kernel_test = self._quantum_kernel(X, self.X_train, False)

        if kernel_test.shape[1] == 0:
            raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
        preds = self.svm.predict(kernel_test)

        printer.print("	FastQSVM testing complete.")
        return preds

    @property
    def n_params(self):
        return None


# =============================================================================
# MPSQSVM
# =============================================================================
class MPSQSVM(Model):
    def __init__(self, nqubits, embedding, *, cores: int = 1, state_dtype=config.state_dtype, kernel_dtype=config.kernel_dtype,
                 C=1.0, class_weight=None, tol=1e-3, cache_size=200, max_iter=-1, shrinking=True, probability=False,
                 random_state=None, decision_function_shape="ovr", break_ties=False, verbose=False):
        super().__init__()
        
        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.embedding         = embedding
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuitMPS(embedding)
        self.cores             = cores
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.train_states      = None
        self.svm               = None
        self.max_bond_dim      = get_max_bond_dim()
        self.mem_budget_mb     = None
        self.state_dtype       = state_dtype
        self.kernel_dtype      = kernel_dtype
        self.numpy_api         = _numpy_math_api()
        self.svc_params        = {
            "C": C,
            "class_weight": class_weight,
            "tol": tol,
            "cache_size": cache_size,
            "max_iter": max_iter,
            "shrinking": shrinking,
            "probability": probability,
            "random_state": random_state,
            "decision_function_shape": decision_function_shape,
            "break_ties": break_ties,
            "verbose": verbose,
        }

    # -------------------------------------------------------------------------
    # Context manager for BLAS threads
    # -------------------------------------------------------------------------
    def _threadpool_ctx(self):
        api = self.numpy_api
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=self.cores, user_api=api)

    def _single_thread_ctx(self):
        api = self.numpy_api
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=1, user_api=api)


    # -------------------------------------------------------------------------
    # returns mps
    # -------------------------------------------------------------------------
    def _build_kernel(self):
        def mps(x):
            psi = qtn.CircuitMPS(N=self.nqubits, max_bond=self.max_bond_dim)

            self.embedding_circuit(psi, x)

            psi_final = psi.psi
            psi_final.normalize()

            return psi_final 
        return mps

    def _build_states(self, X):
        with self._single_thread_ctx():
            return [self.kernel_circ(x) for x in X]

    def _is_analytic_embedding(self):
        return _supports_analytic_kernel(self.embedding)

    def _analytic_kernel(self, X1, X2):
        return _analytic_kernel(self.embedding, X1, X2, self.nqubits, self.kernel_dtype)


    def _mps_overlap(self, X1, X2):
        if len(X1) == 0 or len(X2) == 0:
            raise ValueError("X1 and X2 must be non-empty.")

        N = len(X1)
        M = len(X2)

        K = np.empty((N, M), dtype=self.kernel_dtype)

        if X1 is X2:
            for i in range(N):
                bra_i   = X1[i].H
                K[i, i] = abs(bra_i @ X2[i]) ** 2

                for j in range(i + 1, N):
                    val = abs(bra_i @ X2[j]) ** 2
                    K[i, j] = val
                    K[j, i] = val
        else:
            for i in range(N):
                bra_i = X1[i].H
                for j in range(M):
                    K[i, j] = abs(bra_i @ X2[j]) ** 2
        return K


    def _mps_overlap_generico(self, X1, X2):
        if len(X1) == 0 or len(X2) == 0:
            raise ValueError("X1 and X2 must be non-empty.")

        N = len(X1)
        M = len(X2)

        K = np.zeros((N, M), dtype=self.state_dtype)

        if X1 is X2:
            for i in range(N):
                K[i, i] = abs(X1[i].H @ X2[i]) ** 2
                for j in range(i + 1, N):
                    val = abs(X1[i].H @ X2[j]) ** 2
                    K[i, j] = val
                    K[j, i] = val
        else:
            for i in range(N):
                for j in range(M):
                    K[i, j] = abs(X1[i].H @ X2[j]) ** 2
        return K


    # =============================================================================
    # _quantum_kernel
    # =============================================================================
    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False, *, states_1=None, states_2=None):
        if self._is_analytic_embedding():
            return _finalize_kernel_matrix(self._analytic_kernel(X1, X2), is_symmetric=is_symmetric)

        if states_2 is None:
            states_2 = self._build_states(X2)
        if states_1 is None:
            states_1 = states_2 if is_symmetric else self._build_states(X1)

        K = self._mps_overlap(states_1, states_2)
        K = K.astype(self.kernel_dtype, copy=False)
        return _finalize_kernel_matrix(K, is_symmetric=is_symmetric)


    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X
        printer.print("	Training MPSQSVM...")
        if self._is_analytic_embedding():
            self.train_states = None
            self.qkernel = self._quantum_kernel(X, X, True)
        else:
            self.train_states = self._build_states(X)
            self.qkernel = self._quantum_kernel(X, X, True, states_1=self.train_states, states_2=self.train_states)
        self.svm = _build_precomputed_svc(**self.svc_params)
        self.svm.fit(self.qkernel, y)
        printer.print("	MPSQSVM training complete.")

    # -------------------------------------------------------------------------
    def predict(self, X):
        if self.X_train is None or self.svm is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")
        if (not self._is_analytic_embedding()) and self.train_states is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")

        printer.print("	Testing MPSQSVM...")
        if self._is_analytic_embedding():
            kernel_test = self._quantum_kernel(X, self.X_train, False)
        else:
            kernel_test = self._quantum_kernel(X, self.X_train, False, states_2=self.train_states)
        if kernel_test.shape[1] == 0:
            raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")

        preds = self.svm.predict(kernel_test)

        printer.print("	MPSQSVM testing complete.")
        return preds

    @property
    def n_params(self):
        return None
