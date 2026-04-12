#import warnings 

from contextlib import nullcontext

import numpy     as np
import quimb.tensor as qtn
import pennylane as qml
from sklearn.svm   import SVC
from threadpoolctl import threadpool_limits

from lazyqml.Factories         import CircuitFactory
from lazyqml.Global            import config
from lazyqml.Interfaces.iModel import Model
from lazyqml.Utils             import printer, _numpy_math_api, get_max_bond_dim

#warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# QSVM : QSVM baseline
# -----------------------------------------------------------------------------
class QSVM(Model):
    def __init__(self, nqubits, embedding, backend, shots, seed=1234):
        super().__init__()

        self.nqubits           = nqubits
        self.embedding         = embedding
        self.shots             = shots
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

        printer.print("\t\tTraining the QSVM...")

        assume_norm = (self.shots is None)

        self.qkernel = qml.kernels.square_kernel_matrix(X, self.kernel_circ, assume_normalized_kernel=assume_norm)

        self.qkernel = 0.5 * (self.qkernel + self.qkernel.T)
        np.clip(self.qkernel, 0.0, 1.0, out=self.qkernel)
        np.fill_diagonal(self.qkernel, 1.0)

        # Train the classical SVM with the quantum kernel
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tQSVM training complete.")

    def predict(self, X):
        try:
            if self.X_train is None or self.svm is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")

            if len(X) == 0:
                raise ValueError("X must contain at least one sample.")

            printer.print(f"\t\t\tComputing kernel between test and training data...")
            kernel_test = qml.kernels.kernel_matrix(X, self.X_train, self.kernel_circ)
            np.clip(kernel_test, 0.0, 1.0, out=kernel_test)

            return self.svm.predict(kernel_test)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None


# -----------------------------------------------------------------------------
# FastQSVM : QSVM Blocks based
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
            kernel_dtype=config.kernel_dtype
        ):
        super().__init__()

        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.qnode             = qnode
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(embedding)
        self.cores             = cores
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.svm               = None

        self.mem_budget_mb = mem_budget_mb
        self.state_dtype   = state_dtype
        self.kernel_dtype  = kernel_dtype
        self.numpy_api     = _numpy_math_api()

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
    # MODO C CPU: bloques en RAM
    # -------------------------------------------------------------------------
    def _quantum_kernel_block(self, X1, X2, is_symmetric: bool = False):
        n1           = int(X1.shape[0])  # nº muestras X1 (test)
        n2           = int(X2.shape[0])  # nº muestras X2 (train)
        dim          = 1 << int(self.nqubits)
        bs1, bs2     = min(n1, 4), min(n2, 4)
        ratio        = n1 / (n1 + n2) if (n1 + n2) else 0.5
        frac         = 0.80 if ratio < 0.2 else (0.60 if ratio < 0.4 else 0.50)
        overhead     = config.fast_overhead      
        bytes_state  = np.dtype(self.state_dtype).itemsize
        bytes_kernel = np.dtype(self.kernel_dtype).itemsize

        if self.mem_budget_mb is not None:
            budget_bytes = (float(self.mem_budget_mb) * 1024 * 1024) / overhead

            # kernel_matrix always allocated (n1 x n2)
            bytes_kernel_matrix = n1 * n2 * bytes_kernel

            if bytes_kernel_matrix > budget_bytes:
                raise MemoryError(f"FastQSVM needs kernel_matrix of ~{bytes_kernel_matrix/(1024*1024):.1f} MiB but budget is {float(self.mem_budget_mb):.1f} MiB.")

            # Remaining budget for buffers + temporals
            buffers_budget = budget_bytes - bytes_kernel_matrix

            def buffers_bytes(a: int, b: int) -> int:
                return (a + b) * dim * bytes_state + (a * b) * (bytes_kernel + bytes_state)

            # If (1,1) doesn't fit raise MemoryError
            if buffers_bytes(1, 1) <= buffers_budget:
                total = min(n1 + n2, 100000)  # initial guess, capped
                bs1 = max(1, min(n1, int(total * frac)))
                bs2 = max(1, min(n2, total - bs1))

                # Shrink until it fits
                while buffers_bytes(bs1, bs2) > buffers_budget:
                    if bs1 >= bs2 and bs1 > 1:
                        bs1 = max(1, bs1 // 2)
                    elif bs2 > 1:
                        bs2 = max(1, bs2 // 2)
                    else:
                        break
                if is_symmetric:
                    bs = min(bs1, bs2)
                    bs1 = bs2 = bs
            else:
                raise MemoryError("FastQSVM cannot allocate minimal buffers for block mode.")

        kernel_matrix = np.empty((n1, n2),   dtype=self.kernel_dtype)
        x_buf         = np.empty((bs1, dim), dtype=self.state_dtype)
        y_buf         = np.empty((bs2, dim), dtype=self.state_dtype)
        k_buf         = np.empty((bs1, bs2), dtype=self.kernel_dtype)

        if is_symmetric:
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                m1    = i_end - i_start

                with self._single_thread_ctx():
                    for k in range(m1):
                        x_buf[k, :] = self.kernel_circ(X1[i_start + k])
                x_view = x_buf[:m1, :]

                for j_start in range(i_start, n2, bs2):
                    if j_start == i_start:
                        j_end = i_end
                        m2 = m1
                        y_view = x_view
                    else:
                        j_end = min(j_start + bs2, n2)
                        m2 = j_end - j_start

                        with self._single_thread_ctx():
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
            for j_start in range(0, n2, bs2):
                j_end = min(j_start + bs2, n2)
                m2    = j_end - j_start

                with self._single_thread_ctx():
                    for k in range(m2):
                        y_buf[k, :] = self.kernel_circ(X2[j_start + k])
                y_view = y_buf[:m2, :]

                for i_start in range(0, n1, bs1):
                    i_end = min(i_start + bs1, n1)
                    m1    = i_end - i_start

                    with self._single_thread_ctx():
                        for k in range(m1):
                            x_buf[k, :] = self.kernel_circ(X1[i_start + k])
                    x_view = x_buf[:m1, :]

                    with self._threadpool_ctx():
                        gram = x_view @ y_view.conj().T

                    k_view = k_buf[:m1, :m2]
                    np.square(gram.real, out=k_view)
                    k_view += gram.imag * gram.imag

                    kernel_matrix[i_start:i_end, j_start:j_end] = k_view
        
        if is_symmetric:
            kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
        np.clip(kernel_matrix, 0.0, 1.0, out=kernel_matrix)
        if is_symmetric:
            np.fill_diagonal(kernel_matrix, 1.0)
        return kernel_matrix

    # -------------------------------------------------------------------------
    # Dispatcher: for now only A and C modes in CPU
    # -------------------------------------------------------------------------
    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        """Calculate the quantum kernel matrix for SVM."""
        dim = 1 << int(self.nqubits)
        n1  = int(X1.shape[0])
        n2  = int(X2.shape[0])

        a_mode = False
        if self.mem_budget_mb is not None:
            overhead = config.fast_overhead

            # RAM needed for statevector matrix: 2**nqubits * state_dtype * (n1 or n1+n2) 
            number_states  = n1 if is_symmetric else (n1 + n2)
            bytes_state    = np.dtype(self.state_dtype).itemsize
            byt_state_need = bytes_state * dim * number_states

            # RAM needed for
            n_elems      = (n1 * n1) if is_symmetric else (n1 * n2)
            bytes_kernel = np.dtype(self.kernel_dtype).itemsize
            byt_R_need   = n_elems * (bytes_kernel + bytes_state)

            # Total RAM needed
            MiB_total_need = ((byt_state_need + byt_R_need) * overhead) / (1024 * 1024)
            printer.print(f'MiB necesarios {MiB_total_need} -Budget_MB {float(self.mem_budget_mb)}')
            a_mode = (MiB_total_need <= float(self.mem_budget_mb))
            
        #printer.print(a_mode)
        if a_mode:
            x_sv = np.empty((n1, dim), dtype=self.state_dtype)
            y_sv = x_sv if is_symmetric else np.empty((n2, dim), dtype=self.state_dtype)

            # We can use batching (x_sv = self.kernel_circ(X1)), but not all backend works with it
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

            if is_symmetric:
                R = 0.5 * (R + R.T)
            np.clip(R, 0.0, 1.0, out=R)
            if is_symmetric:
                np.fill_diagonal(R, 1.0)
            return R

        # C Mode in CPU
        return self._quantum_kernel_block(X1, X2, is_symmetric=is_symmetric)

    # -------------------------------------------------------------------------
    # API fit / predict
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X
        printer.print("\t\tTraining the FastQSVM...")

        self.qkernel = self._quantum_kernel(X, X, True)
        self.svm     = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tFastQSVM training complete.")


    def predict(self, X):
        if self.X_train is None or self.svm is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")        

        printer.print(f"\t\t\tComputing kernel between test and training data...")
        kernel_test = self._quantum_kernel(X, self.X_train, False)

        if kernel_test.shape[1] == 0:
            raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
        preds = self.svm.predict(kernel_test)
        return preds

    @property
    def n_params(self):
        return None


# =============================================================================
# MPSQSVM
# =============================================================================
class MPSQSVM(Model):
    def __init__(self, nqubits, embedding, *, cores: int = 1, state_dtype=config.state_dtype, kernel_dtype=config.kernel_dtype):
        super().__init__()
        
        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuitMPS(embedding)
        self.cores             = cores
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.svm               = None
        self.max_bond_dim      = get_max_bond_dim()
        self.mem_budget_mb     = None
        self.state_dtype       = state_dtype
        self.kernel_dtype      = kernel_dtype
        self.numpy_api         = _numpy_math_api()

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
            psi = qtn.CircuitMPS(N=self.nqubits, dtype=self.state_dtype, max_bond=self.max_bond_dim)

            self.embedding_circuit(psi, x)

            psi_final = psi.psi
            #psi_final = psi.psi.canonicalize(where=0)
            psi_final = psi_final.normalize()

            return psi_final 
        return mps


    def _mps_overlap_seg(self, X1, X2):
        if len(X1) == 0 or len(X2) == 0:
            raise ValueError("X1 and X2 must be non-empty.")

        N = len(X1)
        M = len(X2)

        L = getattr(X1[0], "L", None)
        if L is None:
            raise ValueError("Input states do not look like quimb MatrixProductState objects.")

        for x in X1:
            if getattr(x, "L", None) != L:
                raise ValueError("All states in X1 must have the same number of sites.")
        for x in X2:
            if getattr(x, "L", None) != L:
                raise ValueError("All states in X2 must have the same number of sites.")

        K = np.empty((N, M), dtype=self.kernel_dtype)

        same_collection = (X1 is X2)

        with self._threadpool_ctx():
            if same_collection:
                for i in range(N):
                    K[i, i] = 1.0
                    bra_i = X1[i].H
                    for j in range(i + 1, N):
                        amp = qtn.expec_TN_1D(bra_i, X2[j])
                        val = abs(complex(amp)) ** 2
                        K[i, j] = val
                        K[j, i] = val
            else:
                for i in range(N):
                    bra_i = X1[i].H
                    for j in range(M):
                        amp = qtn.expec_TN_1D(bra_i, X2[j])
                        K[i, j] = abs(complex(amp)) ** 2
        return K


    def _mps_overlap(self, X1, X2):
        # Note that it works with convention: tensors have shape (Dl, Dr, d)
        if len(X1) == 0 or len(X2) == 0:
            raise ValueError("X1 and X2 must be non-empty.")

        N = len(X1)
        M = len(X2)
        L = len(X1[0].tensors)

        if any(len(x.tensors) != L for x in X1):
            raise ValueError("All states in X1 must have the same number of tensors.")
        if any(len(x.tensors) != L for x in X2):
            raise ValueError("All states in X2 must have the same number of tensors.")

        def get_array(t):
            arr = t.data if hasattr(t, "data") else t
            if arr.ndim not in (2, 3):
                raise ValueError(f"Unexpected tensor ndim={arr.ndim}, expected 2 or 3.")
            return arr

        def normalize_site_tensor(arr, site):
            if arr.ndim == 3:
                #Observed convention: (Dl, Dr, d)
                return arr
            if arr.ndim != 2:
                raise ValueError(f"Unexpected tensor ndim={arr.ndim}, expected 2 or 3.")
            if site == 0:
                # Observed shape (1, d) -> (1, 1, d)
                if arr.shape[0] != 1:
                    raise ValueError(f"Unexpected left-boundary rank-2 shape at site 0: {arr.shape}")
                d = arr.shape[1]
                return arr.reshape(1, 1, d)
            if site == L - 1:
                # Observed shape (Dl, d) -> (Dl, 1, d)
                if arr.shape[0] < 1:
                    raise ValueError(f"Unexpected right-boundary rank-2 shape at site {site}: {arr.shape}")
                Dl, d = arr.shape
                return arr.reshape(Dl, 1, d)
            raise ValueError(f"Rank-2 tensor found at interior site {site}.")

        env = np.ones((N, M, 1, 1), dtype=self.state_dtype)

        for site in range(L):
            A_list = [normalize_site_tensor(get_array(x.tensors[site]), site) for x in X1]
            B_list = [normalize_site_tensor(get_array(x.tensors[site]), site) for x in X2]

            Dl1_max = max(t.shape[0] for t in A_list)
            Dr1_max = max(t.shape[1] for t in A_list)
            d1_max  = max(t.shape[2] for t in A_list)

            Dl2_max = max(t.shape[0] for t in B_list)
            Dr2_max = max(t.shape[1] for t in B_list)
            d2_max  = max(t.shape[2] for t in B_list)

            if d1_max != d2_max:
                raise ValueError(f"Physical dimensions do not match at site {site}: {d1_max} vs {d2_max}")

            if env.shape[2] > Dl1_max or env.shape[3] > Dl2_max:
                raise ValueError(f"Incompatible bond dimensions at site {site}: env has ({env.shape[2]}, {env.shape[3]}) but tensors expect at most ({Dl1_max}, {Dl2_max})")

            d_max = d1_max

            A_batch = np.zeros((N, Dl1_max, Dr1_max, d_max), dtype=self.state_dtype)
            B_batch = np.zeros((M, Dl2_max, Dr2_max, d_max), dtype=self.state_dtype)

            for i, t in enumerate(A_list):
                Dl, Dr, d = t.shape
                A_batch[i, :Dl, :Dr, :d] = t

            for j, t in enumerate(B_list):
                Dl, Dr, d = t.shape
                B_batch[j, :Dl, :Dr, :d] = t

            env_padded = np.zeros((N, M, Dl1_max, Dl2_max), dtype=self.state_dtype)
            env_padded[:, :, :env.shape[2], :env.shape[3]] = env

            # Convention: tensors have shape (Dl, Dr, d)
            # env'[b, d] = sum_{a, c, s} env[a, c] * conj(A[a, b, s]) * B[c, d, s]
            with self._threadpool_ctx():
                env = np.einsum("ijac,iabs,jcds->ijbd", env_padded, A_batch.conj(), B_batch, optimize=True)

        if env.shape[2] != 1 or env.shape[3] != 1:
            raise ValueError(f"Final environment is not scalar-valued: got shape {env.shape[2:]}")
        return np.abs(env[:, :, 0, 0]) ** 2


    # =============================================================================
    # _quantum_kernel
    # =============================================================================
    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        with self._single_thread_ctx():
            states_2 = [self.kernel_circ(x) for x in X2]
            states_1 = states_2 if is_symmetric else [self.kernel_circ(x) for x in X1]

        K = self._mps_overlap_seg(states_1, states_2)

        if is_symmetric:
            K = 0.5 * (K + K.T)

        K = K.astype(self.kernel_dtype, copy=False)
        np.clip(K, 0.0, 1.0, out=K)

        if is_symmetric:
            np.fill_diagonal(K, 1.0)
        return K


    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X
        printer.print("\t\tTraining the MPSQSVM...")
        self.qkernel = self._quantum_kernel(X, X, True)
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tMPSQSVM training complete.")

    # -------------------------------------------------------------------------
    def predict(self, X):
        if self.X_train is None or self.svm is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")

        printer.print(f"\t\t\tComputing kernel between test and training data...")
        kernel_test = self._quantum_kernel(X, self.X_train, False)
        if kernel_test.shape[1] == 0:
            raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")

        preds = self.svm.predict(kernel_test)
        return preds

    @property
    def n_params(self):
        return None
