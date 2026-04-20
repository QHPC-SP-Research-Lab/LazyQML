#import warnings 

from contextlib import nullcontext

import numpy     as np
import quimb.tensor as qtn
import pennylane as qml
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from lazyqml.Factories         import CircuitFactory
from lazyqml.Global            import config
from lazyqml.Global.globalEnums import Embedding
from lazyqml.Interfaces.iModel import Model
from lazyqml.Utils             import printer, _numpy_math_api, get_max_bond_dim

#warnings.filterwarnings("ignore")

class QKNN(Model):
    def __init__(self, nqubits, embedding, backend, shots, k=5, seed=1234, *, kernel_dtype=config.kernel_dtype):
        """
        Initialize the Quantum KNN model.
        Args:
            nqubits (int): Number of qubits for the quantum kernel.
            backend (enum): Pennylane backend to use.
            shots (int): Number of shots for quantum measurements.
        """
        super().__init__()

        self.nqubits = nqubits
        self.embedding = embedding
        self.k = k
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits, seed=seed, shots=self.shots)
        self.circuit_factory = CircuitFactory(nqubits,nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.kernel_dtype = kernel_dtype
        self.X_train = None
        self.KNN = None

    def _build_kernel(self):
        """Build the quantum kernel circuit."""

         # Get the embedding circuit from the circuit factory
        embedding_circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)
        adj_embedding_circuit = qml.adjoint(embedding_circuit)

        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            adj_embedding_circuit(x2, wires=range(self.nqubits))
            
            return qml.probs(wires = range(self.nqubits))
        
        return kernel


    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.empty((n1, n2), dtype=self.kernel_dtype)

        if is_symmetric:
            for i in range(n1):
                for j in range(i, n2):
                    kij = self.kernel_circ(X1[i], X2[j])[0]
                    K[i, j] = kij
                    K[j, i] = kij
        else:
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel_circ(X1[i], X2[j])[0]

        if is_symmetric:
            K = 0.5 * (K + K.T)
        np.clip(K, 0.0, 1.0, out=K)
        if is_symmetric:
            np.fill_diagonal(K, 1.0)
        return K

    def _compute_distances(self, x1, x2, is_symmetric: bool = False):
        K = self._quantum_kernel(x1, x2, is_symmetric=is_symmetric)
        D = 1.0 - K

        if is_symmetric:
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
        return D


    def fit(self, X, y):
        """
        Fit the Quantum KNN model.
        Args:
            X (ndarray): Training samples (n_samples, n_features).
            y (ndarray): Training labels (n_samples,).
        """
        self.X_train = X

        n_train = len(X)
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.k > n_train:
            raise ValueError(f"k={self.k} cannot be larger than the number of training samples ({n_train}).")

        printer.print("\tTraining QKNN...")
        self.q_distances = self._compute_distances(X, X, is_symmetric=True)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric="precomputed")
        self.KNN.fit(self.q_distances, y)
        printer.print("\tQKNN training complete.")


    def predict(self, X):
        if self.X_train is None or self.KNN is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("X must contain at least one sample.")

        printer.print("\tTesting QKNN...")
        q_distances = self._compute_distances(X, self.X_train, is_symmetric=False)
        printer.print("\tQKNN testing complete.")
        return self.KNN.predict(q_distances)
    
    @property
    def n_params(self):
        return None


class FastQKNN(Model):
    def __init__(
            self,
            nqubits,
            embedding,
            qnode,
            k,
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
        self.k                 = k
        self.cores             = cores
        self.kernel_circ       = self._build_kernel()
        self.X_train           = None
        self.KNN               = None

        self.mem_budget_mb = mem_budget_mb
        self.state_dtype   = state_dtype
        self.kernel_dtype  = kernel_dtype
        self.numpy_api     = _numpy_math_api()


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
    # MODO C CPU: bloques en RAM
    # -------------------------------------------------------------------------
    def _quantum_kernel_block(self, X1, X2, is_symmetric: bool = False):
        n1           = int(X1.shape[0])  # nº muestras X1 (test)
        n2           = int(X2.shape[0])  # nº muestras X2 (train)
        dim          = 1 << int(self.nqubits)
        bs1, bs2     = 1, 1
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
                raise MemoryError(f"FastQKNN needs kernel_matrix of ~{bytes_kernel_matrix/(1024*1024):.1f} MiB but budget is {float(self.mem_budget_mb):.1f} MiB.")

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
            else:
                raise MemoryError("FastQKNN cannot allocate minimal buffers for block mode.")

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
                        m2    = j_end - j_start

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
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                m1    = i_end - i_start

                with self._single_thread_ctx():
                    for k in range(m1):
                        x_buf[k, :] = self.kernel_circ(X1[i_start + k])
                x_view = x_buf[:m1, :]

                for j_start in range(0, n2, bs2):
                    j_end = min(j_start + bs2, n2)
                    m2    = j_end - j_start

                    with self._single_thread_ctx():
                        for k in range(m2):
                            y_buf[k, :] = self.kernel_circ(X2[j_start + k])
                    y_view = y_buf[:m2, :]

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
        """Calculate the quantum kernel matrix for FastQKNN."""
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

            # RAM needed for R and gram matrices:
            n_elems      = (n1 * n1) if is_symmetric else (n1 * n2)
            bytes_kernel = np.dtype(self.kernel_dtype).itemsize
            byt_R_need   = n_elems * (bytes_kernel + bytes_state)

            # Total RAM needed
            MiB_total_need = ((byt_state_need + byt_R_need) * overhead) / (1024 * 1024)
            a_mode = (MiB_total_need <= float(self.mem_budget_mb))
            
        if a_mode:
            #printer.print(f'MiB necesarios {MiB_total_need}, Budget_MB {float(self.mem_budget_mb)}')
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
    

    def _compute_distances(self, x1, x2, is_symmetric: bool = False):
        K = self._quantum_kernel(x1, x2, is_symmetric=is_symmetric)
        D = 1.0 - K

        if is_symmetric:
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
        return D


    # -------------------------------------------------------------------------
    # API fit / predict
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X

        n_train = len(X)
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.k > n_train:
            raise ValueError(f"k={self.k} cannot be larger than the number of training samples ({n_train}).")

        printer.print("\tTraining FastQKNN...")
        self.q_distances = self._compute_distances(X, X, is_symmetric=True)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')
        self.KNN.fit(self.q_distances, y)
        printer.print("\tFastQKNN training complete.")


    def predict(self, X):
        try:
            if self.X_train is None or self.KNN is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")

            if len(X) == 0:
                raise ValueError("X must contain at least one sample.")

            printer.print("\tTesting FastQKNN...")
            q_distances = self._compute_distances(X, self.X_train, is_symmetric=False)

            printer.print("\tFastQKNN testing complete.")
            return self.KNN.predict(q_distances)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None


# =============================================================================
# MPSQKNN
# =============================================================================
class MPSQKNN(Model):
    def __init__(
        self,
            nqubits,
            embedding,
            k,
            *,
            cores: int = 1,
            state_dtype=config.state_dtype,
            kernel_dtype=config.kernel_dtype
        ):
        super().__init__()

        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.embedding         = embedding
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuitMPS(embedding)
        self.cores             = cores
        self.k                 = k
        self.kernel_circ       = self._build_kernel()
        self.X_train           = None
        self.train_states      = None
        self.KNN               = None
        self.max_bond_dim      = get_max_bond_dim()
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
    # Return mps
    # -------------------------------------------------------------------------
    def _build_kernel(self):
        def mps(x):
            psi = qtn.CircuitMPS(N=self.nqubits, dtype=self.state_dtype, max_bond=self.max_bond_dim)
            
            self.embedding_circuit(psi, x)

            psi_final = psi.psi
            #psi_final.canonicalize(where=0)
            psi_final.normalize()

            return psi_final
        return mps

    def _build_states(self, X):
        with self._single_thread_ctx():
            return [self.kernel_circ(x) for x in X]

    def _is_analytic_embedding(self):
        return self.embedding in {Embedding.RX, Embedding.RY, Embedding.RZ}

    def _prepare_angle_features(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] > self.nqubits:
            raise ValueError(f"Features must be of length <= {self.nqubits}; got length {X.shape[1]}.")
        if X.shape[1] < self.nqubits:
            X = np.pad(X, ((0, 0), (0, self.nqubits - X.shape[1])), mode="constant")
        return X

    def _analytic_kernel(self, X1, X2):
        A = self._prepare_angle_features(X1)
        B = self._prepare_angle_features(X2)
        delta = A[:, None, :] - B[None, :, :]
        K = np.prod(np.cos(0.5 * delta) ** 2, axis=2, dtype=np.float64)
        return K.astype(self.kernel_dtype, copy=False)


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

    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False, *, states_1=None, states_2=None):
        if self._is_analytic_embedding():
            K = self._analytic_kernel(X1, X2)
            if is_symmetric:
                K = 0.5 * (K + K.T)
                np.fill_diagonal(K, 1.0)
            np.clip(K, 0.0, 1.0, out=K)
            return K

        if states_2 is None:
            states_2 = self._build_states(X2)
        if states_1 is None:
            states_1 = states_2 if is_symmetric else self._build_states(X1)

        K = self._mps_overlap(states_1, states_2)

        if is_symmetric:
            K = 0.5 * (K + K.T)
        
        K = K.astype(self.kernel_dtype, copy=False)
        np.clip(K, 0.0, 1.0, out=K)

        if is_symmetric:
            np.fill_diagonal(K, 1.0)
        return K


    def _compute_distances(self, x1, x2, is_symmetric: bool = False):
        K = self._quantum_kernel(x1, x2, is_symmetric=is_symmetric)
        D = 1.0 - K

        if is_symmetric:
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
        return D
        

    # -------------------------------------------------------------------------
    def fit(self, X, y):
        self.X_train = X

        n_train = len(X)
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.k > n_train:
            raise ValueError(f"k={self.k} cannot be larger than the number of training samples ({n_train}).")

        printer.print("\t\tTraining the MPSQKNN...")
        self.train_states = self._build_states(X)
        K = self._quantum_kernel(X, X, is_symmetric=True, states_1=self.train_states, states_2=self.train_states)
        self.q_distances = 1.0 - K
        self.q_distances = 0.5 * (self.q_distances + self.q_distances.T)
        np.fill_diagonal(self.q_distances, 0.0)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')
        self.KNN.fit(self.q_distances, y)
        printer.print("\t\tMPSQKNN training complete.")

    # -------------------------------------------------------------------------
    def predict(self, X):
        try:
            if self.X_train is None or self.train_states is None or self.KNN is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")

            if len(X) == 0:
                raise ValueError("X must contain at least one sample.")

            K = self._quantum_kernel(X, self.X_train, is_symmetric=False, states_2=self.train_states)
            q_distances = 1.0 - K
            return self.KNN.predict(q_distances)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None
