import warnings
import numbers

from contextlib import nullcontext
from functools  import partial

import numpy     as np
import quimb.tensor as qtn
import pennylane as qml
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from lazyqml.Factories         import CircuitFactory
from lazyqml.Global            import config
from lazyqml.Interfaces.iModel import Model
from lazyqml.Utils             import printer, _numpy_math_api, get_max_bond_dim


class QKNN(Model):
    def __init__(self, nqubits, embedding, backend, shots, k=5, seed=1234):
        """
        Initialize the Quantum KNN model.
        Args:
            nqubits (int): Number of qubits for the quantum kernel.
            backend (str): Pennylane backend to use.
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
        self.qkernel = None
        self.X_train = None

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

    def _compute_distances(self, x1, x2):
        return 1 - self.kernel_circ(x1, x2)[0]

    def fit(self, X, y):
        """
        Fit the Quantum KNN model.
        Args:
            X (ndarray): Training samples (n_samples, n_features).
            y (ndarray): Training labels (n_samples,).
        """
        self.X_train = X
        self.y_train = y
        self.q_distances = self._compute_distances
        
        printer.print("\t\tTraining the QKNN...")
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric=self.q_distances)
        self.KNN.fit(X, y)

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            return self.KNN.predict(X)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise
        
    @property
    def n_params(self):
        return 0
    

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
        self.qkernel           = None
        self.X_train           = None

        self.mem_budget_mb = mem_budget_mb
        self.state_dtype   = state_dtype
        self.kernel_dtype  = kernel_dtype
        self.numpy_api     = _numpy_math_api()

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
    # MODO C CPU: bloques en RAM
    # -------------------------------------------------------------------------
    def _quantum_kernel_block(self, X1, X2, is_symmetric: bool = False):
        n1           = int(X1.shape[0])  # nº muestras X1 (test)
        n2           = int(X2.shape[0])  # nº muestras X2 (train)
        dim          = 1 << int(self.nqubits)
        bs1, bs2     = 1, 1
        ratio        = n1 / (n1 + n2) if (n1 + n2) else 0.5
        frac         = 0.80 if ratio < 0.2 else (0.60 if ratio < 0.4 else 0.50)
        overhead     = config.fastqsvm_overhead      
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
    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        """Calculate the quantum kernel matrix for SVM."""
        dim = 1 << int(self.nqubits)
        n1  = int(X1.shape[0])
        n2  = int(X2.shape[0])

        a_mode = False
        if self.mem_budget_mb is not None:
            overhead = config.fastqsvm_overhead

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
            printer.print(f'MiB necesarios {MiB_total_need} -Budget_MB {float(self.mem_budget_mb)}')
            a_mode = (MiB_total_need <= float(self.mem_budget_mb))
            
        printer.print(a_mode)
        if a_mode:
            x_sv = np.empty((n1, dim), dtype=self.state_dtype)
            y_sv = x_sv if is_symmetric else np.empty((n2, dim), dtype=self.state_dtype)

            # We can use batching (x_sv = self.kernel_circ(X1)), but not all backend works with it
            with threadpool_limits(limits=1, user_api=self.numpy_api):
                for k in range(n1):
                    x_sv[k, :] = self.kernel_circ(X1[k])

            if not is_symmetric:
                with threadpool_limits(limits=1, user_api=self.numpy_api):
                    for k in range(n2):
                        y_sv[k, :] = self.kernel_circ(X2[k])

            with self._threadpool_ctx():
                gram = x_sv @ y_sv.conj().T

            R = np.empty(gram.shape, dtype=self.kernel_dtype)
            np.square(gram.real, out=R)
            R += gram.imag * gram.imag

            return R
        # C Mode in CPU
        return self._quantum_kernel_block(X1, X2, is_symmetric=is_symmetric)
    
    def _compute_distances(self, x1, x2):
        return 1 - self._quantum_kernel(x1, x2)

    # -------------------------------------------------------------------------
    # API fit / predict
    # -------------------------------------------------------------------------
    def fit(self, X, y):
        
        printer.print("\t\tTraining the SVM...")
        self.X_train = X
        self.q_distances = self._compute_distances(X, X)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')
        self.KNN.fit(self.q_distances, y)
        printer.print("\t\tSVM training complete.")


    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            self.q_distances = self._compute_distances(X, self.X_train)
            return self.KNN.predict(self.q_distances)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None


# =============================================================================
# QSVM (MPS ONLY)
# =============================================================================
class FastQKNN_MPS(Model):
    def __init__(
        self,
            nqubits,
            embedding,
            k,
            *,
            mem_budget_mb = None,
            cores: int = 1,
            state_dtype=config.state_dtype,
            kernel_dtype=config.kernel_dtype
    ):
        self.circuit_factory   = CircuitFactory(nqubits, nlayers=0)
        self.nqubits           = nqubits
        self.embedding_circuit = self.circuit_factory.GetEmbeddingCircuitMPS(embedding)
        self.cores             = cores
        self.k                 = k
        self.kernel_circ       = self._build_kernel()
        self.qkernel           = None
        self.X_train           = None
        self.max_bond_dim      = get_max_bond_dim()

        self.mem_budget_mb = mem_budget_mb
        self.state_dtype   = state_dtype
        self.kernel_dtype  = kernel_dtype
        self.numpy_api     = _numpy_math_api()

    # -------------------------------------------------------------------------
    # returns mps
    # -------------------------------------------------------------------------
    def _build_kernel(self):
        def mps(x):
            psi = qtn.CircuitMPS(
                N=self.nqubits,
                dtype=self.state_dtype,
                max_bond=self.max_bond_dim, 
            )
            
            self.embedding_circuit(psi, x)
            
            psi_final = psi.psi.canonicalize(where=0)
            psi_final.normalize()
            return psi_final
        
        return mps
    
    def _build_kernel_batch(self):
        def mps_batch(X):
            # X.shape = (N, nqubits)
            N = X.shape[0]
            batch_states = [qtn.CircuitMPS(N=self.nqubits, dtype=np.complex128,
                                        max_bond=self.max_bond_dim, cutoff=self.cutoff)
                            for _ in range(N)]

            for i in range(self.nqubits):
                for b in range(N):
                    batch_states[b].apply_gate("RY", X[b,i], i)

            for i in range(self.nqubits - 1):
                for b in range(N):
                    batch_states[b].apply_gate("CZ", i, i+1)

            return [psi.psi.canonicalize(where=0) for psi in batch_states]
        return mps_batch
    
    # -------------------------------------------------------------------------
    # Context manager for BLAS threads
    # -------------------------------------------------------------------------
    def _threadpool_ctx(self):
        api = _numpy_math_api()
        if api is None:
            return nullcontext()
        return threadpool_limits(limits=self.cores, user_api=api)
    
    # -------------------------------------------------------------------------
    # CPU block-wise quantum kernel for MPS
    # -------------------------------------------------------------------------
    def _quantum_kernel_block(self, X1, X2, is_symmetric: bool =False):
        n1           = int(X1.shape[0])  # nº muestras X1 (test)
        n2           = int(X2.shape[0])  # nº muestras X2 (train)
        dim          = 0.25 * self.nqubits * (self.max_bond_dim * self.max_bond_dim)
        bs1, bs2     = 1, 1
        ratio        = n1 / (n1 + n2) if (n1 + n2) else 0.5
        frac         = 0.80 if ratio < 0.2 else (0.60 if ratio < 0.4 else 0.50)
        overhead     = config.fastqsvm_overhead      
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
            else:
                raise MemoryError("FastQSVM cannot allocate minimal buffers for block mode.")
            
        n1 = len(X1)  # nº muestras X1 (test)
        n2 = len(X2)  # nº muestras X2 (train)
        # bs1, bs2 = self._compute_block_sizes(n1, n2)

        # Initialize the kernel matrix
        kernel_matrix = np.empty((n1, n2), dtype=self.kernel_dtype)

        if is_symmetric:
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                block_X1 = [self.kernel_circ(X1[i]) for i in range(i_start, i_end)]

                for j_start in range(i_start, n2, bs2):
                    j_end = min(j_start + bs2, n2)
                    block_X2 = [self.kernel_circ(X1[j]) for j in range(j_start, j_end)]

                    # Compute kernel for this block using MPS overlap
                    with self._threadpool_ctx():
                        K_block = self._mps_overlap(block_X1, block_X2)

                    kernel_matrix[i_start:i_end, j_start:j_end] = K_block

                    # Fill symmetric block
                    if j_start != i_start:
                        kernel_matrix[j_start:j_end, i_start:i_end] = K_block.T

        else:
            for i_start in range(0, n1, bs1):
                i_end = min(i_start + bs1, n1)
                block_X1 = [self.kernel_circ(X1[i]) for i in range(i_start, i_end)]

                for j_start in range(0, n2, bs2):
                    j_end = min(j_start + bs2, n2)
                    block_X2 = [self.kernel_circ(X2[j]) for j in range(j_start, j_end)]

                    with self._threadpool_ctx():
                        K_block = self._mps_overlap(block_X1, block_X2)

                    kernel_matrix[i_start:i_end, j_start:j_end] = K_block

        return kernel_matrix

    
    # =============================================================================
    # MPS OVERLAP
    # =============================================================================
    def _mps_overlap(self, X1, X2):
        """
        Fully vectorized MPS kernel between two batches of MPS:
            K[i,j] = <X1[i] | X2[j]>|^2
        """
        N = len(X1)
        M = len(X2)
        L = len(X1[0].tensors)

        # Initialize environment
        env = np.ones((N, M), dtype=self.state_dtype)

        for site in range(L):
            tensors1 = [t.data for t in [x.tensors[site] for x in X1]]
            tensors2 = [t.data for t in [x.tensors[site] for x in X2]]

            # Determine maximum dimensions
            Dl_max = max(max(t.shape[0] if t.ndim==3 else t.shape[0] for t in tensors1),
                        max(t.shape[0] if t.ndim==3 else t.shape[0] for t in tensors2))
            Dr_max = max(max(t.shape[-1] if t.ndim==3 else 1 for t in tensors1),
                        max(t.shape[-1] if t.ndim==3 else 1 for t in tensors2))
            d_max = max(max(t.shape[1] for t in tensors1),
                        max(t.shape[1] for t in tensors2))

            # Pad tensors
            A_batch = np.zeros((N, Dl_max, d_max, Dr_max), dtype=self.state_dtype)
            B_batch = np.zeros((M, Dl_max, d_max, Dr_max), dtype=self.state_dtype)

            for i, t in enumerate(tensors1):
                Dl, d, Dr = t.shape if t.ndim==3 else (t.shape[0], t.shape[1], 1)
                A_batch[i, :Dl, :d, :Dr] = t if t.ndim==3 else t.reshape(Dl, d, Dr)

            for j, t in enumerate(tensors2):
                Dl, d, Dr = t.shape if t.ndim==3 else (t.shape[0], t.shape[1], 1)
                B_batch[j, :Dl, :d, :Dr] = t if t.ndim==3 else t.reshape(Dl, d, Dr)

            # Fully broadcasted contraction
            env = np.sum(
                A_batch[:, None, :, :, :].conj() * B_batch[None, :, :, :, :] * env[:, :, None, None, None],
                axis=(2, 3, 4)
            )

        # Return squared magnitude as kernel
        return np.abs(env)**2

    # -------------------------------------------------------------------------
    def _quantum_kernel(self, X1, X2, is_symmetric: bool = False):
        dim = 0.25 * self.nqubits * (self.max_bond_dim * self.max_bond_dim)
        n1  = X1.shape[0] # if (X1 != X2) -> X1 is X_test
        n2  = X2.shape[0] # if (X1 != X2) -> X2 is X_train

        A_mode = False
        if self.mem_budget_mb is not None:
            overhead = config.fastqsvm_overhead

            # RAM needed for statevector matrix: 2**nqubits * state_dtype * (n1 or n1+n2) 
            number_states  = 2 * n1 if is_symmetric else (n1 + n2)
            bytes_state    = np.dtype(self.state_dtype).itemsize
            byt_state_need = bytes_state * dim * number_states

            # RAM needed for R and gram matrices:
            n_elems      = (n1 * n1) if is_symmetric else (n1 * n2)
            bytes_kernel = np.dtype(self.kernel_dtype).itemsize
            byt_R_need   = n_elems * (bytes_kernel + bytes_state)

            # Total RAM needed
            MiB_total_need = ((byt_state_need + byt_R_need) * overhead) / (1024 * 1024)
            printer.print(f'MiB necesarios{MiB_total_need}-Budget_MB{float(self.mem_budget_mb)}')
            A_mode = (MiB_total_need <= float(self.mem_budget_mb))

        printer.print(A_mode)
        if A_mode:
            
            states_1 = [self.kernel_circ(x) for x in X1]
            states_2 = states_1 if np.array_equal(X1, X2) else [self.kernel_circ(x) for x in X2]
            
            with self._threadpool_ctx():
                K=self._mps_overlap(states_1, states_2)
            K = np.clip(K, 0.0, 1.0)
            return K

        else:
            # C Mode in CPU
            return self._quantum_kernel_block(X1, X2, is_symmetric=is_symmetric)

    def _compute_distances(self, x1, x2):
        return 1 - self._quantum_kernel(x1, x2)

    # -------------------------------------------------------------------------
    def fit(self, X, y):
        
        printer.print("\t\tTraining the SVM...")
        self.X_train = X
        self.q_distances = self._compute_distances(X, X)
        self.KNN = KNeighborsClassifier(n_neighbors=self.k, metric='precomputed')
        self.KNN.fit(self.q_distances, y)
        printer.print("\t\tSVM training complete.")

    # -------------------------------------------------------------------------
    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            self.q_distances = self._compute_distances(X, self.X_train)
            return self.KNN.predict(self.q_distances)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    @property
    def n_params(self):
        return None