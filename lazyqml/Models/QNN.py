#import warnings 
from time import time

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.utils.data

from lazyqml.Factories import CircuitFactory
from lazyqml.Global.globalEnums import Backend
from lazyqml.Interfaces.iAnsatz import Ansatz
from lazyqml.Interfaces.iCircuit import Circuit
from lazyqml.Interfaces.iModel import Model
from lazyqml.Utils import printer
from lazyqml.Utils.Utils import get_max_bond_dim, get_simulation_type

#warnings.filterwarnings("ignore")

class QNN(Model):
    def __init__(
        self,
        nqubits,
        ansatz,
        embedding,
        n_class,
        layers,
        epochs,
        shots,
        lr,
        batch_size   = 10,
        torch_device = "cpu",
        backend      = "lightning.qubit",
        diff_method  = "best",
        seed         = 1234,
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nqubits      = nqubits
        self.ansatz       = ansatz
        self.embedding    = embedding
        self.n_class      = n_class
        if self.n_class < 2:
            raise ValueError("n_class must be at least 2.")
        if self.n_class > self.nqubits:
            raise ValueError(f"n_class={self.n_class} cannot exceed nqubits={self.nqubits}.")
        self.layers       = layers
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")        
        self.torch_device = torch_device
        self.backend      = backend
        self.diff_method  = diff_method
        self.shots        = shots
        self.circuit_factory = CircuitFactory(nqubits, layers)

        self._build_device()
        self._build_qnode()

        self.criterion = nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()
        self.params    = None
        self.opt       = None

    @property
    def n_params(self):
        return self._n_params

    def _build_device(self):
        if get_simulation_type() == "tensor":
            raise ValueError("QNN does not support tensor-network simulation. Use MPSQNN for tensor-based backends.")

        backend_name = self.backend.value if isinstance(self.backend, Backend) else self.backend
        shots = None if self.shots in (None, 0) else self.shots

        self.dev = qml.device(backend_name, wires=self.nqubits, shots=shots)

    def _build_qnode(self):
        wires     = range(self.nqubits)
        ansatz    = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        def circuit(x, params):
            embedding(x, wires=wires)
            ansatz.getCircuit()(params, wires=wires)

            if self.n_class == 2:
                return qml.expval(qml.PauliZ(0))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n_class))

        # QNode base (sin batch)
        base_qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method=self.diff_method)

        # Batching portable
        self.qnode = qml.batch_input(base_qnode, argnum=0)

        # Retrieve number of parameters from the ansatz
        self._n_params = ansatz.n_total_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.qnode(x, self.params)

        if self.n_class == 2:
            return y.reshape(-1, 1)

        if isinstance(y, (tuple, list)):
            y = torch.stack(list(y), dim=0)  # (n_class, batch)

        if y.ndim == 2 and y.shape[0] == self.n_class:
            y = y.transpose(0, 1)  # (batch, n_class)
        return y

    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        if len(X) != len(y):
            raise ValueError(f"Expected one target label per input sample, but got {len(X)} samples and {len(y)} labels.")

        X_train = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
        y_train = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.torch_device)

        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        self.params = torch.randn((self.n_params,), device=self.torch_device, requires_grad=True)
        self.opt    = torch.optim.Adam([self.params], lr=self.lr)

        ds          = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        printer.print(f"\tTraining QNN...")
        for _epoch in range(self.epochs):
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad(set_to_none=True)
                preds = self.forward(batch_X)
                loss  = self.criterion(preds, batch_y)
                loss.backward()
                self.opt.step()
        self.params = self.params.detach()
        printer.print("\tQNN training complete.")

    def predict(self, X):
        if self.params is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("Prediction data is empty.")

        X_test = torch.tensor(X, dtype=torch.float32).to(self.torch_device)

        printer.print("\tTesting QNN...")
        preds_all = []
        bs        = self.batch_size
        with torch.inference_mode():
            for i in range(0, X_test.shape[0], bs):
                preds_all.append(self.forward(X_test[i:i + bs]))
        y_pred = torch.cat(preds_all, dim=0)

        if self.n_class == 2:
            y_pred = torch.sigmoid(y_pred.view(-1))
            return (y_pred > 0.5).cpu().numpy()

        printer.print("\tQNN testing complete.")
        return torch.argmax(y_pred, dim=1).cpu().numpy()


class QNNBag(Model):
    def __init__(self, nqubits, backend, ansatz, embedding, n_class, layers, epochs, n_features, n_samples, n_estimators, shots, diff_method, lr=0.01, batch_size=50, seed=1234) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.shots = shots
        self.embedding = embedding
        self.n_class = n_class
        if self.n_class < 2:
            raise ValueError("n_class must be at least 2.")
        if self.n_class > self.nqubits:
            raise ValueError(f"n_class={self.n_class} cannot exceed nqubits={self.nqubits}.")

        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.diff_method = diff_method or "best"

        self.batch_size = batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.n_samples = n_samples
        if not (0 < self.n_samples <= 1):
            raise ValueError("n_samples must be in the interval (0, 1].")

        self.n_features = n_features
        if not (0 < self.n_features <= 1):
            raise ValueError("n_features must be in the interval (0, 1].")
        
        self.n_estimators = n_estimators
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")

        self.backend = backend
        backend_name = backend.value if isinstance(backend, Backend) else backend

        shots = None if self.shots in (None, 0) else self.shots
        self.deviceQ = qml.device(backend_name, wires=self.nqubits, shots=shots)
        self.device = None
        self.circuit_factory = CircuitFactory(self.nqubits, nlayers=layers)
        self.qnn = None
        self.params = None
        self.estimator_params = None
        self.random_estimator_features = None
        self._build_circuit()
        # Initialize loss function based on the number of classes
        if self.n_class == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        # Define the quantum circuit as a PennyLane qnode
        @qml.qnode(self.deviceQ, interface='torch', diff_method=self.diff_method)
        def circuit(x, theta):
            # Apply embedding and ansatz circuits
            embedding(x, wires=range(self.nqubits))
            ansatz.getCircuit()(theta, wires=range(self.nqubits))

            if self.n_class==2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]
            
        self.qnn = circuit
        # Retrieve parameters per layer from the ansatz
        self._n_params = ansatz.n_total_params

    def forward(self, x):
        qnn_output = self.qnn(x, self.params)
        if self.n_class == 2:
            return qnn_output.reshape(-1)
            # 16/4/2026: before `return qnn_output.squeeze()` but it exhibits random execution errors
        else:
            return torch.stack(qnn_output).T

    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        if len(X) != len(y):
            raise ValueError(f"Expected one target label per input sample, but got {len(X)} samples and {len(y)} labels.") 

        n_total_features = X.shape[1]
        if n_total_features < self.nqubits:
            raise ValueError(f"Input data has only {n_total_features} features, but nqubits={self.nqubits}.")

        is_gpu_backend = (self.backend == Backend.lightningGPU if isinstance(self.backend, Backend) else self.backend == Backend.lightningGPU.value)
       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and is_gpu_backend else "cpu")

        # Convert training data to torch tensors and transfer to device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.device)

        self.random_estimator_features = []
        self.estimator_params          = []

        for j in range(self.n_estimators):
            # Re-initialize parameters
            self.params = torch.randn((self.n_params,), device=self.device, requires_grad=True)
            self.opt = torch.optim.Adam([self.params], lr=self.lr)

            # Select random samples and features for each estimator
            n_subsamples      = max(1, int(self.n_samples * X.shape[0]))
            estimator_samples = np.random.choice(a=X.shape[0], size=(n_subsamples,), replace=False)
            X_train_est       = X[estimator_samples, :]
            y_train_est       = y[estimator_samples]

            random_estimator_features = np.random.choice(a=X_train_est.shape[1], size=max(1, int(self.n_features * X_train_est.shape[1])), replace=False)
            self.random_estimator_features.append(random_estimator_features)

            # Filter data by selected features
            X_train_est = X_train_est[:, random_estimator_features]

            # Create data loader
            data_loader = torch.utils.data.DataLoader(list(zip(X_train_est, y_train_est)), batch_size=self.batch_size, shuffle=True, drop_last=False)

            start_time = time()

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in data_loader:
                    self.opt.zero_grad()

                    predictions = self.forward(batch_X)
                    loss = self.criterion(predictions, batch_y)
                    loss.backward()

                    self.opt.step()
                    epoch_loss += loss.item()
                printer.print(f"\t\tEpoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(data_loader):.4f}")

            self.estimator_params.append(self.params.detach().clone())
            printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

    def predict(self, X):
        if (self.estimator_params is None or self.random_estimator_features is None or len(self.estimator_params) != self.n_estimators or len(self.random_estimator_features) != self.n_estimators):
            raise ValueError("Model has not been fitted successfully. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("Prediction data is empty.")

        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Initialize y_predictions with the correct shape
        if self.n_class == 2:
            y_predictions = torch.zeros(X_test.shape[0], 1, device=self.device)  # shape (batch_size, 1)
        else:
            y_predictions = torch.zeros(X_test.shape[0], self.n_class, device=self.device)  # shape (batch_size, n_class)

        for j in range(self.n_estimators):
            X_test_features = X_test[:, self.random_estimator_features[j]]
            qnn_output      = self.qnn(X_test_features, self.estimator_params[j])
            
            # Ensure the shape of y_pred matches the expectations
            if self.n_class == 2:
                y_pred = qnn_output.view(-1, 1)
            else:
                y_pred = torch.stack(qnn_output).T 

            y_predictions += y_pred  # Now should work without error

        # Average predictions over all estimators
        y_predictions /= self.n_estimators

        # For binary classification, use sigmoid to get probabilities
        if self.n_class == 2:
            return (torch.sigmoid(y_predictions.detach()).view(-1).cpu().numpy() > 0.5).astype(int)  # Convert to binary predictions
        else:
            return torch.argmax(y_predictions.detach(), dim=1).cpu().numpy()  # For multi-class predictions
        
    @property
    def n_params(self):
        return self._n_params
    

class MPSQNN(Model):
    def __init__(
        self,
        nqubits,
        ansatz,
        embedding,
        n_class,
        layers,
        epochs,
        shots,
        lr,
        batch_size=10,
        torch_device="cpu",
        backend="default.tensor",
        seed=1234,
        diff_method = None
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nqubits = nqubits
        self.ansatz_name = ansatz
        self.embedding_name = embedding
        self.n_class = n_class
        if self.n_class < 2:
            raise ValueError("n_class must be at least 2.")
        if self.n_class > self.nqubits:
            raise ValueError(f"n_class={self.n_class} cannot exceed nqubits={self.nqubits}.")

        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.torch_device = torch_device
        self.torch_dtype = torch.float32
        self.backend = backend
        self.shots = None
        self.diff_method = diff_method
        self.circuit_factory = CircuitFactory(nqubits, layers)

        self._build_device()
        self._build_qnode()

        self.criterion = (
            nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()
        )

        self.params = None
        self.opt = None

    # ============================================================
    # Device
    # ============================================================
    def _build_device(self):
        backend_name = self.backend.value if isinstance(self.backend, Backend) else self.backend
        self.dev     = qml.device(backend_name, wires=self.nqubits, method="mps", shots=self.shots)

    # ============================================================
    # QNode (broadcast-safe)
    # ============================================================
    def _build_qnode(self):
        wires = list(range(self.nqubits))

        ansatz_obj = self.circuit_factory.GetAnsatzCircuit(self.ansatz_name)
        embedding_fn = self.circuit_factory.GetEmbeddingCircuit(self.embedding_name)

        @qml.qnode(self.dev, interface="torch", diff_method=self.diff_method)
        def circuit(x, params):
            """
            x: (batch, features) or single sample
            params: (n_params,)
            """
            embedding_fn(x, wires=wires)
            ansatz_obj.getCircuit()(params, wires=wires)

            if self.n_class == 2:
                return qml.expval(qml.PauliZ(0))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_class)]

        self.qnode = circuit
        self._n_params = ansatz_obj.n_total_params

    # ============================================================
    # Forward for predictions
    # ============================================================
    def _format_preds(self, preds):
        if self.n_class == 2:
            return preds.reshape(-1, 1)

        if isinstance(preds, (tuple, list)):
            preds = torch.stack(list(preds), dim=0)

        if preds.ndim == 2 and preds.shape[0] == self.n_class:
            preds = preds.transpose(0, 1)

        return preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.qnode(x, self.params)
        return self._format_preds(preds)

    # ============================================================
    # Training using SPSAOptimizer
    # ============================================================
    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        if len(X) != len(y):
            raise ValueError(f"Expected one target label per input sample, but got {len(X)} samples and {len(y)} labels.")         

        X_train = torch.tensor(X, dtype=self.torch_dtype, device=self.torch_device)
        y_train = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long, device=self.torch_device)

        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        # SPSA in PennyLane expects trainable array-like inputs and a scalar
        # objective result with NumPy-like shape semantics.
        self.params = qml.numpy.array(np.random.randn(self._n_params), requires_grad=True)

        self.opt = qml.SPSAOptimizer(maxiter=1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        for _ in range(self.epochs):
            for batch_X, batch_y in loader:

                def closure(params, **kwargs):
                    params_t = torch.as_tensor(params, dtype=self.torch_dtype, device=self.torch_device)
                    preds = self._format_preds(self.qnode(batch_X, params_t))

                    loss = self.criterion(preds, batch_y)

                    return np.asarray(loss.detach().cpu().item())
                    # 16/4/2026: before `return qml.numpy.array(loss.detach().cpu().item())` but it exhibits random execution errors

                # pass stepsize here
                self.params = self.opt.step(closure, self.params, stepsize=self.lr)

        self.params = torch.as_tensor(np.asarray(self.params), dtype=self.torch_dtype, device=self.torch_device)

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, X):
        if self.params is None:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("Prediction data is empty.")

        X_test = torch.tensor(X, dtype=self.torch_dtype, device=self.torch_device)

        with torch.inference_mode():
            preds = self.forward(X_test)

        if self.n_class == 2:
            probs = torch.sigmoid(preds.view(-1))
            return (probs > 0.5).cpu().numpy()

        return torch.argmax(preds, dim=1).cpu().numpy()
    
    @property
    def n_params(self):
        return self._n_params
