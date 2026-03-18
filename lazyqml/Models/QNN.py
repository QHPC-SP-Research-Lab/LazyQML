import warnings
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
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nqubits      = nqubits
        self.ansatz       = ansatz
        self.embedding    = embedding
        self.n_class      = n_class
        self.layers       = layers
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        self.torch_device = torch_device
        self.backend      = backend
        self.diff_method  = diff_method
        self.shots        = shots
        self.circuit_factory = CircuitFactory(nqubits, layers)

        warnings.filterwarnings("ignore")

        self._build_device()
        self._build_qnode()

        self.criterion = nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()
        self.params    = None
        self.opt       = None

    @property
    def n_params(self):
        return self._n_params

    def _build_device(self):
        # Create device
        if get_simulation_type() == "tensor":
            if self.backend != Backend.lightningTensor:
                device_kwargs = {
                    "max_bond_dim": get_max_bond_dim(),
                    "cutoff": np.finfo(np.complex128).eps,
                    # "contract": "auto-mps",
                }
            else:
                device_kwargs = {
                    "max_bond_dim": get_max_bond_dim(),
                    "cutoff": 1e-10,
                    "cutoff_mode": "abs",
                }
            
            self.dev = qml.device(self.backend, wires=self.nqubits, method='mps', **device_kwargs)
        else:
            self.dev = qml.device(self.backend, wires=self.nqubits)

    def _build_qnode(self):
        wires = range(self.nqubits)

        ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        def circuit(x, params):
            # self.embedding(x, wires=wires)
            # self.ansatz(params, wires=wires, nlayers=self.layers)

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
        X_train = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
        y_train = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.torch_device)

        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        self.params = torch.randn((self.n_params,), device=self.torch_device, requires_grad=True)
        self.opt    = torch.optim.Adam([self.params], lr=self.lr)

        ds          = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for _epoch in range(self.epochs):
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad(set_to_none=True)
                preds = self.forward(batch_X)
                loss  = self.criterion(preds, batch_y)
                loss.backward()
                self.opt.step()
        self.params = self.params.detach()

    def predict(self, X):
        X_test = torch.tensor(X, dtype=torch.float32).to(self.torch_device)

        preds_all = []
        bs        = max(1, self.batch_size)
        with torch.inference_mode():
            for i in range(0, X_test.shape[0], bs):
                preds_all.append(self.forward(X_test[i:i + bs]))
        y_pred = torch.cat(preds_all, dim=0)

        if self.n_class == 2:
            y_pred = torch.sigmoid(y_pred.view(-1))
            return (y_pred > 0.5).cpu().numpy()

        return torch.argmax(y_pred, dim=1).cpu().numpy()


class QNNBag(Model):
    def __init__(self, nqubits, backend, ansatz, embedding, n_class, layers, epochs, n_features, n_samples, n_estimators, shots, diff_method, lr=0.01, batch_size=50, seed=1234) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.shots = shots
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.diff_method = diff_method
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.backend = backend
        self.deviceQ = qml.device(backend.value, wires=self.nqubits)
        self.device = None
        self.params_per_layer = None
        self.circuit_factory = CircuitFactory(self.nqubits, nlayers=layers)
        self.qnn = None
        self.params = None
        self._build_circuit()
        warnings.filterwarnings("ignore")
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
        @qml.qnode(self.deviceQ, interface='torch', diff_method='best')
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
            return qnn_output.squeeze()
        else:
            return torch.stack(qnn_output).T

    def fit(self, X, y):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.backend == Backend.lightningGPU else "cpu")

        # Convert training data to torch tensors and transfer to device
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.device)

        self.random_estimator_features = []

        for j in range(self.n_estimators):
            # Re-initialize parameters
            self.params = torch.randn((self.n_params,), device=self.device, requires_grad=True)
            self.opt = torch.optim.Adam([self.params], lr=self.lr)

            # Select random samples and features for each estimator
            random_estimator_samples = np.random.choice(a=X.shape[0], size=(int(self.n_samples * X.shape[0]),), replace=False)
            X_train_est = X[random_estimator_samples, :]
            y_train_est = y[random_estimator_samples]

            random_estimator_features = np.random.choice(a=X_train_est.shape[1], size=max(1, int(self.n_features * X_train_est.shape[1])), replace=False)
            self.random_estimator_features.append(random_estimator_features)

            # Filter data by selected features
            X_train_est = X_train_est[:, random_estimator_features]

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                list(zip(X_train_est, y_train_est)), batch_size=self.batch_size, shuffle=True, drop_last=True
            )

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

            printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

    def predict(self, X):
        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Initialize y_predictions with the correct shape
        if self.n_class == 2:
            y_predictions = torch.zeros(X_test.shape[0], 1, device=self.device)  # shape (batch_size, 1)
        else:
            y_predictions = torch.zeros(X_test.shape[0], self.n_class, device=self.device)  # shape (batch_size, n_class)

        for j in range(self.n_estimators):
            X_test_features = X_test[:, self.random_estimator_features[j]]
            y_pred = self.forward(X_test_features)
            
            # Ensure the shape of y_pred matches the expectations
            if self.n_class == 2:
                y_pred = y_pred.view(-1, 1)  # shape (batch_size, 1) for binary classification
            else:
                # For multi-class, ensure it has the shape (batch_size, n_class)
                y_pred = y_pred.view(-1, self.n_class)  # shape (batch_size, n_class)

            y_predictions += y_pred  # Now should work without error

        # Average predictions over all estimators
        y_predictions /= self.n_estimators

        # For binary classification, use sigmoid to get probabilities
        if self.n_class == 2:
            return (torch.sigmoid(y_predictions.detach()).cpu().numpy() > 0.5).astype(int)  # Convert to binary predictions
        else:
            return torch.argmax(y_predictions.detach(), dim=1).cpu().numpy()  # For multi-class predictions
        
    @property
    def n_params(self):
        return self._n_params
    

class QNN_SPSA(Model):
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
        diff_method = 'best'
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.nqubits = nqubits
        self.ansatz_name = ansatz
        self.embedding_name = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.backend = backend
        self.shots = shots

        self.circuit_factory = CircuitFactory(nqubits, layers)

        warnings.filterwarnings("ignore")

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
        self.dev = qml.device(
            self.backend,
            wires=self.nqubits,
            method="mps",
        )

    # ============================================================
    # QNode (broadcast-safe)
    # ============================================================
    def _build_qnode(self):
        wires = list(range(self.nqubits))

        ansatz_obj = self.circuit_factory.GetAnsatzCircuit(self.ansatz_name)
        embedding_fn = self.circuit_factory.GetEmbeddingCircuit(
            self.embedding_name
        )

        @qml.qnode(self.dev, interface="torch", diff_method=None)
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = self.qnode(x, self.params)

        if self.n_class == 2:
            return preds.reshape(-1, 1)

        # multi-class
        preds = torch.stack(preds, dim=-1)
        return preds

    # ============================================================
    # Training using SPSAOptimizer
    # ============================================================
    def fit(self, X, y):
        X_train = torch.tensor(
            X, dtype=torch.float32, device=self.torch_device
        )
        y_train = torch.tensor(
            y,
            dtype=torch.float32 if self.n_class == 2 else torch.long,
            device=self.torch_device,
        )
        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        self.params = torch.randn((self._n_params,), device=self.torch_device)

        self.opt = qml.SPSAOptimizer(maxiter=1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        for _ in range(self.epochs):
            for batch_X, batch_y in loader:

                def closure(params, **kwargs):
                    preds = self.qnode(batch_X, params)
                    loss = self.criterion(preds.view(-1, 1), batch_y)
                    return loss

                # pass stepsize here
                self.params = self.opt.step(closure, self.params, stepsize=self.lr)

        self.params = self.params.detach()

    # ============================================================
    # Prediction
    # ============================================================
    def predict(self, X):
        X_test = torch.tensor(
            X, dtype=torch.float32, device=self.torch_device
        )

        with torch.inference_mode():
            preds = self.forward(X_test)

        if self.n_class == 2:
            probs = torch.sigmoid(preds.view(-1))
            return (probs > 0.5).cpu().numpy()

        return torch.argmax(preds, dim=1).cpu().numpy()
    
    @property
    def n_params(self):
        return self._n_params