import warnings

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
from lazyqml.Utils.Utils import get_max_bond_dim, get_simulation_type


class HybridCNNQNN(Model):
    def __init__(
        self,
        input_shape,
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
        backend="lightning.qubit",
        diff_method="best",
        seed=1234,
        cnn_channels=(8, 16),
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_shape = input_shape
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.shots = shots
        self.lr = lr
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.backend = backend
        self.diff_method = diff_method
        self.cnn_channels = cnn_channels

        self.circuit_factory = CircuitFactory(nqubits, layers)

        warnings.filterwarnings("ignore")

        self._build_device()
        self._build_classical_frontend()
        self._build_qnode()

        self.criterion = nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()

        self.params = None
        self.opt = None


    @property
    def n_params(self):
        return self._n_params


    def _build_device(self):
        if get_simulation_type() == "tensor":
            if self.backend != Backend.lightningTensor:
                device_kwargs = {
                    "max_bond_dim": get_max_bond_dim(),
                    "cutoff": np.finfo(np.complex128).eps,
                }
            else:
                device_kwargs = {
                    "max_bond_dim": get_max_bond_dim(),
                    "cutoff": 1e-10,
                    "cutoff_mode": "abs",
                }

            self.dev = qml.device(self.backend, wires=self.nqubits, method="mps", **device_kwargs)
        else:
            self.dev = qml.device(self.backend, wires=self.nqubits)


    def _build_classical_frontend(self):
        in_channels = self.input_shape[0]
        c1, c2 = self.cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.project = nn.Linear(c2, self.nqubits)

        self.cnn.to(self.torch_device)
        self.project.to(self.torch_device)


    def _build_qnode(self):
        wires = range(self.nqubits)

        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        if ansatz is None:
            raise ValueError(f"Unknown ansatz: {self.ansatz}")

        if embedding is None:
            raise ValueError(f"Unknown embedding: {self.embedding}")

        def circuit(x, params):
            embedding(x, wires=wires)
            ansatz.getCircuit()(params, wires=wires)

            if self.n_class == 2:
                return qml.expval(qml.PauliZ(0))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n_class))

        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method=self.diff_method)

        self._n_params = ansatz.n_total_params


    def _ensure_input_shape(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 3:
            X = X.unsqueeze(1)
        return X


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_input_shape(x)

        x = self.cnn(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.project(x)  # (batch, nqubits)

        y = [self.qnode(xi, self.params) for xi in x]

        if self.n_class == 2:
            y = torch.stack(y, dim=0)
            return y.reshape(-1, 1)

        y = [torch.stack(list(yi), dim=0) if isinstance(yi, (tuple, list)) else yi for yi in y]
        y = torch.stack(y, dim=0)  # (batch, n_class)

        return y


    def fit(self, X, y):
        X_train = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
        X_train = self._ensure_input_shape(X_train)

        y_train = torch.tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long).to(self.torch_device)

        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)

        self.params = torch.randn((self.n_params,), device=self.torch_device, requires_grad=True)

        trainable_params = [self.params]
        trainable_params += list(self.cnn.parameters())
        trainable_params += list(self.project.parameters())

        self.opt = torch.optim.Adam(trainable_params, lr=self.lr)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.cnn.train()
        self.project.train()

        for _epoch in range(self.epochs):
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad(set_to_none=True)
                preds = self.forward(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.opt.step()

        self.params = self.params.detach()


    def predict(self, X):
        X_test = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
        X_test = self._ensure_input_shape(X_test)

        preds_all = []
        bs = max(1, self.batch_size)

        self.cnn.eval()
        self.project.eval()

        with torch.inference_mode():
            for i in range(0, X_test.shape[0], bs):
                preds_all.append(self.forward(X_test[i:i + bs]))

        y_pred = torch.cat(preds_all, dim=0)

        if self.n_class == 2:
            y_pred = torch.sigmoid(y_pred.view(-1))
            return (y_pred > 0.5).cpu().numpy()
        return torch.argmax(y_pred, dim=1).cpu().numpy()