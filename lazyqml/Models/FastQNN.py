import warnings

from lazyqml.Factories import CircuitFactory
from lazyqml.Global.globalEnums import Backend
from lazyqml.Utils.Utils import get_max_bond_dim, get_simulation_type

import numpy     as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.utils.data

class QNNTorch:
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