import numpy as np 
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.utils.data

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from lazyqml.Factories import CircuitFactory
from lazyqml.Global.globalEnums import Backend, Embedding
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
        lr,
        shots = None,
        batch_size=10,
        torch_device="cpu",
        backend=Backend.lightningQubit,
        diff_method="best",
        seed=1234,
        cnn_channels=(8, 16),
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_shape = input_shape
        self.nqubits = nqubits
        self.n_class = n_class
        if self.n_class < 2:
            raise ValueError("n_class must be at least 2.")
        required_outputs = 1 if self.n_class == 2 else self.n_class
        if required_outputs > self.nqubits:
            raise ValueError(f"The model requires at least {required_outputs} qubits for {self.n_class} classes, but got nqubits={self.nqubits}.")

        self.ansatz = ansatz
        self.embedding = embedding
        self.layers = layers
        self.epochs = epochs
        self.shots = shots
        self.lr = lr
        self.batch_size = batch_size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.torch_device = torch_device
        self.backend = backend
        self.diff_method = diff_method
        self.seed = seed
        self.cnn_channels = cnn_channels

        self.circuit_factory = CircuitFactory(nqubits, layers)

        self._build_device()
        self._build_classical_frontend()
        self._build_qnode()

        self.criterion = nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()

        self.params = None
        self.opt = None
        self.train_losses = []
        self.is_fitted = False


    @property
    def n_params(self):
        return self._n_params

    def _init_kwargs(self):
        return {
            "input_shape": self.input_shape,
            "nqubits": self.nqubits,
            "ansatz": self.ansatz,
            "embedding": self.embedding,
            "n_class": self.n_class,
            "layers": self.layers,
            "epochs": self.epochs,
            "lr": self.lr,
            "shots": self.shots,
            "batch_size": self.batch_size,
            "torch_device": self.torch_device,
            "backend": self.backend,
            "diff_method": self.diff_method,
            "seed": self.seed,
            "cnn_channels": self.cnn_channels,
        }

    def _clone(self, *, seed=None):
        params = self._init_kwargs()
        if seed is not None:
            params["seed"] = seed
        return type(self)(**params)

    def _format_preds(self, preds):
        if self.n_class == 2:
            return preds.reshape(-1, 1)

        if isinstance(preds, (tuple, list)):
            preds = torch.stack(list(preds), dim=0)

        if preds.ndim == 2 and preds.shape[0] == self.n_class:
            preds = preds.transpose(0, 1)

        return preds


    def _build_device(self):
        backend_name = self.backend.value if isinstance(self.backend, Backend) else self.backend
        shots        = None if self.shots in (None, 0) else self.shots

        if get_simulation_type() == "tensor":
            allowed_backends = {Backend.lightningTensor.value, "default.tensor"}
            if backend_name not in allowed_backends:
                raise ValueError( f"Tensor simulation requires a tensor-compatible backend, got '{backend_name}'.")

            if backend_name != Backend.lightningTensor.value:
                device_kwargs = {"max_bond_dim": get_max_bond_dim(), "cutoff": np.finfo(np.complex128).eps}
            else:
                device_kwargs = {"max_bond_dim": get_max_bond_dim(), "cutoff": 1e-10, "cutoff_mode": "abs"}

            self.dev = qml.device(backend_name, wires=self.nqubits, method="mps", shots=shots, **device_kwargs)
        else:
            self.dev = qml.device(backend_name, wires=self.nqubits, shots=shots)


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

        proj_dim = 2 ** self.nqubits if self.embedding == Embedding.AMP else self.nqubits
        self.project = nn.Linear(c2, proj_dim)

        self.cnn.to(self.torch_device)
        self.project.to(self.torch_device)


    def _build_qnode(self):
        wires = range(self.nqubits)

        ansatz    = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

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
        if len(self.input_shape) != 3:
            raise ValueError(f"HybridCNNQNN expects input_shape = (C, H, W), got {self.input_shape}.")

        expected_channels = self.input_shape[0]

        # One image grayscale: (H, W) -> (1, 1, H, W)
        if X.ndim == 2:
            if expected_channels != 1:
                raise ValueError(f"Input without channel dimension is only valid for 1-channel images, but expected {expected_channels} channels.")
            X = X.unsqueeze(0).unsqueeze(0)

        # Batch grayscale: (B, H, W) -> (B, 1, H, W)
        elif X.ndim == 3:
            if expected_channels == 1:
                X = X.unsqueeze(1)
            else:
                # Interpretamos (C, H, W) como una sola muestra
                if X.shape[0] == expected_channels:
                    X = X.unsqueeze(0)
                else:
                    raise ValueError(f"Ambiguous 3D input shape {tuple(X.shape)} for expected input_shape {self.input_shape}.")

        # (B, C, H, W) -> OK
        elif X.ndim == 4:
            pass
        else:
            raise ValueError(f"Expected input with 2, 3 or 4 dimensions, got shape {tuple(X.shape)}.")

        if X.shape[1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, but got {X.shape[1]}.")

        return X


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_input_shape(x)

        features = self.cnn(x)
        features = self.pool(features)
        features = self.flatten(features)
        features = self.project(features)
        outputs = self.qnode(features, self.params)
        return self._format_preds(outputs)


    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError("Training data is empty.")

        if len(X) != len(y):
            raise ValueError(f"Expected one target label per input sample, but got {len(X)} samples and {len(y)} labels.")

        self.is_fitted = False

        X_train = torch.as_tensor(X, dtype=torch.float32, device=self.torch_device)
        X_train = self._ensure_input_shape(X_train)

        y_train = torch.as_tensor(y, dtype=torch.float32 if self.n_class == 2 else torch.long, device=self.torch_device)

        if self.n_class == 2 and y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        elif self.n_class > 2 and y_train.ndim != 1:
            y_train = y_train.view(-1)

        self.params = torch.randn((self.n_params,), device=self.torch_device, requires_grad=True)

        trainable_params = [self.params]
        trainable_params += list(self.cnn.parameters())
        trainable_params += list(self.project.parameters())

        self.opt = torch.optim.Adam(trainable_params, lr=self.lr)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.cnn.train()
        self.project.train()

        self.train_losses = []
        for _epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in data_loader:
                self.opt.zero_grad(set_to_none=True)

                preds = self.forward(batch_X)
                loss  = self.criterion(preds, batch_y)

                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()

            epoch_loss /= len(data_loader)
            self.train_losses.append(epoch_loss)

        self.params = self.params.detach().clone()
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        if len(X) == 0:
            raise ValueError("Prediction data is empty.")

        X_test = torch.as_tensor(X, dtype=torch.float32, device=self.torch_device)
        X_test = self._ensure_input_shape(X_test)

        preds_all = []
        bs        = self.batch_size

        self.cnn.eval()
        self.project.eval()

        with torch.no_grad():
            for i in range(0, X_test.shape[0], bs):
                preds_all.append(self.forward(X_test[i:i + bs]))

        y_pred = torch.cat(preds_all, dim=0)

        if self.n_class == 2:
            y_pred = torch.sigmoid(y_pred.view(-1))
            return (y_pred > 0.5).cpu().numpy()
        return torch.argmax(y_pred, dim=1).cpu().numpy()

    def repeated_cross_validation(self, X, y, n_splits=5, n_repeats=1, showTable=True):
        if len(X) == 0:
            raise ValueError("Input data is empty.")
        if len(X) != len(y):
            raise ValueError(f"Expected one target label per input sample, but got {len(X)} samples and {len(y)} labels.")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")

        X = np.asarray(X)
        y = np.asarray(y)

        records = []
        base_seed = self.seed if self.seed is not None else 1234

        for repeat in range(n_repeats):
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=base_seed + repeat,
            )

            for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
                model = self._clone(seed=base_seed + repeat)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])

                records.append({
                    "repeat": repeat + 1,
                    "fold": fold,
                    "accuracy": accuracy_score(y[test_idx], preds),
                    "balanced_accuracy": balanced_accuracy_score(y[test_idx], preds),
                    "f1_weighted": f1_score(y[test_idx], preds, average="weighted"),
                })

        splits_df = pd.DataFrame.from_records(records)
        summary_df = pd.DataFrame([{
            "accuracy_mean": splits_df["accuracy"].mean(),
            "accuracy_std": splits_df["accuracy"].std(ddof=0),
            "balanced_accuracy_mean": splits_df["balanced_accuracy"].mean(),
            "balanced_accuracy_std": splits_df["balanced_accuracy"].std(ddof=0),
            "f1_weighted_mean": splits_df["f1_weighted"].mean(),
            "f1_weighted_std": splits_df["f1_weighted"].std(ddof=0),
            "n_splits": n_splits,
            "n_repeats": n_repeats,
        }])

        if showTable:
            print(splits_df.to_string(index=False))
            print(summary_df.to_string(index=False))

        return {"splits": splits_df, "summary": summary_df}
