import torch
import pennylane as qml
from time import time
import numpy as np
from lazyqml.Interfaces.iModel import Model
from lazyqml.Interfaces.iAnsatz import Ansatz
from lazyqml.Interfaces.iCircuit import Circuit
from lazyqml.Factories import CircuitFactory
from lazyqml.Global.globalEnums import Backend
from lazyqml.Utils import printer, get_simulation_type, get_max_bond_dim

import warnings

from time import time

class QNNTorch(Model):
    def __init__(self, nqubits, ansatz, embedding, n_class, layers, epochs, shots, lr, batch_size, device, backend, diff_method, seed=1234) -> None:
        super().__init__()
        self.nqubits = nqubits
        self.ansatz = ansatz
        self.shots = shots
        self.embedding = embedding
        self.n_class = n_class
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.backend = backend
        self.diff_method = diff_method
        self._n_params = None
        self.circuit_factory = CircuitFactory(nqubits, nlayers=layers)
        self.qnn = None
        self.params = None
        self._build_circuit()

        # Suppress all warnings
        warnings.filterwarnings("ignore")

        # Initialize PyTorch optimizer and loss function
        self.opt = None  # Will initialize in fit method with model parameters
        if self.n_class==2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        ansatz: Ansatz = self.circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding: Circuit = self.circuit_factory.GetEmbeddingCircuit(self.embedding)

        # Define the quantum circuit as a PennyLane qnode
        @qml.qnode(self.device, interface='torch', diff_method=self.diff_method)
        def circuit(x, theta):
            
            embedding(x, wires=range(self.nqubits))
            ansatz.getCircuit()(theta, wires=range(self.nqubits))

            if self.n_class==2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(wires=n)) for n in range(self.n_class)]

        # if self.embedding == embedding.AMP: self.qnn = qml.transforms.broadcast_expand(circuit)
        # self.qnn = circuit
        self.qnn = qml.transforms.broadcast_expand(circuit)
        # Retrieve parameters per layer from the ansatz
        self._n_params = ansatz.n_total_params

    @property
    def n_params(self):
        return self._n_params

    def forward(self, x):

        qnn_output = self.qnn(x, self.params)

        if self.n_class == 2:
            return qnn_output.squeeze()
        else:
            # If qnn_output is a list, apply the transformation to each element
            return torch.stack(qnn_output).T

    def fit(self, X, y):
        # Move the model to the appropriate device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.device == Backend.lightningGPU else "cpu")

        # Convert training data to torch tensors and transfer to device
        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.n_class == 2:
            y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        else:
            y_train = torch.tensor(y, dtype=torch.long).to(self.device)

        # X_train, y_train= X, y

        # Initialize parameters as torch tensors
        printer.print(f"\t\tInitializing {self.n_params} parameters")
        self.params = torch.randn((self.n_params,), device=self.device, requires_grad=True)  # Ensure params are on the same device

        # Define optimizer
        self.opt = torch.optim.Adam([self.params], lr=self.lr)

        # Create data loader for batching
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        
        start_time = time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in data_loader:
                self.opt.zero_grad()

                # Forward pass
                # batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)  # Ensure batch data is on the same device
                predictions = self.forward(batch_X)

                # Compute loss
                loss = self.criterion(predictions, batch_y)  # Ensure all tensors are on the same device
                loss.backward()
                # Optimization step
                self.opt.step()
                epoch_loss += loss.item()

            # Print the average loss for the epoch
            printer.print(f"\t\tEpoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

        printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")
        self.params = self.params.detach().cpu()  # Save trained parameters to CPU

    def predict(self, X):
        # Convert test data to torch tensors
        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Forward pass for prediction
        y_pred = self.forward(X_test)
        
        if self.n_class == 2:
            # For binary classification, apply sigmoid to get probabilities
            y_pred = torch.sigmoid(y_pred.view(-1))  # Ensure shape is [batch_size]
            # Return class labels based on a 0.5 threshold
            return (y_pred > 0.5).cpu().numpy()  # Returns 0 or 1
        else:
            # For multi-class classification, y_pred is logits of shape [batch_size, n_class]
            # Return the class with the highest logit value
            return torch.argmax(y_pred, dim=1).cpu().numpy()  # Returns class indices