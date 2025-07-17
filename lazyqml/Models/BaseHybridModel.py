import pennylane as qml
import torch
from time import time
from abc import abstractmethod

from lazyqml.Utils import printer, get_embedding_expressivity, find_output_shape
from lazyqml.Global.globalEnums import Backend
from lazyqml.Factories import CircuitFactory

class BaseHybridQNNModel:
    __acceptable_keys_list = ['nqubits', 'ansatz', 'embedding', 'n_class', 'layers', 'epochs', 'shots', 'lr', 'batch_size', 'device', 'backend', 'diff_method', 'seed']

    def __init__(self, **kwargs):
        for key in self.__acceptable_keys_list:
            self.__setattr__(key, kwargs.get(key))

    @property
    def n_params(self):
        return self.n_q_params + self.n_c_params

    def _build_circuit(self):
        # Get the ansatz and embedding circuits from the factory
        circuit_factory = CircuitFactory(self.nqubits, nlayers=self.layers)

        ansatz = circuit_factory.GetAnsatzCircuit(self.ansatz)
        embedding = circuit_factory.GetEmbeddingCircuit(self.embedding)

        ansatz_circ = ansatz.getCircuit()

        # Define the quantum circuit as a PennyLane qnode
        @qml.qnode(self.device, interface='torch', diff_method=self.diff_method)
        def circuit(inputs, theta):
            
            embedding(inputs, range(self.nqubits))
            ansatz_circ(theta, range(self.nqubits))

            if self.n_class==2:
                return qml.expval(qml.PauliZ(0))
            else:
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_class)]

        self.qnn = circuit
        self.n_q_params = ansatz.n_total_params

    @abstractmethod
    def _build_model(self, X):
        raise NotImplementedError()

    def fit(self, X, y):
        # Construct model
        self._build_model(X)

        # Select loss function
        if self.n_class==2:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Move the model to the appropriate device (GPU or CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.device == Backend.lightningGPU else "cpu")

        # Convert training data to torch tensors and transfer to device
        X_train = torch.tensor(X, dtype=torch.float32, requires_grad=False).to(self.device)
        if self.n_class == 2:
            y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        else:
            y_train = torch.tensor(y, dtype=torch.long).to(self.device)

        # Define optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Create data loader for batching
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        start_time = time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i, (batch_X, batch_y) in enumerate(data_loader):
                self.opt.zero_grad()

                # Forward pass
                predictions = self.model(batch_X)
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                loss.backward()

                # Optimization step
                self.opt.step()
                epoch_loss += loss.item()

            # Print the average loss for the epoch
            printer.print(f"\t\tEpoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

        printer.print(f"\t\tTraining completed in {time() - start_time:.2f} seconds")

        return self

    def predict(self, X):
        # Convert test data to torch tensors
        X_test = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Forward pass for prediction
        y_pred = self.model(X_test)
        
        if self.n_class == 2:
            # For binary classification, apply sigmoid to get probabilities
            y_pred = torch.sigmoid(y_pred.view(-1))  # Ensure shape is [batch_size]
            # Return class labels based on a 0.5 threshold
            return (y_pred > 0.5).cpu().detach().numpy()  # Returns 0 or 1
        else:
            # For multi-class classification, y_pred is logits of shape [batch_size, n_class]
            # Return the class with the highest logit value
            return torch.argmax(y_pred, dim=1).cpu().detach().numpy()  # Returns class indices

class BasicHybridModel(BaseHybridQNNModel):
    def _build_model(self, _):
        # Builds the quantum circuit of the model
        self._build_circuit()

        # Get number of features needed for dataflow between classic layer and quantum circuit 
        in_features_qcircuit = get_embedding_expressivity(self.nqubits, self.embedding)

        # Setup of the layers of the (torch) sequential model
        weight_shapes = {"theta": self.n_q_params}
        self.qlayer = qml.qnn.TorchLayer(self.qnn, weight_shapes)
        self.clayer = torch.nn.Linear(self.nqubits, in_features_qcircuit)

        # Calculate number of classical parameters
        self.n_c_params = sum(p.numel() for p in self.clayer.parameters())

        # Constructs the final torch model
        self.model_layers = [self.clayer, self.qlayer]
        self.model = torch.nn.Sequential(*self.model_layers)