from lazyqml.Interfaces.iModel import Model
import numpy as np
from sklearn.svm import SVC
import pennylane as qml
from lazyqml.Factories.Circuits.fCircuits import CircuitFactory
from lazyqml.Utils.Utils import printer

from functools import partial

class QSVM(Model):
    def __init__(self, nqubits, embedding, backend, shots, seed=1234):
        super().__init__()
        self.nqubits = nqubits
        self.embedding = embedding
        self.shots = shots
        self.device = qml.device(backend.value, wires=nqubits)
        self.CircuitFactory = CircuitFactory(nqubits,nlayers=0)
        self.kernel_circ = self._build_kernel()
        self.qkernel = None
        self.X_train = None

        # Slower
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=0)(self.kernel_circ)
        # self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.batch_kernel_circ)

        # Faster batching
        self.batch_kernel_circ = partial(qml.batch_input, argnum=1)(self.kernel_circ)

    def _build_kernel(self):
        """Build the quantum kernel using a given embedding and ansatz."""
        # Get the embedding circuit from the circuit factory
        embedding_circuit = self.CircuitFactory.GetEmbeddingCircuit(self.embedding).getCircuit()
        
        # Define the kernel circuit with adjoint embedding for the quantum kernel
        @qml.qnode(self.device, diff_method=None)
        def kernel(x1, x2):
            embedding_circuit(x1, wires=range(self.nqubits))
            qml.adjoint(embedding_circuit)(x2, wires=range(self.nqubits))
            return qml.probs(wires = range(self.nqubits))
        
        return kernel
    
    # Not used at the moment, We might be interested in computing our own kernel.
    def _quantum_kernel(self, X1, X2):
        """Calculate the quantum kernel matrix for SVM."""

        # return np.array([self.batch_kernel_circ(x1, X2) for x1 in X1])[..., 0]
        return np.array([[self.kernel_circ(x1, x2) for x2 in X2]for x1 in X1])[..., 0]
        # return np.array(self.batch_kernel_circ(X1, X2))[..., 0]

    def fit(self, X, y):
        self.X_train = X
        self.qkernel = self._quantum_kernel(X,X)
        # Train the classical SVM with the quantum kernel
        printer.print("\t\tTraining the SVM...")
        self.svm = SVC(kernel="precomputed")
        self.svm.fit(self.qkernel, y)
        printer.print("\t\tSVM training complete.")

    def predict(self, X):
        try:
            if self.X_train is None:
                raise ValueError("Model has not been fitted. Call fit() before predict().")
            
            printer.print(f"\t\t\tComputing kernel between test and training data...")
            
            # Compute kernel between test data and training data
            kernel_test = self._quantum_kernel(X, self.X_train)
            
            if kernel_test.shape[1] == 0:
                raise ValueError(f"Invalid kernel matrix shape: {kernel_test.shape}")
            
            return self.svm.predict(kernel_test)
        except Exception as e:
            printer.print(f"Error during prediction: {str(e)}")
            raise

    def getTrainableParameters(self):
        return "~"
