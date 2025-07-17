#!/usr/bin/env python

"""Tests for `lazyqml` package."""

import unittest

def import_data():
    from sklearn.datasets import load_iris

    # Load data
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

class TestLazyqml(unittest.TestCase):
    """Tests for `lazyqml` package."""

    @classmethod
    def setUpClass(self):
        from lazyqml.Utils import get_simulation_type, get_max_bond_dim, set_simulation_type

    def _test_import(self):
        import lazyqml 
        # print("Imported correctly")

    def _test_simulation_strings(self):
        from lazyqml.Utils import get_simulation_type, get_max_bond_dim, set_simulation_type

        # Verify getter/setter of simulation type flag
        sim = "statevector"
        set_simulation_type(sim)
        self.assertTrue(get_simulation_type(), "statevector")

        sim = "tensor"
        set_simulation_type(sim)
        self.assertTrue(get_simulation_type(), "tensor")

        # Verify that ValueError is raised when number or different string is set
        sim = 3
        with self.assertRaises(ValueError):
            set_simulation_type(sim)

        sim = "tns"
        with self.assertRaises(ValueError):
            set_simulation_type(sim)

    def _test_bdim(self):
        import lazyqml

        # TODO: Desarrollar mas
        lazyqml.get_max_bond_dim()

    def test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model

        X, y = import_data()

        nqubits = {4, 8}
        embeddings = {Embedding.RX, Embedding.DENSE_ANGLE}
        ansatzs = {Ansatzs.TWO_LOCAL}
        models = {Model.QSVM, Model.QNN, Model.QKNN}
        layers = 2
        verbose = True
        sequential = False
        epochs = 10

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers,
                            verbose=verbose, sequential=sequential, epochs=epochs)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        qc.fit(X, y)

    def _test_paper(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Model, Embedding, Ansatzs
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        data = load_breast_cancer()
        X, y = data.data, data.target

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=4241)
        
        # Initialize QuantumClassifier
        classifier = QuantumClassifier(nqubits={4, 8, 16}, embeddings={Embedding.RX, Embedding.RY, Embedding.DENSE_ANGLE, Embedding.ZZ}, ansatzs={Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT, Ansatzs.ANNULAR}, classifiers={Model.QNN}, verbose=True, sequential=False, threshold=20, epochs=5)
    
        # Fit and predict
        classifier.fit(X, y, .25)

    def _test_tensor(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model, Backend
        from lazyqml.Utils import get_simulation_type, get_max_bond_dim, set_simulation_type

        set_simulation_type("tensor")
        assert get_simulation_type() == "tensor"

        X, y = import_data()

        qubits = 4
        nqubits = {qubits}
        embeddings = {Embedding.RX}
        ansatzs = {Ansatzs.TWO_LOCAL}
        models = {Model.QNN}
        epochs = 2
        layers = 1
        verbose = True
        sequential = False
        backend = Backend.lightningTensor

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers, epochs=epochs, verbose=verbose, sequential=sequential, backend=backend)
        
        qc.fit(X, y)

    def _test_qknn(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model

        X, y = import_data()

        qubits = 4
        nqubits = {4, 8}
        embeddings = {Embedding.RX, Embedding.DENSE_ANGLE}
        ansatzs = {Ansatzs.TWO_LOCAL}
        models = {Model.QSVM, Model.QKNN}
        layers = 2
        verbose = True
        sequential = False
        epochs = 10

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers,
                            verbose=verbose, sequential=sequential, epochs=epochs)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        qc.fit(X, y)

# from lazyqml.Factories.Models import GradDescentModel
# import pennylane as qml
# class GDCustomModel(GradDescentModel):
#     # def __init__(self, kwargs):
#     #     super().__init__(**kwargs)

#     def trainable_circuit(self, x, theta):
#         qml.AngleEmbedding(x, wires=range(self.nqubits), rotation='Y')

#         param_count = 0
#         for _ in range(self.layers):
#             for i in range(self.nqubits):
#                 qml.RY(theta[param_count], wires = i)
#                 param_count += 1
#             for i in range(self.nqubits - 1):
#                 qml.CNOT(wires = [i, i + 1])

#     def getTrainableParameters(self):
#         return self.layers*self.nqubits

# class TestCustomModel(unittest.TestCase):
#     def _test_custom_basic(self):
#         from lazyqml import QuantumClassifier
#         from lazyqml.Global.globalEnums import Embedding, Ansatzs, Model, Backend
#         from lazyqml.Factories.Models.modelBlueprint import GenericModel

#         X, y = import_data()

#         qubits = 4
#         nqubits = {qubits}
#         embeddings = {Embedding.RX}
#         ansatzs = {Ansatzs.TWO_LOCAL}
#         models = {Model.QNN}
#         epochs = 10
#         layers = 5
#         verbose = True
#         sequential = False
#         backend = Backend.lightningQubit

#         # {
#         #     "name": custom_circuit_1,
#         #     "type": embedding, ansatz o modelo,
#         #     "circuit": objeto de la clase Model o Ansatz o Circuit,
#         #     "circ_params" (opcional): diccionaro de parametros
#         # },
        
#         # num_params, backend, n_class, epochs, shots, lr, batch_size

#         custom = [
#             {
#                 "name": "custom_1",
#                 "type": "model",
#                 "circuit": GDCustomModel,
#                 "circ_params": {
#                     "layers": layers + 1
#                 }
#             }
#         ]

#         qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers, epochs=epochs, verbose=verbose, sequential=sequential, backend=backend, custom_circuits=custom)
        
#         qc.fit(X, y)

if __name__ == '__main__':
    unittest.main()