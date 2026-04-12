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

    def _test_import(self):
        import lazyqml 
        # print("Imported correctly") 

    def _test_simulation_strings(self):
        from lazyqml.Utils import get_simulation_type, get_max_bond_dim, set_simulation_type

        # Verify getter/setter of simulation type flag
        sim = "statevector"
        set_simulation_type(sim)
        self.assertEqual(get_simulation_type(), "statevector")

        sim = "tensor"
        set_simulation_type(sim)
        self.assertEqual(get_simulation_type(), "tensor")

        # Verify that ValueError is raised when number or different string is set
        sim = 3
        with self.assertRaises(ValueError):
            set_simulation_type(sim)

        sim = "tns"
        with self.assertRaises(ValueError):
            set_simulation_type(sim)

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
        
        # Initialize QuantumClassifier
        classifier = QuantumClassifier(nqubits={4, 8, 16}, embeddings={Embedding.RX, Embedding.RY, Embedding.DENSE_ANGLE, Embedding.ZZ},
                    ansatzs={Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT, Ansatzs.ANNULAR}, classifiers={Model.QNN}, verbose=True, sequential=False, threshold=16, epochs=5)
    
        # Fit and predict
        classifier.fit(X, y, .25)

    def _test_tensor(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model
        from lazyqml.Utils import get_simulation_type, set_simulation_type

        set_simulation_type("tensor")
        assert get_simulation_type() == "tensor"

        X, y = import_data()

        qubits = 4
        nqubits = {qubits}
        embeddings = {Embedding.RX}
        ansatzs = {Ansatzs.TWO_LOCAL}
        models = {Model.MPSQNN}
        epochs = 2
        layers = 1
        verbose = True
        sequential = False

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers, epochs=epochs, verbose=verbose, sequential=sequential)
        
        qc.fit(X, y)

    def _test_qknn(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model

        X, y = import_data()

        qubits = 4
        nqubits = {4, 8}
        embeddings = {Embedding.RX, Embedding.DENSE_ANGLE}
        ansatzs = {Ansatzs.TWO_LOCAL}
        models = {Model.QKNN}
        layers = 2
        verbose = True
        sequential = False
        epochs = 10

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers,
                            verbose=verbose, sequential=sequential, epochs=epochs)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        qc.fit(X, y)

if __name__ == '__main__':
    unittest.main()