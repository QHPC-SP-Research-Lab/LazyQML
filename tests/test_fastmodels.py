#!/usr/bin/env python

import unittest
from sklearn.datasets        import make_classification

def import_data(random_state):
    X, y = make_classification(
        n_samples    = 1000,
        n_features   = 60,
        n_informative= 20,
        n_redundant  = 5,
        n_repeated   = 0,
        n_classes    = 2,
        class_sep    = 2.0, # problema sencillo
        flip_y       = 0.0, # sin ruido, sencillo
        random_state = random_state,
    )

    return X, y

class TestQSVM(unittest.TestCase):
    def test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Model, Ansatzs
        from lazyqml.Utils import set_simulation_type, get_max_bond_dim, set_max_bond_dim

        set_simulation_type("tensor")
        set_max_bond_dim(1)
        # print(get_max_bond_dim())

        random_state = 0
        X, y = import_data(random_state)

        nqubits = {6}
        embeddings = {Embedding.RX}
        ansatz = {Ansatzs.TWO_LOCAL}
        models = {Model.QNN}
        verbose = True
        sequential = False
        # cores = 64

        # It doesn’t matter
        epochs = 2
        batch_size = 8

        qc = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            ansatzs=ansatz,
            classifiers=models,
            verbose=verbose,
            sequential=sequential,
            epochs=epochs,
            batchSize=batch_size,
            random_state=random_state,
            threshold=59)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        # qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        qc.fit(X, y)

class TestFastQSVM(unittest.TestCase):
    def _test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Model
        from lazyqml.Utils import set_simulation_type, get_max_bond_dim

        # set_simulation_type("tensor")
        # print(get_max_bond_dim())

        random_state = 0
        X, y = import_data(random_state)

        nqubits = {14}
        embeddings = {Embedding.DENSE_ANGLE}
        models = {Model.FastQSVM}
        verbose = True
        sequential = False
        # cores = 64

        # It doesn’t matter
        epochs = 10
        batch_size = 8

        qc = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            classifiers=models,
            verbose=verbose,
            sequential=sequential,
            epochs=epochs,
            batchSize=batch_size,
            random_state=random_state)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        # qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        qc.fit(X, y)

class TestQNN(unittest.TestCase):
    def _test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model
        from lazyqml.Utils import set_simulation_type, get_max_bond_dim

        # set_simulation_type("tensor")
        # print(get_max_bond_dim())

        random_state = 0
        X, y = import_data(random_state)

        nqubits = {4}
        embeddings = {Embedding.RX}
        ansatzs = {Ansatzs.HARDWARE_EFFICIENT}
        models = {Model.QNN}
        layers = 2
        epochs = 10
        verbose = True
        sequential = False

        # It doesn’t matter
        # splits = 8
        # repeats = 2
        # threshold = 16
        batch_size = 8
        # cores = 1

        qc = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            ansatzs=ansatzs,
            classifiers=models,
            numLayers=layers,
            verbose=verbose,
            sequential=sequential,
            epochs=epochs,
            batchSize=batch_size,
            random_state=random_state)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        # qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        qc.fit(X, y)

if __name__ == '__main__':
    unittest.main()