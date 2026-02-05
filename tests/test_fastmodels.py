#!/usr/bin/env python

import unittest
from sklearn.datasets        import make_classification

def import_data(random_state):
    X, y = make_classification(
        n_samples    = 1000,
        n_features   = 30,
        n_informative= 20,
        n_redundant  = 5,
        n_repeated   = 0,
        n_classes    = 2,
        class_sep    = 2.0, # problema sencillo
        flip_y       = 0.0, # sin ruido, sencillo
        random_state = random_state,
    )

    return X, y

class TestFastQSVM(unittest.TestCase):
    def _test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Model
        from lazyqml.Utils import set_simulation_type, get_max_bond_dim

        # set_simulation_type("tensor")
        # print(get_max_bond_dim())

        random_state = 0
        X, y = import_data(random_state)

        nqubits = {18}
        embeddings = {Embedding.DENSE_ANGLE}
        models = {Model.QSVM}
        verbose = True
        sequential = False
        # cores = 64

        # No importan
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

class TestFastQNN(unittest.TestCase):
    def test_basic_exec(self):
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

        # No importan
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