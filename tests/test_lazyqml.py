#!/usr/bin/env python

"""Tests for `lazyqml` package."""

import unittest
import numpy as np

def import_data():
    from sklearn.datasets import load_iris

    # Load data
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

class TestLazyqml(unittest.TestCase):
    """Tests for `lazyqml` package."""

    def test_mps_analytic_embeddings_match_tensor_overlap(self):
        from lazyqml.Models.QSVM import MPSQSVM
        from lazyqml.Global import Embedding
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, _ = import_data()
        X = np.pad(X.astype(float), ((0, 0), (0, 4)))[:8]

        for embedding in (Embedding.RX, Embedding.RY, Embedding.RZ):
            with self.subTest(embedding=embedding):
                model = MPSQSVM(nqubits=8, embedding=embedding)
                states = model._build_states(X)
                kernel_mps = model._mps_overlap(states, states)
                kernel_analytic = model._analytic_kernel(X, X)

                np.testing.assert_allclose(kernel_analytic, kernel_mps, rtol=1e-12, atol=1e-12)

    def test_mpsqknn_analytic_embeddings_match_tensor_overlap(self):
        from lazyqml.Models.QKNN import MPSQKNN
        from lazyqml.Global import Embedding
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, _ = import_data()
        X = np.pad(X.astype(float), ((0, 0), (0, 4)))[:8]

        for embedding in (Embedding.RX, Embedding.RY, Embedding.RZ):
            with self.subTest(embedding=embedding):
                model = MPSQKNN(nqubits=8, embedding=embedding, k=3)
                states = model._build_states(X)
                kernel_mps = model._mps_overlap(states, states)
                kernel_analytic = model._analytic_kernel(X, X)

                np.testing.assert_allclose(kernel_analytic, kernel_mps, rtol=1e-12, atol=1e-12)

    def test_mpsqnn_batched_forward_matches_samplewise(self):
        import torch
        from lazyqml.Models.QNN import MPSQNN
        from lazyqml.Global import Embedding, Ansatzs
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, _ = import_data()
        X = X.astype(np.float32)[:6]

        model = MPSQNN(
            nqubits=4,
            ansatz=Ansatzs.TWO_LOCAL,
            embedding=Embedding.RX,
            n_class=2,
            layers=1,
            epochs=1,
            shots=None,
            lr=0.05,
            batch_size=3,
            torch_device="cpu",
            backend="default.tensor",
        )

        model.params = torch.randn((model.n_params,), dtype=model.torch_dtype)
        batch = torch.tensor(X, dtype=model.torch_dtype)

        preds_batched = model.forward(batch)
        preds_samplewise = torch.cat(
            [model._format_preds(model.qnode(xi, model.params)) for xi in batch],
            dim=0,
        )

        torch.testing.assert_close(
            preds_batched.to(torch.float64),
            preds_samplewise.to(torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_hybrid_cnn_qnn_repeated_cross_validation(self):
        import torch
        from lazyqml.Models import HybridCNNQNN
        from lazyqml.Global import Embedding, Ansatzs
        from lazyqml.Utils import set_simulation_type

        np.random.seed(0)
        torch.manual_seed(0)
        set_simulation_type("statevector")

        X = np.random.randn(12, 1, 16, 16).astype(np.float32)
        y = np.array([0, 1] * 6, dtype=int)

        model = HybridCNNQNN(
            input_shape=X.shape[1:],
            nqubits=4,
            ansatz=Ansatzs.HARDWARE_EFFICIENT,
            embedding=Embedding.RY,
            n_class=2,
            layers=1,
            epochs=1,
            shots=0,
            lr=0.01,
            batch_size=4,
            backend="default.qubit",
        )

        scores = model.repeated_cross_validation(X, y, n_splits=3, n_repeats=2, showTable=False)

        self.assertEqual(len(scores["splits"]), 6)
        self.assertEqual(len(scores["summary"]), 1)
        self.assertIn("accuracy_mean", scores["summary"].columns)
        self.assertIn("balanced_accuracy_mean", scores["summary"].columns)
        self.assertIn("f1_weighted_mean", scores["summary"].columns)

    def test_hybrid_cnn_qnn_batched_forward_matches_samplewise(self):
        import torch
        from lazyqml.Models import HybridCNNQNN
        from lazyqml.Global import Embedding, Ansatzs
        from lazyqml.Utils import set_simulation_type

        np.random.seed(0)
        torch.manual_seed(0)
        set_simulation_type("statevector")

        X = np.random.randn(6, 1, 16, 16).astype(np.float32)

        model = HybridCNNQNN(
            input_shape=X.shape[1:],
            nqubits=4,
            ansatz=Ansatzs.HARDWARE_EFFICIENT,
            embedding=Embedding.RY,
            n_class=2,
            layers=1,
            epochs=1,
            shots=0,
            lr=0.01,
            batch_size=3,
            backend="default.qubit",
        )

        model.params = torch.randn((model.n_params,), dtype=torch.float32)
        batch = torch.tensor(X, dtype=torch.float32)

        preds_batched = model.forward(batch)
        features = model.project(model.flatten(model.pool(model.cnn(batch))))
        preds_samplewise = torch.cat(
            [model._format_preds(model.qnode(sample, model.params)) for sample in features],
            dim=0,
        )

        torch.testing.assert_close(
            preds_batched.to(torch.float64),
            preds_samplewise.to(torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )

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
        verbose = False
        sequential = False
        epochs = 10

        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers,
                            verbose=verbose, sequential=sequential, epochs=epochs)

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

        qc.fit(X, y)

if __name__ == '__main__':
    unittest.main()
