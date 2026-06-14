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

        for embedding in (Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.DENSE_ANGLE):
            with self.subTest(embedding=embedding):
                model = MPSQSVM(nqubits=8, embedding=embedding)
                states = model._build_states(X)
                kernel_mps = model._mps_overlap(states, states)
                kernel_analytic = model._analytic_kernel(X, X)

                np.testing.assert_allclose(kernel_analytic, kernel_mps, rtol=1e-12, atol=1e-12)

    def test_quantumclassifier_exposes_qsvm_svm_params(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Ansatzs, Embedding, Model

        qc = QuantumClassifier(
            nqubits={4},
            embeddings={Embedding.RX},
            ansatzs={Ansatzs.TWO_LOCAL},
            classifiers={Model.QSVM},
            svmC=2.5,
            svmClassWeight="balanced",
            svmTol=1e-4,
            svmCacheSize=256,
            svmMaxIter=12,
            svmShrinking=False,
            svmProbability=True,
            svmRandomState=99,
            svmDecisionFunctionShape="ovo",
            svmBreakTies=True,
            svmVerbose=True,
        )

        self.assertEqual(qc._dispatcher.svmC, 2.5)
        self.assertEqual(qc._dispatcher.svmClassWeight, "balanced")
        self.assertEqual(qc._dispatcher.svmTol, 1e-4)
        self.assertEqual(qc._dispatcher.svmCacheSize, 256)
        self.assertEqual(qc._dispatcher.svmMaxIter, 12)
        self.assertFalse(qc._dispatcher.svmShrinking)
        self.assertTrue(qc._dispatcher.svmProbability)
        self.assertEqual(qc._dispatcher.svmRandomState, 99)
        self.assertEqual(qc._dispatcher.svmDecisionFunctionShape, "ovo")
        self.assertTrue(qc._dispatcher.svmBreakTies)
        self.assertTrue(qc._dispatcher.svmVerbose)

    def test_qsvm_svc_parameters_are_propagated(self):
        from lazyqml.Models.QSVM import QSVM, FastQSVM, MPSQSVM
        from lazyqml.Global import Backend, Embedding
        from lazyqml.Utils import set_simulation_type
        import pennylane as qml

        X, y = import_data()
        X = np.concatenate([X[:6], X[50:56]]).astype(float)
        y = np.concatenate([y[:6], y[50:56]])

        set_simulation_type("statevector")
        base_model = QSVM(
            nqubits=4,
            embedding=Embedding.RX,
            backend=Backend.defaultQubit,
            shots=None,
            C=3.5,
            class_weight="balanced",
            tol=1e-4,
            cache_size=512,
            max_iter=25,
            shrinking=False,
            probability=True,
            random_state=7,
            decision_function_shape="ovo",
            break_ties=True,
            verbose=False,
        )
        base_model.fit(X, y)
        params = base_model.svm.get_params()
        self.assertEqual(params["C"], 3.5)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["tol"], 1e-4)
        self.assertEqual(params["cache_size"], 512)
        self.assertEqual(params["max_iter"], 25)
        self.assertFalse(params["shrinking"])
        self.assertTrue(params["probability"])
        self.assertEqual(params["random_state"], 7)
        self.assertEqual(params["decision_function_shape"], "ovo")
        self.assertTrue(params["break_ties"])

        device = qml.device("default.qubit", wires=4)
        qnode = qml.qnode(device, diff_method=None)

        fast_model = FastQSVM(
            nqubits=4,
            embedding=Embedding.DENSE_ANGLE,
            qnode=qnode,
            C=2.25,
            class_weight=None,
            tol=1e-2,
            cache_size=128,
            max_iter=15,
            shrinking=True,
            probability=False,
            random_state=11,
            decision_function_shape="ovr",
            break_ties=False,
            verbose=True,
        )
        fast_model.fit(X, y)
        params = fast_model.svm.get_params()
        self.assertEqual(params["C"], 2.25)
        self.assertEqual(params["tol"], 0.01)
        self.assertEqual(params["cache_size"], 128)
        self.assertEqual(params["max_iter"], 15)
        self.assertTrue(params["shrinking"])
        self.assertFalse(params["probability"])
        self.assertEqual(params["random_state"], 11)
        self.assertEqual(params["decision_function_shape"], "ovr")
        self.assertFalse(params["break_ties"])
        self.assertTrue(params["verbose"])

        set_simulation_type("tensor")
        mps_model = MPSQSVM(
            nqubits=4,
            embedding=Embedding.DENSE_ANGLE,
            C=1.5,
            class_weight="balanced",
            tol=5e-4,
            cache_size=256,
            max_iter=40,
            shrinking=False,
            probability=True,
            random_state=19,
            decision_function_shape="ovo",
            break_ties=True,
            verbose=False,
        )
        mps_model.fit(X, y)
        params = mps_model.svm.get_params()
        self.assertEqual(params["C"], 1.5)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["tol"], 5e-4)
        self.assertEqual(params["cache_size"], 256)
        self.assertEqual(params["max_iter"], 40)
        self.assertFalse(params["shrinking"])
        self.assertTrue(params["probability"])
        self.assertEqual(params["random_state"], 19)
        self.assertEqual(params["decision_function_shape"], "ovo")
        self.assertTrue(params["break_ties"])

    def test_fastqsvm_dense_angle_matches_exact_kernel(self):
        import pennylane as qml
        from lazyqml.Models.QSVM import FastQSVM, QSVM, _finalize_kernel_matrix
        from lazyqml.Global import Backend, Embedding

        X, _ = import_data()
        X = X[:8].astype(float)

        device = qml.device("default.qubit", wires=4)
        qnode = qml.qnode(device, diff_method=None)

        fast_model = FastQSVM(nqubits=4, embedding=Embedding.DENSE_ANGLE, qnode=qnode)
        exact_model = QSVM(nqubits=4, embedding=Embedding.DENSE_ANGLE, backend=Backend.defaultQubit, shots=None)

        kernel_fast = fast_model._quantum_kernel(X, X, True)
        kernel_exact = _finalize_kernel_matrix(
            qml.kernels.square_kernel_matrix(X, exact_model.kernel_circ, assume_normalized_kernel=True),
            is_symmetric=True,
        )

        np.testing.assert_allclose(kernel_fast, kernel_exact, rtol=1e-12, atol=1e-12)

    def test_fastqsvm_non_analytic_falls_back_to_exact_kernel(self):
        import pennylane as qml
        from lazyqml.Models.QSVM import FastQSVM, QSVM, _finalize_kernel_matrix
        from lazyqml.Global import Backend, Embedding

        X, _ = import_data()
        X = X[:6].astype(float)

        device = qml.device("default.qubit", wires=4)
        qnode = qml.qnode(device, diff_method=None)

        fast_model = FastQSVM(nqubits=4, embedding=Embedding.ZZ_LOCAL, qnode=qnode, mem_budget_mb=0.001)
        fast_model._statevector_kernel = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("statevector path should not run"))

        kernel_fast = fast_model._quantum_kernel(X, X, True)

        exact_model = QSVM(nqubits=4, embedding=Embedding.ZZ_LOCAL, backend=Backend.defaultQubit, shots=None)
        kernel_exact = _finalize_kernel_matrix(
            qml.kernels.square_kernel_matrix(X, exact_model.kernel_circ, assume_normalized_kernel=True),
            is_symmetric=True,
        )

        np.testing.assert_allclose(kernel_fast, kernel_exact, rtol=1e-12, atol=1e-12)

    def test_mpsqsvm_dense_angle_fit_predict_skip_state_build(self):
        from lazyqml.Models.QSVM import MPSQSVM
        from lazyqml.Global import Embedding
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, y = import_data()
        X = np.concatenate([X[:6], X[50:56]]).astype(float)
        y = np.concatenate([y[:6], y[50:56]])

        model = MPSQSVM(nqubits=4, embedding=Embedding.DENSE_ANGLE)
        model._build_states = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analytic path should not build MPS states"))

        model.fit(X, y)
        preds = model.predict(X[:3])

        self.assertIsNone(model.train_states)
        self.assertEqual(len(preds), 3)

    def test_fastqknn_dense_angle_matches_exact_distances(self):
        import pennylane as qml
        from lazyqml.Models.QKNN import FastQKNN, QKNN
        from lazyqml.Global import Backend, Embedding

        X, _ = import_data()
        X = X[:8].astype(float)

        device = qml.device("default.qubit", wires=4)
        qnode = qml.qnode(device, diff_method=None)

        fast_model = FastQKNN(nqubits=4, embedding=Embedding.DENSE_ANGLE, qnode=qnode, k=3)
        exact_model = QKNN(nqubits=4, embedding=Embedding.DENSE_ANGLE, backend=Backend.defaultQubit, shots=None, k=3)

        distances_fast = fast_model._compute_distances(X, X, is_symmetric=True)
        distances_exact = exact_model._compute_distances(X, X, is_symmetric=True)

        np.testing.assert_allclose(distances_fast, distances_exact, rtol=1e-12, atol=1e-12)

    def test_fastqknn_non_analytic_falls_back_to_exact_distances(self):
        import pennylane as qml
        from lazyqml.Models.QKNN import FastQKNN, QKNN
        from lazyqml.Global import Backend, Embedding

        X, _ = import_data()
        X = X[:6].astype(float)

        device = qml.device("default.qubit", wires=4)
        qnode = qml.qnode(device, diff_method=None)

        fast_model = FastQKNN(nqubits=4, embedding=Embedding.ZZ_LOCAL, qnode=qnode, k=3, mem_budget_mb=0.001)
        fast_model._statevector_kernel = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("statevector path should not run"))

        distances_fast = fast_model._compute_distances(X, X, is_symmetric=True)

        exact_model = QKNN(nqubits=4, embedding=Embedding.ZZ_LOCAL, backend=Backend.defaultQubit, shots=None, k=3)
        distances_exact = exact_model._compute_distances(X, X, is_symmetric=True)

        np.testing.assert_allclose(distances_fast, distances_exact, rtol=1e-12, atol=1e-12)

    def test_mpsqknn_dense_angle_fit_predict_skip_state_build(self):
        from lazyqml.Models.QKNN import MPSQKNN
        from lazyqml.Global import Embedding
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, y = import_data()
        X = np.concatenate([X[:6], X[50:56]]).astype(float)
        y = np.concatenate([y[:6], y[50:56]])

        model = MPSQKNN(nqubits=4, embedding=Embedding.DENSE_ANGLE, k=3)
        model._build_states = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("analytic path should not build MPS states"))

        model.fit(X, y)
        preds = model.predict(X[:3])

        self.assertIsNone(model.train_states)
        self.assertEqual(len(preds), 3)

    def test_mpsqknn_analytic_embeddings_match_tensor_overlap(self):
        from lazyqml.Models.QKNN import MPSQKNN
        from lazyqml.Global import Embedding
        from lazyqml.Utils import set_simulation_type

        set_simulation_type("tensor")

        X, _ = import_data()
        X = np.pad(X.astype(float), ((0, 0), (0, 4)))[:8]

        for embedding in (Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.DENSE_ANGLE):
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

    def test_zz_embeddings_support_both_backends(self):
        from sklearn.datasets import make_classification
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Ansatzs, Embedding, Model
        from lazyqml.Utils import set_simulation_type

        X, y = make_classification(
            n_samples=12,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            n_repeated=0,
            n_classes=2,
            random_state=0,
        )

        embeddings = {Embedding.ZZ, Embedding.ZZ_LOCAL}
        ansatzs = {Ansatzs.TWO_LOCAL}
        nqubits = {4}

        set_simulation_type("statevector")
        statevector_model = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            ansatzs=ansatzs,
            classifiers={Model.QSVM},
            verbose=False,
            sequential=False,
        )
        statevector_model.fit(X, y)

        set_simulation_type("tensor")
        tensor_model = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            ansatzs=ansatzs,
            classifiers={Model.MPSQSVM},
            verbose=False,
            sequential=False,
        )
        tensor_model.fit(X, y)

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

    def test_hybrid_cnn_qnn_repeated_cross_validation_parallel(self):
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

        scores = model.repeated_cross_validation(
            X,
            y,
            n_splits=3,
            n_repeats=1,
            showTable=False,
            n_jobs=2,
            worker_threads=1,
            interop_threads=1,
        )

        self.assertEqual(len(scores["splits"]), 3)
        self.assertListEqual(scores["splits"]["fold"].tolist(), [1, 2, 3])
        self.assertIn("accuracy_mean", scores["summary"].columns)

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
        classifier = QuantumClassifier(nqubits={4, 8, 16}, embeddings={Embedding.RX, Embedding.RY, Embedding.DENSE_ANGLE, Embedding.ZZ, Embedding.ZZ_LOCAL},
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
