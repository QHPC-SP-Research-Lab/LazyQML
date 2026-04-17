from pathlib import Path

from sklearn.datasets import make_classification

def import_data(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, class_sep, flip_y, random_state):
    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_informative = n_informative,
        n_redundant   = n_redundant,
        n_repeated    = n_repeated,
        n_classes     = n_classes,
        class_sep     = class_sep, # >1 problema sencillo
        flip_y        = flip_y, # 0.0 sin ruido, sencillo
        random_state  = random_state,
    )
    return X, y

class TestALL():
    def test_basic_exec(self):
        from lazyqml.Utils import set_simulation_type
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Ansatzs, Embedding, Model

        threshold    = 16
        verbose      = True
        sequential   = False
        layers       = 2
        epochs       = 1
        batch_size   = 8
        numFeatures  = {0.4, 0.4, 0.4}
        random_state = 0


        n_samples  = 20
        nqubits    = {8}
        n_features = 8
        embeddings = {Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.AMP, Embedding.DENSE_ANGLE, Embedding.HIGHER_ORDER}
        ansatzs    = {Ansatzs.HCZRX, Ansatzs.TREE_TENSOR, Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT, Ansatzs.ANNULAR}
        models     = {Model.QSVM, Model.FastQSVM, Model.QKNN, Model.FastQKNN, Model.QNN, Model.QNNBAG}

        X, y = import_data(n_samples=n_samples, n_features=n_features, n_informative=n_features-2, n_redundant=2, n_repeated=0, n_classes=2, class_sep=2, flip_y=0.0, random_state=random_state)

        set_simulation_type("statevector")
        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, verbose=verbose, sequential=sequential,
                               epochs=epochs,   numLayers=layers, batchSize=batch_size, randomstate=random_state, threshold=threshold, numFeatures=numFeatures)
        qc.fit(X, y)

        embeddings = {Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.DENSE_ANGLE, Embedding.HIGHER_ORDER}
        models     = {Model.MPSQSVM, Model.MPSQKNN, Model.MPSQNN}


        X, y = import_data(n_samples=n_samples, n_features=n_features, n_informative=n_features-2, n_redundant=2, n_repeated=0, n_classes=2, class_sep=2, flip_y=0.0, random_state=random_state)

        set_simulation_type("tensor")
        qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, verbose=verbose, sequential=sequential,
                               epochs=epochs,   numLayers=layers, batchSize=batch_size, randomstate=random_state, threshold=threshold)
        qc.fit(X, y)

if __name__ == '__main__':
    TestALL().test_basic_exec()
