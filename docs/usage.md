# Usage

To use LazyQML:

```python 
from sklearn.datasets import load_iris
from lazyqml import QuantumClassifier

# Load data
data = load_iris()
X = data.data
y = data.target

classifier = QuantumClassifier(nqubits={4}, classifiers={Model.QNN, Model.QSVM}, epochs=10)

# Fit and predict
classifier.fit(X=X, y=y, test_size=0.4)
```

