![LazyQML](./docs/logo.jpg)
---
[![Pypi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)](https://pypi.python.org/pypi/lazyqml)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)



LazyQML is a Python library designed to streamline, automate, and accelerate experimentation with Quantum Machine Learning (QML) architectures, right on classical computers.

With LazyQML, you can:
  - 🛠️ Build, test, and benchmark QML models with minimal effort.
  
  - ⚡ Compare different QML architectures, hyperparameters seamlessly.
  
  - 🧠 Gather knowledge about the most suitable architecture for your problem.

## ✨ Why LazyQML?

- Rapid Prototyping: Experiment with different QML models using just a few lines of code.

- Automated Benchmarking: Evaluate performance and trade-offs across architectures effortlessly.

- Flexible & Modular: From basic quantum circuits to hybrid quantum-classical models—LazyQML has you covered.

## Documentation
For detailed usage instructions, API reference, and code examples, please refer to the official LazyQML [documentation](https://qhpc-sp-research-lab.github.io/LazyQML/).

## Requirements

- Python >= 3.10

> ❗❗ 
> This library is only supported by Linux Systems. It doesn't support Windows nor MacOS. 
> Only supports CUDA compatible devices.

## Installation
To install lazyqml, run this command in your terminal:

```
pip install lazyqml
```

This is the preferred method to install lazyqml, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

### From sources

To install lazyqml from sources, run this command in your terminal:

```
pip install git+https://github.com/QHPC-SP-Research-Lab/LazyQML
```
## Example

```python 
from sklearn.datasets import load_iris
from lazyqml import *

# Load data
data = load_iris()
X = data.data
y = data.target

classifier = QuantumClassifier(nqubits={4}, classifiers={Model.QNN, Model.QSVM}, epochs=10)

# Fit and predict
classifier.fit(X=X, y=y, test_size=0.4)
```

## Quantum and High Performance Computing (QHPC) - University of Oviedo    
- José Ranilla Pastor - ranilla@uniovi.es
- Elías Fernández Combarro - efernandezca@uniovi.es
- Diego García Vega - diegogarciavega@gmail.com
- Fernando Álvaro Plou Llorente - ploufernando@uniovi.es
- Alejandro Leal Castaño - lealcalejandro@uniovi.es
- Group - https://qhpc.uniovi.es

## Citing
If you used LazyQML in your work, please cite:
- García-Vega, D., Plou Llorente, F., Leal Castaño, A., Combarro, E.F., Ranilla, J.: Lazyqml: A python library to benchmark quantum machine learning models. In: 30th European Conference on Parallel and Distributed Processing (2024)

## License
- Free software: MIT License