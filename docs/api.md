# LazyQML API Overview

Welcome to **LazyQML** – your quantum machine learning playground! LazyQML is a cutting-edge Python library designed to simplify the integration of quantum classifiers into your machine learning workflows. With LazyQML, you'll be able to explore quantum neural networks, quantum support vector machines, and other quantum models, all while maintaining a simple and easy to use code.

At the heart of LazyQML is the **QuantumClassifier** – the Swiss Army knife of quantum machine learning. This easy-to-use class empowers you to train, evaluate, and fine-tune quantum classifiers on your data, whether you're a beginner or a seasoned quantum enthusiast. 

## Key Features

LazyQML is packed with tools to streamline quantum classification. Below are the core features that set it apart from the crowd:

### 1. **QuantumClassifier: The Heart of LazyQML**

The **QuantumClassifier** class is the core of LazyQML, offering a variety of methods for training and evaluating quantum models. It provides an elegant and flexible interface for working with quantum circuits, allowing you to explore different types of classifiers, embeddings, and ansatz circuits. The goal? To make quantum classification as intuitive as possible. 

### 2. **Simulation variants**

LazyQML provides **two simulation types**, depending on the qubit representation used underneath. This gives you the freedom to choose the right quantum simulation backend for your specific needs:

- **State Vector Simulation**: This variant simulates the full quantum state of your system, perfect for smaller systems or when you want a more intuitive understanding of quantum behavior.
  
- **Tensor Networks**: This variant uses tensor networks (MPS), providing higher scalability for larger quantum systems. It's optimized for more complex and larger datasets, helping you tackle big problems with ease.

#### **Selecting QuantumClassifier variant**

Choosing the type of simulation is as simple as calling the `set_simulation_type` method with the appropiate string: `'statevector'` for the state vector simulation and `'tensor'` for the tensor network variation.

We also offer methods to change the maximum bond dimension for the `'tensor'` representation, in order to fine-tune and provide more control of the resulting QML models.

```python
from lazyqml.Utils import set_simulation_type, set_max_bond_dim

# Use tensor network qubit representation
set_simulation_type('tensor')
set_max_bond_dim(32)

# Use state vector qubit representation
set_simulation_type('statevector')
```

<!-- #### Importing State Vector Simulation Variant:
```python
from lazyqml.st import *
```

- Use this import to access the **QuantumClassifier** based on **State Vector simulations**, simulating the full quantum state for an intuitive understanding.

#### Importing Tensor Network Variant:
```python
from lazyqml.tn import *
```
- Use this import to access the **QuantumClassifier** based on **Tensor Networks**, offering efficient simulation of larger quantum systems using approximate methods. -->

### 3. **Training and Evaluation Methods**

LazyQML offers you three robust methods to train and evaluate your quantum models. These methods are designed to give you complete control over the classification process:

#### ``QuantumClassifier.fit(...)``
The **fit** method is where the magic happens. It trains your quantum model on your dataset, selecting from different quantum classifiers, embeddings, and ansatz circuits. This method provides a simple interface to quickly train a model, view its results, and get on with your quantum journey.

- **When to use it?** Use ``fit`` when you want to quickly train and evaluate a quantum model with just a few lines of code.

#### ``QuantumClassifier.repeated_cross_validation(...)``
This method performs repeated k-fold cross-validation. It divides your dataset into k subsets, trains the model on k-1 subsets, and tests on the remaining fold. This process is repeated multiple times to provide a more accurate estimate of your model's performance.

- **When to use it?** Use ``repeated_cross_validation`` for a more comprehensive evaluation of your model, especially when working with larger datasets.

#### ``QuantumClassifier.leave_one_out(...)``
**Leave-One-Out Cross Validation (LOOCV)** is a robust technique where each data point is used as the test set exactly once. This method is fantastic for small datasets, providing a deeper understanding of your model’s performance.

- **When to use it?** Choose ``leave_one_out`` when working with small datasets and you need to evaluate every data point for a thorough assessment.

### 4. **Quantum Model Selection**

LazyQML gives you full control over your quantum model's architecture. With a rich set of predefined enums, you can easily select the correct ansatz circuits, embedding strategies, and classification models.

#### ``Ansatzs``
Ansatz circuits set the core structure of your QNN models, defining the trainable parameters that allows learning from data. LazyQML provides a readily selection of ansatz types:

- `ALL`: All available ansatz circuits.
- `HCZRX`, `TREE_TENSOR`, `TWO_LOCAL`, `HARDWARE_EFFICIENT`, `ANNULAR`: Popular ansatz circuits that are ideal for quantum machine learning.

#### ``Embeddings``
Embeddings control how your classical data is encoded onto quantum states. LazyQML offers several types of embedding strategies:

- `ALL`: All available embedding circuits.
- `RX`, `RY`, `RZ`: Common qubit rotation embeddings.
- `ZZ`, `AMP`, `DENSE_ANGLE`, `HIGHER_ORDER`: Embedding strategies based on entanglement and/or amplitude encoding.

#### ``Model``
LazyQML supports a variety of quantum models, each suited for different tasks. Choose the model that best fits your data and problem:

- `ALL`: All available quantum models.
- `QNN`: Quantum Neural Network.
- `QNN_BAG`: Quantum Neural Network with Bagging.
- `QSVM`: Quantum Support Vector Machine.
- `QKNN`: Quantum k-Nearest Neighbors.



## What's Next?

This overview introduces you to the powerful features of **LazyQML** and the **QuantumClassifier**. Whether you’re just getting started or you’re a quantum computing pro, LazyQML simplifies quantum machine learning.

For more detailed documentation on each function, parameter, and quantum algorithm, head over to the full documentation pages. Get ready to dive into the world of quantum classification with LazyQML – your quantum adventure begins here!
