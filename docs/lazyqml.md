## QuantumClassifier Parameters: 
#### Core Parameters:
- **`nqubits`**: `Set[int]`
  - Description: Set of qubit indices, where each value must be greater than 0.
  - Validation: Ensures that all elements are integers > 0.

- **`randomstate`**: `int`
  - Description: Seed value for random number generation.
  - Default: `1234`

- **`predictions`**: `bool`
  - Description: Flag to determine if predictions are enabled.
  - Default: `False`

#### Model Structure Parameters:
- **`numPredictors`**: `int`
  - Description: Number of predictors used in the QNN with bagging.
  - Constraints: Must be greater than 0.
  - Default: `10`

- **`numLayers`**: `int`
  - Description: Number of layers in the Quantum Neural Networks.
  - Constraints: Must be greater than 0.
  - Default: `5`

#### Set-Based Configuration Parameters:
- **`classifiers`**: `Set[Model]`
  - Description: Set of classifier models.
  - Constraints: Must contain at least one classifier.
  - Default: `{Model.ALL}`
  - Options: `{Model.QNN, Model.QSVM, Model.QNN_BAG, Model.QKNN}`

- **`ansatzs`**: `Set[Ansatzs]`
  - Description: Set of quantum ansatz configurations.
  - Constraints: Must contain at least one ansatz.
  - Default: `{Ansatzs.ALL}`
  - Options: `{Ansatzs.RX, Ansatzs.RY, Ansatzs.RZ, Ansatzs.ZZ, Ansatzs.AMP, Ansatzs.DENSE_ANGLE, Ansatzs.HIGHER_ORDER}`

- **`embeddings`**: `Set[Embedding]`
  - Description: Set of embedding strategies.
  - Constraints: Must contain at least one embedding.
  - Default: `{Embedding.ALL}`
  - Options: `{Embedding.HCZRX, Embedding.TREE_TENSOR, Embedding.TWO_LOCAL, Embedding.HARDWARE_EFFICENT, Embedding.ANNULAR}`

- **`features`**: `Set[float]`
  - Description: Set of feature values (must be between 0 and 1).
  - Constraints: Values > 0 and <= 1.
  - Default: `{0.3, 0.5, 0.8}`

#### Training Parameters:
- **`learningRate`**: `float`
  - Description: Learning rate for optimization.
  - Constraints: Must be greater than 0.
  - Default: `0.01`

- **`epochs`**: `int`
  - Description: Number of training epochs.
  - Constraints: Must be greater than 0.
  - Default: `100`

- **`batchSize`**: `int`
  - Description: Size of each batch during training.
  - Constraints: Must be greater than 0.
  - Default: `8`

#### Threshold and Sampling:
- **`threshold`**: `int`
  - Description: Decision threshold for parallelization, if the model is bigger than this threshold it will use GPU.
  - Constraints: Must be greater than 0.
  - Default: `22`

- **`maxSamples`**: `float`
  - Description: Maximum proportion of samples to be used from the dataset characteristics.
  - Constraints: Between 0 and 1.
  - Default: `1.0`

#### Logging and Metrics:
- **`verbose`**: `bool`
  - Description: Flag for detailed output during training.
  - Default: `False`

- **`customMetric`**: `Optional[Callable]`
  - Description: User-defined metric function for evaluation.
  - Validation:
    - Function must accept `y_true` and `y_pred` as the first two arguments.
    - Must return a scalar value (int or float).
    - Function execution is validated with dummy arguments.
  - Default: `None`

#### Custom Preprocessors:
- **`customImputerNum`**: `Optional[Any]`
  - Description: Custom numeric data imputer.
  - Validation:
    - Must be an object with `fit`, `transform`, and optionally `fit_transform` methods.
    - Validated with dummy data.
  - Default: `None`

- **`customImputerCat`**: `Optional[Any]`
  - Description: Custom categorical data imputer.
  - Validation:
    - Must be an object with `fit`, `transform`, and optionally `fit_transform` methods.
    - Validated with dummy data.
  - Default: `None`

## Functions: 

### **`fit`**
```python
fit(self, X, y, test_size=0.4, showTable=True)
```
Fits classification algorithms to `X` and `y` using a hold-out approach. Predicts and scores on a test set determined by `test_size`.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric, e.g., via `LabelEncoder` or `OrdinalEncoder`).
- **`test_size`**: Proportion of the dataset to use as the test set. Default is `0.4`.
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Validates the compatibility of input dimensions.
- Automatically applies PCA transformation for incompatible dimensions.
- Requires all categories to be present in training data.

### **`repeated_cross_validation`**
```python
repeated_cross_validation(self, X, y, n_splits=10, n_repeats=5, showTable=True)
```
Performs repeated cross-validation on the dataset using the specified splits and repeats.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric).
- **`n_splits`**: Number of folds for splitting the dataset. Default is `10`.
- **`n_repeats`**: Number of times cross-validation is repeated. Default is `5`.
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Uses `RepeatedStratifiedKFold` for generating splits.
- Aggregates results from multiple train-test splits.

### **`leave_one_out`**
```python
leave_one_out(self, X, y, showTable=True)
```
Performs leave-one-out cross-validation on the dataset.

#### Parameters:
- **`X`**: Input features (DataFrame or compatible format).
- **`y`**: Target labels (must be numeric).
- **`showTable`**: Display a table with results. Default is `True`.

#### Behavior:
- Uses `LeaveOneOut` for generating train-test splits.
- Evaluates the model on each split and aggregates results.
