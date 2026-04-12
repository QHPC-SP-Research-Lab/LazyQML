import inspect
import warnings
import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict
from typing import Any, Callable, Optional, Set
from typing_extensions import Annotated
from lazyqml.Global.globalEnums import *
from lazyqml.Utils.Utils import *
from lazyqml.Utils.Validator import *
from lazyqml.Dispatchers import Dispatcher

class QuantumClassifier(BaseModel):
    """
    Main public interface of lazyqml. It stores the configuration required to build,
    train and evaluate one or more quantum machine learning models with minimal setup.

    Parameters
    ----------
    nqubits : set[int]
        Set of qubit counts to be explored. A separate configuration is evaluated for each value.
    randomstate : int, optional (default=1234)
        Seed used to make experiments reproducible.
    predictions : bool, optional (default=False)
        If True, stores predictions in the dispatched experiments when supported.
    ignoreWarnings : bool, optional (default=True)
        If True, warnings are suppressed during execution.
    sequential : bool, optional (default=False)
        If True, selected models/configurations are executed sequentially. Otherwise, they may be executed in parallel.
    numPredictors : int, optional (default=10)
        Number of predictors used by the bagging model (`Model.QNNBAG`).
    numLayers : int, optional (default=5)
        Number of ansatz layers used by trainable quantum models.
    classifiers : set[Model], optional (default={Model.ALL})
        Set of model families to evaluate. Supported values include: `Model.ALL`, `Model.QNN`, `Model.QNNBAG`, `Model.MPSQNN`,
        `Model.QSVM`, `Model.FastQSVM`, `Model.MPSQSVM`, `Model.QKNN`, `Model.FastQKNN`, `Model.MPSQKNN`, `Model.HybridCNNQNN`.
    ansatzs : set[Ansatzs], optional (default={Ansatzs.ALL})
        Set of ansatz families to evaluate for trainable quantum models.
    embeddings : set[Embedding], optional (default={Embedding.ALL})
        Set of embeddings to evaluate.
    learningRate : float, optional (default=0.01)
        Learning rate used by trainable models.
    epochs : int, optional (default=100)
        Number of training epochs for trainable models.
    shots : int, optional (default=1)
        Number of shots used in shot-based execution.
    batchSize : int, optional (default=8)
        Batch size used by trainable models.
    threshold : int, optional (default=16)
        Threshold used by the dispatcher to guide execution decisions such as CPU/GPU prioritization when applicable.
    numSamples : float, optional (default=1.0)
        Fraction of the dataset used by each predictor in `Model.QNNBAG`.
    numFeatures : set[float], optional (default={0.3, 0.5, 0.8})
        Fractions of input features to be used by `Model.QNNBAG`. Each value defines a separate configuration.
    verbose : bool, optional (default=False)
        If True, training and execution messages are printed.
    customMetric : callable, optional (default=None)
        Custom evaluation metric. It must accept at least two arguments `(y_true, y_pred)` and return a scalar.
    customImputerNum : object, optional (default=None)
        Custom preprocessor for numeric features. It must provide callable `fit`, `transform` and `fit_transform` methods.
    customImputerCat : object, optional (default=None)
        Custom preprocessor for categorical features. It must provide callable `fit`, `transform` and `fit_transform` methods.
    cores : int, optional (default=-1)
        Number of CPU cores used for parallel execution. If `-1`, all available CPU cores are used.
    """

    # FIXME: These parameters are not used. 
    # backend : Backend enum (default=Backend.lightningQubit)
    # shots : int, optional (default=1)
        
    
    model_config = ConfigDict(strict=True)

    # nqubits: Annotated[int, Field(gt=0)] = 8
    nqubits: Annotated[Set[int], Field(description="Set of qubits, each must be greater than 0")]
    randomstate: int = 1234
    predictions: bool = False
    ignoreWarnings: bool = True
    sequential: bool = False
    numPredictors: Annotated[int, Field(gt=0)] = 10
    numLayers: Annotated[int, Field(gt=0)] = 5
    classifiers: Annotated[Set[Model], Field(min_items=1)] = {Model.ALL}
    ansatzs: Annotated[Set[Ansatzs], Field(min_items=1)] = {Ansatzs.ALL}
    embeddings: Annotated[Set[Embedding], Field(min_items=1)] = {Embedding.ALL}
    # backend: Backend = Backend.lightningQubit
    learningRate: Annotated[float, Field(gt=0)] = 0.01
    epochs: Annotated[int, Field(gt=0)] = 100
    shots: Annotated[int, Field(gt=0)] = 1
    batchSize: Annotated[int, Field(gt=0)] = 8
    threshold: Annotated[int, Field(gt=0)] = 16
    numSamples: Annotated[float, Field(gt=0, le=1)] = 1.0
    numFeatures: Annotated[Set[float], Field(min_items=1)] = {0.3, 0.5, 0.8}
    verbose: bool = False
    customMetric: Optional[Callable] = None
    customImputerNum: Optional[Any]  = None
    customImputerCat: Optional[Any]  = None
    cores: Optional[int] = -1
    _dispatcher: Any = None

    @field_validator('nqubits', mode='before')
    def check_nqubits_positive(cls, value):
        if not isinstance(value, set):
            raise TypeError('nqubits must be a set of integers')
        if any(v <= 0 for v in value):
            raise ValueError('Each value in nqubits must be greater than 0')
        return value

    @field_validator('numFeatures')
    def validate_features(cls, v):
        if not all(0 < x <= 1 for x in v):
            raise ValueError("All features must be greater than 0 and less than or equal to 1")
        return v

    @field_validator('customMetric')
    def validate_custom_metric_field(cls, metric):
        if metric is None:
            return None  # Allow None as a valid value

        # Check the function signature
        sig    = inspect.signature(metric)
        params = list(sig.parameters.values())

        positional_ok = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        has_varargs   = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

        if len(positional_ok) < 2 and not has_varargs:
            raise ValueError(f"Function {metric.__name__} must accept at least two positional arguments.")

        # Test the function by passing dummy arguments
        y_true = np.array([0, 1, 1, 0])  # Example ground truth labels
        y_pred = np.array([0, 1, 0, 0])  # Example predicted labels

        try:
            result = metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Function {metric.__name__} raised an error during execution: {e}")

        # Ensure the result is a scalar (int or float)
        if not isinstance(result, (int, float)):
            raise ValueError(f"Function {metric.__name__} returned {result}, which is not a scalar value.")

        return metric

    @field_validator('customImputerCat', 'customImputerNum')
    def check_preprocessor_methods(cls, preprocessor):
        if preprocessor is None:
            return None

        required_methods = ('fit', 'transform', 'fit_transform')

        missing = [m for m in required_methods if not callable(getattr(preprocessor, m, None))]
        if missing:
            raise ValueError(f"Object {preprocessor.__class__.__name__} is missing required callable methods: {', '.join(missing)}.")

        return preprocessor

    def check_preprocessor_methods_old  (cls, preprocessor):
        if preprocessor is None:
            return None

        # Check if preprocessor is an instance of a class
        if not isinstance(preprocessor, object):
            raise ValueError(f"Expected an instance of a class, but got {type(preprocessor).__name__}.")

        # Ensure the object has 'fit' and 'transform' methods
        if not (hasattr(preprocessor, 'fit') and hasattr(preprocessor, 'transform')):
            raise ValueError(f"Object {preprocessor.__class__.__name__} does not have required methods 'fit' and 'transform'.")

        # Optionally check if the object has 'fit_transform' method
        if not hasattr(preprocessor, 'fit_transform'):
            raise ValueError(f"Object {preprocessor.__class__.__name__} does not have 'fit_transform' method.")

        # Create dummy data for testing the preprocessor methods
        X_dummy = np.array([[1, 2], [3, 4], [5, 6]])  # Example dummy data

        try:
            # Ensure the object can fit on data
            preprocessor.fit(X_dummy)
        except Exception as e:
            raise ValueError(f"Object {preprocessor.__class__.__name__} failed to fit: {e}")

        try:
            # Ensure the object can transform data
            transformed = preprocessor.transform(X_dummy)
        except Exception as e:
            raise ValueError(f"Object {preprocessor.__class__.__name__} failed to transform: {e}")

        # Check the type of the transformed result
        if not isinstance(transformed, (np.ndarray, list)):
            raise ValueError(f"Object {preprocessor.__class__.__name__} returned {type(transformed)} from 'transform', expected np.ndarray or list.")

        return preprocessor

    def model_post_init(self, ctx):
        self._dispatcher = Dispatcher(
            sequential=self.sequential,
            threshold=self.threshold,
            cores=self.cores,
            randomstate=self.randomstate,
            nqubits=self.nqubits,
            predictions=self.predictions,
            numPredictors=self.numPredictors,
            numLayers=self.numLayers,
            classifiers=self.classifiers,
            ansatzs=self.ansatzs,
            # backend=self.backend,
            embeddings=self.embeddings,
            learningRate=self.learningRate,
            epochs=self.epochs,
            numSamples=self.numSamples,
            numFeatures=self.numFeatures,
            customMetric=self.customMetric,
            customImputerNum=self.customImputerNum,
            customImputerCat=self.customImputerCat,
            shots=self.shots,
            batch=self.batchSize
        )


    def _prepare_execution(self, X, y):
        if self.ignoreWarnings:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")

        printer.set_verbose(verbose=self.verbose)
        # Validation model to ensure input parameters are DataFrames and sizes match
        FitParamsValidatorCV(x=X, y=y)
        printer.print("Validation successful, fitting the model...")

        # Fix seed
        fixSeed(self.randomstate)

    def fit(self, X, y, test_size=0.3, showTable=True):
        """
        Main method of the QuantumClassifier class. Divides the input dataset in train and test according to the test_size parameter, creates and builds all the quantum models using the previously introduced parameters and trains them using X as training datapoints and y as target tags. 

        Parameters
        ----------
        X : ndarray
            Complete dataset values to be trained and fitted from.
        y : ndarray
            Target tags for each dataset point for supervised learning.
        test_size : float, optional (default=0.4)
            Floating point number between 0 and 1.0 that indicates which proportion of the dataset to be used to test the trained models.
        showTable : bool, optional (default=True)
            If True, prints the table of results and accuracies in the terminal.
        """

        if not (0 < test_size < 1):
            raise ValueError("test_size must be in the interval (0, 1).")

        self._prepare_execution(X, y)

        scores = self._dispatcher.dispatch(X=X, y=y, folds=1, repeats=1, mode="hold-out", testsize=test_size, showTable=showTable)
        
        return scores

    def repeated_cross_validation(self, X, y, n_splits=10, n_repeats=5, showTable=True):
        """
        Carries out k-fold cross validation based on n_splits (folds) and n_repeats (repeats). 

        Parameters
        ----------
        X : ndarray
            Complete dataset values to be trained and fitted from.
        y : ndarray
            Target tags for each dataset point for supervised learning.
        n_splits : int, optional (default=10)
            Number of folds for k-fold cross validation training.
        n_repeats : int, optional (default=5)
            Number of repetitions for k-fold cross validation.
        showTable : bool, optional (default=True)
            If True, prints the table of results and accuracies in the terminal.
        """
        self._prepare_execution(X, y)

        scores = self._dispatcher.dispatch(X=X, y=y, folds=n_splits, repeats=n_repeats, mode="cross-validation", showTable=showTable)
        
        return scores

    def leave_one_out(self, X, y, showTable=True):
        """
        Similar method to repeated_cross_validation. Carries out leave-one-out cross validation. Equivalent to repeated_cross_validation using n_splits=len(X) and n_repeats=1. 

        Parameters
        ----------
        X : ndarray
            Complete dataset values to be trained and fitted from.
        y : ndarray
            Target tags for each dataset point for supervised learning.
        n_splits : int, optional (default=10)
            Number of folds for k-fold cross validation training.
        n_repeats : int, optional (default=5)
            Number of repetitions for k-fold cross validation.
        showTable : bool, optional (default=True)
            If True, prints the table of results and accuracies in the terminal.
        """
        self._prepare_execution(X, y)

        scores = self._dispatcher.dispatch(X=X, y=y, folds=len(X), repeats=1, mode="leave-one-out", showTable=showTable)

        # self.repeated_cross_validation(X, y, len(X), 1, showTable) wouldn’t work because the mode has to be set inside the dispatch.

        return scores
    

from sklearn.datasets import load_iris
import numpy as np

def import_data():
    dataset = load_iris()
    X, y = dataset.data, dataset.target

    return X, y

if __name__ == '__main__':
    X, y = import_data()

    qubits = 4
    nqubits = {4, 8}
    embeddings = {Embedding.RX, Embedding.DENSE_ANGLE}
    ansatzs = {Ansatzs.TWO_LOCAL}
    models = {Model.QSVM, Model.FastQSVM, Model.QKNN}
    layers = 2
    verbose = True
    sequential = False
    epochs = 10

    qc = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, ansatzs=ansatzs, classifiers=models, numLayers=layers,
                        verbose=verbose, sequential=sequential, epochs=epochs)
    
    # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
    # else: qc.fit(X, y)
    qc.fit(X, y)