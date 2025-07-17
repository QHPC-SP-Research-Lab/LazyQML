import inspect
import warnings
import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict
from typing import Callable, Optional, Set
from typing_extensions import Annotated, Set, List
from lazyqml.Global.globalEnums import *
from lazyqml.Utils.Utils import *
from lazyqml.Utils.Validator import *
from lazyqml.Dispatchers import Dispatcher

class QuantumClassifier(BaseModel):
    """
    Main class of lazyqml that serves as an inteface to build and train a wide variety of quantum machine learning models with little setup. It stores model configurations and and functions as a starting point for model fitting.

    Parameters
    ----------
    nqubits : set of ints
        Set of qubits to be used in the circuits of the quantum models
    randomSate : int, optional (default=1234)
        This integer is used as a seed for the repeatability of the experiments.
    ignoreWarnings : bool, optional (default=True)
        When set to True, the warning related to algorithms that are not able to run are ignored.
    sequential : bool, optional (default=False)
        If set to True, executes selected models and circuits in a sequential manner. Otherwise, they are executed in parallel.
    numPredictors : int, optional (default=10)
        The number of different predictoras that the Quantum Neural Networks with Bagging (QNN_Bag) will use.
    numLayers : int, optional (default=5)
        The number of layers that the QNN models will use.
    classifiers : set of Model enums, optional (default={Model.ALL})
        Selects the quantum models to build and train. Possible values are: Model.ALL, Model.QNN, Model.QNN_BAG and Model.QSVM
    ansatzs : set of Ansatzs enums, optional (default={Ansatzs.ALL})
        Selects the ansatzs to build the QNN and QNNBag quantum models. Possible values are: Ansatzs.ALL, Ansatzs.HCZRX, Ansatzs.TREE_TENSOR, Ansatzs.TWO_LOCAL, Ansatzs.HARDWARE_EFFICIENT, Ansatzs.ANNULAR.
    embeddings : list of strings, optional (default={Embedding.ALL})
        Selects the embeddings for all available quantum models. Possible values are: Embedding.ALL, Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.ZZ, Embedding.AMP, Embedding.DENSE_ANGLE, Embedding.HIGHER_ORDER.
    features : set of floats, optional (default={0.3, 0.5, 0.8})
        Set of floating point numbers between 0 and 1.0 that indicates the percentage of data features to be used for each predictor in the QNNBag quantum model. For each value, a new QNNBag model will be trained.
    learningRate : int, optional (default=0.01)
        The parameter that will be used for the optimization process of all the QNN and QNNBag models in the gradient descent.
    epochs : int, optional (default=100)
        Number of complete passes that will be done over the dataset while fitting the models.
    batchSize : int, optional (default=8)
        Number of datapoints per batch when training QNN and QNNBag models.
    threshold : int, optional (default=22)
        This parameter partially determines when to use GPU over CPU. If number of qubits surpases this threshold, GPU execution will be prioritized over CPU, but its not guaranteed. Only used for QNN models.
    maxSamples : float, optional (default=1.0)
        A floating point number between 0 and 1.0 that indicates the percentage of the dataset that will be used for each predictor in the QNNBag quantum model.
    verbose : bool, optional (default=False)
        If True, shows all training messages during the fitting of the selected models.
    customMetric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    customImputerNum : function, optional (default=None)
        When function is provided, models are imputed based on the custom numeric imputer provided.
    customImputerCat : function, optional (default=None)
        When function is provided, models are imputed based on the custom categorical imputer provided.
    cores : int, optional (default=-1)
        Number of cores used for parallel execution. If cores = -1, maximum cores available in CPU will be used.
    """

    # FIXME: Estos parametros no se usan
    # runs : int, optional (default=1)
    #    The number of training runs that will be done with the Quantum Neural Network (QNN) models.
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
    backend: Backend = Backend.lightningQubit
    learningRate: Annotated[float, Field(gt=0)] = 0.01
    epochs: Annotated[int, Field(gt=0)] = 100
    shots: Annotated[int, Field(gt=0)] = 1
    runs: Annotated[int, Field(gt=0)] = 1
    batchSize: Annotated[int, Field(gt=0)] = 8
    threshold: Annotated[int, Field(gt=0)] = 22
    numSamples: Annotated[float, Field(gt=0, le=1)] = 1.0
    numFeatures: Annotated[Set[float], Field(min_items=1)] = {0.3, 0.5, 0.8}
    verbose: bool = False
    customMetric: Optional[Callable] = None
    customImputerNum: Optional[Any] = None
    customImputerCat: Optional[Any] = None
    cores: Optional[int] = -1
    _dispatcher: Any = None

    @field_validator('nqubits', mode='before')
    def check_nqubits_positive(cls, value):
        # TODO: Funciona aunque el set no sea de enteros?
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
        sig = inspect.signature(metric)
        params = list(sig.parameters.values())

        if len(params) < 2 or params[0].name != 'y_true' or params[1].name != 'y_pred':
            raise ValueError(
                f"Function {metric.__name__} does not have the required signature. "
                f"Expected first two arguments to be 'y_true' and 'y_pred'."
            )

        # Test the function by passing dummy arguments
        y_true = np.array([0, 1, 1, 0])  # Example ground truth labels
        y_pred = np.array([0, 1, 0, 0])  # Example predicted labels

        try:
            result = metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(f"Function {metric.__name__} raised an error during execution: {e}")

        # Ensure the result is a scalar (int or float)
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Function {metric.__name__} returned {result}, which is not a scalar value."
            )

        return metric

    @field_validator('customImputerCat', 'customImputerNum')
    def check_preprocessor_methods(cls, preprocessor):
        # Check if preprocessor is an instance of a class
        if not isinstance(preprocessor, object):
            raise ValueError(
                f"Expected an instance of a class, but got {type(preprocessor).__name__}."
            )

        # Ensure the object has 'fit' and 'transform' methods
        if not (hasattr(preprocessor, 'fit') and hasattr(preprocessor, 'transform')):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} does not have required methods 'fit' and 'transform'."
            )

        # Optionally check if the object has 'fit_transform' method
        if not hasattr(preprocessor, 'fit_transform'):
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} does not have 'fit_transform' method."
            )

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
            raise ValueError(
                f"Object {preprocessor.__class__.__name__} returned {type(transformed)} from 'transform', expected np.ndarray or list."
            )

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
            backend=self.backend,
            embeddings=self.embeddings,
            learningRate=self.learningRate,
            epochs=self.epochs,
            runs=self.runs,
            numSamples=self.numSamples,
            numFeatures=self.numFeatures,
            customMetric=self.customMetric,
            customImputerNum=self.customImputerNum,
            customImputerCat=self.customImputerCat,
            shots=self.shots,
            batch=self.batchSize
        )


    def _prepare_execution(self, X, y):
        warnings.filterwarnings("ignore")
        printer.set_verbose(verbose=self.verbose)
        # Validation model to ensure input parameters are DataFrames and sizes match
        FitParamsValidatorCV(
            x=X,
            y=y
        )
        printer.print("Validation successful, fitting the model...")

        # Fix seed
        fixSeed(self.randomstate)

    def fit(self, X, y, test_size=0.4, showTable=True):
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

        self._prepare_execution(X, y)

        scores = self._dispatcher.dispatch(
                        X=X,
                        y=y,
                        folds=1,
                        repeats=1,
                        mode="hold-out",
                        testsize=test_size,
                        showTable=showTable
                    )
        
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

        scores = self._dispatcher.dispatch(
                        X=X,
                        y=y,
                        folds=n_splits,
                        repeats=n_repeats,
                        mode="cross-validation",
                        showTable=showTable
                    )
        
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

        scores = self._dispatcher.dispatch(
                        X=X,
                        y=y,
                        folds=len(X),
                        repeats=1,
                        mode="leave-one-out",
                        showTable=showTable
                    )

        # No funcionaria porque hay que poner el modo dentro del dispatch
        # self.repeated_cross_validation(X, y, len(X), 1, showTable)

        return scores