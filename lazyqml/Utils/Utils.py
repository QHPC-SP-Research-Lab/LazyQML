# Standard library
import math
from itertools import product

# Third-party libraries
import GPUtil
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits, threadpool_info

# Local imports
from lazyqml.Global.globalEnums import *
import lazyqml.Global.config as cfg

"""
------------------------------------------------------------------------------------------------------------------
    Verbose printer class
        - This class implements the functionlity to print or not depending on a boolean flag
        - The message is preceded by "[VERBOSE] {message}"
        - It is implemented as a Singleton Object
------------------------------------------------------------------------------------------------------------------
"""
class VerbosePrinter:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VerbosePrinter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not VerbosePrinter._initialized:
            self.verbose = False
            VerbosePrinter._initialized = True

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def print(self, message: str):
        if self.verbose:
            print(f"[VERBOSE] {message}")
        else:
            pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VerbosePrinter()
        return cls._instance

"""
------------------------------------------------------------------------------------------------------------------
                                            Miscelaneous Utils
------------------------------------------------------------------------------------------------------------------
"""

def adjustQubits(nqubits, numClasses):
    """
        Adjust the number of qubits to be able to solve the problem
    """
    # Find the next power of 2 greater than numClasses
    power = np.ceil(np.log2(numClasses))
    nqubits = 2 ** power
    # Ensure nqubits is greater than numClasses
    if nqubits <= numClasses:
        nqubits *= 2
    return int(nqubits)

def calculate_quantum_memory(num_qubits):
    """
        Estimates the memory in MiB used by the quantum circuits.
    """
    # Complex number bytes, defined in config.py
    bytes_per_qubit_state = np.dtype(cfg.state_dtype).itemsize

    # Number of possible states is 2^n, where n is the number of qubits
    if get_simulation_type() == "statevector":
        num_states = 1 << num_qubits
    else:
        num_states =  num_qubits * (get_max_bond_dim() ** 2)

    # Total memory in bytes
    total_memory_bytes = num_states * bytes_per_qubit_state * cfg.ram_overhead

    return total_memory_bytes / (1024 * 1024)

def calculate_free_memory():
    """
        Calculates the amount of free RAM
    """
    # Use psutil to get available system memory (in MiB)
    mem = psutil.virtual_memory()
    free_ram_mb = mem.available / (1024 ** 2)  # Convert bytes to MiB
    return free_ram_mb

def calculate_free_video_memory(verbose=False):
    """
    Calculates the amount of free Video Memory.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            raise ValueError("No GPUs found.")
        return gpus[0].memoryFree
    except Exception as e:
        if verbose:
            print(f"Error calculating free video memory: {e}", flush=True)
        return 0  # Return None or an appropriate default value


def _estimate_split_sizes(n, mode, folds, test_size):
    if mode == "leave-one-out":
        n_test = 1 if n > 0 else 0
        n_train = max(0, n - n_test)
        return n_train, n_test

    if mode == "hold-out":
        n_test = int(round(n * float(test_size)))
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        n_train = max(0, n - n_test)
        return n_train, n_test

    # default: cross-validation
    if folds is None or folds <= 1:
        n_test = max(1, n // 2) if n >= 2 else n
    else:
        n_test = int(math.ceil(n / int(folds)))
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
    n_train = max(0, n - n_test)
    return n_train, n_test


def calculate_min_memory_FastQSVM(nqubits):
    bytes_per_complex = np.dtype(cfg.state_dtype).itemsize
    overhead          = cfg.fastqsvm_overhead  
    dim               = 1 << int(nqubits)

    # States vector MiB
    MiB_state = (bytes_per_complex * dim) / (1024 * 1024)
    
    # FastQSMV needs to store at least two state vectors
    return MiB_state * overhead * 2

def calculate_quantum_memory_FastQSVM(nqubits, n, mode, folds, test_size, free_ram_mb):
    bytes_per_complex = np.dtype(cfg.state_dtype).itemsize
    bytes_per_kernel  = np.dtype(cfg.kernel_dtype).itemsize
    n_train, n_test   = _estimate_split_sizes(n, mode, folds, test_size) 
    overhead          = cfg.fastqsvm_overhead  
    dim               = 1 << int(nqubits)

    # States vector MiB
    MiB_state = (bytes_per_complex * dim) / (1024 * 1024)
    
    # FastQSMV needs to store at least two state vectors
    if (MiB_state * overhead *2 ) > free_ram_mb: 
        return MiB_state * overhead

    # Worst-case: n_train x n_train kernel
    n_elems = n_train * n_train

    # Always allocated in current implementation
    MiB_kernel_matrix = (n_elems * bytes_per_kernel) / (1024 * 1024) 

    # A-mode extra (worst-case): gram (complex) + cached statevectors (complex)
    MiB_gram   = (n_elems * bytes_per_complex) / (1024 * 1024) 
    MiB_states = n_train * MiB_state

    MiB_need_all = (MiB_kernel_matrix + MiB_gram + MiB_states) * overhead

    return min(MiB_need_all, free_ram_mb)


_NVML_INITIALIZED = False
def gpu_can_run_my_jobs(verbose=False):
    global _NVML_INITIALIZED
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            if verbose:
                print("No GPUs found.", flush=True)
            return False

        gpu = gpus[0]
        try:
            import pynvml

            if not _NVML_INITIALIZED:
                pynvml.nvmlInit()
                _NVML_INITIALIZED = True
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.id)

            mode = pynvml.nvmlDeviceGetComputeMode(handle)

            if mode == pynvml.NVML_COMPUTEMODE_PROHIBITED:
                if verbose:
                    print("GPU compute mode is PROHIBITED. Using CPU.", flush=True)
                return False

            if mode == pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)

                if procs and len(procs) > 0:
                    if verbose:
                        print(f"GPU is EXCLUSIVE_PROCESS and busy (compute procs={len(procs)}). Using CPU.", flush=True)
                    return False
            # DEFAULT (or other) modes: allow
            return True
        except Exception as e:
            if verbose:
                print(f"Warning: NVML check failed ({e}). Assuming GPU is usable.", flush=True)
            return True
    except Exception as e:
        if verbose:
            print(f"Error checking GPU status: {e}", flush=True)
        return False

def create_combinations(classifiers, embeddings, ansatzs, features, qubits, folds, repeats, n_samples_total, mode, test_size, free_ram_mb):
    classifier_list = []
    embedding_list = []
    ansatzs_list = []

    # Make sure we don't have duplicated items
    classifiers = list(classifiers)
    embeddings = list(embeddings)
    ansatzs = list(ansatzs)
    qubit_values = sorted(list(qubits))
    repeat_range = range(repeats)
    folds_range = range(folds)
    
    cv_size = repeats*folds

    if Model.ALL in classifiers:
        classifier_list = Model.list()
        classifier_list.remove(Model.ALL)
    else:
        classifier_list = classifiers

    if Embedding.ALL in embeddings:
        embedding_list = Embedding.list()
        embedding_list.remove(Embedding.ALL)
    else:
        embedding_list = embeddings

    if Ansatzs.ALL in ansatzs:
        ansatzs_list = Ansatzs.list()
        ansatzs_list.remove(Ansatzs.ALL)
    else:
        ansatzs_list = ansatzs

    combo_counter = 0
    combinations = []
    # Create all base combinations first
    for qubits in qubit_values:
        for classifier in classifier_list:
            temp_combinations = []
            if classifier == Model.QSVM or classifier == Model.FastQSVM or classifier == Model.QKNN:
                # QSVM doesn't use ansatzs or features but uses qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, [None], [None], repeat_range, folds_range))
            elif classifier == Model.QNN:
                # QNN uses ansatzs and qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, ansatzs_list, [None], repeat_range, folds_range))
            elif classifier == Model.QNN_BAG:
                # QNN_BAG uses ansatzs, features, and qubits
                temp_combinations = list(product([qubits], [classifier], embedding_list, ansatzs_list, features, repeat_range, folds_range))

            # Add memory calculation for each combination
            for combo in temp_combinations:
                if combo[1] == Model.FastQSVM:
                    memory = calculate_quantum_memory_FastQSVM(combo[0], n_samples_total, mode, folds, test_size, free_ram_mb)
                else:
                    memory = calculate_quantum_memory(combo[0])  # MiB memory used based on number of qubits
                combinations.append((combo_counter // cv_size, *combo, memory))
                combo_counter += 1
    return combinations

def fixSeed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def generate_cv_indices(X, y, mode="cross-validation", test_size=0.4, n_splits=5, n_repeats=1, random_state=None):
    """
    Generate train and test indices for either cross-validation, holdout split, or leave-one-out.

    Parameters:
        X (pd.DataFrame or np.ndarray): The features matrix.
        y (pd.Series or np.ndarray): The target vector.
        mode (str): "cross-validation", "holdout", or "leave-one-out".
        test_size (float): Test set proportion for the holdout split (ignored for CV and LOO).
        n_splits (int): Number of folds in StratifiedKFold (ignored for holdout and LOO).
        n_repeats (int): Number of repeats for cross-validation (ignored for holdout and LOO).
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary of train/test indices.
    """
    cv_indices = {}

    if mode == "hold-out":
        # Single train-test split for holdout
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        cv_indices[(0, 0)] = {
            'train_idx': train_idx,
            'test_idx': test_idx
        }

    elif mode == "cross-validation":
        # StratifiedKFold for cross-validation splits
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat if random_state is not None else None)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                cv_indices[(repeat, fold)] = {
                    'train_idx': train_idx,
                    'test_idx': test_idx
                }

    elif mode == "leave-one-out":
        # LeaveOneOut cross-validation
        loo = LeaveOneOut()
        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            cv_indices[(0, fold)] = {
                'train_idx': train_idx,
                'test_idx': test_idx
            }

    else:
        raise ValueError("Invalid mode. Choose 'hold-out', 'cross-validation', or 'leave-one-out'.")

    return cv_indices

def get_train_test_split(cv_indices, repeat_id=0, fold_id=0):
    """
    Retrieve the train and test indices for a given repeat and fold ID.

    Parameters:
        cv_indices (dict): The cross-validation indices dictionary.
        repeat_id (int): The repeat ID (0 to n_repeats-1 or 0 for holdout/LOO).
        fold_id (int): The fold ID within the specified repeat.

    Returns:
        tuple: (train_idx, test_idx) arrays for the specified fold and repeat.
    """
    indices = cv_indices.get((repeat_id, fold_id))
    if indices is None:
        print(f"RepeatID {repeat_id}, FoldID{fold_id}")
        raise ValueError("Invalid repeat_id or fold_id specified.")

    return indices['train_idx'], indices['test_idx']

def dataProcessing(X, y, prepFactory, customImputerCat, customImputerNum,
                train_idx, test_idx, ansatz=None, embedding=None):
    """
    Process data for specific train/test indices.

    Parameters:
    - X: Input features
    - y: Target variable
    - prepFactory: Preprocessing factory object
    - customImputerCat: Categorical imputer
    - customImputerNum: Numerical imputer
    - train_idx: Training set indices
    - test_idx: Test set indices
    - ansatz: Optional preprocessing ansatz
    - embedding: Optional embedding method

    Returns:
    Tuple of (X_train_processed, X_test_processed, y_train, y_test)
    """
    # Split the data using provided indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create sanitizer and preprocess
    sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)
    X_train = pd.DataFrame(sanitizer.fit_transform(X_train))
    X_test = pd.DataFrame(sanitizer.transform(X_test))

    # Apply additional preprocessing if ansatz/embedding provided
    if ansatz is not None or embedding is not None:
        preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
        X_train_processed = np.array(preprocessing.fit_transform(X_train, y=y_train))
        X_test_processed = np.array(preprocessing.transform(X_test))
    else:
        X_train_processed = np.array(X_train)
        X_test_processed = np.array(X_test)

    # Convert target variables to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train_processed, X_test_processed, y_train, y_test

######
def get_embedding_expressivity(nqubits, embedding):
    if embedding == Embedding.AMP:
        return 2**nqubits
    elif embedding == Embedding.DENSE_ANGLE:
        return 2*nqubits
    else:
        return nqubits
    
def find_output_shape(model, sample):
    sample = torch.Tensor(sample)

    output = torch.flatten(model(sample))
    return output.shape[0]


######
def _numpy_math_api():
    runtime_cfg = threadpool_info()
    for item in runtime_cfg:
        filepath = item.get("filepath", "")
        if "numpy" in filepath:
            return item.get("user_api", None)
    return None


######
def set_max_bond_dim(dim: int):
    """
    Sets the maximum bond dimension for tensor network simulation.

    Parameters
    ----------
    dim : int
        Maximum bond dimension
    """
    cfg._max_bond_dim = dim

def get_max_bond_dim():
    return cfg._max_bond_dim

def set_simulation_type(sim):
    """
    Sets the qubit representation for quantum circuit simulation.

    Parameters
    ----------
    sim : str
        String that represents the type of simulation. 'tensor' for tensor network simulation and 'statevector' for state vector simulation.
    """
    try:
        assert sim == "statevector" or sim == "tensor"
        cfg._simulation = sim

    except Exception as e:
        raise ValueError(f"Simulation type must be \"statevector\" or \"tensor\". Got \"{sim}\"")
    

def get_simulation_type():
    return cfg._simulation
######

printer = VerbosePrinter()
