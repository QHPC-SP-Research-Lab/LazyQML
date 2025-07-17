# Importing from
    # Internal Dependencies
from lazyqml.Factories import ModelFactory, PreprocessingFactory
from lazyqml.Global.globalEnums import Model, Backend
from .Tasks import QMLTask
from lazyqml.Utils import printer, calculate_free_memory, get_simulation_type, calculate_free_video_memory, generate_cv_indices, create_combinations, calculate_quantum_memory, get_train_test_split, dataProcessing
    # External Libraries
import numpy as np
import pandas as pd
import psutil

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from multiprocessing import Queue, Process, Pool, Manager
import queue
from time import time, sleep

class Dispatcher:
    def __init__(self, nqubits, randomstate, predictions, shots, numPredictors, numLayers, classifiers, ansatzs, backend, embeddings, numFeatures, learningRate, epochs, runs, batch, numSamples, customMetric, customImputerNum, customImputerCat, sequential=False, threshold=22, time=True, cores=-1):
        self.sequential = sequential
        self.threshold = threshold
        self.timeM = time
        self.cores = cores

        self.nqubits = nqubits
        self.randomstate = randomstate
        self.shots = shots
        self.numPredictors = numPredictors
        self.numLayers = numLayers
        self.classifiers = classifiers
        self.ansatzs = ansatzs
        self.backend = backend
        self.embeddings = embeddings
        self.learningRate = learningRate
        self.epochs = epochs
        self.batch = batch
        self.numSamples = numSamples
        self.numFeatures = numFeatures
        self.customMetric = customMetric
        self.customImputerNum = customImputerNum
        self.customImputerCat = customImputerCat
        self.predictions = predictions

    def execute_model(self, id, model_params, X_train, y_train, X_test, y_test, customMetric):
        model = ModelFactory().getModel(**model_params)
        preds = []
        accuracy, b_accuracy, f1, custom = 0, 0, 0, 0

        start = time()

        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_test)

        accuracy += accuracy_score(y_test, y_pred, normalize=True)
        b_accuracy += balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        if customMetric is not None:
            custom = customMetric(y_test, y_pred)

        exeT = time() - start

        # Construct dataframe with results
        dict_keys = ['nqubits', 'model', 'embedding', 'ansatz', 'n_features', 'n_samples']
        model_attr = {key: model_params[key] for key in dict_keys}

        metric_results = {
            "Time taken": exeT,
            "Accuracy": accuracy,
            "Balanced Accuracy": b_accuracy,
            "F1 Score": f1,
            "Custom Metric": custom,
            "Predictions": 0
        }

        result = pd.DataFrame([{'id': id, **model_attr, 'features': model.n_params, **metric_results}])

        return result
    
    def _print_exception(self, e: Exception):
        printer.print(f"Error in the batch: {str(e)}")

    def process_gpu_task(self, gpu_queue, results):
        while not gpu_queue.empty():
            try:
                qmltask = gpu_queue.get_nowait()

                partial_result = self.execute_model(*qmltask.get_task_params())

                # Store results
                results.append(partial_result)

            except queue.Empty:
                break
            except Exception as e:
                self._print_exception(e)

    def process_cpu_task(self, cpu_queue, gpu_queue, results):
        numProcs = psutil.cpu_count(logical=False)
        total_memory = calculate_free_memory()
        available_memory = total_memory
        if not self.sequential:
            if self.cores == -1:
                available_cores = numProcs
            else:
                available_cores = self.cores
        else:
            available_cores = 1

        # Lock para el acceso seguro a los recursos compartidos
        manager = Manager()
        resource_lock = manager.Lock()

        while not cpu_queue.empty():
            try:
                # Determinar número de cores a usar basado en el estado de gpu_queue
                if gpu_queue.empty():
                    max_cores = numProcs
                else:
                    max_cores = max(1, numProcs - 1)

                current_batch = []
                current_cores = 0

                # Recolectar items para procesar mientras haya recursos disponibles
                while current_cores < max_cores and not cpu_queue.empty():
                    try:
                        qmltask = cpu_queue.get_nowait()
                        # printer.print(f"ITEM CPU: {item[0]}")
                        mem_model = qmltask.model_memory

                        # Verificar si hay recursos suficientes
                        with resource_lock:
                            if available_memory >= mem_model and available_cores >= 1:
                                # printer.print(f"Available Resources - Memory: {available_memory}, Cores: {available_cores}")
                                available_memory -= mem_model
                                available_cores -= 1
                                current_batch.append(qmltask)
                                current_cores += 1
                            else:
                                # printer.print(f"Unavailable Resources - Requirements: {mem_model}, Available: {available_memory}")
                                cpu_queue.put(qmltask)
                                break

                    except queue.Empty:
                        break

                # Procesar el batch actual si no está vacío
                if current_batch:
                    # printer.print(f"Executing Batch of {len(current_batch)} Jobs")
                    with Pool(processes=len(current_batch)) as pool:
                        # Usamos map de forma síncrona para asegurar que todos los items se procesen
                        batch_results = pool.starmap(self.execute_model, [qmltask.get_task_params() for qmltask in current_batch])

                        # Filtramos los resultados None (errores) y los añadimos a results
                        # valid_results = [r for r in batch_results if r is not None]
                        results.extend(batch_results)

                    # Liberar recursos después del procesamiento
                    with resource_lock:
                        # printer.print("Freeing Up Resources")
                        for qmltask in current_batch:
                            mem_model = qmltask.model_memory
                            available_memory += mem_model
                            available_cores += 1
                            # printer.print(f"Freed - Memory: {available_memory}MB, Cores: {available_cores}")

                # printer.print("Waiting for next batch")
                sleep(0.1)

            except Exception as e:
                self._print_exception(e)
                import traceback
                traceback.print_exc()
                break

    def dispatch(self, X, y, showTable, folds=10, repeats=5, mode="cross-validation", testsize=0.4):
        """
        ################################################################################
        Preparing Data Structures & Initializing Variables
        ################################################################################
        """
        # Replace the list-based queues with multiprocessing queues
        manager = Manager()
        gpu_queue = Queue()
        cpu_queue = Queue()
        # Shared list for results if needed
        results = manager.list()
        # Also keep track of items for printing
        cpu_items = []
        gpu_items = []

        tensor_sim = get_simulation_type() == "tensor"

        RAM = calculate_free_memory()
        VRAM = calculate_free_video_memory()

        """
        ################################################################################
        Generate CV indices
        ################################################################################
        """
        cv_indices = generate_cv_indices(
            X, y,
            mode=mode,
            n_splits=folds,
            n_repeats=repeats,
            random_state=self.randomstate,
            test_size=testsize
        )

        # print(cv_indices)

        """
        ################################################################################
        Generating Combinations
        ################################################################################
        """

        t_pre = time()
        combinations = create_combinations(qubits=self.nqubits,
                                        classifiers=self.classifiers,
                                        embeddings=self.embeddings,
                                        features=self.numFeatures,
                                        ansatzs=self.ansatzs,
                                        repeats=repeats,
                                        folds=folds)
        cancelledQubits = set()
        to_remove = []

        # print(combinations)

        for _, combination in enumerate(combinations):
            modelMem = combination[-1]
            if modelMem > RAM and modelMem > VRAM:
                to_remove.append(combination)

        for combination in to_remove:
            combinations.remove(combination)
            cancelledQubits.add(combination[0])

        for val in cancelledQubits:
            printer.print(f"Execution with {val} Qubits are cancelled due to memory constrains -> Memory Required: {calculate_quantum_memory(val)/1024:.2f}GB Out of {calculate_free_memory()/1024:.2f}GB")

        X = pd.DataFrame(X)

        # Prepare all model executions
        for combination in combinations:
            id, qubits, name, embedding, ansatz, n_features, repeat, fold, memModel = combination
            # feature = feature if feature is not None else "~"

            # Get indices for this repeat/fold combination
            train_idx, test_idx = get_train_test_split(cv_indices, repeat, fold)

            n_classes = len(np.unique(y))
            adjustedQubits = qubits  # or use adjustQubits if needed
            prepFactory = PreprocessingFactory(adjustedQubits)

            # Process data for this specific combination using pre-generated indices
            X_train_processed, X_test_processed, y_train_processed, y_test_processed = dataProcessing(
                X,
                y,
                prepFactory,
                self.customImputerCat,
                self.customImputerNum,
                train_idx,
                test_idx,
                ansatz=ansatz,
                embedding=embedding
            )

            model_factory_params = {
                "nqubits": adjustedQubits,
                "model": name,
                "embedding": embedding,
                "ansatz": ansatz,
                "n_class": n_classes,
                "shots": self.shots,
                "seed": self.randomstate*repeat,
                "layers": self.numLayers,
                "n_samples": self.numSamples,
                "n_features": n_features,
                "lr": self.learningRate,
                "batch_size": self.batch,
                "epochs": self.epochs,
                "numPredictors": self.numPredictors
            }

            qmltask = QMLTask(
                id=id,
                model_memory=memModel,
                X_train=X_train_processed,
                X_test=X_test_processed,
                y_train=y_train_processed,
                y_test=y_test_processed,
                custom_metric=self.customMetric
            )

            # When adding items to queues
            if name == Model.QNN and qubits >= self.threshold and VRAM > memModel:
                model_factory_params["backend"] = Backend.lightningGPU if not tensor_sim else Backend.lightningTensor

                qmltask.model_params = model_factory_params
                gpu_queue.put(qmltask)

                gpu_items.append(combination)
            else:
                model_factory_params["backend"] = Backend.lightningQubit if not tensor_sim else Backend.defaultTensor

                qmltask.model_params = model_factory_params
                cpu_queue.put(qmltask)

                cpu_items.append(combination)

        if self.timeM:
            printer.print(f"PREPROCESSING TIME: {time()-t_pre}")

        """
        ################################################################################
        Creating processes
        ################################################################################
        """
        # Wait a bit to add remaining tasks to queue
        sleep(0.1)

        executionTime = time()
        gpu_process = None
        # Start GPU process
        if not gpu_queue.empty():
            gpu_process = Process(target=self.process_gpu_task, args=(gpu_queue, results))
            gpu_process.start()

        # Start CPU processes
        if not cpu_queue.empty():
            self.process_cpu_task(cpu_queue, gpu_queue, results)

        # Wait for all processes to complete
        if gpu_process is not None:
            gpu_process.join()

        executionTime = time()-executionTime
        printer.print(f"Execution TIME: {executionTime}")

        """
        ################################################################################
        Processing results
        ################################################################################
        """
        t_res = time()

        all_results = pd.concat(list(results)).reset_index(drop=True)

        scores = all_results.groupby(['id']).agg({
            'nqubits': 'first',
            'model': 'first',
            'embedding': 'first',
            'ansatz': 'first',
            'features': 'first',
            'n_features': 'first',
            'n_samples': 'first',
            'Time taken': 'sum',
            'Accuracy': 'mean',
            'Balanced Accuracy': 'mean',
            'F1 Score': 'mean',
            'Custom Metric': 'mean'
        })

        scores.loc[scores['model'] != Model.QNN_BAG, 'n_samples'] = np.nan
 
        # Clean and format dataframe
        if not self.customMetric:
            scores = scores.drop(columns=['Custom Metric'])

        scores.columns = ["Qubits", "Model", "Embedding", "Ansatz", "Features", "% Features", "% Samples", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score"]

        # Remove columns if all empty
        scores = scores.dropna(how='all', axis=1)

        # Fill values if empty
        scores = scores.fillna("~")

        # Sort scores
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).reset_index(drop=True)

        if showTable:
            print(scores.to_markdown())

        if self.timeM:
            printer.print(f"RESULTS TIME: {time() - t_res}")

        return scores
