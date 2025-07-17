from lazyqml.GUI._layout import *

import sys

# Generate code function
@output_code_display.capture(clear_output=True)
def generate_code(button):
    # General options
    epochs = epochs_widget.value
    layers = layers_widget.value
    qubits = nqubits_widget.value
    dataset_ix = dataset_widget.value
    test_size = 0.7

    # Cross validation
    cv = cv_checkbox.value
    splits = splits_widget.value
    repeats = repeats_widget.value

    # Classifiers
    qsvm = qsvm_checkbox.value
    qnn = qnn_checkbox.value
    qnn_bag = qnn_bag_checkbox.value

    # Ansatzs
    hczrx = hp_checkbox.value
    tree_tensor = tt_checkbox.value
    two_local = two_checkbox.value
    hardware_efficient = hwe_checkbox.value
    annular = annular_checkbox.value

    # Embeddings
    rx = rx_checkbox.value
    ry = ry_checkbox.value
    rz = rz_checkbox.value
    zz = zz_checkbox.value
    amplitude = amp_checkbox.value
    higher_order = ho_checkbox.value
    dense_angle = dense_checkbox.value

    # Bagging options
    estimators = nestimators_widget.value
    samples = nsamples_widget.value
    features = nfeatures_widget.value

    # Other options
    runs = runs_widget.value
    seed = randomstate_widget.value
    verbose = verbose_widget.value
    sim = tn_widget.value

    tn_sim = sim != 'State vector'

    selected_embeddings = []
    selected_ansatzs = []
    selected_classifiers = []

    if qsvm: selected_classifiers.append("Model.QSVM")
    if qnn: selected_classifiers.append("Model.QNN")
    if qnn_bag: selected_classifiers.append("Model.QNN_BAG")

    if hczrx: selected_ansatzs.append("Ansatzs.HCZRX")
    if tree_tensor: selected_ansatzs.append("Ansatzs.TREE_TENSOR")
    if two_local: selected_ansatzs.append("Ansatzs.TWO_LOCAL")
    if hardware_efficient: selected_ansatzs.append("Ansatzs.HARDWARE_EFFICIENT")
    if annular: selected_ansatzs.append("Ansatzs.ANNULAR")

    if rx: selected_embeddings.append("Embedding.RX")
    if ry: selected_embeddings.append("Embedding.RY")
    if rz: selected_embeddings.append("Embedding.RZ")
    if zz: selected_embeddings.append("Embedding.ZZ")
    if amplitude: selected_embeddings.append("Embedding.AMP")
    if higher_order: selected_embeddings.append("Embedding.HIGHER_ORDER")
    if dense_angle: selected_embeddings.append("Embedding.DENSE_ANGLE")

    # Classifier parameters
    classifier_params = {
        "nqubits": {qubits}
    }

    if selected_classifiers:
        selected_classifiers = "{" + ", ".join(selected_classifiers) + "}"

        classifier_params = {
            **classifier_params,
            "classifiers": selected_classifiers
        }

    if selected_embeddings:
        selected_embeddings = "{" + ", ".join(selected_embeddings) + "}"

        classifier_params = {
            **classifier_params,
            "embeddings": selected_embeddings
        }

    if qnn:
        if selected_ansatzs:
            selected_ansatzs = "{" + ", ".join(selected_ansatzs) + "}"

            classifier_params = {
                **classifier_params,
                "ansatzs": selected_ansatzs,
                "numLayers": layers
            }
        else:
            classifier_params = {
                **classifier_params,
                "numLayers": layers
            }

    if qnn_bag:
        classifier_params = {
            **classifier_params,
            "maxSamples": samples,
            "numPredictors": estimators,
            "features": features,
        }

    # Rest of the parameters
    classifier_params = {
        **classifier_params,
        "epochs": epochs,
        "randomstate": seed,
        "runs": runs,
        "verbose": verbose
    }

    # Code snippet preparation
    fit_params = {
        "X": "X",
        "y": "y"
    }

    fit = ""
    if cv:
        fit_params["n_splits"] = splits
        fit_params["n_repeats"] = repeats

        fit += "classifier.repeated_cross_validation("
    else:
        fit_params["test_size"] = test_size
        fit += "classifier.fit("

    for k, v in fit_params.items():
        fit += f"{k}={v},"

    fit = fit[:-1] + ")"

    # Dataset
    datasets = {
        "Iris": "iris",
        "Breast Cancer": "breast_cancer",
        "Wine": "wine"
    }

    selected_dataset = datasets[dataset_ix]

    imports = "\n".join(
        [
            f"from sklearn.datasets import load_{selected_dataset}",
            "from lazyqml import QuantumClassifier",
            "from lazyqml.Global import *"
        ]
    )

    change_qbitrepr = ""
    if tn_sim:
        change_qbitrepr = "set_simulation_type('tensor')"

        imports += f"\nfrom lazyqml.Utils import set_simulation_type\n\n{change_qbitrepr}"

    data_loading = "\n".join(
        [
            f"data = load_{selected_dataset}()",
            "X, y = data.data, data.target"
        ]
    )

    classifier_params_code = "classifier = QuantumClassifier(\n"
    for k, v in classifier_params.items():
        classifier_params_code += f"    {k}={v},\n"
    classifier_params_code = classifier_params_code[:-2] + ")"

    # Code snippet
    code_snippet = "\n\n".join([imports, data_loading, classifier_params_code, fit])

    out_code.value = code_snippet

    return code_snippet


class OSNotSupported(Exception):
    """Raised when OS isn't linux"""

# Save to .py file
def save2file(button):
    if sys.platform == 'linux':
        code = generate_code(None)

        with open("output.py", "w") as f:
            f.write(code)
    else:
        raise OSNotSupported("OS not supported")