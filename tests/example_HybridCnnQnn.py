# ----------------------------------------------------
# Example: acoustic features + HybridCNNQNN
# ----------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pennylane as qml
import torch

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from lazyqml.Preprocessing import MelSpectrogram
from lazyqml.Models import HybridCNNQNN
from lazyqml.Global.globalEnums import Ansatzs, Embedding, Backend
from lazyqml.Utils import set_simulation_type

np.random.seed(1234)
torch.manual_seed(1234)

# --------------------------------------------------
# I/O
# --------------------------------------------------
def collect_wavs_by_class(data_root: Path):
    # Input: data_root/ -> clase_0/ -> a.wav 
    # Return clase_x: [Path(...), Path(...)],

    classes = {}

    if not data_root.exists():
        raise FileNotFoundError(f"The folder does not exist: {data_root}")

    if not data_root.is_dir():
        raise NotADirectoryError(f"The path is not a directory: {data_root}")

    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            wavs = sorted(subdir.glob("*.wav"))
            if wavs:
                classes[subdir.name] = wavs

    if not classes:
        raise ValueError(f"No subdirectories with .wav files were found in {data_root}")
    return classes


def flatten_labeled_wavs(classes_dict):
    items = []
    class_names = sorted(classes_dict.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        label = class_to_idx[class_name]
        for wav_path in classes_dict[class_name]:
            items.append((wav_path, label, class_name))
    return items


def pick_statevector_backend(torch_device: str, nqubits: int):
    if torch_device == "cuda":
        try:
            qml.device(Backend.lightningGPU.value, wires=nqubits)
            return Backend.lightningGPU
        except Exception:
            pass
    return Backend.lightningQubit


def main():
    ap = argparse.ArgumentParser(description="Example of: acoustic_features + QNN with lazyqml.")
    ap.add_argument("--data_root", type=str, required=True, help="Folder with subdirectories per class (each containing .wav files).")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    classes   = collect_wavs_by_class(data_root)
    n_classes = len(classes)
    items     = flatten_labeled_wavs(classes)
    wav_files = [str(path) for path, _, _ in items]
    y         = np.array([label for _, label, _ in items], dtype=int)

    # --------------------------------------------------
    # 1. Extract acoustic features
    # --------------------------------------------------
    ext = MelSpectrogram(sr=8000, duration=2.0, n_mels=64, n_fft=256, hop_length=128)
    X   = ext.fit_transform(wav_files)

    nqubits = 8
    set_simulation_type("statevector")
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = pick_statevector_backend(torch_device, nqubits)

    # --------------------------------------------------
    # 2. Train/test HybridCNNQNN
    # --------------------------------------------------
    model = HybridCNNQNN(input_shape=X.shape[1:], nqubits=nqubits, ansatz=Ansatzs.HARDWARE_EFFICIENT, embedding=Embedding.RY, n_class=n_classes,
        layers=2, epochs=10, shots=0, lr=0.01, batch_size=4, torch_device=torch_device, backend=backend)
    model.fit(X, y)

    # --------------------------------------------------
    # 3. Predict
    # --------------------------------------------------
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))

    accuracy   = accuracy_score(y, preds)
    b_accuracy = balanced_accuracy_score(y, preds)
    f1         = f1_score(y, preds, average="weighted")

    print(f"{accuracy:.3f}, {b_accuracy:.3f}, {f1:.3f}", flush=True)

    # --------------------------------------------------
    # 4. Repeated cross validation
    # --------------------------------------------------
    scores = model.repeated_cross_validation(X, y, n_splits=5, n_repeats=2, showTable=True)
    print(scores["summary"].to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
