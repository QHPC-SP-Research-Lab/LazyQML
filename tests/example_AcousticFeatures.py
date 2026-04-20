# ----------------------------------------------------
# Example: acoustic features + QNN with lazyqml 
# ----------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import torch

from lazyqml.Preprocessing import AcousticFeatures, PCAHelper
from lazyqml.Utils  import set_simulation_type
from lazyqml        import QuantumClassifier
from lazyqml.Global import Ansatzs, Embedding, Model

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
        raise NotADirectoryError(f"he path is not a directory: {data_root}")

    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            wavs = sorted(subdir.glob("*.wav"))
            if wavs:
                classes[subdir.name] = wavs

    if not classes:
        raise ValueError(f"NNo subdirectories with .wav files were found in {data_root}")
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


def main():
    ap = argparse.ArgumentParser(description="Example of: acoustic_features + QNN with lazyqml.")
    ap.add_argument("--data_root", type=str, required=True, help="Folder with subdirectories per class (each containing .wav files).")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    classes   = collect_wavs_by_class(data_root)
    items     = flatten_labeled_wavs(classes)
    wav_files = [str(path) for path, _, _ in items]
    y         = np.array([label for _, label, _ in items], dtype=int)

    # --------------------------------------------------
    # Config parameters
    # --------------------------------------------------
    verbose    = False
    sequential = False
    embeddings = {Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.AMP, Embedding.DENSE_ANGLE, Embedding.HIGHER_ORDER}
    nFeatures  = 8
    nqubits    = {nFeatures}
    random_st  = 0

    # --------------------------------------------------
    # 1. Extract acoustic features
    # --------------------------------------------------
    extrc = AcousticFeatures(sr=4000, duration=2.0, n_mfcc=20, n_mels=32)
    X     = extrc.fit_transform(wav_files)
    pca   = PCAHelper(nqubits=nFeatures, ncomponents=nFeatures)
    X_red = pca.fit_transform(X, y)

    # --------------------------------------------------
    # 2. Statevector
    # --------------------------------------------------
    sim_type="statevector"
    set_simulation_type(sim_type)
    models = {Model.FastQSVM}
    model  = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, classifiers=models, verbose=verbose, sequential=sequential, randomstate=random_st)

    print(f"{sim_type} train/test con {models}")
    scores = model.fit(X_red, y, test_size=0.3, showTable=True)

    print(f"\n{sim_type} con CV (10, 5) con {models}")
    scores = model.repeated_cross_validation(X_red, y, n_splits=5, n_repeats=10, showTable=True)


    sim_type   = "tensor"
    embeddings = {Embedding.RX}
    models     = {Model.MPSQSVM}

    set_simulation_type(sim_type)
    model  = QuantumClassifier(nqubits=nqubits, embeddings=embeddings, classifiers=models, verbose=verbose, sequential=sequential, randomstate=random_st)
    print(f"\n\n{sim_type} train/test con {models}")
    scores = model.fit(X_red, y, test_size=0.3, showTable=True)

    print(f"\n{sim_type} con CV (10, 5) con {models}")
    scores = model.repeated_cross_validation(X_red, y, n_splits=5, n_repeats=10, showTable=True)

if __name__ == "__main__":
    main()
