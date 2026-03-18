# ---------------------------------------------------------
# Tests for lazyqml package using acoustic features + QNN
# ---------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np

from lazyqml.Preprocessing import AcousticFeatures, PCAHelper
from lazyqml.Models.QNN import QNN
from lazyqml.Global.globalEnums import Ansatzs, Embedding

# --------------------------------------------------
# I/O
# --------------------------------------------------
def collect_wavs_by_class(data_root: Path):
    # Espera una estructura: data_root/ -> clase_0/ -> a.wav
    # Retorna clase_x: [Path(...), Path(...)],

    classes = {}

    if not data_root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {data_root}")

    if not data_root.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {data_root}")

    for subdir in sorted(data_root.iterdir()):
        if subdir.is_dir():
            wavs = sorted(subdir.glob("*.wav"))
            if wavs:
                classes[subdir.name] = wavs

    if not classes:
        raise ValueError(f"No se encontraron subcarpetas con archivos .wav en {data_root}")
    return classes


def flatten_labeled_wavs(classes_dict):
    # Convierte el diccionario por clase en una lista de tuplas: [(wav_path, label_int, class_name), ...]

    items = []
    class_names = sorted(classes_dict.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        label = class_to_idx[class_name]
        for wav_path in classes_dict[class_name]:
            items.append((wav_path, label, class_name))
    return items


def main():
    ap = argparse.ArgumentParser(description="Ejemplo uso acustic_features + QNN con lazyqml.")
    ap.add_argument("--data_root", type=str, required=True, help="Carpeta con subcarpetas por clase (cada una con .wav).")

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
    extractor = AcousticFeatures(sr=4000, duration=2.0, n_mfcc=20, n_mels=32)
    X         = extractor.fit_transform(wav_files)

    nqubits = 8
    pca = PCAHelper(nqubits=nqubits, ncomponents=nqubits)
    X_red = pca.fit_transform(X, y)

    # --------------------------------------------------
    # 2. Train QNN
    # --------------------------------------------------
    model = QNN(nqubits=nqubits, ansatz=Ansatzs.HARDWARE_EFFICIENT, embedding=Embedding.RY, n_class=n_classes, layers=2, epochs=10, shots=0, lr=0.01, batch_size=4, backend="lightning.qubit")
    model.fit(X_red, y)

    # --------------------------------------------------
    # 3. Predict
    # --------------------------------------------------
    y_pred = model.predict(X_red)

    print("Predictions:", y_pred)

if __name__ == "__main__":
    main()