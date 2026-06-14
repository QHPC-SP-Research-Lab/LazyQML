# ----------------------------------------------------
# Example: MelSpectrogram + CNN encoder + FastQSVM
# Leak-free hold-out and repeated CV
# ----------------------------------------------------
import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from lazyqml.Factories import ModelFactory, PreprocessingFactory
from lazyqml.Preprocessing import MelSpectrogram
from lazyqml.Global.globalEnums import Embedding, Model
from lazyqml.Utils import (
    calculate_free_memory,
    calculate_quantum_memory_Fast,
    get_embedding_expressivity,
    set_simulation_type,
)

np.random.seed(1234)
torch.manual_seed(1234)


def collect_wavs_by_class(data_root: Path):
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


class CNNEncoderClassifier(nn.Module):
    def __init__(self, input_shape, latent_dim, n_classes, channels=(8, 16)):
        super().__init__()

        in_channels = input_shape[0]
        c1, c2 = channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding  = nn.Linear(c2, latent_dim)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        z = self.features(x)
        z = self.embedding(z)
        return self.classifier(z)

    def encode(self, x):
        z = self.features(x)
        z = self.embedding(z)
        return z


def ensure_nchw(X):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 3:
        X = X[:, None, :, :]
    elif X.ndim != 4:
        raise ValueError(f"Expected X with 3 or 4 dims, got {X.shape}.")

    return X


def train_cnn_encoder(model, X_train, y_train, *, device, epochs=10, batch_size=16, lr=1e-3, log_prefix=""):
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(loader))
        if log_prefix is not None:
            print(f"{log_prefix}epoch {epoch + 1:02d}/{epochs} cnn_loss={epoch_loss:.4f}", flush=True)


@torch.no_grad()
def extract_embeddings(model, X, *, device, batch_size=32):
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    outputs = []

    for start in range(0, len(X_tensor), batch_size):
        batch = X_tensor[start:start + batch_size].to(device)
        outputs.append(model.encode(batch).cpu().numpy())

    return np.concatenate(outputs, axis=0).astype(np.float32)


def embedding_list_statevector():
    return [Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.ZZ, Embedding.ZZ_LOCAL, Embedding.AMP, Embedding.DENSE_ANGLE, Embedding.HIGHER_ORDER]


def embedding_list_tensor():
    return [Embedding.RX, Embedding.RY, Embedding.RZ, Embedding.ZZ, Embedding.ZZ_LOCAL]


def build_model(*, model_kind, nqubits, embedding, mem_budget_mb):
    return ModelFactory().getModel(model=model_kind, nqubits=nqubits, embedding=embedding, ansatz=None, n_class=2, mem_budget_mb=mem_budget_mb)


def evaluate_split(*, X, y, train_idx, test_idx, nqubits, n_classes, epochs, batch_size, torch_device, split_label,
                   embeddings, verbose_cnn, simulation_type, model_kind, latent_dim):

    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    cnn = CNNEncoderClassifier(input_shape=X.shape[1:], latent_dim=latent_dim, n_classes=n_classes).to(torch_device)

    train_cnn_encoder(cnn, X_train, y_train, device=torch_device, epochs=epochs, batch_size=batch_size, lr=1e-3,
                      log_prefix=(f"[{split_label}] " if verbose_cnn else None))

    raw_train = extract_embeddings(cnn, X_train, device=torch_device, batch_size=batch_size)
    raw_test  = extract_embeddings(cnn, X_test, device=torch_device, batch_size=batch_size)

    if model_kind == Model.FastQSVM:
        free_ram_mb   = calculate_free_memory()
        mem_budget_mb = calculate_quantum_memory_Fast(nqubits=nqubits, n=len(train_idx) + len(test_idx), mode="hold-out", folds=1,
                        test_size=len(test_idx) / float(len(train_idx) + len(test_idx)), free_ram_mb=free_ram_mb)
    else:
        mem_budget_mb = None

    prep_factory = PreprocessingFactory(nqubits)
    rows = []
    set_simulation_type(simulation_type)

    for embedding in embeddings:
        preproc = prep_factory.GetPreprocessing(embedding=embedding, ansatz=None)
        Z_train = np.asarray(preproc.fit_transform(raw_train, y_train), dtype=np.float32)
        Z_test  = np.asarray(preproc.transform(raw_test), dtype=np.float32)

        start = time.perf_counter()
        model = build_model(model_kind=model_kind, nqubits=nqubits, embedding=embedding, mem_budget_mb=mem_budget_mb)
        model.fit(Z_train, y_train)
        preds   = model.predict(Z_test)
        elapsed = time.perf_counter() - start

        rows.append({"simulation": simulation_type, "model": model_kind.name, "split": split_label, "embedding": embedding.name, "latent_dim": latent_dim,
                     "feature_dim": int(Z_train.shape[1]), "mem_budget_mb": (float(mem_budget_mb) if mem_budget_mb is not None else np.nan), "time_taken": elapsed,
                     "accuracy": accuracy_score(y_test, preds), "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                     "f1_weighted": f1_score(y_test, preds, average="weighted")})
    return pd.DataFrame(rows)


def _run_cv_fold(*, X, y, train_idx, test_idx, nqubits, n_classes, epochs, batch_size, split_label, embeddings, verbose_cnn, simulation_type, model_kind, latent_dim):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    return evaluate_split(X=X, y=y, train_idx=train_idx, test_idx=test_idx, nqubits=nqubits, n_classes=n_classes, epochs=epochs, batch_size=batch_size,
                          torch_device=torch_device, split_label=split_label, embeddings=embeddings, verbose_cnn=verbose_cnn, simulation_type=simulation_type,
                          model_kind=model_kind, latent_dim=latent_dim)


def run_pipeline(*, X, y, nqubits, n_classes, epochs, batch_size, test_size, n_splits, n_repeats, n_jobs, verbose_cnn, simulation_type, model_kind, 
                 embeddings, latent_dim, show_cv_folds):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    set_simulation_type(simulation_type)

    # train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size, random_state=1234, stratify=y)

    # holdout_df = evaluate_split(X=X, y=y, train_idx=np.asarray(train_idx), test_idx=np.asarray(test_idx), nqubits=nqubits, n_classes=n_classes, epochs=epochs,
    #                             batch_size=batch_size, torch_device=torch_device, split_label="holdout", embeddings=embeddings, verbose_cnn=verbose_cnn,
    #                             simulation_type=simulation_type, model_kind=model_kind, latent_dim=latent_dim)

    # print(f"\nHold-out - {simulation_type} / {model_kind.name}", flush=True)
    # print(holdout_df.to_string(index=False), flush=True)

    cv       = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
    cv_tasks = [{"X": X, "y": y, "train_idx": cv_train_idx, "test_idx": cv_test_idx, "nqubits": nqubits, "n_classes": n_classes, "epochs": epochs, "batch_size": batch_size,
                 "split_label": f"cv_{fold_idx}", "embeddings": embeddings, "verbose_cnn": verbose_cnn, "simulation_type": simulation_type, "model_kind": model_kind, "latent_dim": latent_dim,}
                for fold_idx, (cv_train_idx, cv_test_idx) in enumerate(cv.split(X, y), start=1)]

    print(f"\nRepeated CV ({n_splits}, {n_repeats}) - summary - {simulation_type} / {model_kind.name} {nqubits} qubits", flush=True)
    if n_jobs == 1:
        cv_frames = [_run_cv_fold(**task) for task in cv_tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as executor:
            futures   = [executor.submit(_run_cv_fold, **task) for task in cv_tasks]
            cv_frames = [future.result() for future in futures]

    cv_df      = pd.concat(cv_frames, ignore_index=True)
    cv_summary = (cv_df.groupby(["simulation", "model", "embedding"], as_index=False).agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), 
                                 balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
                                 f1_weighted_mean=("f1_weighted", "mean"), f1_weighted_std=("f1_weighted", "std"), time_mean=("time_taken", "mean"),
                                 feature_dim=("feature_dim", "first")).sort_values("accuracy_mean", ascending=False).reset_index(drop=True))

    if show_cv_folds:
        print(f"\nRepeated CV - per fold - {simulation_type} / {model_kind.name}", flush=True)
        print(cv_df.to_string(index=False), flush=True)
    print(cv_summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Leak-free example of: MelSpectrogram + CNN encoder + FastQSVM with repeated CV.")
    ap.add_argument("--data_root",     type=str,   required=True,                     help="Folder with subdirectories per class (each containing .wav files).")
    ap.add_argument("--nqubits",       type=int,   default=8,                         help="Number of qubits for FastQSVM.")
    ap.add_argument("--epochs",        type=int,   default=10,                        help="CNN epochs per split/fold.")
    ap.add_argument("--batch_size",    type=int,   default=16,                        help="CNN batch size.")
    ap.add_argument("--test_size",     type=float, default=0.3,                       help="Hold-out test split proportion.")
    ap.add_argument("--n_splits",      type=int,   default=5,                         help="Number of CV folds.")
    ap.add_argument("--n_repeats",     type=int,   default=10,                        help="Number of repeated CV rounds.")
    ap.add_argument("--verbose_cnn",   action="store_true",                           help="Print per-epoch CNN training logs.")
    ap.add_argument("--show_cv_folds", action="store_true",                           help="Print the detailed per-fold CV table.")
    ap.add_argument("--n_jobs",        type=int, default=max(1, os.cpu_count() // 2), help="Number of CV folds executed in parallel.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    classes   = collect_wavs_by_class(data_root)
    n_classes = len(classes)
    items     = flatten_labeled_wavs(classes)
    wav_files = [str(path) for path, _, _ in items]
    y         = np.array([label for _, label, _ in items], dtype=int)

    extractor = MelSpectrogram(sr=4000, duration=2.0, n_mels=64, n_fft=256, hop_length=128)
    X         = ensure_nchw(extractor.fit_transform(wav_files))

    all_embeddings = embedding_list_statevector()
    latent_dim     = max(get_embedding_expressivity(args.nqubits, emb) for emb in all_embeddings)

    run_pipeline(X=X, y=y, nqubits=args.nqubits, n_classes=n_classes, epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size,
                 n_splits=args.n_splits, n_repeats=args.n_repeats, n_jobs=args.n_jobs, verbose_cnn=args.verbose_cnn, simulation_type="statevector",
                 model_kind=Model.FastQSVM, embeddings=embedding_list_statevector(), latent_dim=latent_dim, show_cv_folds=args.show_cv_folds)

    run_pipeline(X=X, y=y, nqubits=args.nqubits, n_classes=n_classes, epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size,
                 n_splits=args.n_splits, n_repeats=args.n_repeats, n_jobs=args.n_jobs, verbose_cnn=args.verbose_cnn, simulation_type="tensor",
                 model_kind=Model.MPSQSVM, embeddings=embedding_list_tensor(), latent_dim=latent_dim, show_cv_folds=args.show_cv_folds)

    run_pipeline(X=X, y=y, nqubits=latent_dim // 2, n_classes=n_classes, epochs=args.epochs, batch_size=args.batch_size, test_size=args.test_size,
                 n_splits=args.n_splits, n_repeats=args.n_repeats, n_jobs=args.n_jobs, verbose_cnn=args.verbose_cnn, simulation_type="tensor",
                 model_kind=Model.MPSQSVM, embeddings=embedding_list_tensor(), latent_dim=latent_dim, show_cv_folds=args.show_cv_folds)

if __name__ == "__main__":
    main()
