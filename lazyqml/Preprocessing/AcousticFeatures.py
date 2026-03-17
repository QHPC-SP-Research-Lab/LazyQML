import warnings
from typing import Iterable, List, Sequence, Union

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise ImportError("AcousticFeatures necesita librosa.") from exc

from lazyqml.Interfaces.iPreprocessing import Preprocessing

ArrayLike1D = Union[np.ndarray, Sequence[float]]
AudioInput  = Union[str, ArrayLike1D]

class AcousticFeatures(Preprocessing):
    """
    Extract fixed-size acoustic descriptors from WAV files or raw 1D audio signals.
    Main output blocks:
      - MFCC mean/std
      - log-mel mean/std
      - zero-crossing rate mean/std
      - RMS energy mean/std
      - spectral centroid mean/std
      - spectral bandwidth mean/std
      - spectral rolloff mean/std
      - spectral flatness mean/std
      - chroma mean/std (optional)
    Usage:
      X_features = AcousticFeatures(...).transform(wav_paths)
      qnn.fit(X_features, y)
    """
    def __init__(self, sr: int = 8000, duration: float | None = 2.0, mono: bool = True, n_mfcc: int = 20, n_mels: int = 32, n_fft: int = 256, hop_length: int = 128,
        include_chroma: bool = False, normalize_waveform: bool = True, eps: float = 1e-10):
        self.sr = sr
        self.duration = duration
        self.mono = mono
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_chroma = include_chroma
        self.normalize_waveform = normalize_waveform
        self.eps = eps

        self._fitted = False
        self._feature_names = self._build_feature_names()


    def fit(self, X, y=None):
        self._fitted = True
        return self


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


    def transform(self, X: Union[AudioInput, Iterable[AudioInput]]) -> np.ndarray:
        """
        Transform a list of WAV paths or raw waveforms into a 2D feature matrix.
        Parameters
        ----------
        X : str | 1D array | iterable of them
            Each element can be:
              - path to a WAV/audio file
              - raw 1D waveform
        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        if isinstance(X, (str, np.ndarray)) or not hasattr(X, "__iter__"):
            X = [X]

        features = [self._extract_one(x) for x in X]
        return np.asarray(features, dtype=np.float32)


    @property
    def n_features(self) -> int:
        return len(self._feature_names)


    def get_feature_names(self) -> List[str]:
        return list(self._feature_names)


    def _extract_one(self, x: AudioInput) -> np.ndarray:
        y = self._load_audio(x)
        y = self._prepare_audio(y)

        # Core time-frequency representations
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)

        mel     = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, power=2.0)
        log_mel = librosa.power_to_db(mel + self.eps, ref=np.max)

        # Scalar/frame-level descriptors
        zcr       = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        rms       = librosa.feature.rms               (y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        centroid  = librosa.feature.spectral_centroid (y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        rolloff   = librosa.feature.spectral_rolloff  (y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        flatness  = librosa.feature.spectral_flatness (y=y, n_fft=self.n_fft, hop_length=self.hop_length)

        blocks = [self._summary_stats(mfcc), self._summary_stats(log_mel), self._summary_stats(zcr), self._summary_stats(rms),
            self._summary_stats(centroid), self._summary_stats(bandwidth), self._summary_stats(rolloff), self._summary_stats(flatness)]

        if self.include_chroma:
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            blocks.append(self._summary_stats(chroma))
        return np.concatenate(blocks, axis=0)


    def _load_audio(self, x: AudioInput) -> np.ndarray:
        if isinstance(x, str):
            y, _ = librosa.load(x, sr=self.sr, mono=self.mono)
            return y.astype(np.float32)

        y = np.asarray(x, dtype=np.float32).squeeze()
        if y.ndim != 1:
            raise ValueError("Raw audio input must be a 1D waveform.")
        return y


    def _prepare_audio(self, y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            raise ValueError("Empty audio input.")

        # Fix duration to obtain stable feature dimensions/statistics
        if self.duration is not None:
            target_len = int(round(self.sr * self.duration))

            if y.shape[0] < target_len:
                y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
            elif y.shape[0] > target_len:
                y = y[:target_len]

        if self.normalize_waveform:
            peak = np.max(np.abs(y))
            if peak > self.eps:
                y = y / peak
        return y.astype(np.float32)


    @staticmethod
    def _summary_stats(M: np.ndarray) -> np.ndarray:
        """
        Summarize a framewise descriptor matrix M of shape (n_features, n_frames)
        using mean and std over time.
        """
        if M.ndim == 1:
            M = M[None, :]

        mean = np.mean(M, axis=1)
        std = np.std(M, axis=1)
        return np.concatenate([mean, std], axis=0)


    def _build_feature_names(self) -> List[str]:
        names: List[str] = []

        names += [f"mfcc_{i}_mean" for i in range(self.n_mfcc)]
        names += [f"mfcc_{i}_std"  for i in range(self.n_mfcc)]

        names += [f"logmel_{i}_mean" for i in range(self.n_mels)]
        names += [f"logmel_{i}_std"  for i in range(self.n_mels)]

        scalar_blocks = ["zcr", "rms", "centroid", "bandwidth", "rolloff", "flatness"]
        for name in scalar_blocks:
            names += [f"{name}_mean", f"{name}_std"]

        if self.include_chroma:
            names += [f"chroma_{i}_mean" for i in range(12)]
            names += [f"chroma_{i}_std" for i in range(12)]
        return names