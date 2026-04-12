from typing import Iterable, List, Sequence, Union

import numpy as np 

try:
    import librosa
except ImportError as exc:
    raise ImportError("MelSpectrogram requires librosa."
    ) from exc

from lazyqml.Interfaces.iPreprocessing import Preprocessing


ArrayLike1D = Union[np.ndarray, Sequence[float]]
AudioInput  = Union[str, ArrayLike1D]


class MelSpectrogram(Preprocessing):
    """
    Convert WAV files or raw 1D audio signals into fixed-size log-mel spectrograms.
    Output format:
        (n_samples, 1, n_mels, n_frames)
    Usage:
        X_spec = MelSpectrogram(...).transform(wav_paths)
        hybrid_model.fit(X_spec, y)
    """

    def __init__(self, sr: int = 8000, duration: float | None = 2.0, mono: bool = True, n_mels: int = 64, n_fft: int = 256, hop_length: int = 128, power: float = 2.0,
        normalize_waveform: bool = True, normalize_spectrogram: bool = True, eps: float = 1e-10):
        self.sr = sr
        self.duration = duration
        self.mono = mono
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.normalize_waveform = normalize_waveform
        self.normalize_spectrogram = normalize_spectrogram
        self.eps = eps

        self._fitted = False

    def fit(self, X, y=None):
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: Union[AudioInput, Iterable[AudioInput]]) -> np.ndarray:
        """
        Transform WAV paths or raw waveforms into log-mel spectrogram tensors.
        Parameters
        ----------
        X : str | 1D array | iterable of them
        Returns
        -------
        np.ndarray
            Shape: (n_samples, 1, n_mels, n_frames)
        """
        if isinstance(X, (str, np.ndarray)) or not hasattr(X, "__iter__"):
            X = [X]

        specs = [self._extract_one(x) for x in X]
        return np.stack(specs, axis=0).astype(np.float32)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """
        Shape of one sample: (1, n_mels, n_frames)
        """
        return (1, self.n_mels, self._expected_n_frames())

    def _extract_one(self, x: AudioInput) -> np.ndarray:
        y = self._load_audio(x)
        y = self._prepare_audio(y)

        mel     = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        log_mel = librosa.power_to_db(mel + self.eps, ref=np.max)

        if self.normalize_spectrogram:
            mean = np.mean(log_mel)
            std = np.std(log_mel)
            if std > self.eps:
                log_mel = (log_mel - mean) / std
            else:
                log_mel = log_mel - mean

        # Add channel dimension -> (1, n_mels, n_frames)
        return log_mel[np.newaxis, :, :]

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

    def _expected_n_frames(self) -> int:
        if self.duration is None:
            raise ValueError("output_shape requires a fixed duration.")

        target_len = int(round(self.sr * self.duration))
        mel = librosa.feature.melspectrogram(y=np.zeros(target_len, dtype=np.float32), sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        return int(mel.shape[1])