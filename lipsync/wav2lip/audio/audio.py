import os

import librosa
import librosa.filters
import numpy as np
from scipy import signal

AudioDType: type = np.float32
Audio: type = np.ndarray[AudioDType]

_num_mels: int = 80  # Number of mel-spectrogram channels and local conditioning dimensionality
_n_fft: int = 800  # Extra window size is filled with 0 paddings to match this parameter
_hop_size: int = 200  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
_win_size: int = 800  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
_sample_rate: float = 16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
_max_abs_value: float = 4.0  # # Scales data symmetrically around 0 and expands output range for faster convergence.

_min_level_db: int = -100
_ref_level_db = 20
_fmin_male = 55  # Set this to 55 if your speaker is male!
_fmin_female = 55  # Set this to 95, if your speaker is female! It should help taking off noise.
_fmax = 7600  # To be increased/reduced depending on data.


def load_wav(path: str | os.PathLike) -> Audio:
	wav, _ = librosa.core.load(
		path,
		dtype=AudioDType,
		sr=_sample_rate
	)
	return Audio(wav)


# noinspection PyPep8Naming
def melspectrogram(wav: Audio, female: bool) -> Audio:
	D = np.abs(_stft(_preemphasis(wav)))
	S = _amp_to_db(_linear_to_mel(D, female)) - _ref_level_db
	return _normalize(S)


def _preemphasis(wav: Audio, coef: float = 0.97) -> Audio:
	return Audio(signal.lfilter([1.0, -coef], [1], wav))


def _stft(y: Audio) -> Audio:
	return Audio(librosa.stft(
		y=y,
		n_fft=_n_fft,
		hop_length=_hop_size,
		win_length=_win_size
	))


# Conversions
_mel_basis_male = librosa.filters.mel(
	_sample_rate,
	_n_fft,
	n_mels=_num_mels,
	fmin=_fmin_male,
	fmax=_fmax
)

_mel_basis_female = librosa.filters.mel(
	_sample_rate,
	_n_fft,
	n_mels=_num_mels,
	fmin=_fmin_female,
	fmax=_fmax
)


def _linear_to_mel(
	spectrogram: Audio,
	female: bool
) -> Audio:
	mel_basis: np.ndarray
	if female:
		mel_basis = _mel_basis_female
	else:
		mel_basis = _mel_basis_male
	return np.dot(mel_basis, spectrogram)


def _amp_to_db(x: Audio) -> Audio:
	min_level = np.exp(_min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))


# noinspection PyPep8Naming
def _normalize(S: Audio) -> Audio:
	return np.clip(
		(2 * _max_abs_value) * ((S - _min_level_db) / -_min_level_db) - _max_abs_value,
		-_max_abs_value,
		_max_abs_value,
	)
