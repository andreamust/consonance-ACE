"""
This module provides basic audio processing pipeline for loading, resampling,
and transforming audio files. The pipeline includes the following steps:
    - Load audio file using librosa
    - Resample to target sample rate if needed
    - Convert stereo to mono
    - Apply custom transformations (e.g., CQT)
    - Normalize audio to [-1, 1] range
    - Pitch shift audio if needed
"""

import logging
from pathlib import Path

import gin
import librosa
import numpy as np
import torch
from torchaudio.transforms import Resample
from transforms import CQTransform

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@gin.configurable
class AudioProcessor:
    """Audio processing class with configurable transformations.

    This class provides a comprehensive pipeline for audio processing, including:
    - Loading audio files
    - Resampling to target sample rate
    - Converting stereo to mono
    - Applying various transformations (CQT, custom transforms)
    - Pitch shifting and normalization

    Attributes
    ----------
    target_sample_rate : int
        Target sample rate in Hz (e.g., 44100, 22050)
    hop_length : int
        Number of samples between successive frames for spectral transforms
    max_sequence_length : float
        Maximum duration of audio sequence in seconds
    device : torch.device
        Device for processing ('cpu' or 'cuda')
    transform : Optional[torch.nn.Module]
        Transform to apply as a PyTorch module
    normalize : bool
        Whether to normalize audio to [-1, 1] range
    """

    def __init__(
        self,
        target_sample_rate: int = gin.REQUIRED,  # type: ignore
        hop_length: int = gin.REQUIRED,  # type: ignore
        max_sequence_length: float = gin.REQUIRED,  # type: ignore
        device: str = "cpu",
        normalize: bool = True,
        transform: torch.nn.Module | None = None,
    ) -> None:
        """Initialize the AudioProcessor with specified configuration.

        Parameters
        ----------
        target_sample_rate : int, optional
            Target sample rate in Hz, by default 44100
        hop_length : int, optional
            Hop length for spectral transforms, by default 512
        max_sequence_length : float, optional
            Maximum audio length in seconds, by default 15.0
        device : str, optional
            Computing device, by default "cpu"
        transform : Optional[Union[str, torch.nn.Module]], optional
            Transform to apply ('cqt' or torch.nn.Module), by default None
        normalize : bool, optional
            Whether to normalize audio to [-1, 1], by default True
        """

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.max_sequence_length = max_sequence_length
        self.device = torch.device(device)
        self.transform = transform
        self.normalize = normalize

        self.num_samples = int(target_sample_rate * max_sequence_length)

    @torch.no_grad()
    def process_audio(
        self, audio_path: Path, onset: float = 0.0, augment: int = 0
    ) -> torch.Tensor:
        """Process an audio file through the complete transformation pipeline.

        Parameters
        ----------
        audio_path : Union[str, Path]
            Path to the audio file
        onset : float, optional
            Start time in seconds for processing, by default 0.0
        augment : int, optional
            Number of semitones to pitch shift, by default 0

        Returns
        -------
        torch.Tensor
            Processed audio tensor. Shape depends on the transforms applied.

        Raises
        ------
        FileNotFoundError
            If the audio file doesn't exist
        RuntimeError
            If audio processing fails for any reason
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            sample_onset = int(onset * self.target_sample_rate)
            signal, sr = self._load_audio(audio_path)
            signal = self._process_signal(signal, sr, sample_onset, augment)
            return signal

        except Exception as e:
            raise RuntimeError(f"Error processing {audio_path}: {str(e)}")

    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int | float]:
        """Load an audio file using librosa.

        Parameters
        ----------
        audio_path : Path
            Path to the audio file

        Returns
        -------
        Tuple[np.ndarray, int]
            Tuple containing:
            - signal: Audio signal as numpy array
            - sr: Sample rate of the loaded audio

        Raises
        ------
        RuntimeError
            If audio loading fails
        """
        try:
            signal, sr = librosa.load(str(audio_path), sr=None, mono=False)
            return signal, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")

    @torch.no_grad()
    def _process_signal(
        self,
        signal: np.ndarray,
        sr: int | float,
        sample_onset: int,
        augment: int,
    ) -> torch.Tensor:
        """Apply the complete processing chain to an audio signal.

        Parameters
        ----------
        signal : Union[torch.Tensor, np.ndarray]
            Input audio signal
        sr : int
            Sample rate of the input signal
        sample_onset : int
            Starting sample for processing
        augment : int
            Number of semitones for pitch shifting

        Returns
        -------
        torch.Tensor
            Processed audio signal
        """
        tensor_signal = torch.from_numpy(signal).to(self.device)
        tensor_signal = self._resample_if_necessary(tensor_signal, sr)
        tensor_signal = self._mix_down_if_necessary(tensor_signal)
        tensor_signal = self._cut_if_necessary(tensor_signal, sample_onset)
        tensor_signal = self._right_pad_if_necessary(tensor_signal)

        if self.normalize:
            tensor_signal = self._normalize(tensor_signal)

        if augment != 0:
            tensor_signal = self._pitch_shift(tensor_signal, augment)

        if self.transform is not None:
            tensor_signal = self.transform(tensor_signal)

        return tensor_signal

    def _resample_if_necessary(
        self, signal: torch.Tensor, sr: int | float
    ) -> torch.Tensor:
        """Resample signal to target sample rate if needed."""
        if sr != self.target_sample_rate:
            resampler = Resample(
                int(sr), self.target_sample_rate, dtype=signal.dtype
            ).to(self.device)
            signal = resampler(signal)
        return signal

    @staticmethod
    def _mix_down_if_necessary(signal: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono by averaging channels if needed."""
        if signal.ndim == 1:
            return signal.unsqueeze(0)
        if signal.shape[0] > 1:
            return torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(
        self, signal: torch.Tensor, sample_onset: int
    ) -> torch.Tensor:
        """Cut signal to specified length starting from onset."""
        if signal.shape[1] > self.num_samples:
            end_idx = min(sample_onset + self.num_samples, signal.shape[1])
            signal = signal[:, sample_onset:end_idx]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """Pad signal with zeros if shorter than required length."""
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            pad_length = self.num_samples - length_signal
            signal = torch.nn.functional.pad(signal, (0, pad_length))
        return signal

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        if torch.abs(signal).max() > 0:
            signal = signal / torch.abs(signal).max()
        return signal

    @torch.no_grad()
    def _pitch_shift(self, signal: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Apply pitch shift transformation."""
        # Move to CPU for librosa processing
        signal_np = signal.cpu().numpy()

        try:
            shifted = librosa.effects.pitch_shift(
                signal_np, sr=self.target_sample_rate, n_steps=n_steps
            )
            return torch.from_numpy(shifted).to(self.device)
        except Exception as e:
            logging.warning(f"Pitch shift failed: {str(e)}. Returning original signal.")
            return signal


@gin.configurable
class AudioChunkProcessor(AudioProcessor):
    """Process multiple variations of a single audio track."""

    def __init__(self, audio_path: Path, **kwargs):
        """
        Initialize with an audio file that stays in memory.

        Parameters
        ----------
        audio_path : Path
            Path to audio file
        **kwargs :
            Arguments passed to parent AudioProcessor
        """
        super().__init__(**kwargs)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and preprocess audio once
        signal, sr = self._load_audio(audio_path)
        self.base_signal = torch.from_numpy(signal).to(self.device)
        self.base_signal = self._resample_if_necessary(self.base_signal, sr)
        self.base_signal = self._mix_down_if_necessary(self.base_signal)

    @torch.no_grad()
    def process_chunk(
        self,
        onset: float = 0.0,
        augment: int = 0,
    ) -> torch.Tensor:
        """
        Get processed chunk from the loaded audio.

        Parameters
        ----------
        onset : float
            Start time in seconds
        augment : int
            Semitones to shift pitch

        Returns
        -------
        torch.Tensor
            Processed audio chunk
        """
        sample_onset = int(onset * self.target_sample_rate)
        signal = self._cut_if_necessary(self.base_signal, sample_onset)
        signal = self._right_pad_if_necessary(signal)

        if augment != 0:
            signal = self._pitch_shift(signal, augment)
        if self.normalize:
            signal = self._normalize(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        return signal


if __name__ == "__main__":
    # initialize the gin configuration
    gin_config = """
                AudioProcessor.target_sample_rate = 22050
                AudioProcessor.hop_length = 512
                AudioProcessor.max_sequence_length = 20.0
                AudioProcessor.device = 'cpu'
                AudioProcessor.normalize = True
                AudioProcessor.transform = None
                """
    # Test the audio processor
    gin.parse_config(gin_config)
    transform = CQTransform(22050, 512)
    processor = AudioProcessor(transform=transform, normalize=True)
    audio_path = Path("/media/data/andrea/choco_audio_marl/audio/isophonics_0.flac")
    audio_tensor = processor.process_audio(audio_path)
    print(audio_tensor.shape)
    # print(audio_tensor[0, 0, :])
