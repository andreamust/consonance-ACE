"""
Scripts for computing audio transforms. Both standard transforms and custom embedding
models are included in this file.
"""

import sys
from pathlib import Path

import gin
import librosa
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

sys.path.append(str(Path(__file__).parents[3] / "control-transfer-diffusion"))


class AudioTransform(torch.nn.Module):
    """Base class for all audio transforms."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the input audio.

        Args:
            x: Input tensor of shape (channels, samples)

        Returns:
            Transformed audio tensor
        """
        raise NotImplementedError


@gin.configurable
class CQTransform(AudioTransform):
    """Constant-Q Transform implementation as a torch.nn.Module."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        bins_per_octave: int = 24,
        num_octaves: int = 6,
        start_note: str = "C1",
    ):
        """Initialize CQT transform.

        Args:
            sample_rate: Audio sample rate in Hz
            hop_length: Number of samples between frames
            bins_per_octave: Number of bins per octave
            num_octaves: Number of octaves to analyze
            start_note: Starting note for the transform
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.num_octaves = num_octaves
        self.start_note = start_note

        # Store parameters for logging/debugging
        self.transform_params = {
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "bins_per_octave": bins_per_octave,
            "num_octaves": num_octaves,
            "start_note": start_note,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CQT transform.

        Args:
            x: Input tensor of shape (channels, samples)

        Returns:
            CQT tensor of shape (bins, time)
        """
        # Move to CPU for librosa
        x_np = x.cpu().numpy()

        # Apply CQT
        cqt = np.abs(
            librosa.cqt(
                x_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                n_bins=self.bins_per_octave * self.num_octaves,
                fmin=librosa.note_to_hz(self.start_note),
            )
        )

        return torch.from_numpy(cqt).to(x.device)

    def get_output_shape(self, sequence_length: int) -> tuple[int, int]:
        """Calculate output shape for a given input length.

        Args:
            sequence_length: Length of input sequence in samples

        Returns:
            Tuple of (num_bins, num_frames)
        """
        n_bins = self.bins_per_octave * self.num_octaves
        n_frames = int(np.ceil(sequence_length / self.hop_length))
        return (n_bins, n_frames)


mel = MelSpectrogram(
    sample_rate=22050,
    n_fft=2048,
    win_length=2048,
    hop_length=2048,
    f_min=0.0,
    f_max=11025.0,
    n_mels=128,
    window_fn=torch.hann_window,
    power=2.0,
    normalized=False,
)

if __name__ == "__main__":
    # Test Transforms
    audio_path = Path("/media/data/andrea/choco_audio_marl/audio/uspop2002_148.flac")
    transform_cqt = CQTransform(
        sample_rate=22050, hop_length=2048, bins_per_octave=24, num_octaves=6
    )

    # Load audio
    audio, _ = librosa.load(audio_path, sr=22050, duration=10)
    audio = torch.from_numpy(audio)
    audio = audio.unsqueeze(0)

    # Apply CQT
    cqt = transform_cqt(audio)
    print(cqt.shape)

    # # Test Mel
    # mel = mel(audio)
    # print(mel.shape)
