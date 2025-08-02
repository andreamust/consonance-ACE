"""Audio preprocessing module for chord recognition.

This module provides functionality for preprocessing audio files and their corresponding
JAMS annotations for chord recognition tasks. It includes classes for audio processing,
dataset management, and caching utilities.

Key components:
- ChocoAudioPreprocessor: Dataset class for managing audio files and annotations
- cache_preprocessor: Utility for caching preprocessed data
"""

import logging
from pathlib import Path

import gin
import joblib
import numpy as np
import torch
from audio_processor import AudioChunkProcessor
from chord_processor import JAMSProcessor
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_chords_vocab(jams_path: Path, output_file: str | Path) -> Path:
    """
    Create a vocabulary of chords from the JAMS files in the specified directory.
    This function scans all JAMS files and collects unique chord labels.
    """
    jams_files = list(Path(jams_path).glob("*.jams"))
    if not jams_files:
        raise ValueError(f"No JAMS files found in the directory: {jams_path}")

    # Initialize a set to collect unique chord labels
    unique_chords = set()
    for jams_file in tqdm(
        jams_files, desc="Collecting unique chords", disable=len(jams_files) == 0
    ):
        jams_processor = JAMSProcessor(str(jams_file))
        for augmentation in np.arange(-5, 7, 1):
            chords = jams_processor.get_unique_chords(augmentation)
            unique_chords.update(chords)

    # Create a LabelEncoder and fit it to the unique chords
    encoder = LabelEncoder()
    encoder.fit(sorted(list(unique_chords)))

    # Save the encoder to a file
    joblib.dump(encoder, Path(output_file))

    # log infor about the created vocabulary
    print(f"Chords vocabulary created with {len(unique_chords)} unique chords.")

    return Path(output_file)


@gin.configurable
class ChoCoProcessor:
    """Process multiple audio files with JAMS annotations in parallel."""

    def __init__(
        self,
        audio_path: str | Path = gin.REQUIRED,  # type: ignore
        jams_path: str | Path = gin.REQUIRED,  # type: ignore
        cache_path: str | Path = gin.REQUIRED,  # type: ignore
        excerpt_per_song: int = gin.REQUIRED,  # type: ignore
        excerpt_distance: int = gin.REQUIRED,  # type: ignore
        augmentation_range: tuple[int, int, int] = gin.REQUIRED,  # type: ignore
        allowed_datasets: tuple[str, ...] = gin.REQUIRED,  # type: ignore
        audio_extensions: tuple[str, ...] = gin.REQUIRED,  # type: ignore
        n_jobs: int = -1,
    ) -> None:
        """Initialize the processor.

        Parameters
        ----------
        audio_path : str | Path
            Directory containing audio files
        jams_path : str | Path
            Directory containing JAMS files
        cache_path : str | Path
            Directory to save processed chunks
        excerpt_per_song : int
            Number of excerpts per song
        excerpt_distance : int
            Time distance between excerpts in seconds
        augmentation_range : tuple[int, int, int]
            Range for pitch shifting (start, stop, step)
        allowed_datasets : tuple[str, ...]
            Names of allowed datasets
        audio_extensions : tuple[str, ...]
            Allowed audio file extensions
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        self.audio_path = Path(audio_path)
        self.jams_path = Path(jams_path)
        self.cache_path = Path(cache_path)

        self.excerpt_per_song = excerpt_per_song
        self.excerpt_distance = excerpt_distance
        self.augmentation = np.arange(*augmentation_range)
        self.allowed_datasets = allowed_datasets
        self.audio_extensions = audio_extensions
        self.n_jobs = n_jobs

        # Clean the cache directory
        self._clean_cache()

        # Get the audio files
        self.audio_files: list = self._get_audio_files()

    def _get_audio_files(self) -> list:
        """Get all audio files in the audio directory.

        Returns
        -------
        list
            List of audio files
        """
        return [
            file
            for file in self.audio_path.iterdir()
            if file.suffix in self.audio_extensions
            # and file.stem.split("_")[0] in self.allowed_datasets
        ]

    def _clean_cache(self) -> None:
        """
        Utility function that:
            - Checks if cache directory exists; if not, creates it
            - Deletes all files in the cache directory
        """
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # if the cache directory is not empty, log a info message
        if any(self.cache_path.iterdir()):
            logger.info(f"Cleaning cache directory {self.cache_path}")

        for file in self.cache_path.iterdir():
            file.unlink()

    def _process_song(self, audio_file: Path) -> None:
        """Process all chunks of a single song.

        Parameters
        ----------
        audio_file : Path
            Path to audio file
        """
        file_name = audio_file.stem
        jams_file = self.jams_path / f"{file_name}.jams"

        # Create processors inside the function
        audio_processor = AudioChunkProcessor(audio_file)
        try:
            jams_processor = JAMSProcessor(jams_file)
            dataset_name = jams_processor.dataset_name
        except Exception as e:
            logger.error(f"Error processing JAMS file {jams_file}: {e}")
            return

        chunks = [
            (i * self.excerpt_distance, augment)
            for i in range(self.excerpt_per_song)
            for augment in self.augmentation
        ]

        dataset_name = f"{dataset_name}_" if dataset_name else ""

        for onset, augment in chunks:
            chunk_name = f"{dataset_name}{file_name}_t{onset:04d}_p{augment:+d}"
            jams_chunk = jams_processor.process_chunk(onset, augment)

            while jams_chunk:
                audio_chunk = audio_processor.process_chunk(onset, augment)

                if audio_chunk is not None:
                    logger.debug(f"Processing {chunk_name}")

                    torch.save(
                        (
                            audio_chunk,
                            jams_chunk,
                        ),
                        self.cache_path / f"{chunk_name}.pt",
                    )
                    break

    def process(self) -> None:
        """Process all songs in parallel."""
        # Process songs in parallel
        Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=0)(
            delayed(self._process_song)(audio_file)
            for audio_file in tqdm(self.audio_files)
        )


def main(config_file_name: str = "dataset.gin") -> None:
    """
    Main function for preprocessing audio files and JAMS annotations. It reads the
    configuration from the gin file and processes the audio files in parallel.

    Parameters
    ----------
    config_file_name : str
        Name of the gin configuration file
    """
    # Get current file's directory and find gin config
    config_path = Path(__file__).parent / config_file_name
    gin.parse_config_file(config_path.__str__())

    # Encode the original chord vocabulary
    create_chords_vocab(
        gin.query_parameter("ChoCoProcessor.jams_path"),
        gin.query_parameter("ChordProcessor.vocab_path"),
    )

    processor = ChoCoProcessor()
    processor.process()

    # Save a copy of the config file in the cache directory
    cache_path = processor.cache_path / "config.gin"
    if not cache_path.exists():
        cache_path.write_text(gin.operative_config_str())
        logger.info(f"Saved config to {cache_path}")


if __name__ == "__main__":
    # logger.setLevel(logging.DEBUG)
    main()
