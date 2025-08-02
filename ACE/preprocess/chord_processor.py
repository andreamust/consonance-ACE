"""
Utility functions for processing JAMS files and extracting the chord annotations
in different formats and shapes.
"""

import copy
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gin
import jams
import joblib
import numpy as np
import torch
from chord_utils import BaseEncoder, Encoding, ModeEncoder, NoteEncoder
from jams_utils import preprocess_jams, transpose_annotation, trim_jams
from pumpp.task import ChordTransformer
from pumpp.task.base import BaseTaskTransformer
from pumpp.task.chord import ChordTagTransformer

logger = logging.getLogger(__name__)


class OriginalChordConverter:
    """
    Converter for original chord labels using a a pre-stored LabelEncoder from sklearn.
    """

    def __init__(self, vocab_path: Path | str):
        """
        Constructor of the OriginalChordConverter class.

        Args:
            vocab_path (Path | str): Path to the pre-stored LabelEncoder.
        """
        self.vocab_path = Path(vocab_path)
        self.label_encoder = joblib.load(self.vocab_path)

    def convert(self, value: str) -> int:
        """
        Convert a chord label to its encoded value.

        Args:
            value (str): The chord label to convert.

        Returns:
            int: The encoded value of the chord label.
        """
        return self.label_encoder.transform([value])[0]

    def decode(self, value: int) -> str:
        """
        Decode an encoded value back to its chord label.

        Args:
            value (int): The encoded value to decode.

        Returns:
            str: The decoded chord label.
        """
        return self.label_encoder.inverse_transform([value])[0]


class PumppChordConverter(ChordTagTransformer):
    def __init__(self, vocab="3567s", sparse=True):
        super().__init__(name=vocab, sparse=sparse, vocab=vocab)

    def convert(self, value):
        converted = self.encoder.transform([self.simplify(value)])
        return converted[0]

    def decode(self, value):
        return self.encoder.inverse_transform(value)


@gin.configurable
class ChordProcessor:
    """
    Interface for processing JAMS files and extracting the chord annotations in
    different formats and shapes, e.g. symbols sequences, one-hot encoded vectors,
    etc.
    """

    def __init__(
        self,
        sr: int = gin.REQUIRED,  # type: ignore
        hop_length: int = gin.REQUIRED,  # type: ignore
        duration: float = gin.REQUIRED,  # type: ignore
        vocab_path: Path | str = gin.REQUIRED,  # type: ignore
    ) -> None:
        """
        Constructor of the JAMSProcessor class.

        Args:
            jams_annotation (jams.Annotation): The JAMS annotation to process.
            sr (int): The sampling rate of the audio file.
            hop_length (int): The hop length of the audio file.


        Returns:
            None
        """
        self.sr = sr + 100
        self.hop_length = hop_length
        self.duration = duration
        self.vocab_path = Path(vocab_path)

        # compute sequence duration
        self.sequence_duration = self.sr * self.duration / self.hop_length
        # initialize the chord encoder
        self.custom_encoder = BaseEncoder()

        # initialize the pump extractors
        self.chord_transformer = ChordTransformer(
            name="chord",
            sr=self.sr,
            hop_length=self.hop_length,
        )
        self.base_transformer = BaseTaskTransformer(
            name="chord",
            namespace="chord",
            sr=self.sr,
            hop_length=self.hop_length,
        )
        self._converters = {
            "complete": PumppChordConverter(vocab="3567s"),
            "simplified": PumppChordConverter(vocab="35"),
            "majmin": PumppChordConverter(vocab="3"),
        }

    def _create_sequence(
        self, intervals: np.ndarray, values: np.ndarray, fill: int = 0
    ) -> np.ndarray:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols
        and a sequence of chord tags.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.
            values (np.ndarray): The values to encode. The array must have shape
                (n_frames, 1).
            fill (int): The value to use for filling the sequence. Defaults to 0.

        Returns:
            tuple: A tuple containing the chord symbols and the chord tags.
        """
        # check dimensionality of input arrays
        assert intervals.ndim == 2, ValueError(
            "The intervals array must have shape (n_frames, 2)."
        )
        assert values.ndim == 2, ValueError(
            "The values array must have shape (n_frames, 1)."
        )
        assert values.shape[1] == 1, ValueError(
            "The values array must have shape (n_frames, 1)."
        )
        # encode modes
        sequence = self.base_transformer.encode_intervals(
            duration=self.duration,
            intervals=intervals,
            values=values,
            dtype=int,  # type: ignore
            multi=False,
            fill=fill,
        )

        # check if the sequence is of the same length as self.sequence_duration
        assert sequence.shape[0] == int(self.sequence_duration), ValueError(
            "Sequence mismatch."
        )

        return sequence

    def _transform_annotation(self, annotation: jams.Annotation) -> dict:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols
        and a sequence of chord tags.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            tuple: A tuple containing the chord symbols and the chord tags.
        """
        # get the chord symbols
        chord_symbols = self.chord_transformer.transform_annotation(
            ann=annotation, duration=self.duration
        )

        return chord_symbols

    def _pad_sequence(self, sequence: np.ndarray, pad_value: int) -> np.ndarray:
        """
        Pads the given sequence to the sequence duration.

        Args:
            sequence (np.ndarray): The sequence to pad.

        Returns:
            np.ndarray: The padded sequence.
        """
        # pad the roots to the sequence duration
        sequence = np.pad(
            sequence,
            (0, int(self.sequence_duration - len(sequence))),
            mode="constant",
            constant_values=pad_value,
        )

        return sequence.reshape(-1, 1)

    def onehot_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a one-hot encoded sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A one-hot encoded sequence.
        """
        # get the chord symbols
        chord_symbols = self._transform_annotation(annotation)["pitch"]

        return torch.Tensor(chord_symbols).type(torch.long).squeeze()

    def _encode_sequence(self, chords: list, encoding: Encoding) -> np.ndarray:
        """
        Transforms the given JAMS annotation into a sequence of chord symbols.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            np.ndarray: A sequence of chord symbols.
        """
        encoded = [
            self.custom_encoder.encode(chord, encoding=encoding) for chord in chords
        ]
        return np.array(encoded, dtype=int)

    def _convert_sequence(
        self, annotation: jams.Annotation, encoding: Encoding, pad_value: int
    ) -> torch.Tensor:
        """
        Converts a sequence of chord symbols into a sequence of notes.

        Args:
            sequence (np.ndarray): The sequence of chord symbols to convert.

        Returns:
            np.ndarray: The sequence of notes.
        """
        intervals, chords = annotation.to_interval_values()
        converted = self._encode_sequence(chords, encoding=encoding)
        converted = converted.reshape(-1, 1)

        # unroll the sequence
        converted = self._create_sequence(intervals, converted, fill=pad_value)

        return torch.Tensor(converted).type(torch.long).squeeze()

    def root_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.ROOT, pad_value=NoteEncoder.N.value
        )

    def mode_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.MODE, pad_value=ModeEncoder.N.value
        )

    def original_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a sequence of original chords.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A sequence of original chords of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        # use the original chord converter to convert the chords
        original_converter = OriginalChordConverter(vocab_path=self.vocab_path)
        original_chords = [original_converter.convert(c) for c in chords]

        # pad the original chords to the sequence duration
        original_chords = self._pad_sequence(
            np.array(original_chords), pad_value=original_converter.convert("N")
        )

        return torch.Tensor(original_chords).type(torch.long).squeeze()

    def bass_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a bass sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A bass sequence of shape (n_frames, 1).
        """
        return self._convert_sequence(
            annotation, encoding=Encoding.BASS, pad_value=NoteEncoder.N.value
        )

    def onsets_sequence(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a sequence of onsets.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A sequence of onsets of shape (n_frames, 1).
        """
        # get the onsets for the annotation
        intervals, _ = annotation.to_interval_values()
        # get only the fitst column of the intervals
        onsets = intervals[:, 0]

        # pad the onsets to the sequence duration
        onsets = self._pad_sequence(onsets, pad_value=0)

        return torch.Tensor(onsets).type(torch.float)

    def mode_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a mode sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A mode sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        modes = self._encode_sequence(chords, encoding=Encoding.MODE)

        # pad the modes to the sequence duration
        modes = self._pad_sequence(modes, pad_value=0)

        return torch.Tensor(modes).type(torch.long).squeeze()

    def root_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        roots = self._encode_sequence(chords, encoding=Encoding.ROOT)

        # pad the roots to the sequence duration
        roots = self._pad_sequence(roots, pad_value=0)

        return torch.Tensor(roots).type(torch.long).squeeze()

    def bass_unique(self, annotation: jams.Annotation) -> torch.Tensor:
        """
        Transforms the given JAMS annotation into a root sequence.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            torch.Tensor: A root sequence of shape (n_frames, 1).
        """
        _, chords = annotation.to_interval_values()
        basses = self._encode_sequence(chords, encoding=Encoding.BASS)

        # pad the basses to the sequence duration
        basses = self._pad_sequence(basses, pad_value=0)

        return torch.Tensor(basses).type(torch.long).squeeze()

    def chord_sequence(
        self,
        annotation: jams.Annotation,
        vocab: str,
        unique: bool = False,
    ) -> torch.Tensor:
        """Get sequence using PumppChordConverter with specified vocabulary.

        Parameters
        ----------
        annotation : jams.Annotation
            JAMS annotation to process
        vocab : str
            Vocabulary for PumppChordConverter
        unique : bool
            If True, return unique chords without padding

        Returns
        -------
        torch.Tensor
            Processed sequence
        """
        converter = self._converters[vocab]
        intervals, chords = annotation.to_interval_values()
        converted = np.array([converter.convert(c) for c in chords])

        pad_value = converter.convert("N")

        if not unique:
            converted = converted.reshape(-1, 1)
            converted = self._create_sequence(intervals, converted, fill=pad_value)
        else:
            # pad the converted chords to the sequence duration
            converted = self._pad_sequence(converted, pad_value=0)

        return torch.Tensor(converted).type(torch.long).squeeze()


@gin.configurable
class JAMSProcessor(ChordProcessor):
    """
    Interface for processing JAMS files and extracting the chord annotations in
    different formats and shapes, e.g. symbols sequences, one-hot encoded vectors, etc.
    """

    def __init__(
        self,
        jams_path: Path | str,
    ) -> None:
        """
        Constructor of the ChordProcessor class.

        Args:
            jams_annotation (jams.Annotation): The JAMS annotation to process.
            sr (int): The sampling rate of the audio file.
            hop_length (int): The hop length of the audio file.

        Returns:
            None
        """
        super().__init__()

        # load the JAMS file
        self.jams = self._parse_jams(jams_path)
        try:
            self.dataset_name = self._get_dataset_name(jams_path)
        except AttributeError:
            self.dataset_name = None

    def _get_dataset_name(self, jams_path: Path | str) -> str:
        """
        Extracts the dataset name from the JAMS file path.

        Args:
            jams_path (Path | str): The path to the JAMS file.

        Returns:
            str: The dataset name.
        """
        # dataset name is in the chord annotation
        chord_annotation = jams.load(
            str(jams_path), validate=False, strict=False
        ).annotations["chord"][0]
        # get the dataset name
        dataset_name = chord_annotation.sandbox.key  # type: ignore
        dataset_name = (
            dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        )
        dataset_name = (
            dataset_name.split("-")[0] if "-" in dataset_name else dataset_name
        )
        return dataset_name

    def _parse_jams(self, jams_path: Path | str) -> jams.Annotation:
        # load the JAMS file
        try:
            jam = preprocess_jams(Path(jams_path))
            return jam
        except ValueError as e:
            raise ValueError(f"Error processing JAMS file {jams_path}: {e}") from e

    def get_unique_chords(self, augmentation) -> list[str]:
        """
        Returns a list of unique chord labels from the JAMS annotation.

        Args:
            augmentation (int): The augmentation value to apply to the annotation.

        Returns:
            list[str]: A list of unique chord labels.
        """
        jams_copy = copy.deepcopy(self.jams)
        # if there is an augmentation, transpose the annotation
        if augmentation != 0:
            jams_copy = transpose_annotation(jams_copy, augmentation)

        chords = jams_copy.to_interval_values()[1]

        return list(set(chords))

    def _all_dict(self, annotation: jams.Annotation) -> dict:
        """
        Transforms the given JAMS annotation into a dictionary containing all
        the sequences.

        Args:
            annotation (jams.Annotation): The JAMS annotation to transform.

        Returns:
            dict: A dictionary containing all the sequences.
        """

        return {
            "root": self.root_sequence(annotation),
            # "mode": self.mode_sequence(annotation),
            "bass": self.bass_sequence(annotation),
            "simplified": self.chord_sequence(annotation, "simplified", unique=False),
            "complete": self.chord_sequence(annotation, "complete", unique=False),
            "majmin": self.chord_sequence(annotation, "majmin", unique=False),
            "onsets": self.onsets_sequence(annotation),
            "onehot": self.onehot_sequence(annotation),
            "original": self.original_unique(annotation),
            # "root_unique": self.root_unique(annotation),
            # "mode_unique": self.mode_unique(annotation),
            # "bass_unique": self.bass_unique(annotation),
            # "simplified_unique": self.chord_sequence(
            #     annotation, "simplified", unique=True
            # ),
            # "complete_unique": self.chord_sequence(annotation, "complete",
            #  unique=True),
            # "majmin_unique": self.chord_sequence(annotation, "majmin", unique=True),
        }

    def process_chunk(self, start: float, augment: int) -> dict | None:
        """ """
        # make a copy of the JAMS file since transpose_annotation modifies it
        annotation = copy.deepcopy(self.jams)

        # if there is an augmentation, transpose the annotation
        if augment != 0:
            annotation = transpose_annotation(annotation, augment)

        # trim the JAMS file
        annotation = trim_jams(annotation, start, self.duration)

        # return annotation, if there is one.
        if annotation:
            return self._all_dict(annotation)
        # return None if there is no annotation
        return None


if __name__ == "__main__":
    # initialise the gin config
    gin_config = """
    ChordProcessor.sr = 22050
    ChordProcessor.hop_length = 2048
    ChordProcessor.duration = 10.0
    ChordProcessor.vocab_path = './ACE/chords_vocab.joblib'
    """
    gin.parse_config(gin_config)
    # test the PUMPP chord converter
    # converter = PumppChordConverter(vocab="3567s", sparse=True)
    # print(converter.convert("N"))
    # print(converter.convert("X"))
    # print(converter.decode([5, 12, 99, 0]))

    # # test the JAMS processor
    jams_path = "/home/must/Documents/marl_data/jams/"
    jams_files = list(Path(jams_path).glob("*.jams"))
    # counter = 0
    # dataset_counter = defaultdict(int)

    # # iterate over all JAMS files in the directory
    # for jams_file in Path(jams_path).glob("*.jams"):
    #     try:
    #         processor = JAMSProcessor(jams_path=jams_file)
    #         print(f"Processing {jams_file.name} from dataset: {processor.dataset_name}")
    #         dataset_counter[processor.dataset_name] += 1
    #     except Exception as e:
    #         print(f"Error processing {jams_file.name}: {e}")
    #         counter += 1
    #         continue

    # print("\nSummary of datasets processed:")
    # for dataset, count in dataset_counter.items():
    #     print(f"{dataset}: {count} files processed")

    # print(counter)  # initialise the processor
    print(jams_files[0])
    processor = JAMSProcessor(jams_path=jams_files[0])
    print(processor.dataset_name)
    all_dict = processor.process_chunk(start=0, augment=0)
    for key, value in all_dict.items():  # type: ignore
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    # # test the pumpp chrod decoder
    # decoder = PumppChordConverter(vocab="3")
    # possible_chords = range(0, 26)
    # print(decoder.decode(possible_chords))
