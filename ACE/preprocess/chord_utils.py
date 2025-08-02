"""
Main script for encoding chord as numerical values in different formats.
The implementations available are:
    - root: encodes the root of the chord as a number from 1 to 12, plus 13 for "N"
    - mode: encodes the mode of the chord as a number from 1 to 7, plus 8 for "N"
    - bass: encodes the bass of the chord as a number from 1 to 12, plus 13 for "N"
    - majmin: encodes the chord as a number from 1 to 25, plus 26 for "N"
    - simplified: encodes the chord as a product of the root and the mode, as a
      number from 1 to 84, plus 85 for "N"
"""

from enum import CONTINUOUS, Enum, verify

from harte.harte import Harte
from music21 import pitch


class Encoding(Enum):
    """
    Enum for the different encodings available.
    """

    ROOT = 1
    BASS = 2
    MODE = 3


@verify(CONTINUOUS)
class ModeEncoder(Enum):
    """
    Enum for the different encodings available.
    """

    MAJOR = 1
    MINOR = 2
    MAJOR_SEVENTH = 3
    MINOR_SEVENTH = 4
    AUGMENTED = 5
    AUGMENTED_SEVENTH = 5
    DIMINISHED = 6
    DIMINISHED_SEVENTH = 6
    OTHER = 7
    OTHER_SEVENTH = 7
    N = 8


@verify(CONTINUOUS)
class NoteEncoder(Enum):
    """
    Enum for the different encodings available.
    """

    C = 1
    C_SHARP = 2
    D = 3
    D_SHARP = 4
    E = 5
    F = 6
    F_SHARP = 7
    G = 8
    G_SHARP = 9
    A = 10
    A_SHARP = 11
    B = 12
    N = 13


class BaseEncoder:
    """
    Class for encoding chords as numerical values.
    """

    def __init__(self):
        """
        Initializes the encoder with the given encoding.

        Args:
            encoding (Encoding): The encoding to use.
        """

    def encode(self, chord: str, encoding: Encoding = Encoding.ROOT) -> int:
        """
        Encodes the given chord as a numerical value.

        Args:
            chord (str): The chord to encode.

        Returns:
            int: The numerical value of the chord.
        """
        if encoding == Encoding.ROOT:
            return self._encode_root(chord)
        elif encoding == Encoding.BASS:
            return self._encode_bass(chord)
        elif encoding == Encoding.MODE:
            return self._encode_mode(chord)
        else:
            raise ValueError("Invalid encoding.")

    def _open_harte(self, chord: str) -> Harte:
        """
        Creates a Harte chord object for the given chord.

        Args:
            chord (str): The chord to create a Harte object for.

        Returns:
            Harte: The Harte object for the given chord.
        """
        if "/bb1" in chord:
            chord = chord.replace("/bb1", "/b7")

        return Harte(chord)

    def _encode_note(self, pitch: pitch.Pitch) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        pitch.octave = 0
        return pitch.pitchClass + 1

    def _encode_root(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        if chord == "N" or chord == "X":
            return NoteEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get root from m21
        root: pitch.Pitch = harte_chord.root()

        return self._encode_note(root)

    def _encode_bass(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """

        if chord == "N" or chord == "X":
            return NoteEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get root from m21
        root: pitch.Pitch = harte_chord.bass()

        return self._encode_note(root)

    def _encode_mode(self, chord: str) -> int:
        """
        Creates a chord embedding for the given chord.

        Args:
            chord (str): The chord to create an embedding for.

        Returns:
            int: The encoded chord embedding.
        """
        if chord == "N" or chord == "X":
            return ModeEncoder.N.value

        harte_chord = self._open_harte(chord)
        # get mode from m21
        mode = harte_chord.quality
        # get seventh from m21
        is_seventh = harte_chord.isSeventh()
        # merge mode and seventh
        is_seventh = "_seventh" if is_seventh else ""
        mode = mode + is_seventh

        # get encoding from enum
        mode = ModeEncoder[mode.upper()].value

        return mode


if __name__ == "__main__":
    # test cases
    encoder = BaseEncoder()
    print(encoder.encode("F:maj6", Encoding.ROOT))
    print(encoder.encode("F#:(2,3,6)/5", Encoding.BASS))
    print(encoder.encode("C:maj/#1", Encoding.BASS))
