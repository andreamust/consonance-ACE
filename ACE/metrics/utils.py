"""
Utility functions for distance metrics in ACE. Part of this code is taken from the
`chord_eval` toolkit, which is licensed under the MIT License.
The original implementation can be found at:
* GitHub: https://github.com/DCMLab/chord-eval
* Paper: McLeod, Andrew, Suermondt, Xavier, Rammos, Yannis, Herff, Steffen, and
Rohrmeier, Martin A. 2022. Three Metrics for Musical Chord Label Evaluation. In Forum
for Information Retrieval Evaluation (FIRE ’22), December 9–13, 2022, Kolkata, India.
ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3574318.3574335
"""

import numpy as np
from mir_eval.chord import encode_many


def get_smallest_interval(note1: int, note2: int) -> int:
    """
    Find the smallest interval between tow given midi notes by shifting them by
    one or more octaves.

    Parameters
    ----------
    note1 : int
        Midi number of the first note.
    note2 : int
        Midi number of the second note

    Returns
    -------
    int
        Midi number corresponding to the smallest interval.

    """
    diff = np.abs(note1 - note2) % 12
    return min(diff, 12 - diff)


def chord_to_midi(
    root: int, semitone_bitmap: np.ndarray, bass: int | None = None
) -> dict:
    """
    Converts a chord's root and interval bitmap into a set of MIDI pitches.
    - root: int in [0,11], -1 for N.C.
    - semitone_bitmap: 12D binary vector
    - bass: optional int in [0,11]
    """
    if root < 0:
        return {"root": None, "bass": None, "pitches": set()}

    # Extract the MIDI number for the root note
    root_midi = 60 + root
    # Create a set of relative semitones from the bitmap
    rel_semitones = np.where(semitone_bitmap > 0)[0]
    pitches = set(root_midi + rel for rel in rel_semitones)
    pitches = sorted(list(pitches))  # Sort pitches for consistency
    bass_midi = root_midi - 12 + bass if bass is not None and bass >= 0 else root_midi

    return {"root": root_midi, "bass": bass_midi, "pitches": pitches}


def encode_chords(labels):
    roots, semitones, bass = encode_many(labels, reduce_extended_chords=False)
    return roots, semitones, bass
