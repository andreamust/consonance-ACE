"""
Non-binary metrics for evaluating chord label pairs (prediction vs ground truth).
Part of this code is taken from the `chord_eval` toolkit, which is licensed under the
MIT License.
The original implementation can be found at:
* GitHub: https://github.com/DCMLab/chord-eval
* Paper: McLeod, Andrew, Suermondt, Xavier, Rammos, Yannis, Herff, Steffen, and
Rohrmeier, Martin A. 2022. Three Metrics for Musical Chord Label Evaluation. In Forum
for Information Retrieval Evaluation (FIRE ’22), December 9–13, 2022, Kolkata, India.
ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3574318.3574335
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path

# add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from harte.harte import Harte
from metrics.distances import Distance, find_notes_matching

TONAL_VECTOR = [0, 7, 5, 1, 1, 2, 3, 1, 2, 2, 4, 6]

GCT_DISTANCE = Distance(
    custom_distance=TONAL_VECTOR,
)


@lru_cache()
def tone_by_tone_distance(
    ground_truth: str,
    estimated: str,
    root_bonus: int = 1,
    bass_bonus: int = 1,
) -> float:
    """
    Compute the tone-by-tone distance between two chords.

    Parameters
    ----------
    ground_truth : str
        Ground truth chord.
    estimated : str
        Estimated chord.

    Returns
    -------
    float
        Tone-by-tone distance between the two chords.
    """

    def one_sided_tbt(
        note_set1: set,
        note_set2: set,
        root_matches: bool,
        bass_matches: bool,
        root_bonus: int,
        bass_bonus: int,
    ) -> float:
        """
        Get the one-sided tbt. That is, the proportion of chord 1's notes which are
        missing from chord2, including root and bass bonus.

        Parameters
        ----------
        note_set1 : set
            The set of pitch classes contained in chord 1.
        note_set2 : np.ndarray
            The set of pitch classes in chord 2.
        root_matches : bool
            True if the chord roots match. False otherwise.
        bass_matches : bool
            True if the chord bass notes match. False otherwise.
        root_bonus : int
            The root bonus to use (see the tone_by_tone_distance).
        bass_bonus : int
            The bass bonus to use (see the tone_by_tone_distance).

        Returns
        -------
        distance : float
            The one-sided tone-by-tone distance.
        """
        matches = len(note_set1.intersection(note_set2))

        if root_matches:
            matches += root_bonus
        if bass_matches:
            matches += bass_bonus

        return 1 - matches / (len(note_set1) + bass_bonus + root_bonus)

    chord1 = Harte(ground_truth)
    chord2 = Harte(estimated)

    root1 = chord1.root().midi
    root2 = chord2.root().midi

    bass1 = chord1.bass().midi
    bass2 = chord2.bass().midi

    notes1 = np.array(chord1.get_midi_pitches())
    notes2 = np.array(chord2.get_midi_pitches())

    root_matches = root1 == root2
    bass_matches = bass1 == bass2

    distance = np.mean([
        one_sided_tbt(
            set(notes1),
            set(notes2),
            root_matches,
            bass_matches,
            root_bonus,
            bass_bonus,
        ),
        one_sided_tbt(
            set(notes2),
            set(notes1),
            root_matches,
            bass_matches,
            root_bonus,
            bass_bonus,
        ),
    ])

    return float(distance)


def tone_by_tone_distance_from_midi(
    ground_truth: list[int],
    estimated: list[int],
    ground_truth_root: int,
    estimated_root: int,
    ground_truth_bass: int | None = None,
    estimated_bass: int | None = None,
    root_bonus: int = 1,
    bass_bonus: int = 1,
) -> float:
    """
    Compute the tone-by-tone distance between two chords.

    Parameters
    ----------
    ground_truth : str
        Ground truth chord.
    estimated : str
        Estimated chord.

    Returns
    -------
    float
        Tone-by-tone distance between the two chords.
    """

    def one_sided_tbt(
        note_set1: set,
        note_set2: set,
        root_matches: bool,
        bass_matches: bool,
        root_bonus: int,
        bass_bonus: int,
    ) -> float:
        """
        Get the one-sided tbt. That is, the proportion of chord 1's notes which are
        missing from chord2, including root and bass bonus.

        Parameters
        ----------
        note_set1 : set
            The set of pitch classes contained in chord 1.
        note_set2 : np.ndarray
            The set of pitch classes in chord 2.
        root_matches : bool
            True if the chord roots match. False otherwise.
        bass_matches : bool
            True if the chord bass notes match. False otherwise.
        root_bonus : int
            The root bonus to use (see the tone_by_tone_distance).
        bass_bonus : int
            The bass bonus to use (see the tone_by_tone_distance).

        Returns
        -------
        distance : float
            The one-sided tone-by-tone distance.
        """
        matches = len(note_set1.intersection(note_set2))

        if root_matches:
            matches += root_bonus
        if bass_matches:
            matches += bass_bonus

        return 1 - matches / (len(note_set1) + bass_bonus + root_bonus)

    root1 = ground_truth_root
    root2 = estimated_root
    notes1 = np.array(ground_truth)
    notes2 = np.array(estimated)

    root_matches = root1 == root2

    if ground_truth_bass is not None and estimated_bass is not None:
        bass_matches = ground_truth_bass == estimated_bass
    else:
        bass_matches = root1 == root2

    distance = np.mean([
        one_sided_tbt(
            set(notes1),
            set(notes2),
            root_matches,
            bass_matches,
            root_bonus,
            bass_bonus,
        ),
        one_sided_tbt(
            set(notes2),
            set(notes1),
            root_matches,
            bass_matches,
            root_bonus,
            bass_bonus,
        ),
    ])

    return float(distance)


def tone_by_tone_similarity(
    ground_truth: str,
    estimated: str,
    root_bonus: int = 1,
    bass_bonus: int = 1,
) -> float:
    """
    Compute a similarity value between 0 and 1 based on tone-by-tone distance.

    Parameters
    ----------
    ground_truth : str
        Ground truth chord.
    estimated : str
        Estimated chord.

    Returns
    -------
    float
        Similarity value between 0 and 1.
    """
    dist = tone_by_tone_distance(ground_truth, estimated, root_bonus, bass_bonus)
    return float(1.0 - dist)  # Clamp between 0 and 1


def tone_by_tone_similarity_from_midi(
    ground_truth: list[int],
    estimated: list[int],
    ground_truth_root: int,
    estimated_root: int,
    ground_truth_bass: int | None = None,
    estimated_bass: int | None = None,
    root_bonus: int = 1,
    bass_bonus: int = 1,
) -> float:
    """
    Compute a similarity value between 0 and 1 based on tone-by-tone distance.

    Parameters
    ----------
    ground_truth : list[int]
        Ground truth chord as a list of MIDI pitches.
    estimated : list[int]
        Estimated chord as a list of MIDI pitches.

    Returns
    -------
    float
        Similarity value between 0 and 1.
    """
    dist = tone_by_tone_distance_from_midi(
        ground_truth,
        estimated,
        ground_truth_root,
        estimated_root,
        ground_truth_bass,
        estimated_bass,
        root_bonus,
        bass_bonus,
    )
    return float(1.0 - dist)  # Clamp between 0 and 1


@lru_cache()
def mechanical_distance(
    ground_truth: str,
    estimated: str,
    distance: Distance = Distance(),
    bass_distance: Distance | None = None,
    bass_weight: int = 1,
) -> float:
    """
    Compute the mechanical distance between two chords.

    Parameters
    ----------
    ground_truth : str
        Ground truth chord.
    estimated : str
        Estimated chord.

    Returns
    -------
    float
        Mechanical distance between the two chords.
    """
    assert distance.is_valid(), (
        "Given distance is not valid (some distances are negative)."
    )
    if bass_distance is None:
        bass_distance = distance
    else:
        if not bass_distance.is_valid():
            logging.warning(
                "bass_distance is not valid (some values are negative). "
                "Defaulting to given distance."
            )
            bass_distance = distance

    # Get the notes from the chords
    notes1 = np.array(Harte(ground_truth).get_midi_pitches())
    notes2 = np.array(Harte(estimated).get_midi_pitches())

    # Get the basses (first note in the chords in Harte().get_midi_pitches())
    bass1 = notes1[0]
    bass2 = notes2[0]

    # Bass distance
    bass_steps = bass_distance.distance_between(bass1, bass2) * bass_weight

    # Other pitches
    upper_steps, match_ids = find_notes_matching(notes1, notes2, distance)

    # If the bass notes have been matched together, do not count again
    if (0, 0) in match_ids:
        upper_steps -= distance.distance_between(bass1, bass2)

    # Handle chords of different lengths
    if len(notes1) != len(notes2):
        big_chord, bc_idx, small_chord = (
            (notes1, 0, notes2) if len(notes1) > len(notes2) else (notes2, 1, notes1)
        )

        # Remove all pitches from the biggest chord that have already been matched
        big_chord = [
            big_chord[idx]
            for idx in range(len(big_chord))
            if idx not in set([pair[bc_idx] for pair in match_ids] + [0])
        ]

        # Find corresponding note in smaller chord for each additional big chord note
        upper_steps += sum([
            min([
                distance.distance_between(bc_pitch, sc_pitch)
                for sc_pitch in small_chord
            ])
            for bc_pitch in big_chord
        ])

    return float(upper_steps + bass_steps)


def mechanical_distance_from_midi(
    ground_truth: list[int],
    estimated: list[int],
    ground_truth_root: int,
    estimated_root: int,
    ground_truth_bass: int | None = None,
    estimated_bass: int | None = None,
    distance: Distance = Distance(),
    bass_distance: Distance | None = None,
    bass_weight: int = 1,
) -> float:
    """
    Compute the mechanical distance between two chords.

    Parameters
    ----------
    ground_truth : str
        Ground truth chord.
    estimated : str
        Estimated chord.

    Returns
    -------
    float
        Mechanical distance between the two chords.
    """
    assert distance.is_valid(), (
        "Given distance is not valid (some distances are negative)."
    )
    if bass_distance is None:
        bass_distance = distance
    else:
        if not bass_distance.is_valid():
            logging.warning(
                "bass_distance is not valid (some values are negative). "
                "Defaulting to given distance."
            )
            bass_distance = distance

    notes1 = np.array(ground_truth)
    notes2 = np.array(estimated)

    if ground_truth_bass is not None and estimated_bass is not None:
        bass1 = ground_truth_bass
        bass2 = estimated_bass
    else:
        bass1 = ground_truth_root
        bass2 = estimated_root

    # Bass distance
    bass_steps = bass_distance.distance_between(bass1, bass2) * bass_weight

    # Other pitches
    upper_steps, match_ids = find_notes_matching(notes1, notes2, distance)

    # If the bass notes have been matched together, do not count again
    if (0, 0) in match_ids:
        upper_steps -= distance.distance_between(bass1, bass2)

    # Handle chords of different lengths
    if len(notes1) != len(notes2):
        big_chord, bc_idx, small_chord = (
            (notes1, 0, notes2) if len(notes1) > len(notes2) else (notes2, 1, notes1)
        )

        # Remove all pitches from the biggest chord that have already been matched
        big_chord = [
            big_chord[idx]
            for idx in range(len(big_chord))
            if idx not in set([pair[bc_idx] for pair in match_ids] + [0])  # type: ignore
        ]

        # Find corresponding note in smaller chord for each additional big chord note
        upper_steps += sum([
            min([
                distance.distance_between(bc_pitch, sc_pitch)
                for sc_pitch in small_chord
            ])
            for bc_pitch in big_chord
        ])

    return float(upper_steps + bass_steps)
