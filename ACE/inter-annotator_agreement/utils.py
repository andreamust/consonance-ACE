"""Utility functions for inter-annotator agreement analysis."""

import glob
import os
from collections import OrderedDict
from pathlib import Path

import jams
import numpy as np


def get_files(annotation_path: str | Path) -> list:
    """
    Get all the .jams files in the given directory.

    Parameters
    ----------
    annotation_path : str or Path
        Path to the directory containing the .jams files.

    Returns
    -------
    list
        List of .jams files in the given directory.
    """
    annotation_path = Path(annotation_path)
    assert annotation_path.exists(), f"{annotation_path} does not exist."

    return glob.glob(os.path.join(annotation_path, "*casd*.jams"))


def get_file_annotations(jams_file: str | Path) -> dict[str, jams.Annotation]:
    """
    Get the annotations from a .jams file.

    Parameters
    ----------
    jams_file : str or Path
        Path to the .jams file.

    Returns
    -------
    dict
        Dictionary containing the annotations from the .jams file.
    """
    jams_file = Path(jams_file)
    assert jams_file.exists(), f"{jams_file} does not exist."

    # initialize annotation list
    annotations = {}

    # Load the .jams file
    jam = jams.load(str(jams_file), validate=False)
    # Search for the chord annotations
    for annotation in jam.annotations:
        if annotation.namespace == "chord":
            annotator_id = annotation.annotation_metadata.annotator.id
            annotations[annotator_id] = annotation
        if annotation.namespace == "key_mode":
            annotations["key"] = annotation

    return annotations


def remove_repeat_chords(chords: list[str]) -> list[str]:
    """
    Remove successive repeated chords from a list of chords.

    Parameters
    ----------
    chords : list
        List of chords.

    Returns
    -------
    list
        List of chords with successive repeated chords removed.
    """
    return [
        chord
        for idx, chord in enumerate(chords)
        if idx == 0 or chord != chords[idx - 1]
    ]


def get_dataset_statistics(jams_files: list[str | Path]) -> dict:
    """
    Get the statistics of the dataset.

    Parameters
    ----------
    jams_files : list
        List of .jams files.

    Returns
    -------
    dict
        Dictionary containing the statistics of the dataset. The following statistics
        are computed:
            - Max number of chords
            - Min number of chords
            - Average number of chords
            - Chord vocabulary
    """
    # Initialize the statistics
    statistics = {
        "max_chords": 0,
        "min_chords": 1000,
        "chord_vocabulary": set(),
    }

    for jams_file in jams_files[:1]:
        annotations = get_file_annotations(jams_file)
        for annotator in annotations.keys():
            if annotator == "key":
                continue
            _, chords = annotations[annotator].to_interval_values()
            chords = remove_repeat_chords(chords)
            statistics["max_chords"] = max(statistics["max_chords"], len(chords))
            statistics["min_chords"] = min(statistics["min_chords"], len(chords))
            statistics["chord_vocabulary"].update(chords)

    return statistics


def generate_random_sequences(
    num_sequences: int, statistics: dict, seed: int = 42
) -> list[dict[str, jams.Annotation]]:
    """
    Generate random sequences of chords based on the dataset statistics.

    Parameters
    ----------
    num_sequences : int
        Number of sequences to generate.
    statistics : dict
        Dictionary containing the statistics of a reference dataset.

    Returns
    -------
    list
        List of dictioraries, each of which containingh 4 random sequences of chords
        faking annotators (A1, A2, A3, A4).
    """
    # set the random seed for reproducibility
    np.random.seed(seed)

    random_sequences = []
    for _ in range(num_sequences):
        random_sequence = dict()
        num_chords = np.random.randint(
            statistics["min_chords"], statistics["max_chords"] + 1
        )
        for annotator in ["A1", "A2", "A3", "A4"]:
            chords = np.random.choice(list(statistics["chord_vocabulary"]), num_chords)

            # convert random sequence to jams annotation
            annotation = jams.Annotation(namespace="chord", time=1, duration=1)

            # Add the chords to the annotation
            for idx, chord in enumerate(chords):
                # print(idx, chord)
                annotation.append(
                    time=idx,
                    duration=1,
                    value=chord,
                    confidence=1,
                )

            random_sequence[annotator] = annotation

        random_sequences.append(random_sequence)

    return random_sequences


def sum_ordered_dict_values(d: dict) -> OrderedDict:
    """
    Takes a dictionary dict[str, OrderedDict] and returns the sum of the values
    of the OrderedDicts, key by key.

    Parameters
    ----------
    d : dict
        Dictionary containing OrderedDicts.

    Returns
    -------
    OrderedDict
        Sum of the values of the OrderedDicts.
    """
    result = OrderedDict()
    for couple in d.keys():
        couple_dict = d[couple]
        for key in couple_dict.keys():
            if key not in result.keys():
                result[key] = couple_dict[key]
            else:
                result[key] += couple_dict[key]

    return result


def average_ordered_dict_values(d: dict) -> OrderedDict:
    """
    Takes a dictionary dict[str, OrderedDict] and returns the average of the values
    of the OrderedDicts.

    Parameters
    ----------
    d : dict
        Dictionary containing OrderedDicts.

    Returns
    -------
    OrderedDict
        Average of the values of the OrderedDicts.
    """
    result = sum_ordered_dict_values(d)
    for key in result.keys():
        result[key] = result[key] / len(d)

    return result
