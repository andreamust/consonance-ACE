"""
Evaluation of inter-annotator agreement using different chord representations.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm

# append the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import OrderedDict

import jams
from metrics.distances import GCT_DISTANCE, LINEAR_DISTANCE

# from converters import harte_to_roman_harmtrace
from metrics.mir_eval_nonbinary import (
    evaluate_mechanical,
    evaluate_tone_by_tone,
)
from mir_eval.chord import evaluate
from mir_eval.util import intervals_to_durations
from utils import get_file_annotations


def evaluate_chord_agreement(
    annotations: dict[str, jams.Annotation], eval_type: str = "harte"
) -> dict[str, OrderedDict]:
    """
    Evaluate the inter-annotator agreement of the CASD dataset using different metrics.

    Parameters
    ----------
    annotations : dict
        Dictionary containing the annotations from the .jams file.
    chord_type : str
        The chord representation to use. Options are "harte" or "roman".

    Returns
    -------
    pd.DataFrame
        Dataframe containing the inter-annotator agreement.
    """
    annotators = list(annotations.keys())

    # remove the key annotation
    if "key" in annotators:
        annotators.remove("key")

    # Get all possible combinations of annotators
    combinations = [
        (annotators[i], annotators[j])
        for i in range(len(annotators))
        for j in range(i + 1, len(annotators))
    ]

    # Get the evaluation for each combination
    results = {}
    for combination in combinations:
        annotator1 = combination[0]
        annotator2 = combination[1]
        intervals_a, chords_a = annotations[annotator1].to_interval_values()
        intervals_b, chords_b = annotations[annotator2].to_interval_values()

        # Check intervals_a and intervals_b are equal
        assert len(intervals_a) == len(intervals_b), (
            f"Intervals mismatch between annotators {annotator1} and {annotator2}. "
        )

        # Convert intervals to durations
        durations = intervals_to_durations(intervals_a)

        # Get the agreement
        if eval_type == "mir_eval":
            evaluations = evaluate(intervals_a, chords_a, intervals_b, chords_b)
        elif eval_type == "tone-by-tone":
            evaluations = evaluate_tone_by_tone(
                reference_labels=chords_a,
                estimated_labels=chords_b,
                durations=durations,
            )
        elif eval_type == "mechanical":
            evaluations = evaluate_mechanical(
                reference_labels=chords_a,
                estimated_labels=chords_b,
                durations=durations,
                distance=LINEAR_DISTANCE,
            )
        elif eval_type == "mechanical_consonance":
            evaluations = evaluate_mechanical(
                reference_labels=chords_a,
                estimated_labels=chords_b,
                durations=durations,
                distance=GCT_DISTANCE,
            )
        else:
            raise ValueError(
                f"Unknown evaluation type: {eval_type}. "
                "Available options are 'mir_eval', 'tone-by-tone', 'mechanical', or"
                " 'mechanical_consonance'."
            )

        results[combination] = evaluations

    return results


def evaluate_dataset_agreement(
    jams_annotations: list, eval_type: str = "mir_eval"
) -> dict:
    """
    Evaluate the inter-annotator agreement of the CASD dataset using different metrics.

    Parameters
    ----------
    jams_path : str
        Path to the .jams file containing the annotations.
    eval_type : str
        The chord representation to use. Options are "harte" or "roman".

    Returns
    -------
    pd.DataFrame
        Dataframe containing the inter-annotator agreement.
    """
    results = []
    # Iterate over all jams annotations
    for annotation in tqdm(jams_annotations, desc="Evaluating annotations"):
        # Get the evaluation
        evaluation = evaluate_chord_agreement(annotation, eval_type=eval_type)
        for value in evaluation.values():
            results.append(value)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Average results by metric (which are the columns)
    results = results_df.mean().to_dict()

    return results


if __name__ == "__main__":
    # Load the annotations
    jam_file = "/media/data/andrea/casd/casd_16.jams"

    # Get the annotations
    annotations = get_file_annotations(jam_file)
    # print(annotations.keys())
    # print(annotations["A1"])
    # harmtrace = convert_harmtrace(jams_path)
    m21_eval = evaluate_chord_agreement(annotations, eval_type="mir_eval")
    print(m21_eval)
