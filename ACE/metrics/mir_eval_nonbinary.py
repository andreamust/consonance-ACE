"""
Re-implementation of the `mir_eval` metrics using the `chord_eval` toolkit.
This module provides functions to compute non-binary chord distances using
`mir_eval`-style metrics, specifically for tone-by-tone (TBT) and mechanical
distances.

In order to provide a fair comparison, for each metric, the comparisons that are skipped
in the original `mir_eval` implementation are also skipped here.
"""

import sys
from pathlib import Path

# append the parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from metrics.distances import GCT_DISTANCE, LINEAR_DISTANCE, Distance
from metrics.metric import (
    mechanical_distance,
    mechanical_distance_from_midi,
    tone_by_tone_similarity,
    tone_by_tone_similarity_from_midi,
)
from metrics.utils import (
    chord_to_midi,
    encode_chords,
)
from mir_eval.chord import (
    majmin,
    majmin_inv,
    mirex,
    root,
    sevenths,
    sevenths_inv,
    tetrads,
    tetrads_inv,
    thirds,
    thirds_inv,
    triads,
    triads_inv,
    weighted_accuracy,
)

MIR_EVAL_METRICS = {
    "root": root,
    "majmin": majmin,
    "majmin_inv": majmin_inv,
    "triads": triads,
    "triads_inv": triads_inv,
    "thirds": thirds,
    "thirds_inv": thirds_inv,
    "sevenths": sevenths,
    "sevenths_inv": sevenths_inv,
    "tetrads": tetrads,
    "tetrads_inv": tetrads_inv,
    "mirex": mirex,
}

MIR_EVAL_STOPS = {
    "root": 1,
    "majmin": 8,
    "majmin_inv": 8,
    "triads": 8,
    "triads_inv": 8,
    "thirds": 3,
    "thirds_inv": 3,
    "sevenths": 12,
    "sevenths_inv": 12,
    "tetrads": 12,
    "tetrads_inv": 12,
    "mirex": 12,
}


def evaluate_nonbinary(
    reference_labels: list[str],
    estimated_labels: list[str],
    mir_eval_metric: str = "majmin",
    metric: str = "tbt",
    root_bonus: int = 1,
    bass_bonus: int = 1,
    distance: Distance = LINEAR_DISTANCE,
    bass_distance: Distance | None = None,
) -> np.ndarray:
    """
    Compute a non-binary distance measure between estimated and reference chords
    using either a tone-by-tone (TBT) or mechanical distance metric.

    This function applies the `mir_eval`-style majmin filtering for fair evaluation.
    It returns a list of distances per frame, with `-1` for invalid comparisons
    (e.g. out-of-vocabulary or ambiguous chords), and direct binary comparison values
    for unclassifiable segments (e.g. 'N' chords).

    Parameters
    ----------
    reference_labels : list of str
        Ground truth chord labels in `mir_eval` format.
    estimated_labels : list of str
        Predicted chord labels in `mir_eval` format.
    root_bonus : int, optional
        Bonus to apply when the root pitch matches (TBT only). Default is 1.
    bass_bonus : int, optional
        Bonus to apply when the bass pitch matches (TBT only). Default is 1.
    distance : Distance, optional
        Distance function to use for pitch comparison in the mechanical metric.
        Default is `LINEAR_DISTANCE`.
    bass_distance : Distance or None, optional
        Optional separate distance function for bass note comparison in mechanical dist.
        If None, the same `distance` function is used. Default is None.
    metric : {'tbt', 'mechanical'}, optional
        Distance computation strategy:
        - 'tbt': Tone-by-tone distance
        - 'mechanical': Weighted pitch-level distance
        Default is 'tbt'.
    inversions : bool, optional
        Whether to consider chord inversions (i.e., bass notes). Default is False.

    Returns
    -------
    distances : list of float
        A list of per-frame distances. Values can be:
        - A float ∈ [0, 1] for valid comparisons
        - `-1` for invalid comparisons (e.g., undefined chords)
    """
    # Store whether to use inversions
    inversions = True if mir_eval_metric.endswith("_inv") else False

    # Get the binary comparisons using `mir_eval`. This is needed to filter out
    # comparisons that are not valid for ensuring a fair comparison.
    assert mir_eval_metric in MIR_EVAL_METRICS, (
        f"Invalid mir_eval metric: {mir_eval_metric}. "
        f"Choose from {list(MIR_EVAL_METRICS.keys())}."
    )
    comparisons = MIR_EVAL_METRICS[mir_eval_metric](
        reference_labels=reference_labels, estimated_labels=estimated_labels
    )

    # Ensure the sequences are of the same length
    assert len(reference_labels) == len(estimated_labels), "Sequences mismatch!"

    # Encode the reference and estimated labels
    ref_roots, ref_semitones, ref_basses = encode_chords(reference_labels)
    est_roots, est_semitones, est_basses = encode_chords(estimated_labels)

    # Filter semitones according to the `mir_eval` metric
    stop = MIR_EVAL_STOPS[mir_eval_metric]
    ref_semitones = ref_semitones[:, :stop]
    est_semitones = est_semitones[:, :stop]

    # Compute distances
    distances = []
    for i in range(len(reference_labels)):
        # Get the reference and estimated chords
        comparison = comparisons[i]

        # If `mir_eval` comparison is not valid, keep the -1 value
        # This does not apply to `mirex` metric
        if comparison < 0 and mir_eval_metric != "mirex":
            distances.append(-1)

        # `N` chords are not comparable using tone-by-tone and mechanical distance
        # Use this heuristic instead
        elif reference_labels[i] == "N" or estimated_labels[i] == "N":
            if reference_labels[i] == estimated_labels[i]:
                distances.append(1)
            else:
                distances.append(0)

        # Otherwise, compute the selected distance
        else:
            midi_ref = chord_to_midi(ref_roots[i], ref_semitones[i], ref_basses[i])
            midi_est = chord_to_midi(est_roots[i], est_semitones[i], est_basses[i])

            # If inversions are enabled use the actual bass, otherwise use the root
            ref_bass = midi_ref["bass"] if inversions else None
            est_bass = midi_est["bass"] if inversions else None

            if metric == "tbt":
                # Compute the tone-by-tone distance
                dist = tone_by_tone_similarity_from_midi(
                    ground_truth=midi_ref["pitches"],
                    estimated=midi_est["pitches"],
                    ground_truth_root=midi_ref["root"],
                    estimated_root=midi_est["root"],
                    ground_truth_bass=ref_bass,
                    estimated_bass=est_bass,
                    root_bonus=root_bonus,
                    bass_bonus=bass_bonus,
                )
            elif metric == "mechanical_distance":
                # Compute the mechanical distance
                dist = mechanical_distance_from_midi(
                    ground_truth=midi_ref["pitches"],
                    estimated=midi_est["pitches"],
                    ground_truth_root=midi_ref["root"],
                    estimated_root=midi_est["root"],
                    ground_truth_bass=ref_bass,
                    estimated_bass=est_bass,
                    distance=distance,
                    bass_distance=bass_distance,
                    bass_weight=1,
                )
            else:
                raise ValueError(
                    f"Invalid metric: {metric}. Choose 'tbt' or 'mechanical'."
                )

            distances.append(dist)

    return np.array(distances, dtype=float)


def evaluate_mechanical(
    reference_labels: list[str],
    estimated_labels: list[str],
    durations: np.ndarray,
    distance: Distance = LINEAR_DISTANCE,
    bass_distance: Distance | None = None,
) -> dict:
    """
    Wrapper that computes the mechanical similarity for a pair of chord sequences for
    all the `mir_eval` metrics available.

    Parameters
    ----------
    reference_labels : list of str
        Ground truth chord labels in `mir_eval` format.
    estimated_labels : list of str
        Predicted chord labels in `mir_eval` format.
    distance : Distance, optional
        Distance function to use for pitch comparison. Default is `LINEAR_DISTANCE`.
    bass_distance : Distance or None, optional
        Optional separate distance function for bass note comparison.
        If None, the same `distance` function is used. Default is None.

    Returns
    -------
    distances : list of float
        A list of per-frame distances.
    """
    # Mechanical distance metrics
    results = {}
    for metric in MIR_EVAL_METRICS.keys():
        # get the comparisons for the current metric
        comparisons = evaluate_nonbinary(
            reference_labels=reference_labels,
            estimated_labels=estimated_labels,
            mir_eval_metric=metric,
            metric="mechanical_distance",
            distance=distance,
            bass_distance=bass_distance,
        )
        results[metric] = weighted_accuracy(comparisons=comparisons, weights=durations)
    return results


def evaluate_tone_by_tone(
    reference_labels: list[str],
    estimated_labels: list[str],
    durations: np.ndarray,
    root_bonus: int = 1,
    bass_bonus: int = 1,
) -> dict:
    """
    Wrapper that computes the tone-by-tone similarity for a pair of chord sequences for
    all the `mir_eval` metrics available.

    Parameters
    ----------
    reference_labels : list of str
        Ground truth chord labels in `mir_eval` format.
    estimated_labels : list of str
        Predicted chord labels in `mir_eval` format.
    root_bonus : int, optional
        Bonus to apply when the root pitch matches. Default is 1.
    bass_bonus : int, optional
        Bonus to apply when the bass pitch matches. Default is 1.

    Returns
    -------
    distances : list of float
        A list of per-frame distances.
    """
    # Tone-by-tone distance metrics
    results = {}
    for metric in MIR_EVAL_METRICS.keys():
        # get the comparisons for the current metric
        comparisons = evaluate_nonbinary(
            reference_labels=reference_labels,
            estimated_labels=estimated_labels,
            mir_eval_metric=metric,
            metric="tbt",
            root_bonus=root_bonus,
            bass_bonus=bass_bonus,
        )
        results[metric] = weighted_accuracy(comparisons=comparisons, weights=durations)
    return results


def evaluate_nonbinary_labels(
    reference_labels: list[str],
    estimated_labels: list[str],
    durations: np.ndarray,
    metric: str = "tbt",
    root_bonus: int = 1,
    bass_bonus: int = 1,
    distance: str | None = None,
) -> float:
    """
    Compute a non-binary distance measure between estimated and reference chords
    using either a tone-by-tone (TBT) or mechanical distance metric.
    This function applies the `mir_eval`-style majmin filtering for fair evaluation.
    It returns a numerical value representing the distance between the two chord
    sequences, using the weighted accuracy metric from `mir_eval`.

    Parameters
    ----------
    reference_labels : list of str
        Ground truth chord labels in `mir_eval` format.
    estimated_labels : list of str
        Predicted chord labels in `mir_eval` format.
    root_bonus : int, optional
        Bonus to apply when the root pitch matches (TBT only). Default is 1.
    bass_bonus : int, optional
        Bonus to apply when the bass pitch matches (TBT only). Default is 1.
    distance : Distance, optional
        Distance function to use for pitch comparison in the mechanical metric.
        Default is `LINEAR_DISTANCE`.
    bass_distance : Distance or None, optional
        Optional separate distance function for bass note comparison in mechanical dist.
        If None, the same `distance` function is used. Default is None.
    metric : {'tbt', 'mechanical'}, optional
        Distance computation strategy. Default is 'tbt'.
    root_bonus : int, optional
        Bonus to apply when the root pitch matches (TBT only). Default is 1.
    bass_bonus : int, optional
        Bonus to apply when the bass pitch matches (TBT only). Default is 1.
    Returns
    -------
    float
        A numerical value representing the distance between the two chord sequences.
        The value is in the range [0, 1] for tbt. For mechanical distance, it is
        a distance value that can be any non-negative integer.
    """
    # Ensure that if mechanical distance is used, a distance function is provided
    if metric == "mechanical":
        assert distance in ["linear", "consonance"], (
            f"Invalid distance: {distance}. Choose 'linear' or 'consonance'."
        )
        if distance == "linear":
            distance_function = LINEAR_DISTANCE
        elif distance == "consonance":
            distance_function = GCT_DISTANCE

    distances = []
    # Iterate over the labels
    for reference, estimated in zip(reference_labels, estimated_labels):
        # Since neither tbt nor mechanical distance can handle N chords,
        # implement a heuristic to handle them
        if reference in ["N", "X"] or estimated in ["N", "X"]:
            # If both are the same, return 1.0, otherwise return 0.
            if reference == estimated:
                distances.append(1.0)
            else:
                distances.append(0.0)
            continue
        # print(f"reference: {reference}, estimated: {estimated}")
        # Get the comparisons for the current metric
        if metric == "tbt":
            comparisons = tone_by_tone_similarity(
                ground_truth=reference,
                estimated=estimated,
                root_bonus=root_bonus,
                bass_bonus=bass_bonus,
            )
        elif metric == "mechanical":
            comparisons = mechanical_distance(
                ground_truth=reference,
                estimated=estimated,
                distance=distance_function,  # type: ignore
                bass_distance=distance_function,  # type: ignore
                bass_weight=1,
            )
        else:
            raise ValueError(f"Invalid metric: {metric}. Choose 'tbt' or 'mechanical'.")
        distances.append(comparisons)

    # Compute the weighted accuracy
    return weighted_accuracy(comparisons=np.array(distances), weights=durations)
