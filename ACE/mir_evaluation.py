"""
Scripts for the evaluation of the chord recognition task.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "preprocess"))

# suppress mir_eval warnings
import warnings
from collections import Counter
from pathlib import Path

import mir_eval
import numpy as np
import torch
from harte.harte import Harte

from ACE.metrics.mir_eval_nonbinary import evaluate_nonbinary_labels
from ACE.preprocess.chord_processor import (
    NoteEncoder,
    OriginalChordConverter,
    PumppChordConverter,
)

warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval")

INTERVAL_MAP = {
    0: "1",
    1: "b9",
    2: "9",
    3: "b3",
    4: "3",
    5: "4",
    6: "b5",
    7: "5",
    8: "b6",
    9: "6",
    10: "b7",
    11: "7",
}


def mode_filter(seq: np.ndarray, window_size: int = 5) -> np.ndarray:
    smoothed = []
    T = len(seq)
    for i in range(T):
        start = max(0, i - window_size // 2)
        end = min(T, i + window_size // 2 + 1)
        window = seq[start:end]
        counts = Counter(window)
        most_common = counts.most_common(1)[0][0]
        smoothed.append(most_common)
    return np.array(smoothed)


def mean_filter_sequence(seq: np.ndarray, window_size: int = 5) -> np.ndarray:
    pad_width = window_size // 2
    padded = np.pad(seq, ((pad_width, pad_width), (0, 0)), mode="edge")
    smoothed = np.zeros_like(seq)
    for i in range(seq.shape[0]):
        smoothed[i] = padded[i : i + window_size].mean(axis=0)
    return smoothed


def remove_short_chords(intervals, labels, min_duration=0.3):
    if len(intervals) == 0:
        return intervals, labels

    filtered_intervals = []
    filtered_labels = []

    for i, (interval, label) in enumerate(zip(intervals, labels)):
        duration = interval[1] - interval[0]
        if duration < min_duration:
            # Merge with previous if it exists
            if len(filtered_intervals) > 0:
                # Extend the previous interval
                filtered_intervals[-1] = (filtered_intervals[-1][0], interval[1])
            elif i + 1 < len(intervals):
                # Merge with next interval
                intervals[i + 1] = (interval[0], intervals[i + 1][1])
            # If neither previous nor next exist (edge case), drop it
        else:
            filtered_intervals.append(interval)
            filtered_labels.append(label)

    return np.array(filtered_intervals), filtered_labels


def convert_predictions(
    predictions: np.ndarray, vocabulary: str = "majmin", segment_duration: float = 30.0
):
    """
    Convert per-frame predictions to intervals and labels (single segment only).

    Args:
        predictions (np.ndarray): shape (T,) - per-frame predictions.
        label_mapping (list of str): index-to-label mapping.
        segment_duration (float): total duration of the segment (in seconds).

    Returns:
        intervals (np.ndarray): shape (n_events, 2)
        labels (list of str)
    """
    T = predictions.shape[0]
    frame_duration = segment_duration / T
    times = np.arange(T) * frame_duration

    intervals = []
    labels = []
    start_idx = 0

    # instantiate the converter based on vocabulary
    if vocabulary == "majmin":
        label_mapping = PumppChordConverter(vocab="3")
    elif vocabulary == "complete":
        label_mapping = PumppChordConverter(vocab="3567s")
    else:
        raise ValueError(f"Unsupported vocabulary: {vocabulary}")

    # Convert chord predictions to labels
    predictions = label_mapping.decode(predictions)  # type: ignore
    current_label = predictions[0]

    for i in range(1, T):
        if predictions[i] != current_label:
            intervals.append((times[start_idx], times[i]))
            labels.append(current_label)
            start_idx = i
            current_label = predictions[i]

    # Final interval
    intervals.append((times[start_idx], segment_duration))
    labels.append(current_label)

    return np.array(intervals), labels


def convert_predictions_decomposed(
    root_predictions: np.ndarray,
    bass_predictions: np.ndarray,
    chord_predictions: np.ndarray,
    segment_duration: float = 30.0,
    threshold: float = 0.5,
    remove_short_min_duration: float = 0.3,
):
    """
    Convert per-frame chord component predictions to intervals and labels (single
    segment only).

    Args:
        root_predictions (np.ndarray): shape (T,) - per-frame root predictions
        (class indices).
        bass_predictions (np.ndarray): shape (T,) - per-frame bass predictions
        (class indices).
        chord_predictions (np.ndarray): shape (T, n_chord_classes) - per-frame chord
        predictions (probabilities/logits).
        segment_duration (float): total duration of the segment (in seconds).
        threshold (float): threshold for chord component detection.

    Returns:
        intervals (np.ndarray): shape (n_events, 2) - start and end times of each chord
        labels (list of str): chord labels corresponding to each interval
    """
    T = root_predictions.shape[0]
    frame_duration = segment_duration / T
    times = np.arange(T) * frame_duration

    intervals = []
    labels = []
    start_idx = 0

    # --- STEP 1: Decode each frame ---
    chord_labels = [
        decode_chord(
            root=root_predictions[i],
            bass=bass_predictions[i],
            chord=chord_predictions[i],
            threshold=threshold,
        )
        for i in range(T)
    ]

    # --- STEP 2: Group consecutive identical chord labels into intervals ---
    intervals = []
    labels = []
    start_idx = 0
    current_label = chord_labels[0]

    for i in range(1, T):
        if chord_labels[i] != current_label:
            intervals.append((times[start_idx], times[i]))
            labels.append(current_label)
            start_idx = i
            current_label = chord_labels[i]

    intervals.append((times[start_idx], segment_duration))
    labels.append(current_label)

    # --- STEP 3: Remove short segments ---
    intervals, labels = remove_short_chords(
        intervals, labels, min_duration=remove_short_min_duration
    )

    return np.array(intervals), labels


def convert_ground_truth(
    onsets: torch.Tensor,
    label_ids: torch.Tensor,
    segment_duration: float = 30.0,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
):
    """
    Convert ground truth onsets and label ids to (intervals, labels) format.

    Args:
        onsets (torch.Tensor): shape (N,), onset times in seconds.
        label_ids (torch.Tensor): shape (N,), integer-encoded chord labels.
        label_mapping (list of str): mapping from class index to label string.
        segment_duration (float): max segment duration in seconds (used to cap final
        interval).

    Returns:
        intervals (np.ndarray): shape (n_events, 2)
        labels (list of str)
    """

    # Keep only valid events (nonzero onsets, OR first one which might be zero)
    valid = (onsets > 0) | (np.arange(len(onsets)) == 0)
    onsets = onsets[valid]
    label_ids = label_ids[valid]

    # Build intervals
    end_times = np.concatenate([onsets[1:], [segment_duration]])
    intervals = np.stack([onsets, end_times], axis=1)

    # Map label ids
    original_converter = OriginalChordConverter(vocab_path=vocab_path)
    labels = [original_converter.decode(int(i)) for i in label_ids]

    return intervals, labels


def decode_chord(
    root: int, bass: int, chord: np.ndarray, threshold: float = 0.5
) -> str:
    """
    Decode the chord label from the one-hot encoding, root and bass labels.
    :param root: root label
    :param bass: bass label
    :param chord: chord label
    :return: chord label
    """
    if root == 12:
        return "N"
    root_name = NoteEncoder(int(root) + 1).name  # type: ignore
    root_name = root_name.replace("_SHARP", "#")
    bass_degree = ""
    if int(bass) != 12 and int(bass) != int(root):
        bass_interval = (int(bass) - int(root)) % 12
        bass_degree = f"/{INTERVAL_MAP[int(bass_interval)]}"

    # chord processing
    chord = np.roll(chord, -int(root))
    chord_degree = ""
    for c in np.where(chord > threshold)[0]:
        if c < 12:
            chord_degree += f",{INTERVAL_MAP[c]}"
        else:
            return "N"

    if len(chord_degree) > 0:
        chord_degree = f":({chord_degree.lstrip(',')})"

    # if 1,3 in chord_degree but not 5, add 5
    if ("1,3" in chord_degree or "1,b3" in chord_degree) and "5" not in chord_degree:
        chord_degree = chord_degree.replace(")", ",5)")

    # prettify and validate chord
    chord_label = f"{root_name}{chord_degree}{bass_degree}"

    # fix a bug from harte-lib
    if ":/" in chord_label:
        chord_label = chord_label.replace(":/", ":maj/")
    if chord_label.endswith(":"):
        chord_label += "maj"

    chord_label = Harte(chord_label).prettify()

    # remove recurrent errors
    # chord_label = chord_label.replace("(4)/4", "/3")
    # chord_label = chord_label.replace("(b6)/b6", "")

    return chord_label


def compute_scores(ref_intervals, ref_labels, est_intervals, est_labels) -> dict:
    """
    Compute multiple chord evaluation scores using mir_eval.

    Args:
        ref_intervals (np.ndarray): shape (n_ref, 2), reference intervals.
        ref_labels (list of str): reference chord labels.
        est_intervals (np.ndarray): shape (n_est, 2), estimated intervals.
        est_labels (list of str): estimated chord labels.

    Returns:
        dict: scores from mir_eval (majmin, mirex, sevenths, thirds, triads)
    """
    # Align intervals
    (intervals, ref_aligned, est_aligned) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )

    durations = mir_eval.util.intervals_to_durations(intervals)

    # Compute comparison arrays for each metric
    root_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.root(ref_aligned, est_aligned), durations
        ),
    )
    majmin_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.majmin(ref_aligned, est_aligned), durations
        ),
    )
    majmin_inv_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.majmin_inv(ref_aligned, est_aligned), durations
        ),
    )
    mirex_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.mirex(ref_aligned, est_aligned), durations
        ),
    )
    sevenths_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.sevenths(ref_aligned, est_aligned), durations
        ),
    )
    sevenths_inv_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.sevenths_inv(ref_aligned, est_aligned), durations
        ),
    )
    thirds_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.thirds(ref_aligned, est_aligned), durations
        ),
    )
    thirds_inv_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.thirds_inv(ref_aligned, est_aligned), durations
        ),
    )
    triads_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.triads(ref_aligned, est_aligned), durations
        ),
    )
    triads_inv_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.triads_inv(ref_aligned, est_aligned), durations
        ),
    )
    tetrads_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.tetrads(ref_aligned, est_aligned), durations
        ),
    )
    tetrads_inv_score = (
        mir_eval.chord.weighted_accuracy(
            mir_eval.chord.tetrads_inv(ref_aligned, est_aligned), durations
        ),
    )
    # Non-binary evaluation
    tbt_score = evaluate_nonbinary_labels(
        reference_labels=ref_aligned,
        estimated_labels=est_aligned,
        durations=durations,
        metric="tbt",
    )
    mechanical_score = evaluate_nonbinary_labels(
        reference_labels=ref_aligned,
        estimated_labels=est_aligned,
        durations=durations,
        metric="mechanical",
        distance="linear",
    )
    mechanical_consonance_score = evaluate_nonbinary_labels(
        reference_labels=ref_aligned,
        estimated_labels=est_aligned,
        durations=durations,
        metric="mechanical",
        distance="consonance",
    )

    # Return all scores in a dictionary
    return {
        "root": root_score[0],
        "majmin": majmin_score[0],
        "mirex": mirex_score[0],
        "sevenths": sevenths_score[0],
        "thirds": thirds_score[0],
        "triads": triads_score[0],
        "tetrads": tetrads_score[0],
        "majmin_inv": majmin_inv_score[0],
        "sevenths_inv": sevenths_inv_score[0],
        "thirds_inv": thirds_inv_score[0],
        "triads_inv": triads_inv_score[0],
        "tetrads_inv": tetrads_inv_score[0],
        "tbt": tbt_score,
        "mechanical": mechanical_score,
        "mechanical_consonance": mechanical_consonance_score,
    }


def evaluate_batch(
    batched_predictions: np.ndarray,
    batched_onsets: np.ndarray,
    batched_gt_labels: np.ndarray,
    vocabulary: str = "majmin",
    segment_duration: float = 30.0,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
) -> dict:
    """
    Compute chord recognition scores over a batch of predictions and ground truth.

    Args:
        batched_predictions (np.ndarray): shape (B, T,), predicted probabilities.
        batched_onsets (np.ndarray): shape (B, T), onset times for each segment.
        batched_gt_labels (np.ndarray): shape (B, T), ground truth chord labels.
        vocabulary (str): vocabulary type, either "majmin" or "complete" (default
        "majmin").
        segment_duration (float): duration of each segment (default 30s).

    Returns:
        dict: scores from mir_eval (majmin, mirex, sevenths, thirds, triads)
    """
    B = batched_predictions.shape[0]

    # initialize empty scores dictionary
    scores = {
        "root": 0.0,
        "majmin": 0.0,
        "mirex": 0.0,
        "sevenths": 0.0,
        "thirds": 0.0,
        "triads": 0.0,
        "tetrads": 0.0,
        "majmin_inv": 0.0,
        "sevenths_inv": 0.0,
        "thirds_inv": 0.0,
        "triads_inv": 0.0,
        "tetrads_inv": 0.0,
        "tbt": 0.0,
        "mechanical": 0.0,
        "mechanical_consonance": 0.0,
    }
    # print(f"Shape batched_predictions: {batched_predictions.shape}")
    # print(f"Shape batched_onsets: {batched_onsets.shape}")
    # print(f"Shape batched_gt_labels: {batched_gt_labels.shape}")

    for i in range(B):
        # Estimated
        est_int, est_lab = convert_predictions(
            batched_predictions[i], vocabulary, segment_duration
        )

        # Ground truth
        ref_int, ref_lab = convert_ground_truth(
            batched_onsets[i],
            batched_gt_labels[i],
            segment_duration,
            vocab_path=vocab_path,
        )

        # Compute scores
        results = compute_scores(ref_int, ref_lab, est_int, est_lab)

        # Accumulate scores
        scores["root"] += results["root"]
        scores["majmin"] += results["majmin"]
        scores["mirex"] += results["mirex"]
        scores["sevenths"] += results["sevenths"]
        scores["thirds"] += results["thirds"]
        scores["triads"] += results["triads"]
        scores["tetrads"] += results["tetrads"]
        scores["majmin_inv"] += results["majmin_inv"]
        scores["sevenths_inv"] += results["sevenths_inv"]
        scores["thirds_inv"] += results["thirds_inv"]
        scores["triads_inv"] += results["triads_inv"]
        scores["tetrads_inv"] += results["tetrads_inv"]
        scores["tbt"] += results["tbt"]
        scores["mechanical"] += results["mechanical"]
        scores["mechanical_consonance"] += results["mechanical_consonance"]

        # print an example
        if i == 0:
            print(f"Estimated labels: {est_lab}")
            print(f"Reference labels: {ref_lab}")
            print(f"Scores: {results}")

    # Average scores over the batch
    for key in scores:
        scores[key] /= B

    return scores


def evaluate_batch_decomposed(
    batched_predictions_root: np.ndarray,
    batched_predictions_bass: np.ndarray,
    batched_predictions_chord: np.ndarray,
    batched_onsets: np.ndarray,
    batched_gt_labels: np.ndarray,
    segment_duration: float = 30.0,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
) -> dict:
    """
    Compute chord recognition scores over a batch of predictions and ground truth.

    Args:
        batched_predictions (np.ndarray): shape (B, T,), predicted probabilities.
        batched_onsets (np.ndarray): shape (B, T), onset times for each segment.
        batched_gt_labels (np.ndarray): shape (B, T), ground truth chord labels.
        vocabulary (str): vocabulary type, either "majmin" or "complete" (default
        "majmin").
        segment_duration (float): duration of each segment (default 30s).

    Returns:
        dict: scores from mir_eval (majmin, mirex, sevenths, thirds, triads)
    """
    B = batched_predictions_root.shape[0]

    # initialize empty scores dictionary
    scores = {
        "root": 0.0,
        "majmin": 0.0,
        "mirex": 0.0,
        "sevenths": 0.0,
        "thirds": 0.0,
        "triads": 0.0,
        "tetrads": 0.0,
        "majmin_inv": 0.0,
        "sevenths_inv": 0.0,
        "thirds_inv": 0.0,
        "triads_inv": 0.0,
        "tetrads_inv": 0.0,
        "tbt": 0.0,
        "mechanical": 0.0,
        "mechanical_consonance": 0.0,
    }

    for i in range(B):
        # Estimated
        est_int, est_lab = convert_predictions_decomposed(
            batched_predictions_root[i],
            batched_predictions_bass[i],
            batched_predictions_chord[i],
            segment_duration=segment_duration,
            threshold=0.0,
        )

        # Ground truth
        ref_int, ref_lab = convert_ground_truth(
            batched_onsets[i],
            batched_gt_labels[i],
            segment_duration,
            vocab_path=vocab_path,
        )

        # Compute scores
        results = compute_scores(ref_int, ref_lab, est_int, est_lab)

        # Accumulate scores
        scores["root"] += results["root"]
        scores["majmin"] += results["majmin"]
        scores["mirex"] += results["mirex"]
        scores["sevenths"] += results["sevenths"]
        scores["thirds"] += results["thirds"]
        scores["triads"] += results["triads"]
        scores["tetrads"] += results["tetrads"]
        scores["majmin_inv"] += results["majmin_inv"]
        scores["sevenths_inv"] += results["sevenths_inv"]
        scores["thirds_inv"] += results["thirds_inv"]
        scores["triads_inv"] += results["triads_inv"]
        scores["tetrads_inv"] += results["tetrads_inv"]
        scores["tbt"] += results["tbt"]
        scores["mechanical"] += results["mechanical"]
        scores["mechanical_consonance"] += results["mechanical_consonance"]

        # print an example
        if i in range(10):
            print(f"\nExample {i}:")
            print(f"Estimated intervals: {est_int}")
            print(f"Reference intervals: {ref_int}")
            print(f"Estimated labels: {est_lab}")
            print(f"Reference labels: {ref_lab}")
            print(f"Scores: {results}")

    # Average scores over the batch
    for key in scores:
        scores[key] /= B

    return scores


if __name__ == "__main__":
    c = decode_chord(
        root=0,
        bass=3,
        chord=np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        threshold=0.5,
    )
    print(c)
