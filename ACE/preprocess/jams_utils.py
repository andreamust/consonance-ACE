"""
Module containing functions to preprocess JAMS files.

Functions:
    _get_jams_annotation: Loads a JAMS file from the given file path and extracts
        the chord annotations from it.
    _recompute_duration: Recomputes the duration of the given annotation by
        taking into account the onsets of the chords.
    _remove_duplicates: Removes consecutive duplicate chords from the given
        annotation.
    _array_to_annotation: Creates a JAMS annotation object from the given arrays.
    _x_to_n: Replaces all "X" values in the given array with "N".
    preprocess_jams: Preprocesses a given JAMS file by removing duplicate chords
        and taking care of the chord durations and onsets. Moreover, it converts
        any "X" chord to "N".
    _right_pad_annotation: Pads the given annotation to the given duration.
    _left_pad_annotation: Checks if the times array starts from 0 and, if not,
        pads the array with a "N" chord.
    _remove_short_chords: Removes chords shorter than the given threshold.
    preprocess_jams: Preprocesses a given JAMS file, polishes the observations contained
        in the annotation, and returns a new JAMS annotation object. If no chord
        annotations are found in the given JAMS file, it returns None.
    trim_jams: Trims a given JAMS Annotation to the given start and duration.
"""

from pathlib import Path

import jams
import numpy as np
from music21 import pitch


def _get_jams_annotation(jams_path: Path | str) -> jams.Annotation | None:
    """
    Loads a JAMS file from the given file path and extracts the chord
    annotations from it.

    Args:
        jams_path (Path | str): The path to the JAMS file.

    Returns:
        jams.Annotation: A JAMS annotation object.

    Raises:
        ValueError: If no chord annotations are found in the given JAMS file.
    """
    jam = jams.load(str(jams_path), strict=False, validate=False)
    chord_namespaces = ["chord", "chord_harte"]
    annotation = None
    for namespace in chord_namespaces:
        annotation = jam.search(namespace=namespace)
        if annotation:
            return annotation[0]  # type: ignore
    if annotation is None:
        raise ValueError(f"No chord annotations found in {jams_path}")


def _recompute_duration(start_times, annotation) -> np.ndarray:
    """
    Recomputes the duration of the given annotation by taking into account the
    onsets of the chords.

    Args:
        annotation (jams.Annotation): The annotation to recompute the duration of.

    Returns:
        jams.Annotation: The annotation with the recomputed duration.
    """
    # get the duration of the last chord
    last_duration = annotation.data[-1].duration  # type: ignore
    last_time = start_times[-1]
    last_duration = (
        annotation.duration - last_time if annotation.duration else last_duration
    )
    # duration[i] = start_times[i + 1] - start_times[i]
    duration = np.diff(start_times)
    # add the last duration
    duration = np.append(duration, last_duration)

    return duration


def _remove_duplicates(annotation: jams.Annotation) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes consecutive duplicate chords from the given annotation.

    Args:
        annotation (jams.Annotation): The annotation to remove duplicates from.

    Returns:
        jams.Annotation: The annotation without consecutive duplicates.
    """
    # get all sequences from the annotation
    start_times, values = annotation.to_event_values()

    # remove consecutive duplicates
    clean_values, clean_start_times = [], []
    for i, value in enumerate(values):
        if i != 0 and value == values[i - 1]:
            continue
        clean_values.append(value)
        clean_start_times.append(start_times[i])

    assert len(clean_values) == len(clean_start_times)

    return np.array(clean_start_times), np.array(clean_values)


def _array_to_annotation(
    times: np.ndarray, duration: np.ndarray, values: np.ndarray
) -> jams.Annotation:
    """
    Creates a JAMS annotation object from the given arrays.

    Args:
        times (np.ndarray): The array of times.
        duration (np.ndarray): The array of durations.
        values (np.ndarray): The array of values.

    Returns:
        jams.Annotation: A JAMS annotation object.
    """
    # create data as a list of observations
    data = [
        jams.Observation(
            time=times[i], duration=duration[i], value=values[i], confidence=1.0
        )
        for i in range(len(times))
    ]

    # create a new annotation
    new_annotation = jams.Annotation(
        namespace="chord",
        time=times[0],
        duration=times[-1] + duration[-1],
        data=data,
    )

    return new_annotation


def _x_to_n(values: np.ndarray) -> np.ndarray:
    """
    Replaces all "X" values in the given array with "N".

    Args:
        values (np.ndarray): The array of values.

    Returns:
        np.ndarray: The array of values with "X" replaced by "N".
    """
    return np.array([value if value != "X" else "N" for value in values])


def _right_pad_annotation(
    annotation: jams.Annotation,
    start_pad: float,
    duration_pad: float,
) -> jams.Annotation:
    """
    Pads the given annotation to the given duration.

    Args:
        annotation (jams.Annotation): The annotation to pad.
        duration (float): The duration to pad to.

    Returns:
        jams.Annotation: A JAMS annotation object.
    """
    # if last chord is "N", extend it
    if annotation.data[-1].value == "N":  # type: ignore
        n_chord: jams.Observation = annotation.data[-1]  # type: ignore
        n_chord = n_chord._replace(duration=duration_pad + n_chord.duration)
        annotation.data = annotation.data[:-1] + [n_chord]  # type: ignore
        annotation.duration = annotation.duration + duration_pad  # type: ignore
        return annotation

    # otherwise, add a new "N" chord
    annotation.append(time=start_pad, duration=duration_pad, value="N", confidence=1.0)
    annotation.duration = annotation.duration + duration_pad  # type: ignore

    return annotation


def _left_pad_annotation(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Checks if the times array starts from 0 and, if not, pads the array with a
    "N" chord.

    Args:
        values (np.ndarray): The array of values.
        times (np.ndarray): The array of times.

    Returns:
        tuple[np.ndarray, np.ndarray]: The padded arrays.
    """
    # if the first chord is not at time 0, add a "N" chord
    if times[0] != 0:
        values = np.insert(values, 0, "N")
        times = np.insert(times, 0, 0.0)

    return times, values


def _remove_short_chords(
    values: np.ndarray,
    times: np.ndarray,
    duration: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes chords shorter than the given threshold.

    Args:
        values (np.ndarray): The array of values.
        times (np.ndarray): The array of times.
        duration (np.ndarray): The array of durations.
        threshold (float): The minimum duration of a chord.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The arrays without the short chords.
    """
    # get the indices of the chords shorter than the threshold
    short_chords = np.where(duration < threshold)[0]

    # remove the short chords
    values = np.delete(values, short_chords)
    times = np.delete(times, short_chords)
    duration = np.delete(duration, short_chords)

    return values, times, duration


def preprocess_jams(jams_path: Path) -> jams.Annotation:
    """
    Preprocesses a given JAMS file by removing duplicate chords and taking care
    of the chord durations and onsets. Moreover, it converts any "X" chord to
    "N".
    It returns a JAMS annotation object.

    Args:
        jams_path (Path): The path to the JAMS file.

    Returns:
        jams.Annotation: A JAMS annotation object.

    Raises:
        ValueError: If no chord annotations are found in the given JAMS file.
    """
    annotation = _get_jams_annotation(jams_path)
    # handle possible absence of chord annotations
    if not annotation or len(annotation) == 0:
        raise ValueError(f"No chord annotations found in {jams_path}")
    # remove consecutive duplicates and transforms Annotation to a numpy array
    times, values = _remove_duplicates(annotation)
    times, values = _left_pad_annotation(times, values)

    # recomputes the duration of the annotation (fixes bugs in some datasets)
    duration = _recompute_duration(times, annotation)
    # # remove short chords
    # values, times, duration = _remove_short_chords(
    #     values, times, duration, threshold=0.01
    # )

    # # replace "X" with "N"
    # values = _x_to_n(values)

    assert len(times) == len(duration) == len(values)

    # create data as a list of observations
    new_annotation = _array_to_annotation(times, duration, values)

    return new_annotation


def trim_jams(
    jams_annotation: jams.Annotation, start: float, duration: float
) -> jams.Annotation | None:
    """
    Trims a given JAMS Annotation to the given start and duration.
    It returns a new JAMS annotation object if the trimmed annotation is not empty.
    If the trimmed annotation is empty, it returns None.

    Args:
        jams_annotation (jams.Annotation): The JAMS annotation to trim.
        start (float): The start time of the trimmed annotation.
        duration (float): The duration of the trimmed annotation.

    Returns:
        jams.Annotation | None: A JAMS annotation object or None.
    """
    end: float = start + duration
    assert start >= 0 and duration > 0, "Start and duration must be positive"

    annotation_duration: float = float(jams_annotation.duration)  # type: ignore
    # check if the annotation is longer than the given start
    if start >= annotation_duration:
        return None
    # check if the annotation ends before the given start
    elif end >= annotation_duration:
        # pad the annotation adding "N" with the left duration
        pad_duration = end - annotation_duration
        jams_annotation = _right_pad_annotation(
            jams_annotation, start_pad=annotation_duration, duration_pad=pad_duration
        )

    # trim the annotation
    trimmed = jams_annotation.slice(start_time=start, end_time=end, strict=False)

    return trimmed


def transpose_chord(chord: str, augment: int = 0) -> str:
    """
    Transpose the chord by a given number of semitones.

    Args:
        chord (str): The chord to transpose.
        augment (int): The number of semitones to augment the chord by.

    Returns:
        str: The transposed chord.
    """
    augment = int(augment)

    if chord == "N" or chord == "X":
        return chord
    elif ":" in chord:
        root, etc = chord.split(":")
        etc = f":{etc}"
    elif "/" in chord:
        root, etc = chord.split("/")
        etc = f"/{etc}"
    else:
        root = chord
        etc = ""

    root = str(root).replace("b", "-")
    m21_root = pitch.Pitch(root)
    m21_root.transpose(augment, inPlace=True)
    return f"{m21_root.name.replace('-', 'b')}{etc}"


def transpose_annotation(
    annotation: jams.Annotation,
    augment: int = 0,
) -> jams.Annotation:
    """
    Transpose the annotation by a given number of semitones.

    Args:
        annotation (jams.Annotation): The annotation to transpose.
        augment (int): The number of semitones to augment the annotation by.

    Returns:
        jams.Annotation: The transposed annotation.
    """
    # Create new annotation with same metadata
    new_annotation = jams.Annotation(namespace=annotation.namespace)
    new_annotation.duration = annotation.duration

    # Add observations using JAMS methods
    for data in annotation.data:
        new_annotation.append(
            time=data.time,
            duration=data.duration,
            value=transpose_chord(data.value, augment),
            confidence=data.confidence,
        )

    return new_annotation


if __name__ == "__main__":
    jams_path = "/media/data/andrea/choco_audio/jams/jaah_16.jams"

    jam = preprocess_jams(Path(jams_path))
    print(jam.to_interval_values())
    # print(jam)
    trim_jams = trim_jams(jam, 250, 15)  # type: ignore
    dc = trim_jams.to_interval_values() if trim_jams else None  # type: ignore
    print(dc)
