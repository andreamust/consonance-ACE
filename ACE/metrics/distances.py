"""
Distances module whih provides a Distance class for measuring distances to be used
in the mechanical distance metric.
"""

import networkx as nx
import numpy as np
from metrics.utils import get_smallest_interval


class Distance:
    """
    Distance objects define a spatial distance measure in semitone space.
    """

    def __init__(
        self, adjacent_step: int = 1, custom_distance: list[int] | None = None
    ):
        """
        Create a new Distance object, either by defining which step should be adjacent,
        or by defining a custom distance List manually.

        Parameters
        ----------
        adjacent_step : int
            How many semitones apart should be adjacent according to this distance.
            For example, to measure distance in (enharmonic) fifths, set this to 7.

            No factor of 12 (greater than 1) will produce a valid, usable Distance
            measure, since they will not assign a difference to every semitone.
            For example, 4 (a minor 3rd) is not valid, since you cannot reach ever
            enharmonic interval via minor thirds (C to D, for example).

            However, the object can still be created, and then combined with another
            Distance object via the combine function.

        custom_distance : List[int]
            Manually define a distance list. This should be a list of length 12,
            and if given supercedes adjacency_step. Invalid distances (which must be
            combined with another Distance object) should be assigned -1. The th element
            represents the distance to the pitch class up i semitones.
        """
        if custom_distance is None:
            distance_up = [0] + [-1] * 11
            idx = adjacent_step % 12
            steps = 1
            while distance_up[idx] == -1:
                distance_up[idx] = steps
                steps += 1
                idx = (idx + adjacent_step) % 12

            distance_down = [0] + [-1] * 11
            idx = -adjacent_step % 12
            steps = 1
            while distance_down[idx] == -1:
                distance_down[idx] = steps
                steps += 1
                idx = (idx - adjacent_step) % 12

            self.distance = (
                Distance(custom_distance=distance_up)
                .combine(Distance(custom_distance=distance_down))
                .distance
            )

        else:
            assert len(custom_distance) == 12, "custom_distance must be of length 12."
            self.distance = custom_distance

        self.custom_distance = custom_distance

    def is_valid(self) -> bool:
        """
        Check whether this Distance object is valid for measuring distances. A valid
        Distance object must have all distances >= 0.

        Returns
        -------
        is_valid : bool
            True if this Distance object is valid. False otherwise.
        """
        return min(self.distance) >= 0

    def combine(self, other_distance: "Distance") -> "Distance":
        """
        Combine this Distance object with another, returning a 3rd whose distances are
        the minimums of the distances from each Distance object at each step.

        Parameters
        ----------
        other_distance : Distance
            The Distance object to combine with.

        Returns
        -------
        distance : Distance
            A new Distance object, whose distance at each step is the minimum of
            self.distance and other_distance.distance.
        """
        return Distance(
            custom_distance=[
                min(d1, d2) if -1 not in [d1, d2] else max(d1, d2)
                for d1, d2 in zip(self.distance, other_distance.distance)
            ]
        )

    def distance_between(self, pitch1: int, pitch2: int) -> int:
        """
        Get the distance between two pitches according to this Distance object.

        Parameters
        ----------
        pitch1 : int
            The first pitch.
        pitch2 : int
            The second pitch.

        Returns
        -------
        distance : int
            The distance between pitch1 and pitch2.
        """
        return self.distance[get_smallest_interval(pitch1, pitch2)]

    def __hash__(self) -> int:
        return tuple(self.distance).__hash__()


def find_notes_matching(
    notes1: np.ndarray, notes2: np.ndarray, distance: "Distance"
) -> tuple[int, list]:
    """
    Find the smallest distance between two lists of notes.

    Parameters
    ----------
    notes1 : np.ndarray
        List of notes.
    notes2 : np.ndarray
        List of notes.
    distance : Distance
        The distance metric to use.

    Returns
    -------
    total_steps : int
        The minimum length between the two sets of notes according to distance metric.
    matching : List[Tuple[int, int]]
        A List of matched pairs of ids from notes1 and notes2.

    """
    # Matrix that keep the distance between each pair of notes
    similarities = np.zeros((len(notes1), len(notes2)))

    for idx1, pitch1 in enumerate(notes1):
        for idx2, pitch2 in enumerate(notes2):
            similarities[idx1, idx2] = distance.distance_between(pitch1, pitch2)

    # Graph for the matching
    G = nx.Graph()
    # The second node ID is the length of the first chord plus its index
    G.add_weighted_edges_from([
        (idx1, idx2 + len(notes1), similarities[idx1, idx2])
        for idx1 in range(len(notes1))
        for idx2 in range(len(notes2))
    ])

    matching = nx.bipartite.minimum_weight_full_matching(G)

    # Subtract the length of the first chord from the 2nd node IDs
    match_ids = [
        ((pair[0], pair[1] - len(notes1)))
        for pair in matching.items()
        if pair[0] < pair[1]
    ]

    # Total distance
    total_steps = int(sum([similarities[pair[0], pair[1]] for pair in match_ids]))

    return total_steps, match_ids


TONAL_VECTOR = [0, 7, 5, 1, 1, 2, 3, 1, 2, 2, 4, 6]

LINEAR_DISTANCE = Distance()

GCT_DISTANCE = Distance(
    custom_distance=TONAL_VECTOR,
)
