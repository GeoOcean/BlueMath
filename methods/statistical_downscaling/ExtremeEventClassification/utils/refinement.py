from typing import Any, List

import numpy as np


def filter_clusters_by_removal_and_centroid(
    merged_cluster_labels: List[List[int]],
    hydrographs_list: List[np.ndarray],
    md_reduced: Any,
    threshold: float = 1.0,
) -> List[List[int]]:
    """
    Filter clusters by removing unwanted indices and applying centroid threshold criteria.

    Parameters
    ----------
    merged_cluster_labels : list of list of int
        List of lists, where each sublist contains indices of a cluster.
    hydrographs_list : list of np.ndarray
        List of hydrograph series per station.
    md_reduced : Any
        Pandas DataFrame or Series containing original indices of data points.
    threshold : float, optional
        Minimum required max value in cluster centroid to keep it. Default is 1.0.

    Returns
    -------
    list of list of int
        Filtered cluster labels after removal and threshold filtering.
    """

    array_NTR = np.concatenate(hydrographs_list)

    labels_to_remove = [
        64,
        65,
        197,
        171,
        170,
        111,
        258,
        252,
        104,
        84,
        267,
        278,
        123,
        56,
        79,
        59,
        305,
        224,
        11,
        72,
        208,
        244,
        309,
        286,
        282,
        142,
        235,
        127,
        233,
        307,
        14,
        185,
        263,
        228,
        88,
        108,
        300,
        95,
        239,
        159,
        85,
        223,
        124,
        241,
        168,
        94,
        156,
        52,
        181,
        77,
        152,
        120,
        270,
        162,
        274,
        187,
        245,
        251,
        198,
        163,
        100,
        238,
        86,
    ]

    # Remove clusters listed in labels_to_remove
    filtered_clusters_labels = [
        label
        for index, label in enumerate(merged_cluster_labels)
        if index not in labels_to_remove
    ]

    # Compute centroids and filter by threshold
    labels_max_threshold = []
    for cluster_indices in filtered_clusters_labels:
        mapped_indices = [md_reduced.index[index] for index in cluster_indices]
        cluster_data = array_NTR[mapped_indices]
        centroid = np.median(cluster_data, axis=0)
        if np.max(centroid) > threshold:
            labels_max_threshold.append(cluster_indices)

    return labels_max_threshold


def manual_merge_step_one(filtered_clusters: List[List[int]]) -> List[List[int]]:
    """
    Perform first manual merge step by combining specified cluster pairs.

    Parameters
    ----------
    filtered_clusters : list of list of int
        List of cluster labels to merge.

    Returns
    -------
    list of list of int
        Merged cluster labels after step one.
    """

    labels = filtered_clusters.copy()

    merge_map = [
        (75, 87),
        (16, 64),
        (47, 82),
        (47, 72),
        (21, 41),
        (23, 50),
        (26, 18),
        (33, 60),
        (55, 73),
        (40, 79),
        (20, 159),
        (1, 120),
        (35, 93),
        (48, 156),
        (12, 5),
        (9, 87),
        (9, 153),
        (9, 165),
        (61, 89),
        (52, 74),
        (59, 99),
        (62, 102),
        (63, 113),
        (83, 162),
        (94, 111),
        (92, 121),
        (100, 158),
        (100, 71),
        (106, 151),
        (2, 128),
        (98, 137),
        (11, 157),
        (15, 162),
    ]

    for target, source in merge_map:
        if source < len(labels) and target < len(labels):
            labels[target].extend(labels[source])
            labels[source] = []

    return [label for label in labels if label]


def manual_merge_step_two(filtered_clusters: List[List[int]]) -> List[List[int]]:
    """
    Perform second manual merge step by combining specified cluster pairs.

    Parameters
    ----------
    filtered_clusters : list of list of int
        List of cluster labels to merge.

    Returns
    -------
    list of list of int
        Merged cluster labels after step two.
    """

    labels = filtered_clusters.copy()

    merge_map = [
        (65, 54),
        (91, 105),
        (26, 27),
        (62, 18),
        (3, 16),
        (5, 31),
        (29, 97),
        (12, 52),
        (11, 116),
        (44, 45),
        (44, 42),
        (118, 134),
        (118, 14),
        (69, 2),
        (69, 9),
        (7, 24),
        (7, 122),
    ]

    for target, source in merge_map:
        if source < len(labels) and target < len(labels):
            labels[target].extend(labels[source])
            labels[source] = []

    return [label for label in labels if label]


def manual_merge_step_three(filtered_clusters: List[List[int]]) -> List[List[int]]:
    """
    Perform third manual merge step by combining specified cluster pairs.

    Parameters
    ----------
    filtered_clusters : list of list of int
        List of cluster labels to merge.

    Returns
    -------
    list of list of int
        Merged cluster labels after step three.
    """

    labels = filtered_clusters.copy()

    merge_map = [
        (75, 97),
        (79, 84),
        (2, 28),
        (3, 103),
        (17, 30),
        (7, 22),
        (4, 87),
        (4, 16),
        (38, 85),
        (38, 55),
    ]

    for target, source in merge_map:
        if source < len(labels) and target < len(labels):
            labels[target].extend(labels[source])
            labels[source] = []

    return [label for label in labels if label]
