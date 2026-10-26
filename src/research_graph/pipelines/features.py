"""Feature calculations reused in the notebook experiments."""

import numpy as np

from research_graph.pipelines.preprocessing import USED_INDICES


def raw_and_person_channels(
    filtered: np.ndarray,
    cadence: np.ndarray,
    person_mean: np.ndarray,
    person_std: np.ndarray,
) -> np.ndarray:
    original = np.column_stack((cadence, filtered))
    normalized = (original - person_mean) / person_std
    columns = np.r_[0, 1 + np.array(USED_INDICES)]
    return np.column_stack((original[:, columns], normalized[:, columns]))


def stride_statistics(window: np.ndarray) -> np.ndarray:
    q75, q25 = np.percentile(window, [75, 25], axis=0)
    return np.column_stack(
        (
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            np.sqrt(np.mean(window**2, axis=0)),
            np.median(window, axis=0),
            q75 - q25,
        )
    ).ravel()
