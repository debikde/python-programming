"""Signal steps shared by the notebooks and the saved model."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.signal import butter, filtfilt  # type: ignore[import-untyped]

SAMPLING_HZ = 100
SOURCE_CHANNELS = (
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "FreeAcc_E",
    "FreeAcc_N",
    "FreeAcc_U",
    "Gyr_X",
    "Gyr_Y",
    "Gyr_Z",
    "Mag_X",
    "Mag_Y",
    "Mag_Z",
    "VelInc_X",
    "VelInc_Y",
    "VelInc_Z",
)
FILTERED_CHANNELS = SOURCE_CHANNELS + ("Accrsn", "FreeAccrsn")
USED_CHANNELS = tuple(
    name for name in FILTERED_CHANNELS if not name.startswith("VelInc")
)
USED_INDICES = tuple(FILTERED_CHANNELS.index(name) for name in USED_CHANNELS)
MIRROR_CHANNELS = ("Acc_Y", "Gyr_X", "Mag_Y", "FreeAcc_N", "VelInc_X")
RIGHT_GROUPS = frozenset(
    {"P3", "P4", "P5", "P6", "P7_TT", "P11", "P13", "P14", "P19", "P20"}
)


def filter_imu(frame: pd.DataFrame, *, right: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return channels before and after the 5 Hz filter."""
    signals = frame.loc[:, list(SOURCE_CHANNELS)].to_numpy(dtype=float, copy=True)
    if not np.isfinite(signals).all():
        raise ValueError("Non-finite IMU signal")
    if right:
        for name in MIRROR_CHANNELS:
            signals[:, SOURCE_CHANNELS.index(name)] *= -1
    acceleration = np.linalg.norm(signals[:, :3], axis=1)
    free_acceleration = np.linalg.norm(signals[:, 3:6], axis=1)
    augmented = np.column_stack((signals, 9.81 - acceleration, -free_acceleration))
    filter_b, filter_a = butter(2, 5, fs=SAMPLING_HZ, btype="low")
    return augmented, filtfilt(filter_b, filter_a, augmented, axis=0)


def cadence_from_steps(steps: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(steps, return_inverse=True, return_counts=True)
    return SAMPLING_HZ / counts[inverse]


def person_mean_and_std(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0)
    standard_deviation = samples.std(axis=0, ddof=1)
    standard_deviation[~np.isfinite(standard_deviation) | (standard_deviation == 0)] = 1
    return mean, standard_deviation
