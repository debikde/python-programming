"""Reproducible descriptive EDA of every imported NTNU CSV; no model training."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from research_graph.sources.ntnu_loader import participant_id, recording_trial_id

CHANNELS = [
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
]
LABELS = ["Steps", "Terrain", "Turn"]
TERRAINS = {
    1: "Flat",
    2: "Grass",
    4: "Stair ascent",
    5: "Stair descent",
    6: "Slope ascent",
    7: "Slope descent",
    8: "Gravel (README)",
    9: "Uneven (README)",
}
FS = 100.0  # Source README assumption; CSV contains no timestamps.
NAME = re.compile(
    r"(?P<course>[A-Za-z]+?)(?P<aid>wi|wo|w0)(?P<repeat>\d+)(?P<sensor>PS|TH|TR|OS|OH)$"
)


def save_table(rows, name, output):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(output / "tables" / f"{name}.csv", index=False)
    return table


def save_figure(fig, name, output):
    fig.savefig(output / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def longest_run(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    return int(lengths.max()) if len(lengths) else 0


def valid_turn_fraction(series):
    """Unknown turn codes are not converted into binary turning observations."""
    valid = series[series.isin([0, 1])]
    return float(valid.mean()) if len(valid) else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ntnu"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/eda_ntnu")
    )
    args = parser.parse_args()
    output = args.output
    for directory in (output / "tables", output / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.data.glob("*/*.csv"))
    if not paths:
        raise ValueError("No participant CSV files found; import the archive first")
    file_rows, trials, stride_rows, windows, pairs, channel_quality = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    label_counts, transitions, spectra, ratio_rows = [], [], [], []
    ps_frames, sensor_arrays = [], defaultdict(list)
    groups, byte_hashes = defaultdict(dict), defaultdict(list)
    trial_signatures = {}
    tr_by_content = defaultdict(list)
    for path in paths:
        if path.stem.endswith("TR"):
            tr_by_content[
                (path.parent.name, hashlib.sha256(path.read_bytes()).hexdigest())
            ].append(path)
    for path in paths:
        match = NAME.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Unrecognized recording name: {path}")
        meta = match.groupdict()
        meta["course_raw"] = meta["course"]
        if meta["course"].lower() != "step":
            meta["course"] = meta["course"].upper()
        meta["aid_raw"] = meta["aid"]
        trial_stem = f"{meta['course']}{meta['aid']}{meta['repeat']}"
        if meta["aid"] == "w0":
            sibling_stem = f"{meta['course']}wo{meta['repeat']}"
            if any(
                (path.parent / f"{sibling_stem}{sensor}.csv").exists()
                for sensor in ("PS", "TH", "TR", "OH")
            ):
                trial_stem = sibling_stem
                meta["aid"] = "wo"
            else:
                meta["aid"] = "unknown_w0"
        meta["grouping_adjustment"] = ""
        if meta["sensor"] == "OS":
            matches = tr_by_content.get(
                (path.parent.name, hashlib.sha256(path.read_bytes()).hexdigest()), []
            )
            if len(matches) == 1:
                candidate = matches[0].stem[:-2]
                if candidate.upper() != trial_stem.upper():
                    meta["grouping_adjustment"] = (
                        f"exact content match: {matches[0].name}"
                    )
                    trial_stem = candidate
        meta["recording_group"] = path.parent.name
        key = (
            participant_id(path.parent.name),
            recording_trial_id(path.parent.name, trial_stem),
        )
        if meta["sensor"] in groups[key]:
            raise ValueError(f"Ambiguous sensor grouping: {key}, {path}")
        groups[key][meta["sensor"]] = (path, meta)
    for index, ((participant, trial_id), sensors) in enumerate(sorted(groups.items())):
        frames = {}
        for sensor, (path, meta) in sorted(sensors.items()):
            raw = pd.read_csv(path)
            if list(raw.columns) != CHANNELS + LABELS:
                raise ValueError(f"Unexpected schema: {path}")
            df = raw.apply(pd.to_numeric, errors="coerce")
            frames[sensor] = df
            values = df[CHANNELS].to_numpy(dtype=float)
            sensor_arrays[sensor].append(values)
            relative = path.relative_to(args.data).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            byte_hashes[digest].append(relative)
            delta_steps = df.Steps.diff().dropna()
            file_rows.append(
                {
                    "path": relative,
                    "participant": participant,
                    "trial": trial_id,
                    **meta,
                    "rows": len(df),
                    "duration_s_assuming_100hz": len(df) / FS,
                    "sha256": digest,
                    "missing_cells": int(raw.isna().sum().sum()),
                    "nonnumeric_cells": int((df.isna() & raw.notna()).sum().sum()),
                    "infinite_cells": int(np.isinf(df.to_numpy()).sum()),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "consecutive_identical_signal_rows": int(
                        np.all(np.diff(values, axis=0) == 0, axis=1).sum()
                    ),
                    "max_frozen_signal_samples": longest_run(
                        np.all(np.diff(values, axis=0) == 0, axis=1)
                    )
                    + 1,
                    "step_decreases": int((delta_steps < 0).sum()),
                    "step_jumps_gt1": int((delta_steps > 1).sum()),
                    "invalid_steps": int(
                        (
                            ~np.isfinite(df.Steps)
                            | (df.Steps < 1)
                            | (df.Steps % 1 != 0)
                        ).sum()
                    ),
                    "invalid_terrain": int((~df.Terrain.isin(TERRAINS)).sum()),
                    "invalid_turn": int((~df.Turn.isin([0, 1])).sum()),
                    "constant_channels": ";".join(
                        df[CHANNELS].columns[df[CHANNELS].nunique() <= 1]
                    ),
                }
            )
            for column in CHANNELS:
                x = df[column].to_numpy()
                finite = x[np.isfinite(x)]
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                iqr = q3 - q1
                channel_quality.append(
                    {
                        "path": relative,
                        "sensor": sensor,
                        "channel": column,
                        "min": finite.min(),
                        "max": finite.max(),
                        "mean": finite.mean(),
                        "std": finite.std(),
                        "zero_fraction": float(np.mean(finite == 0)),
                        "iqr_outlier_count": int(
                            np.sum(
                                (finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)
                            )
                        ),
                        "max_constant_run_samples": longest_run(np.diff(x) == 0) + 1,
                        "extreme_value_fraction": float(
                            max(
                                np.mean(finite == finite.min()),
                                np.mean(finite == finite.max()),
                            )
                        ),
                    }
                )
            acceleration = values[:, :3].ravel()
            increments = values[:, 12:15].ravel()
            valid = np.isfinite(acceleration) & np.isfinite(increments)
            acceleration, increments = acceleration[valid], increments[valid]
            slope = float(acceleration @ increments / (acceleration @ acceleration))
            ratio_rows.append(
                {
                    "participant": participant,
                    "trial": trial_id,
                    "sensor": sensor,
                    "velinc_acc_scale_s": slope,
                    "correlation": float(np.corrcoef(acceleration, increments)[0, 1]),
                    "relative_rmse": float(
                        np.sqrt(np.mean((increments - slope * acceleration) ** 2))
                        / np.sqrt(np.mean(increments**2))
                    ),
                }
            )
        reference = frames.get("PS")
        if reference is None:
            raise ValueError(f"No PS reference for trial: {participant}/{trial_id}")
        meta = sensors["PS"][1]
        signature = tuple(
            (sensor, hashlib.sha256(frame.to_numpy().tobytes()).hexdigest())
            for sensor, frame in sorted(frames.items())
        )
        canonical = trial_signatures.setdefault(signature, f"{participant}/{trial_id}")
        duplicate_trial = canonical != f"{participant}/{trial_id}"
        labels_equal = all(
            len(df) == len(reference) and df[LABELS].equals(reference[LABELS])
            for df in frames.values()
        )
        trials.append(
            {
                "participant": participant,
                "trial": trial_id,
                "recording_group": meta["recording_group"],
                "course": meta["course"],
                "aid": meta["aid"],
                "files": len(frames),
                "sensors": ";".join(sorted(frames)),
                "samples": len(reference),
                "duration_s_assuming_100hz": len(reference) / FS,
                "equal_lengths": len({len(df) for df in frames.values()}) == 1,
                "labels_equal": labels_equal,
                "terrain_classes": reference.Terrain.nunique(),
                "turn_fraction": valid_turn_fraction(reference.Turn),
                "unknown_turn_samples": int((~reference.Turn.isin([0, 1])).sum()),
                "unknown_terrain_samples": int(
                    (~reference.Terrain.isin(TERRAINS)).sum()
                ),
                "filename_anomaly_files": sum(
                    item[1]["aid_raw"] == "w0" for item in sensors.values()
                ),
                "canonical_trial": canonical,
                "duplicate_trial": duplicate_trial,
            }
        )
        for left, right in itertools.combinations(sorted(frames), 2):
            a, b = frames[left], frames[right]
            if len(a) != len(b):
                pairs.append(
                    {
                        "participant": participant,
                        "trial": trial_id,
                        "left": left,
                        "right": right,
                        "same_length": False,
                    }
                )
                continue
            x, y = a[CHANNELS].to_numpy(), b[CHANNELS].to_numpy()
            pairs.append(
                {
                    "participant": participant,
                    "trial": trial_id,
                    "left": left,
                    "right": right,
                    "same_length": True,
                    "all_values_equal": a.equals(b),
                    "signal_values_equal": bool(np.array_equal(x, y, equal_nan=True)),
                    "equal_signal_fraction": float(np.mean(x == y)),
                    "gyr_y_correlation": float(a.Gyr_Y.corr(b.Gyr_Y)),
                }
            )
        if duplicate_trial:
            for sensor in frames:
                sensor_arrays[sensor].pop()
            continue
        ps = reference.assign(
            sample_index=np.arange(len(reference)),
            participant=participant,
            trial=trial_id,
            aid=meta["aid"],
            course=meta["course"],
        )
        ps_frames.append(ps)
        for (terrain, turn), count in (
            reference.groupby(["Terrain", "Turn"]).size().items()
        ):
            label_counts.append(
                {
                    "participant": participant,
                    "trial": trial_id,
                    "course": meta["course"],
                    "aid": meta["aid"],
                    "terrain": int(terrain),
                    "turn": int(turn),
                    "samples": int(count),
                }
            )
        terrain = reference.Terrain.to_numpy()
        for left, right in zip(terrain[:-1], terrain[1:], strict=True):
            if left != right:
                transitions.append(
                    {
                        "participant": participant,
                        "trial": trial_id,
                        "from": int(left),
                        "to": int(right),
                    }
                )
        # Contiguous runs, rather than grouping repeated step IDs across gaps.
        runs = reference.Steps.ne(reference.Steps.shift()).cumsum()
        for _, frame in reference.groupby(runs, sort=False):
            counts = frame.Terrain.value_counts()
            gyr = frame[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
            acc = frame[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
            stride_rows.append(
                {
                    "participant": participant,
                    "trial": trial_id,
                    "aid": meta["aid"],
                    "course": meta["course"],
                    "step_id": frame.Steps.iloc[0],
                    "start_sample": int(frame.index[0]),
                    "end_sample_exclusive": int(frame.index[-1] + 1),
                    "samples": len(frame),
                    "duration_s_assuming_100hz": len(frame) / FS,
                    "terrain": int(counts.index[0]),
                    "terrain_purity": counts.iloc[0] / len(frame),
                    "terrain_classes": frame.Terrain.nunique(),
                    "turn_classes": frame.Turn.nunique(),
                    "turn_fraction": valid_turn_fraction(frame.Turn),
                    "unknown_turn_samples": int((~frame.Turn.isin([0, 1])).sum()),
                    "gyr_norm_rms": float(np.sqrt(np.mean(np.sum(gyr**2, axis=1)))),
                    "acc_norm_rms": float(np.sqrt(np.mean(np.sum(acc**2, axis=1)))),
                    "gyr_y_std": float(frame.Gyr_Y.std(ddof=0)),
                }
            )
        for start in range(0, len(reference) - 200 + 1, 100):
            frame = reference.iloc[start : start + 200]
            counts = frame.Terrain.value_counts()
            windows.append(
                {
                    "participant": participant,
                    "trial": trial_id,
                    "start_sample": start,
                    "terrain": int(counts.index[0]),
                    "purity": counts.iloc[0] / 200,
                    "turn_fraction": valid_turn_fraction(frame.Turn),
                    "unknown_turn_samples": int((~frame.Turn.isin([0, 1])).sum()),
                }
            )
        freq, power = welch(
            reference.Gyr_Y.to_numpy(), fs=FS, nperseg=min(512, len(reference))
        )
        grid = np.linspace(0, 50, 257)
        spectra.append(
            {"participant": participant, "power": np.interp(grid, freq, power)}
        )
        if index % 40 == 0:
            print(f"Analyzed {index + 1}/{len(groups)} trials", flush=True)

    files = save_table(file_rows, "files", output)
    save_table(
        files.loc[
            files.aid_raw.eq("w0")
            | files.grouping_adjustment.ne("")
            | files.course_raw.ne(files.course)
        ],
        "filename_grouping_audit",
        output,
    )
    trials_df = save_table(trials, "trials", output)
    unique_trials = trials_df.loc[~trials_df.duplicate_trial]
    strides = save_table(stride_rows, "strides_ps", output)
    window_df = save_table(windows, "windows_2s_stride1s", output)
    pair_df = save_table(pairs, "sensor_pair_checks", output)
    os_tr_pairs = pair_df.loc[pair_df.left.eq("OS") & pair_df.right.eq("TR")].copy()
    file_hashes = files.set_index(["participant", "trial", "sensor"]).sha256
    os_tr_pairs["byte_identical"] = [
        file_hashes.loc[(row.participant, row.trial, "OS")]
        == file_hashes.loc[(row.participant, row.trial, "TR")]
        for row in os_tr_pairs.itertuples()
    ]
    save_table(os_tr_pairs, "os_tr_identity_by_trial", output)
    save_table(
        os_tr_pairs.groupby("participant", as_index=False).agg(
            compared_trials=("trial", "size"),
            byte_identical_trials=("byte_identical", "sum"),
            value_identical_trials=("all_values_equal", "sum"),
        ),
        "os_tr_identity_by_participant",
        output,
    )
    save_table(channel_quality, "channel_quality_by_file", output)
    ratios = save_table(ratio_rows, "velinc_acc_consistency", output)
    save_table(
        strides.loc[
            (strides.duration_s_assuming_100hz < 0.4)
            | (strides.duration_s_assuming_100hz > 3)
            | (strides.unknown_turn_samples > 0)
            | (~strides.terrain.isin(TERRAINS))
        ],
        "strides_for_review",
        output,
    )
    save_table(
        files.loc[
            (files.invalid_turn > 0)
            | (files.invalid_terrain > 0)
            | (files.step_jumps_gt1 > 0)
            | (files.step_decreases > 0)
            | (files.missing_cells > 0)
        ],
        "files_for_review",
        output,
    )
    registry = trials_df.copy()
    identity = pair_df.loc[
        pair_df.left.eq("OS") & pair_df.right.eq("TR"),
        ["participant", "trial", "all_values_equal"],
    ].rename(columns={"all_values_equal": "os_tr_identity_unresolved"})
    registry = registry.merge(identity, on=["participant", "trial"], how="left")
    registry["os_tr_identity_unresolved"] = registry.os_tr_identity_unresolved.astype(
        "boolean"
    )
    registry = registry.merge(
        ratios.loc[
            ratios.sensor.eq("PS"), ["participant", "trial", "velinc_acc_scale_s"]
        ],
        on=["participant", "trial"],
        how="left",
    )
    # A consistency flag, not an estimate of the actual sampling frequency.
    registry["sampling_scale_review"] = (
        registry.velinc_acc_scale_s - 1 / FS
    ).abs() > 0.0025
    registry["numeric_terrain_labels_only"] = True
    registry["long_stride_runs"] = registry.apply(
        lambda row: int(
            (
                (strides.participant == row.participant)
                & (strides.trial == row.trial)
                & (strides.duration_s_assuming_100hz > 3)
            ).sum()
        ),
        axis=1,
    )
    save_table(registry, "trial_registry_with_qc", output)
    processed = args.data.parent.parent / "processed" / "ntnu"
    processed.mkdir(parents=True, exist_ok=True)
    registry.to_csv(processed / "trial_registry.csv", index=False)
    counts_df = save_table(label_counts, "label_counts_ps", output)
    save_table(
        pd.DataFrame(transitions, columns=["participant", "trial", "from", "to"]),
        "terrain_transitions_ps",
        output,
    )
    duplicate_groups = [group for group in byte_hashes.values() if len(group) > 1]
    (output / "duplicate_groups.json").write_text(
        json.dumps(duplicate_groups, indent=2), encoding="utf-8"
    )
    ps_all = pd.concat(ps_frames, ignore_index=True)
    norms = np.linalg.norm(ps_all[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(), axis=1)
    extremes = ps_all.assign(acc_norm=norms).nlargest(30, "acc_norm")
    save_table(
        extremes[
            [
                "participant",
                "trial",
                "sample_index",
                "Steps",
                "Terrain",
                "Turn",
                "Acc_X",
                "Acc_Y",
                "Acc_Z",
                "acc_norm",
            ]
        ],
        "largest_acceleration_samples_ps",
        output,
    )
    participant_order = sorted(trials_df.participant.unique(), key=lambda p: int(p[1:]))
    summary_rows = []
    for sensor, arrays in sensor_arrays.items():
        matrix = np.concatenate(arrays)
        for j, column in enumerate(CHANNELS):
            x = matrix[:, j]
            x = x[np.isfinite(x)]
            quantiles = np.quantile(x, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
            summary_rows.append(
                {
                    "sensor": sensor,
                    "channel": column,
                    "count": len(x),
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    **dict(
                        zip(
                            ["min", "p01", "p25", "median", "p75", "p99", "max"],
                            quantiles,
                            strict=True,
                        )
                    ),
                }
            )
    save_table(summary_rows, "channel_statistics", output)
    spectral_table = pd.DataFrame({"frequency_hz_assuming_100hz": grid})
    for participant in participant_order:
        spectral_table[participant] = np.mean(
            [
                entry["power"]
                for entry in spectra
                if entry["participant"] == participant
            ],
            axis=0,
        )
    save_table(spectral_table, "welch_psd_by_participant", output)
    participants = unique_trials.groupby("participant").agg(
        trials=("trial", "size"),
        samples=("samples", "sum"),
        seconds=("duration_s_assuming_100hz", "sum"),
    )
    participants["strides"] = strides.groupby("participant").size()
    participants = participants.reindex(participant_order)
    save_table(participants.reset_index(), "participants", output)
    balance = counts_df.groupby("terrain").samples.sum().to_frame()
    balance["seconds"] = balance.samples / FS
    balance["sample_percent"] = balance.samples / balance.samples.sum() * 100
    balance["strides"] = strides.groupby("terrain").size()
    balance["participants"] = counts_df.groupby("terrain").participant.nunique()
    balance["label_readme"] = [TERRAINS.get(i, "Undocumented") for i in balance.index]
    save_table(balance.reset_index(), "class_balance_ps", output)
    support = counts_df.pivot_table(
        index="participant",
        columns="terrain",
        values="samples",
        aggfunc="sum",
        fill_value=0,
    ).reindex(participant_order)
    save_table(support.reset_index(), "participant_terrain_samples", output)
    course_labels = counts_df.pivot_table(
        index="course", columns="terrain", values="samples", aggfunc="sum", fill_value=0
    )
    save_table(course_labels.reset_index(), "filename_vs_terrain", output)
    aid_support = unique_trials.pivot_table(
        index="participant",
        columns="aid",
        values="trial",
        aggfunc="count",
        fill_value=0,
    ).reindex(participant_order)
    save_table(aid_support.reset_index(), "participant_aid_trials", output)
    save_table(
        strides.groupby(["terrain", "aid"])[
            ["duration_s_assuming_100hz", "gyr_norm_rms", "acc_norm_rms"]
        ]
        .agg(["count", "median", "mean", "std"])
        .reset_index(),
        "stride_features_by_class_aid",
        output,
    )
    correlation = ps_all[CHANNELS].corr()
    save_table(correlation.reset_index(), "ps_channel_correlation", output)
    # Leave-one-participant-out feasibility, not a fitted model or accuracy estimate.
    fold_rows = []
    for participant in participant_order:
        train = set(strides.loc[strides.participant != participant, "terrain"])
        test = set(strides.loc[strides.participant == participant, "terrain"])
        fold_rows.append(
            {
                "held_out": participant,
                "train_classes": ";".join(map(str, sorted(train))),
                "test_classes": ";".join(map(str, sorted(test))),
                "test_classes_absent_in_train": ";".join(
                    map(str, sorted(test - train))
                ),
                "classes_absent_in_test": ";".join(
                    map(str, sorted(set(TERRAINS) - test))
                ),
            }
        )
    save_table(fold_rows, "loso_feasibility", output)
    valid_fold_rows = []
    for participant in participant_order:
        valid = strides.loc[strides.terrain.isin(TERRAINS)]
        train = set(valid.loc[valid.participant != participant, "terrain"])
        test = set(valid.loc[valid.participant == participant, "terrain"])
        valid_fold_rows.append(
            {
                "held_out": participant,
                "train_classes": ";".join(map(str, sorted(train))),
                "test_classes": ";".join(map(str, sorted(test))),
                "test_classes_absent_in_train": ";".join(
                    map(str, sorted(test - train))
                ),
                "classes_absent_in_test": ";".join(
                    map(str, sorted(set(TERRAINS) - test))
                ),
            }
        )
    save_table(valid_fold_rows, "loso_feasibility_documented_classes", output)
    import_manifest = json.loads((args.data / "import_manifest.json").read_text())
    summary = {
        "files": len(files),
        "participants": participant_order,
        "recording_groups": sorted(files.recording_group.unique()),
        "trials": len(trials_df),
        "unique_trials": len(unique_trials),
        "duplicate_trials": int(trials_df.duplicate_trial.sum()),
        "unknown_aid_trials": int((~trials_df.aid.isin(["wi", "wo"])).sum()),
        "filename_anomaly_files": int(trials_df.filename_anomaly_files.sum()),
        "sensor_rows": int(files.rows.sum()),
        "ps_samples": len(ps_all),
        "nominal_trial_hours_at_100hz": float(
            unique_trials.duration_s_assuming_100hz.sum() / 3600
        ),
        "strides": len(strides),
        "terrain_counts": {str(k): int(v) for k, v in balance.samples.items()},
        "duplicate_file_groups": len(duplicate_groups),
        "unique_file_hashes": len(byte_hashes),
        "os_tr_equal_trials": int(
            pair_df.loc[
                (pair_df.left == "OS") & (pair_df.right == "TR"), "all_values_equal"
            ].sum()
        ),
        "os_tr_byte_identical_trials": int(os_tr_pairs.byte_identical.sum()),
        "missing_cells": int(files.missing_cells.sum()),
        "nonnumeric_cells": int(files.nonnumeric_cells.sum()),
        "infinite_cells": int(files.infinite_cells.sum()),
        "invalid_terrain_rows": int(files.invalid_terrain.sum()),
        "unknown_terrain_ps_samples": int((~ps_all.Terrain.isin(TERRAINS)).sum()),
        "unknown_terrain_strides": int((~strides.terrain.isin(TERRAINS)).sum()),
        "invalid_turn_rows": int(files.invalid_turn.sum()),
        "invalid_step_rows": int(files.invalid_steps.sum()),
        "duplicate_rows_within_files": int(files.duplicate_rows.sum()),
        "step_decreases": int(files.step_decreases.sum()),
        "step_jumps_gt1": int(files.step_jumps_gt1.sum()),
        "unequal_length_trials": int((~trials_df.equal_lengths).sum()),
        "unequal_label_trials": int((~trials_df.labels_equal).sum()),
        "mixed_terrain_strides": int((strides.terrain_classes > 1).sum()),
        "mixed_turn_strides": int((strides.turn_classes > 1).sum()),
        "nominal_stride_seconds_quantiles": {
            str(k): float(v)
            for k, v in strides.duration_s_assuming_100hz.quantile(
                [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
            ).items()
        },
        "heuristic_short_strides_lt_0_4s": int(
            (strides.duration_s_assuming_100hz < 0.4).sum()
        ),
        "heuristic_long_strides_gt_3s": int(
            (strides.duration_s_assuming_100hz > 3).sum()
        ),
        "turn_sample_fraction_valid_labels": valid_turn_fraction(ps_all.Turn),
        "unknown_turn_ps_samples": int((~ps_all.Turn.isin([0, 1])).sum()),
        "unknown_turn_strides": int((strides.unknown_turn_samples > 0).sum()),
        "windows_2s": len(window_df),
        "mixed_terrain_window_fraction": float((window_df.purity < 1).mean()),
        "windows_below_0_9_purity": int((window_df.purity < 0.9).sum()),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ["numpy", "pandas", "matplotlib", "scipy"]
        },
        "sampling_hz_assumption": FS,
        "source_archive_sha256": import_manifest["source_archive_sha256"],
        "source_archives": [
            {key: item[key] for key in ("name", "sha256", "recording_groups")}
            for item in import_manifest.get("source_archives", [])
        ],
        "dataset_sha256": import_manifest.get("dataset_sha256", ""),
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    (participants.seconds / 60).plot.bar(ax=axes[0], color="#247b92", rot=0)
    axes[0].set(
        title="Recording time by participant (100 Hz assumed)",
        ylabel="Minutes",
        xlabel="",
    )
    aid_support.plot.bar(stacked=True, ax=axes[1], color=["#247b92", "#e7a34b"], rot=0)
    axes[1].set(
        title="Walking-aid coverage: wi=with, wo=without", ylabel="Trials", xlabel=""
    )
    fig.tight_layout()
    save_figure(fig, "01_participant_coverage", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    balance.sample_percent.plot.bar(ax=axes[0], color="#247b92", rot=0)
    axes[0].set(
        title="Class balance (PS reference only)",
        ylabel="% of time samples",
        xlabel="Terrain code",
    )
    balance.strides.plot.bar(ax=axes[1], color="#e7a34b", rot=0)
    axes[1].set(
        title="Stride counts by terrain code",
        ylabel="Contiguous stride runs",
        xlabel="Terrain code",
    )
    fig.tight_layout()
    save_figure(fig, "02_class_balance", output)
    fig, ax = plt.subplots(figsize=(10, max(5, len(participant_order) * 0.42)))
    im = ax.imshow(support / FS / 60, aspect="auto", cmap="YlGnBu")
    ax.set(
        xticks=range(len(support.columns)),
        xticklabels=support.columns,
        yticks=range(len(support)),
        yticklabels=support.index,
        title="Terrain support per participant (minutes at assumed 100 Hz)",
        xlabel="Terrain code",
    )
    for row in range(len(support)):
        for col in range(len(support.columns)):
            ax.text(
                col,
                row,
                f"{support.iloc[row, col] / FS / 60:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white"
                if support.iloc[row, col] > support.to_numpy().max() * 0.55
                else "black",
            )
    fig.colorbar(im, ax=ax, label="Minutes")
    save_figure(fig, "03_participant_terrain", output)
    fig, ax = plt.subplots(figsize=(9, 4))
    normalized = course_labels.div(course_labels.sum(axis=1), axis=0) * 100
    normalized.plot.bar(stacked=True, ax=ax, colormap="tab10", rot=0)
    ax.set(
        title="Filename course versus numeric Terrain label",
        xlabel="Course in filename",
        ylabel="% of samples",
    )
    ax.legend(title="Terrain", bbox_to_anchor=(1.02, 1))
    save_figure(fig, "04_filename_label_audit", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(strides.duration_s_assuming_100hz, bins=60, color="#247b92")
    axes[0].set(
        title="Stride duration (full range; no removal)",
        xlabel="Seconds at assumed 100 Hz",
        ylabel="Strides",
    )
    classes = sorted(strides.terrain.unique())
    axes[1].boxplot(
        [
            strides.loc[strides.terrain == c, "duration_s_assuming_100hz"]
            for c in classes
        ],
        tick_labels=classes,
        showfliers=False,
    )
    axes[1].set(
        title="Stride duration by terrain (fliers hidden only here)",
        xlabel="Terrain code",
        ylabel="Seconds at assumed 100 Hz",
    )
    fig.tight_layout()
    save_figure(fig, "05_stride_duration", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, title in zip(
        axes,
        ["gyr_norm_rms", "acc_norm_rms"],
        ["PS angular-speed RMS (rad/s)", "PS acceleration RMS (m/s², with gravity)"],
        strict=True,
    ):
        ax.boxplot(
            [strides.loc[strides.terrain == c, column] for c in classes],
            tick_labels=classes,
            showfliers=False,
        )
        ax.set(title=title, xlabel="Terrain code")
    fig.tight_layout()
    save_figure(fig, "06_descriptive_signal_features", output)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(
        xticks=range(15),
        xticklabels=CHANNELS,
        yticks=range(15),
        yticklabels=CHANNELS,
        title="PS channel Pearson correlation (all reference samples)",
    )
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right")
    fig.colorbar(im, ax=ax)
    save_figure(fig, "07_channel_correlation", output)
    representative = max(ps_frames, key=lambda frame: frame.Terrain.nunique())
    key = (representative.participant.iloc[0], representative.trial.iloc[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    for sensor, (path, _) in groups[key].items():
        frame = pd.read_csv(path)
        axes[0].plot(
            np.arange(len(frame)) / FS, frame.Gyr_Y, label=sensor, alpha=0.7, lw=0.8
        )
    axes[0].set(
        ylabel="Gyr_Y (rad/s)",
        title=f"Example trial {key[0]}/{key[1]}: maximum terrain diversity",
    )
    axes[0].legend(ncol=4)
    axes[1].step(
        np.arange(len(representative)) / FS,
        representative.Terrain,
        where="post",
        color="#247b92",
    )
    axes[1].set(ylabel="Terrain code")
    axes[2].step(
        np.arange(len(representative)) / FS,
        representative.Turn,
        where="post",
        color="#e7a34b",
    )
    axes[2].set(ylabel="Turn", xlabel="Seconds at assumed 100 Hz")
    fig.tight_layout()
    save_figure(fig, "08_signal_and_labels", output)
    fig, ax = plt.subplots(figsize=(10, 4))
    for participant in participant_order:
        powers = [
            entry["power"] for entry in spectra if entry["participant"] == participant
        ]
        ax.semilogy(grid, np.mean(powers, axis=0), label=participant)
    ax.set(
        xlim=(0, 20),
        title="PS Gyr_Y Welch PSD: equal weight per trial, then participant",
        xlabel="Frequency (Hz, assuming 100 Hz sampling)",
        ylabel="PSD ((rad/s)²/Hz)",
    )
    ax.legend(ncol=4, fontsize=8)
    save_figure(fig, "09_spectral_profiles", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(
        [
            ratios.loc[
                (ratios.participant == p) & (ratios.sensor == "PS"),
                "velinc_acc_scale_s",
            ]
            for p in participant_order
        ],
        tick_labels=participant_order,
    )
    axes[0].axhline(0.01, ls="--", color="gray")
    axes[0].set(
        title="PS: fitted VelInc / Acc scale (not a clock measurement)",
        ylabel="Scale (s)",
    )
    turns = counts_df.groupby(["terrain", "turn"]).samples.sum().unstack(fill_value=0)
    (
        turns.get(1, 0) / turns.reindex(columns=[0, 1], fill_value=0).sum(axis=1) * 100
    ).plot.bar(ax=axes[1], color="#e7a34b", rot=0)
    axes[1].set(
        title="Turning fraction by terrain (valid 0/1 labels only)",
        xlabel="Terrain code",
        ylabel="% of samples",
    )
    fig.tight_layout()
    save_figure(fig, "10_consistency_and_turns", output)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    longest = strides.loc[strides.duration_s_assuming_100hz.idxmax()]
    frame = ps_all.loc[
        (ps_all.participant == longest.participant) & (ps_all.trial == longest.trial)
    ]
    axes[0].plot(frame.sample_index / FS, frame.Gyr_Y, lw=0.7, color="#247b92")
    axes[0].axvspan(
        longest.start_sample / FS,
        longest.end_sample_exclusive / FS,
        color="#e7a34b",
        alpha=0.3,
        label=f"Stride ID {longest.step_id:g}",
    )
    axes[0].set(
        title=f"Long annotated stride: {longest.participant}/{longest.trial}",
        xlabel="Seconds at assumed 100 Hz",
        ylabel="Gyr_Y (rad/s)",
    )
    axes[0].legend()
    peak = extremes.iloc[0]
    frame = ps_all.loc[
        (ps_all.participant == peak.participant)
        & (ps_all.trial == peak.trial)
        & ps_all.sample_index.between(peak.sample_index - 100, peak.sample_index + 100)
    ]
    for channel in ["Acc_X", "Acc_Y", "Acc_Z"]:
        axes[1].plot(frame.sample_index / FS, frame[channel], label=channel, lw=0.9)
    axes[1].set(
        title=f"Largest PS acceleration norm: {peak.participant}/{peak.trial}",
        xlabel="Seconds at assumed 100 Hz",
        ylabel="Acceleration (m/s²)",
    )
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, "11_anomaly_examples", output)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved EDA tables and figures to {output}")


if __name__ == "__main__":
    main()
