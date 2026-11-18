
"""Convert Member 1's processed trips into the Member 2 ML pipeline format."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline import KEY, load_tables


# Original Member 1 feature names -> Member 2 feature names
CANDIDATE_COLUMNS = {
    "candidateRank": "candidate_rank",
    "distanceMeters": "distance_m",
    "segmentFraction": "segment_fraction",
    "headingDifferenceDegrees": "heading_diff_deg",
    "vehicleSpeedMetersPerSecond": "gps_speed_mps",
    "timeGapSeconds": "time_gap_s",
}

LABEL_COLUMNS = {
    "tripId": "trip_id",
    "partition": "split",
    "observationIndex": "observation_idx",
    "referenceArcId": "reference_arc_id",
}

CANDIDATE_ID_COLUMNS = {
    "tripId": "trip_id",
    "partition": "split",
    "observationIndex": "observation_idx",
    "candidateArcId": "arc_id",
}


def _read_trip_csv(path, string_columns):
    return pd.read_csv(
        path,
        dtype={column: "string" for column in string_columns},
    )


def _convert_features(frame):
    """Create numeric feature columns from Member 1's original columns."""

    for source, target in CANDIDATE_COLUMNS.items():
        if source not in frame.columns:
            raise ValueError(f"Missing feature column: {source}")

        values = frame[source]
        converted = pd.to_numeric(values, errors="coerce")

        invalid = (
            values.notna()
            & values.astype("string").str.strip().ne("")
            & converted.isna()
        )

        if invalid.any():
            raise ValueError(
                f"Non-numeric values in {source}: "
                f"{values[invalid].head().tolist()}"
            )

        if not np.isfinite(
            converted.dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Non-finite values in {source}")

        frame[target] = converted.astype(float)

    return frame


def _trip_paths(source, split):
    directory = source / split

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing split directory: {directory}"
        )

    trips = sorted(
        path for path in directory.iterdir()
        if path.is_dir()
    )

    if not trips:
        raise ValueError(
            f"No trip folders found in {directory}"
        )

    return trips


def prepare(source, out):
    source = Path(source)
    out = Path(out)

    candidate_frames = []
    label_frames = []
    trip_ids_by_split = {}

    for split in ("train", "student_test"):

        trips = _trip_paths(source, split)
        trip_ids_by_split[split] = set()

        print(f"\nProcessing {split}: {len(trips)} trips")

        for trip_dir in trips:

            candidate_path = trip_dir / "candidateRecords.csv"
            label_path = trip_dir / "observationLabels.csv"

            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)

            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            # Read original CSVs
            candidates = _read_trip_csv(
                candidate_path,
                ("tripId", "partition", "candidateArcId"),
            )

            labels = _read_trip_csv(
                label_path,
                ("tripId", "partition", "referenceArcId"),
            )

            # Validate common columns and identifiers
            for frame, path in (
                (candidates, candidate_path),
                (labels, label_path),
            ):

                required = {
                    "tripId",
                    "partition",
                    "observationIndex",
                }

                missing = required - set(frame.columns)

                if missing:
                    raise ValueError(
                        f"{path} missing columns: {sorted(missing)}"
                    )

                if (
                    frame["tripId"].isna().any()
                    or frame["partition"].isna().any()
                ):
                    raise ValueError(
                        f"{path} contains missing trip ID or partition"
                    )

                if set(frame["partition"].dropna()) != {split}:
                    raise ValueError(
                        f"{path} partition does not match {split}"
                    )

                if set(frame["tripId"].dropna()) != {trip_dir.name}:
                    raise ValueError(
                        f"{path} trip ID does not match folder"
                    )

                frame["observationIndex"] = pd.to_numeric(
                    frame["observationIndex"],
                    errors="raise",
                ).astype("int64")

            # Check candidate columns
            required_candidates = (
                set(CANDIDATE_ID_COLUMNS)
                | set(CANDIDATE_COLUMNS)
            )

            missing = required_candidates - set(candidates.columns)

            if missing:
                raise ValueError(
                    f"{candidate_path} missing columns: {sorted(missing)}"
                )

            # Check label columns
            required_labels = (
                set(LABEL_COLUMNS)
                | {"alignmentStatus"}
            )

            missing = required_labels - set(labels.columns)

            if missing:
                raise ValueError(
                    f"{label_path} missing columns: {sorted(missing)}"
                )

            trip_id = trip_dir.name

            if trip_id in trip_ids_by_split[split]:
                raise ValueError(
                    f"Duplicate trip ID in {split}: {trip_id}"
                )

            trip_ids_by_split[split].add(trip_id)

            # Confirm every candidate observation exists in labels
            key_columns = [
                "partition",
                "tripId",
                "observationIndex",
            ]

            all_label_keys = labels[
                key_columns
            ].drop_duplicates()

            candidate_keys = candidates[
                key_columns
            ].drop_duplicates()

            check = candidate_keys.merge(
                all_label_keys,
                on=key_columns,
                how="left",
                indicator=True,
            )

            if (check["_merge"] == "left_only").any():
                raise ValueError(
                    f"{candidate_path} contains observations "
                    "without label records"
                )

            # Keep only aligned observations for supervised learning
            aligned_labels = labels.loc[
                labels["alignmentStatus"].eq("aligned")
            ].copy()

            if aligned_labels["referenceArcId"].isna().any():
                raise ValueError(
                    f"{label_path} has aligned observations "
                    "without reference arc IDs"
                )

            if (
                aligned_labels["referenceArcId"]
                .astype("string")
                .str.strip()
                .eq("")
                .any()
            ):
                raise ValueError(
                    f"{label_path} contains blank reference arc IDs"
                )

            # Keep candidate records belonging to aligned observations
            eligible_keys = aligned_labels[
                key_columns
            ].drop_duplicates()

            candidates = candidates.merge(
                eligible_keys,
                on=key_columns,
                how="inner",
                validate="many_to_one",
            )

            # IMPORTANT:
            # Keep aligned labels even if their correct arc is absent
            # from the candidates. These count as candidate misses.
            candidate_frames.append(candidates)
            label_frames.append(aligned_labels)

    # Combine all trips
    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    labels = pd.concat(
        label_frames,
        ignore_index=True,
    )

    print("\nConverting columns and numerical features...")

    # Rename identifier columns
    candidates = candidates.rename(
        columns=CANDIDATE_ID_COLUMNS
    )

    labels = labels.rename(
        columns=LABEL_COLUMNS
    )

    # FIX:
    # Create numerical feature columns BEFORE selecting them
    candidates = _convert_features(candidates)

    # Select only approved ML features and identifiers
    candidates = candidates[
        [
            "split",
            "trip_id",
            "observation_idx",
            "arc_id",
            *CANDIDATE_COLUMNS.values(),
        ]
    ].copy()

    # Keep ground-truth labels separately
    labels = labels[
        [
            "split",
            "trip_id",
            "observation_idx",
            "reference_arc_id",
        ]
    ].copy()

    # Ensure observation indexes are integers
    candidates["observation_idx"] = candidates[
        "observation_idx"
    ].astype("int64")

    labels["observation_idx"] = labels[
        "observation_idx"
    ].astype("int64")

    # Validate identifiers
    candidate_keys = [
        "split",
        "trip_id",
        "observation_idx",
        "arc_id",
    ]

    if candidates[candidate_keys].isna().any().any():
        raise ValueError(
            "Candidate records contain missing keys or arc IDs"
        )

    if labels[KEY + ["reference_arc_id"]].isna().any().any():
        raise ValueError(
            "Labels contain missing keys or reference arc IDs"
        )

    # Check duplicates
    if candidates.duplicated(candidate_keys).any():
        raise ValueError(
            "Duplicate candidate (split, trip, observation, arc)"
        )

    if labels.duplicated(KEY).any():
        raise ValueError(
            "Duplicate observation labels"
        )

    # Prevent train/test leakage
    if (
        trip_ids_by_split["train"]
        & trip_ids_by_split["student_test"]
    ):
        raise ValueError(
            "Trip IDs overlap between training and testing"
        )

    # Create output folder
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = out / "real_candidates.csv"
    label_path = out / "real_labels.csv"

    # Save converted CSVs
    print("Saving converted datasets...")

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    labels.to_csv(
        label_path,
        index=False,
    )

    # Validate using Member 2 ML pipeline
    print("Validating against ml_pipeline.py...")

    merged, loaded_labels, features = load_tables(
        candidate_path,
        label_path,
    )

    for feature in features:
        if not pd.api.types.is_numeric_dtype(
            merged[feature]
        ):
            raise ValueError(
                f"Non-numeric ML feature detected: {feature}"
            )

    # Print data statistics
    print("\n========== DATASET SUMMARY ==========")

    for split in ("train", "student_test"):

        split_labels = loaded_labels[
            loaded_labels["split"] == split
        ]

        split_candidates = merged[
            merged["split"] == split
        ]

        # Identify correct-road candidate availability
        hits = split_candidates.groupby(
            KEY,
            sort=False,
        )["target"].max()

        labeled_keys = pd.MultiIndex.from_frame(
            split_labels[KEY]
        )

        hits = hits.reindex(
            labeled_keys,
            fill_value=0,
        )

        miss_rate = (
            1.0 - float(hits.mean())
            if len(hits)
            else 0.0
        )

        print(f"\nSplit: {split}")
        print(
            f"Trips: {len(trip_ids_by_split[split])}"
        )
        print(
            f"Candidate rows: {len(split_candidates):,}"
        )
        print(
            f"Labelled observations: {len(split_labels):,}"
        )
        print(
            f"Candidate miss rate: {miss_rate:.4%}"
        )

    print("\nFeatures used:")
    for feature in features:
        print(f"  - {feature}")

    print("\nConversion and validation successful!")

    return candidate_path, label_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    candidate_path, label_path = prepare(
        args.source,
        args.out,
    )

    print(f"\nCreated: {candidate_path}")
    print(f"Created: {label_path}")


if __name__ == "__main__":
    main()
