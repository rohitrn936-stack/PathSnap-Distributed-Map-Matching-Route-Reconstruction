
import os
import glob
import csv
import numpy as np

DATASET_DIR = os.path.join("data", "Student folder", "train")
CANDIDATE_DIR = "candidate_results"
OUTPUT_FILE = os.path.join(
    CANDIDATE_DIR,
    "candidate_validation_summary.csv"
)

MAX_REFERENCE_DISTANCE = 100
CHUNK_SIZE = 500


def project_coordinates(coords, reference_latitude):
    coords = np.asarray(coords, dtype=float)

    x = (
        coords[:, 0]
        * 111320
        * np.cos(np.radians(reference_latitude))
    )

    y = coords[:, 1] * 111320

    return np.column_stack((x, y))


def load_candidates(trip_id):
    candidate_file = os.path.join(
        CANDIDATE_DIR,
        f"candidates_{trip_id}.csv"
    )

    if not os.path.exists(candidate_file):
        raise FileNotFoundError(candidate_file)

    candidate_map = {}

    with open(candidate_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            gps_index = int(row["gps_index"])
            arc_index = int(row["arc_index"])

            candidate_map.setdefault(
                gps_index, set()
            ).add(arc_index)

    return candidate_map


def calculate_nearest_route_arcs(
    gps_xy, nodes_xy, arcs, route
):
    route_arcs = arcs[route]

    starts = nodes_xy[route_arcs[:, 0]]
    ends = nodes_xy[route_arcs[:, 1]]

    segment_vectors = ends - starts

    length_squared = np.sum(
        segment_vectors ** 2,
        axis=1
    )

    nearest_indices = []
    nearest_distances = []

    for start in range(0, len(gps_xy), CHUNK_SIZE):
        points = gps_xy[start:start + CHUNK_SIZE]

        differences = (
            points[:, None, :] - starts[None, :, :]
        )

        numerators = np.sum(
            differences * segment_vectors[None, :, :],
            axis=2
        )

        denominators = np.where(
            length_squared == 0,
            1,
            length_squared
        )

        t = numerators / denominators[None, :]
        t = np.clip(t, 0, 1)

        projections = (
            starts[None, :, :]
            + t[:, :, None] * segment_vectors[None, :, :]
        )

        distances = np.linalg.norm(
            points[:, None, :] - projections,
            axis=2
        )

        zero_length = length_squared == 0

        if np.any(zero_length):
            distances[:, zero_length] = np.linalg.norm(
                points[:, None, :]
                - starts[None, zero_length, :],
                axis=2
            )

        nearest = np.argmin(distances, axis=1)

        nearest_indices.extend(
            route[nearest].tolist()
        )

        nearest_distances.extend(
            distances[np.arange(len(points)), nearest].tolist()
        )

    return (
        np.asarray(nearest_indices, dtype=int),
        np.asarray(nearest_distances, dtype=float)
    )


def validate_trip(track_path):
    trip_id = os.path.splitext(
        os.path.basename(track_path)
    )[0]

    trip_folder = os.path.dirname(track_path)

    nodes_path = os.path.join(
        trip_folder, trip_id + ".nodes"
    )

    arcs_path = os.path.join(
        trip_folder, trip_id + ".arcs"
    )

    route_path = os.path.join(
        trip_folder, trip_id + ".route"
    )

    gps = np.loadtxt(
        track_path,
        usecols=(0, 1),
        ndmin=2
    )

    nodes = np.loadtxt(
        nodes_path,
        ndmin=2
    )

    arcs = np.loadtxt(
        arcs_path,
        dtype=int,
        ndmin=2
    )

    route = np.loadtxt(
        route_path,
        dtype=int,
        ndmin=1
    )

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Invalid node index in road network.")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid arc index in reference route.")

    candidate_map = load_candidates(trip_id)

    reference_latitude = np.mean(gps[:, 1])

    gps_xy = project_coordinates(
        gps, reference_latitude
    )

    nodes_xy = project_coordinates(
        nodes, reference_latitude
    )

    nearest_arcs, nearest_distances = (
        calculate_nearest_route_arcs(
            gps_xy, nodes_xy, arcs, route
        )
    )

    eligible_points = 0
    matched_points = 0

    for gps_index, distance in enumerate(nearest_distances):

        if distance <= MAX_REFERENCE_DISTANCE:
            eligible_points += 1

            if nearest_arcs[gps_index] in candidate_map.get(
                gps_index, set()
            ):
                matched_points += 1

    missing_points = eligible_points - matched_points

    recall = (
        matched_points / eligible_points * 100
        if eligible_points else 0
    )

    points_with_candidates = len(candidate_map)

    return {
        "trip_id": trip_id,
        "status": "SUCCESS",
        "gps_points": len(gps),
        "reference_route_arcs": len(route),
        "points_with_candidates": points_with_candidates,
        "points_without_candidates": (
            len(gps) - points_with_candidates
        ),
        "eligible_points": eligible_points,
        "matched_reference_arcs": matched_points,
        "missing_reference_arcs": missing_points,
        "candidate_recall_percent": round(recall, 2),
        "error": ""
    }


def main():
    track_files = sorted(
        glob.glob(
            os.path.join(DATASET_DIR, "**", "*.track"),
            recursive=True
        )
    )

    print("Total training trips:", len(track_files))

    results = []

    for number, track_path in enumerate(track_files, start=1):
        trip_id = os.path.splitext(
            os.path.basename(track_path)
        )[0]

        print(
            f"\nValidating trip {number}/{len(track_files)}: {trip_id}"
        )

        try:
            result = validate_trip(track_path)
            results.append(result)

            print(
                "Candidate recall:",
                result["candidate_recall_percent"],
                "%"
            )

            print(
                "Matched:",
                result["matched_reference_arcs"],
                "/",
                result["eligible_points"]
            )

        except Exception as error:
            print("ERROR:", error)

            results.append({
                "trip_id": trip_id,
                "status": "FAILED",
                "gps_points": 0,
                "reference_route_arcs": 0,
                "points_with_candidates": 0,
                "points_without_candidates": 0,
                "eligible_points": 0,
                "matched_reference_arcs": 0,
                "missing_reference_arcs": 0,
                "candidate_recall_percent": 0,
                "error": str(error)
            })

    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    successful = [
        row for row in results
        if row["status"] == "SUCCESS"
    ]

    failed = len(results) - len(successful)

    total_eligible = sum(
        row["eligible_points"] for row in successful
    )

    total_matched = sum(
        row["matched_reference_arcs"] for row in successful
    )

    overall_recall = (
        total_matched / total_eligible * 100
        if total_eligible else 0
    )

    print("\n" + "=" * 45)
    print("ALL-TRIP CANDIDATE VALIDATION COMPLETED")
    print("Total trips:", len(results))
    print("Successful:", len(successful))
    print("Failed:", failed)
    print("Total eligible GPS points:", total_eligible)
    print("Total matched reference arcs:", total_matched)
    print("Total missing reference arcs:", total_eligible - total_matched)
    print("Overall candidate recall:", round(overall_recall, 2), "%")
    print("Summary saved to:", OUTPUT_FILE)
    print("=" * 45)

    print(
        "\nNote: Recall is a nearest-reference-arc coverage proxy, "
        "not exact GPS-to-route-point alignment."
    )


if __name__ == "__main__":
    main()
