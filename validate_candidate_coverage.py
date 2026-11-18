
import os
import glob
import csv
import numpy as np

# Configuration
DATASET_DIR = os.path.join("data", "Student folder", "train")
TRIP_ID = "00000002"

CANDIDATE_FILE = f"candidates_{TRIP_ID}.csv"
MAX_REFERENCE_DISTANCE = 100


def find_trip_folder():
    pattern = os.path.join(
        DATASET_DIR, "**", TRIP_ID + ".track"
    )

    matches = glob.glob(pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(
            f"Could not find {TRIP_ID}.track"
        )

    return os.path.dirname(matches[0]), matches[0]


def project_coordinates(coords, reference_latitude):
    coords = np.asarray(coords, dtype=float)

    x = coords[:, 0] * 111320 * np.cos(
        np.radians(reference_latitude)
    )

    y = coords[:, 1] * 111320

    return np.column_stack((x, y))


def point_to_segment_distance(point, start, end):
    segment = end - start
    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / length_squared
    t = np.clip(t, 0, 1)

    projection = start + t * segment

    return np.linalg.norm(point - projection)


def load_candidates():
    candidate_map = {}

    if not os.path.exists(CANDIDATE_FILE):
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATE_FILE}"
        )

    with open(CANDIDATE_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            gps_index = int(row["gps_index"])
            arc_index = int(row["arc_index"])

            candidate_map.setdefault(
                gps_index, set()
            ).add(arc_index)

    return candidate_map


def main():
    trip_folder, track_path = find_trip_folder()

    nodes_path = os.path.join(
        trip_folder, TRIP_ID + ".nodes"
    )

    arcs_path = os.path.join(
        trip_folder, TRIP_ID + ".arcs"
    )

    route_path = os.path.join(
        trip_folder, TRIP_ID + ".route"
    )

    gps = np.loadtxt(
        track_path, usecols=(0, 1), ndmin=2
    )

    nodes = np.loadtxt(nodes_path, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)
    route = np.loadtxt(route_path, dtype=int, ndmin=1)

    candidate_map = load_candidates()

    reference_latitude = np.mean(gps[:, 1])

    gps_xy = project_coordinates(
        gps, reference_latitude
    )

    nodes_xy = project_coordinates(
        nodes, reference_latitude
    )

    # Validate route arc indexes
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid arc index in reference route.")

    route_arcs = arcs[route]

    eligible_points = 0
    matched_points = 0
    unmatched_points = 0

    for gps_index, point in enumerate(gps_xy):

        nearest_arc_index = None
        nearest_distance = float("inf")

        for arc_index in route:
            start_node, end_node = arcs[arc_index]

            distance = point_to_segment_distance(
                point,
                nodes_xy[start_node],
                nodes_xy[end_node]
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_arc_index = int(arc_index)

        if nearest_distance <= MAX_REFERENCE_DISTANCE:
            eligible_points += 1

            if nearest_arc_index in candidate_map.get(
                gps_index, set()
            ):
                matched_points += 1
            else:
                unmatched_points += 1

    recall = (
        matched_points / eligible_points * 100
        if eligible_points else 0
    )

    print("\n========== CANDIDATE VALIDATION ==========")
    print("Trip:", TRIP_ID)
    print("Total GPS points:", len(gps))
    print("Reference route arcs:", len(route))
    print(
        "GPS points within 100m of reference route:",
        eligible_points
    )
    print(
        "Reference nearest arcs found in candidate lists:",
        matched_points
    )
    print("Reference nearest arcs missing:", unmatched_points)
    print("Candidate recall:", round(recall, 2), "%")
    print("==========================================")

    print(
        "\nNote: This is a nearest-reference-arc coverage "
        "proxy, not exact GPS-to-route-point alignment."
    )


if __name__ == "__main__":
    main()
