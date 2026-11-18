
import os
import csv
import numpy as np

TRIP_ID = "00000033"

TRIP_FOLDER = os.path.join(
    "data", "Student folder", "train", TRIP_ID
)

CANDIDATE_FILE = os.path.join(
    "candidate_results",
    f"candidates_{TRIP_ID}.csv"
)

NODE_RADIUS = 200
SEGMENT_RADIUS = 100
MAX_CANDIDATES = 10


def project_coordinates(coords, reference_latitude):
    coords = np.asarray(coords, dtype=float)

    x = (
        coords[:, 0]
        * 111320
        * np.cos(np.radians(reference_latitude))
    )

    y = coords[:, 1] * 111320

    return np.column_stack((x, y))


def point_to_segment_distance(point, starts, ends):
    vectors = ends - starts
    lengths = np.sum(vectors ** 2, axis=1)

    t = np.sum(
        (point - starts) * vectors,
        axis=1
    )

    t = np.divide(
        t,
        lengths,
        out=np.zeros_like(t),
        where=lengths != 0
    )

    t = np.clip(t, 0, 1)

    projections = starts + t[:, None] * vectors

    return np.linalg.norm(
        point - projections,
        axis=1
    )


def main():
    gps = np.loadtxt(
        os.path.join(TRIP_FOLDER, TRIP_ID + ".track"),
        usecols=(0, 1),
        ndmin=2
    )

    nodes = np.loadtxt(
        os.path.join(TRIP_FOLDER, TRIP_ID + ".nodes"),
        ndmin=2
    )

    arcs = np.loadtxt(
        os.path.join(TRIP_FOLDER, TRIP_ID + ".arcs"),
        dtype=int,
        ndmin=2
    )

    route = np.loadtxt(
        os.path.join(TRIP_FOLDER, TRIP_ID + ".route"),
        dtype=int,
        ndmin=1
    )

    candidate_map = {}

    with open(CANDIDATE_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            gps_index = int(row["gps_index"])
            arc_index = int(row["arc_index"])

            candidate_map.setdefault(
                gps_index, set()
            ).add(arc_index)

    latitude = np.mean(gps[:, 1])

    gps_xy = project_coordinates(gps, latitude)
    nodes_xy = project_coordinates(nodes, latitude)

    route_arcs = arcs[route]

    route_starts = nodes_xy[route_arcs[:, 0]]
    route_ends = nodes_xy[route_arcs[:, 1]]

    route_start_nodes = route_arcs[:, 0]
    route_end_nodes = route_arcs[:, 1]

    missing = []
    eligible = 0

    for gps_index, point in enumerate(gps_xy):

        distances = point_to_segment_distance(
            point,
            route_starts,
            route_ends
        )

        nearest_position = np.argmin(distances)
        nearest_distance = distances[nearest_position]

        if nearest_distance > SEGMENT_RADIUS:
            continue

        eligible += 1

        reference_arc = int(route[nearest_position])

        if reference_arc in candidate_map.get(gps_index, set()):
            continue

        start_node = int(route_start_nodes[nearest_position])
        end_node = int(route_end_nodes[nearest_position])

        start_distance = np.linalg.norm(
            point - nodes_xy[start_node]
        )

        end_distance = np.linalg.norm(
            point - nodes_xy[end_node]
        )

        candidate_count = len(
            candidate_map.get(gps_index, set())
        )

        missing.append({
            "gps_index": gps_index,
            "reference_arc": reference_arc,
            "reference_distance": nearest_distance,
            "start_node_distance": start_distance,
            "end_node_distance": end_distance,
            "candidate_count": candidate_count,
            "candidate_arc_ids": candidate_map.get(
                gps_index, set()
            )
        })

    endpoint_far = sum(
        min(item["start_node_distance"],
            item["end_node_distance"]) > NODE_RADIUS
        for item in missing
    )

    reference_too_far = sum(
        item["reference_distance"] > SEGMENT_RADIUS
        for item in missing
    )

    top_limit_suspects = sum(
        item["candidate_count"] >= MAX_CANDIDATES
        for item in missing
    )

    print("\n========== CANDIDATE MISS DIAGNOSTICS ==========")
    print("Trip:", TRIP_ID)
    print("Eligible GPS points:", eligible)
    print("Missing reference candidates:", len(missing))

    print("\nPossible reasons:")
    print(
        "Both reference endpoints farther than 200m:",
        endpoint_far
    )
    print(
        "Missing points with 10 existing candidates:",
        top_limit_suspects
    )

    print("\nFirst 15 missing examples:")

    for item in missing[:15]:
        print("\nGPS index:", item["gps_index"])
        print("Reference arc:", item["reference_arc"])
        print(
            "Reference segment distance:",
            round(float(item["reference_distance"]), 2),
            "m"
        )
        print(
            "Start endpoint distance:",
            round(float(item["start_node_distance"]), 2),
            "m"
        )
        print(
            "End endpoint distance:",
            round(float(item["end_node_distance"]), 2),
            "m"
        )
        print("Existing candidates:", item["candidate_count"])

    print("\n===============================================")


if __name__ == "__main__":
    main()
