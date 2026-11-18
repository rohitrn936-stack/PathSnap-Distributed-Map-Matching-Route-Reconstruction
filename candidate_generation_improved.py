
import os
import csv
import glob
import numpy as np
from scipy.spatial import cKDTree

# Dataset and output folders
DATASET_DIR = os.path.join("data", "Student folder", "train")
OUTPUT_DIR = "candidate_results_improved"

# Improved experimental configuration
NODE_SEARCH_RADIUS = 500
SEGMENT_SEARCH_RADIUS = 100
MAX_CANDIDATES = 20


def project_coordinates(coords, reference_latitude):
    coords = np.asarray(coords, dtype=float)

    x = (
        coords[:, 0]
        * 111320
        * np.cos(np.radians(reference_latitude))
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


def load_data(track_path):
    trip_id = os.path.splitext(os.path.basename(track_path))[0]
    trip_folder = os.path.dirname(track_path)

    gps = np.loadtxt(
        track_path,
        usecols=(0, 1),
        ndmin=2
    )

    nodes = np.loadtxt(
        os.path.join(trip_folder, trip_id + ".nodes"),
        ndmin=2
    )

    arcs = np.loadtxt(
        os.path.join(trip_folder, trip_id + ".arcs"),
        dtype=int,
        ndmin=2
    )

    return trip_id, gps, nodes, arcs


def generate_candidates(trip_id, gps, nodes, arcs):

    reference_latitude = np.mean(gps[:, 1])

    gps_xy = project_coordinates(gps, reference_latitude)
    nodes_xy = project_coordinates(nodes, reference_latitude)

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Invalid node index in road network.")

    # Build spatial index for road nodes
    tree = cKDTree(nodes_xy)

    # Build node-to-road-segment mapping
    node_to_arcs = {}

    for arc_index, (start_node, end_node) in enumerate(arcs):
        node_to_arcs.setdefault(start_node, []).append(arc_index)
        node_to_arcs.setdefault(end_node, []).append(arc_index)

    candidates = []

    for gps_index, point in enumerate(gps_xy):

        # Search a larger area around each GPS point
        nearby_nodes = tree.query_ball_point(
            point,
            NODE_SEARCH_RADIUS
        )

        nearby_arcs = set()

        for node_id in nearby_nodes:
            nearby_arcs.update(
                node_to_arcs.get(node_id, [])
            )

        point_candidates = []

        for arc_index in nearby_arcs:

            start_node, end_node = arcs[arc_index]

            distance = point_to_segment_distance(
                point,
                nodes_xy[start_node],
                nodes_xy[end_node]
            )

            if distance <= SEGMENT_SEARCH_RADIUS:
                point_candidates.append(
                    (arc_index, distance)
                )

        # Sort by distance
        point_candidates.sort(
            key=lambda item: item[1]
        )

        # Keep up to 20 closest road segments
        for arc_index, distance in point_candidates[:MAX_CANDIDATES]:

            candidates.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "arc_index": arc_index,
                "distance_metres": round(float(distance), 3)
            })

    return candidates


def save_candidates(trip_id, candidates):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"candidates_{trip_id}.csv"
    )

    with open(output_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trip_id",
                "gps_index",
                "arc_index",
                "distance_metres"
            ]
        )

        writer.writeheader()
        writer.writerows(candidates)

    return output_file


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    track_files = sorted(
        glob.glob(
            os.path.join(DATASET_DIR, "**", "*.track"),
            recursive=True
        )
    )

    print("IMPROVED CANDIDATE GENERATION")
    print("Node search radius:", NODE_SEARCH_RADIUS, "metres")
    print("Segment search radius:", SEGMENT_SEARCH_RADIUS, "metres")
    print("Maximum candidates:", MAX_CANDIDATES)
    print("Output folder:", OUTPUT_DIR)

    print("\nTotal training trips found:", len(track_files))

    if not track_files:
        raise FileNotFoundError(
            "No training trips found. Check the dataset folder."
        )

    summary = []

    for number, track_path in enumerate(track_files, start=1):

        trip_id = os.path.splitext(
            os.path.basename(track_path)
        )[0]

        print("\n" + "=" * 45)
        print(
            f"Processing trip {number}/{len(track_files)}: {trip_id}"
        )

        try:

            trip_id, gps, nodes, arcs = load_data(track_path)

            candidates = generate_candidates(
                trip_id,
                gps,
                nodes,
                arcs
            )

            covered_points = len({
                item["gps_index"] for item in candidates
            })

            total_candidates = len(candidates)

            uncovered_points = len(gps) - covered_points

            average = (
                total_candidates / len(gps)
                if len(gps) else 0
            )

            output_file = save_candidates(
                trip_id,
                candidates
            )

            summary.append({
                "trip_id": trip_id,
                "status": "SUCCESS",
                "gps_points": len(gps),
                "road_nodes": len(nodes),
                "road_segments": len(arcs),
                "points_with_candidates": covered_points,
                "points_without_candidates": uncovered_points,
                "total_candidates": total_candidates,
                "average_candidates": round(average, 2),
                "output_file": output_file,
                "error": ""
            })

            print("GPS points:", len(gps))
            print("Points with candidates:", covered_points)
            print("Points without candidates:", uncovered_points)
            print("Total candidates:", total_candidates)
            print("Average candidates:", round(average, 2))
            print("Saved:", output_file)

        except Exception as error:

            print("ERROR:", error)

            summary.append({
                "trip_id": trip_id,
                "status": "FAILED",
                "gps_points": 0,
                "road_nodes": 0,
                "road_segments": 0,
                "points_with_candidates": 0,
                "points_without_candidates": 0,
                "total_candidates": 0,
                "average_candidates": 0,
                "output_file": "",
                "error": str(error)
            })

    summary_file = os.path.join(
        OUTPUT_DIR,
        "candidate_generation_summary.csv"
    )

    with open(summary_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary)

    successful = sum(
        item["status"] == "SUCCESS"
        for item in summary
    )

    failed = len(summary) - successful

    print("\n" + "=" * 45)
    print("IMPROVED CANDIDATE GENERATION COMPLETED")
    print("Total trips:", len(summary))
    print("Successful trips:", successful)
    print("Failed trips:", failed)
    print("Summary saved to:", summary_file)


if __name__ == "__main__":
    main()
