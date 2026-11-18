
import csv
import os
import math
import numpy as np

# PATHSNAP - HMM + VITERBI ROUTE RECONSTRUCTION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRIP_ID = "00000001"

DATA_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train", TRIP_ID
)

CANDIDATE_FILE = os.path.join(
    BASE_DIR, "candidate_results",
    f"candidates_{TRIP_ID}.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# HMM parameters
GPS_SIGMA = 30.0
TRANSITION_BETA = 50.0
DISCONNECTED_PENALTY = 5.0


def load_data():
    track = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.track"),
        usecols=(0, 1),
        ndmin=2
    )

    arcs = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.arcs"),
        dtype=int,
        ndmin=2
    )

    nodes = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.nodes"),
        ndmin=2
    )

    return track, arcs, nodes


def load_candidates():
    grouped = {}

    with open(CANDIDATE_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            gps_index = int(row["gps_index"])
            arc_index = int(row["arc_index"])
            distance = float(row["distance_metres"])

            grouped.setdefault(gps_index, []).append(
                (arc_index, distance)
            )

    return grouped


def distance_metres(point1, point2):
    lon1, lat1 = point1
    lon2, lat2 = point2

    mean_lat = math.radians((lat1 + lat2) / 2)

    dx = (lon2 - lon1) * 111320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110540

    return math.hypot(dx, dy)


def emission_log_probability(distance):
    """
    Gaussian GPS observation model.
    Smaller GPS-to-road distance gives higher probability.
    """
    return -0.5 * (distance / GPS_SIGMA) ** 2


def transition_log_probability(
    previous_arc,
    current_arc,
    previous_gps,
    current_gps,
    track,
    arcs,
    nodes,
    arc_midpoints
):
    """
    Transition model based on:
    1. Difference between GPS movement and road movement.
    2. Road-network connectivity between consecutive arcs.
    """

    if previous_arc == current_arc:
        topology_penalty = 0.0

    else:
        previous_end = arcs[previous_arc][1]
        current_start = arcs[current_arc][0]

        if previous_end == current_start:
            topology_penalty = 0.0
        else:
            gap = distance_metres(
                nodes[previous_end],
                nodes[current_start]
            )
            topology_penalty = (
                DISCONNECTED_PENALTY + gap / TRANSITION_BETA
            )

    gps_distance = distance_metres(
        track[previous_gps],
        track[current_gps]
    )

    road_distance = distance_metres(
        arc_midpoints[previous_arc],
        arc_midpoints[current_arc]
    )

    movement_error = abs(road_distance - gps_distance)

    return (
        -movement_error / TRANSITION_BETA
        - topology_penalty
    )


def build_arc_midpoints(arcs, nodes):
    midpoints = []

    for start_node, end_node in arcs:
        start = nodes[start_node]
        end = nodes[end_node]

        midpoints.append([
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2
        ])

    return np.asarray(midpoints)


def viterbi_decode(track, candidates, arcs, nodes):
    gps_indices = sorted(candidates.keys())

    if not gps_indices:
        raise ValueError("No candidate GPS points found.")

    arc_midpoints = build_arc_midpoints(arcs, nodes)

    # Viterbi state scores and backpointers
    previous_scores = {}
    backpointers = []

    # Initialize first GPS observation
    first_gps = gps_indices[0]

    for arc_index, emission_distance in candidates[first_gps]:
        previous_scores[arc_index] = (
            emission_log_probability(emission_distance)
        )

    # Process remaining GPS observations
    for position in range(1, len(gps_indices)):

        previous_gps = gps_indices[position - 1]
        current_gps = gps_indices[position]

        current_scores = {}
        current_backpointers = {}

        for current_arc, emission_distance in candidates[current_gps]:

            emission_score = emission_log_probability(
                emission_distance
            )

            best_score = -float("inf")
            best_previous_arc = None

            for previous_arc, previous_score in previous_scores.items():

                transition_score = transition_log_probability(
                    previous_arc,
                    current_arc,
                    previous_gps,
                    current_gps,
                    track,
                    arcs,
                    nodes,
                    arc_midpoints
                )

                total_score = (
                    previous_score
                    + transition_score
                    + emission_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_previous_arc = previous_arc

            current_scores[current_arc] = best_score
            current_backpointers[current_arc] = best_previous_arc

        backpointers.append(current_backpointers)
        previous_scores = current_scores

    # Select the most likely final state
    current_arc = max(
        previous_scores,
        key=previous_scores.get
    )

    selected_arcs = [current_arc]

    # Viterbi backtracking
    for position in range(len(backpointers) - 1, -1, -1):
        current_arc = backpointers[position][current_arc]
        selected_arcs.append(current_arc)

    selected_arcs.reverse()

    return gps_indices, selected_arcs


def main():
    print("\nPATHSNAP - HMM + VITERBI")
    print("=" * 40)

    track, arcs, nodes = load_data()
    candidates = load_candidates()

    print("Trip ID:", TRIP_ID)
    print("GPS points:", len(track))
    print("Road arcs:", len(arcs))
    print("Road nodes:", len(nodes))
    print("GPS points with candidates:", len(candidates))

    gps_indices, selected_arcs = viterbi_decode(
        track,
        candidates,
        arcs,
        nodes
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"hmm_reconstructed_{TRIP_ID}.csv"
    )

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "gps_index",
            "selected_arc_index",
            "candidate_distance_metres"
        ])

        for gps_index, arc_index in zip(
            gps_indices, selected_arcs
        ):
            distance = next(
                d for a, d in candidates[gps_index]
                if a == arc_index
            )

            writer.writerow([
                gps_index,
                arc_index,
                distance
            ])

    print("-" * 40)
    print("VITERBI RECONSTRUCTION COMPLETED")
    print("Selected GPS points:", len(selected_arcs))
    print("Unique selected arcs:", len(set(selected_arcs)))
    print("Output:", output_file)
    print("=" * 40)


if __name__ == "__main__":
    main()
