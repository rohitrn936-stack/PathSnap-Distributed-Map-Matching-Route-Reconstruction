
import csv
import os
import math
import numpy as np

# PATHSNAP - Route Reconstruction
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


def load_data():
    track = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.track"),
        usecols=(0, 1)
    )

    arcs = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.arcs"),
        dtype=int
    )

    nodes = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.nodes")
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


def transition_cost(previous_arc, current_arc, arcs, nodes):

    previous_start, previous_end = arcs[previous_arc]
    current_start, current_end = arcs[current_arc]

    if previous_arc == current_arc:
        return 0.0

    if previous_end == current_start:
        return 1.0

    gap = distance_metres(
        nodes[previous_end],
        nodes[current_start]
    )

    return 10.0 + gap / 10.0


def reconstruct_route(candidates, arcs, nodes):

    gps_indices = sorted(candidates.keys())

    if not gps_indices:
        raise ValueError("No candidate GPS points found.")

    previous_costs = {}
    backpointers = []

    # Initial GPS point
    for arc_index, emission in candidates[gps_indices[0]]:
        previous_costs[arc_index] = emission

    # Dynamic programming
    for position in range(1, len(gps_indices)):

        current_candidates = candidates[gps_indices[position]]

        current_costs = {}
        current_backpointers = {}

        for current_arc, emission in current_candidates:

            best_cost = float("inf")
            best_previous = None

            for previous_arc, previous_cost in previous_costs.items():

                cost = (
                    previous_cost
                    + emission
                    + transition_cost(
                        previous_arc,
                        current_arc,
                        arcs,
                        nodes
                    )
                )

                if cost < best_cost:
                    best_cost = cost
                    best_previous = previous_arc

            current_costs[current_arc] = best_cost
            current_backpointers[current_arc] = best_previous

        backpointers.append(current_backpointers)
        previous_costs = current_costs

    # Find best final candidate
    current_arc = min(
        previous_costs,
        key=previous_costs.get
    )

    selected_arcs = [current_arc]

    # Backtracking
    for position in range(len(backpointers) - 1, -1, -1):
        current_arc = backpointers[position][current_arc]
        selected_arcs.append(current_arc)

    selected_arcs.reverse()

    return gps_indices, selected_arcs


def main():

    print("\nPATHSNAP ROUTE RECONSTRUCTION")
    print("--------------------------------")

    track, arcs, nodes = load_data()
    candidates = load_candidates()

    print("Trip ID:", TRIP_ID)
    print("GPS points:", len(track))
    print("Road arcs:", len(arcs))
    print("Road nodes:", len(nodes))
    print("GPS points with candidates:", len(candidates))

    gps_indices, selected_arcs = reconstruct_route(
        candidates, arcs, nodes
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"reconstructed_{TRIP_ID}.csv"
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

    print("--------------------------------")
    print("RECONSTRUCTION COMPLETED")
    print("Selected GPS points:", len(selected_arcs))
    print("Unique selected arcs:", len(set(selected_arcs)))
    print("Output:", output_file)
    print("--------------------------------")


if __name__ == "__main__":
    main()
