
import csv
import os
import math
import glob
import numpy as np

# PATHSNAP - Batch Route Reconstruction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train"
)

CANDIDATE_DIR = os.path.join(
    BASE_DIR, "candidate_results"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "batch_reconstructed"
)


def load_data(trip_id):

    trip_dir = os.path.join(TRAIN_DIR, trip_id)

    track = np.loadtxt(
        os.path.join(trip_dir, f"{trip_id}.track"),
        usecols=(0, 1),
        ndmin=2
    )

    arcs = np.loadtxt(
        os.path.join(trip_dir, f"{trip_id}.arcs"),
        dtype=int,
        ndmin=2
    )

    nodes = np.loadtxt(
        os.path.join(trip_dir, f"{trip_id}.nodes"),
        ndmin=2
    )

    return track, arcs, nodes


def load_candidates(candidate_file):

    grouped = {}

    with open(candidate_file, "r", newline="") as file:

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

    for arc_index, emission in candidates[gps_indices[0]]:

        previous_costs[arc_index] = emission

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

    current_arc = min(
        previous_costs,
        key=previous_costs.get
    )

    selected_arcs = [current_arc]

    for position in range(len(backpointers) - 1, -1, -1):

        current_arc = backpointers[position][current_arc]
        selected_arcs.append(current_arc)

    selected_arcs.reverse()

    return gps_indices, selected_arcs


def process_trip(trip_id, candidate_file):

    track, arcs, nodes = load_data(trip_id)

    candidates = load_candidates(candidate_file)

    if not candidates:
        raise ValueError("No candidates available.")

    # Validate candidate references
    for gps_index, records in candidates.items():

        if gps_index < 0 or gps_index >= len(track):
            raise ValueError("Invalid GPS index.")

        for arc_index, distance in records:

            if arc_index < 0 or arc_index >= len(arcs):
                raise ValueError("Invalid arc index.")

            if not np.isfinite(distance):
                raise ValueError("Invalid candidate distance.")

    gps_indices, selected_arcs = reconstruct_route(
        candidates, arcs, nodes
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"reconstructed_{trip_id}.csv"
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

    return {
        "trip_id": trip_id,
        "status": "success",
        "gps_points": len(track),
        "matched_points": len(selected_arcs),
        "unique_arcs": len(set(selected_arcs)),
        "output_file": output_file
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_files = sorted(
        glob.glob(
            os.path.join(
                CANDIDATE_DIR,
                "candidates_*.csv"
            )
        )
    )

    print("\nPATHSNAP - BATCH ROUTE RECONSTRUCTION")
    print("--------------------------------------")
    print("Candidate files found:", len(candidate_files))

    results = []

    for candidate_file in candidate_files:

        filename = os.path.basename(candidate_file)

        trip_id = filename.replace(
            "candidates_", ""
        ).replace(".csv", "")

        print(f"\nProcessing trip: {trip_id}")

        try:

            result = process_trip(
                trip_id,
                candidate_file
            )

            results.append(result)

            print("Status: SUCCESS")
            print("Matched GPS points:", result["matched_points"])
            print("Unique selected arcs:", result["unique_arcs"])

        except Exception as error:

            print("Status: FAILED")
            print("Reason:", str(error))

            results.append({
                "trip_id": trip_id,
                "status": "failed",
                "gps_points": 0,
                "matched_points": 0,
                "unique_arcs": 0,
                "output_file": str(error)
            })

    summary_file = os.path.join(
        OUTPUT_DIR,
        "batch_reconstruction_summary.csv"
    )

    with open(summary_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trip_id",
                "status",
                "gps_points",
                "matched_points",
                "unique_arcs",
                "output_file"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    failed = len(results) - successful

    print("\n--------------------------------------")
    print("BATCH RECONSTRUCTION COMPLETED")
    print("Total trips:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Summary:", summary_file)
    print("Output folder:", OUTPUT_DIR)
    print("--------------------------------------")


if __name__ == "__main__":
    main()
