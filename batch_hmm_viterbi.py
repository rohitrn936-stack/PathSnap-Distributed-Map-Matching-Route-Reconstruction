
import csv
import os
import math
import glob
import numpy as np

# PATHSNAP - BATCH HMM + VITERBI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train"
)

CANDIDATE_DIR = os.path.join(
    BASE_DIR, "candidate_results"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "batch_hmm"
)

GPS_SIGMA = 30.0
TRANSITION_BETA = 50.0
DISCONNECTED_PENALTY = 5.0


def distance_metres(point1, point2):
    lon1, lat1 = point1
    lon2, lat2 = point2

    mean_lat = math.radians((lat1 + lat2) / 2)

    dx = (lon2 - lon1) * 111320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110540

    return math.hypot(dx, dy)


def emission_log_probability(distance):
    return -0.5 * (distance / GPS_SIGMA) ** 2


def build_arc_midpoints(arcs, nodes):
    start = nodes[arcs[:, 0]]
    end = nodes[arcs[:, 1]]

    return (start + end) / 2


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


def viterbi_decode(track, candidates, arcs, nodes):
    gps_indices = sorted(candidates.keys())

    if not gps_indices:
        raise ValueError("No candidate GPS points found")

    arc_midpoints = build_arc_midpoints(arcs, nodes)

    previous_scores = {}
    backpointers = []

    first_gps = gps_indices[0]

    for arc_index, distance in candidates[first_gps]:
        previous_scores[arc_index] = (
            emission_log_probability(distance)
        )

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

    current_arc = max(
        previous_scores,
        key=previous_scores.get
    )

    selected_arcs = [current_arc]

    for position in range(len(backpointers) - 1, -1, -1):
        current_arc = backpointers[position][current_arc]
        selected_arcs.append(current_arc)

    selected_arcs.reverse()

    return gps_indices, selected_arcs


def process_trip(trip_id, candidate_file):
    trip_dir = os.path.join(TRAIN_DIR, trip_id)

    track_file = os.path.join(
        trip_dir, f"{trip_id}.track"
    )
    arcs_file = os.path.join(
        trip_dir, f"{trip_id}.arcs"
    )
    nodes_file = os.path.join(
        trip_dir, f"{trip_id}.nodes"
    )

    for path in (track_file, arcs_file, nodes_file):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    track = np.loadtxt(
        track_file, usecols=(0, 1), ndmin=2
    )

    arcs = np.loadtxt(
        arcs_file, dtype=int, ndmin=2
    )

    nodes = np.loadtxt(
        nodes_file, ndmin=2
    )

    candidates = load_candidates(candidate_file)

    gps_indices, selected_arcs = viterbi_decode(
        track, candidates, arcs, nodes
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"hmm_reconstructed_{trip_id}.csv"
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

    return len(track), len(selected_arcs), len(set(selected_arcs))


def main():
    print("\nPATHSNAP - BATCH HMM + VITERBI")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_files = sorted(
        glob.glob(
            os.path.join(CANDIDATE_DIR, "candidates_*.csv")
        )
    )

    print("Candidate files found:", len(candidate_files))
    print("-" * 50)

    successful = 0
    failed = 0
    total_selected = 0
    errors = []

    for number, candidate_file in enumerate(candidate_files, 1):
        filename = os.path.basename(candidate_file)
        trip_id = filename.replace(
            "candidates_", ""
        ).replace(".csv", "")

        print(
            f"\n[{number}/{len(candidate_files)}] "
            f"Processing trip {trip_id}"
        )

        try:
            gps_count, selected_count, unique_count = process_trip(
                trip_id, candidate_file
            )

            successful += 1
            total_selected += selected_count

            print("Status: SUCCESS")
            print("GPS points:", gps_count)
            print("Selected points:", selected_count)
            print("Unique selected arcs:", unique_count)

        except Exception as error:
            failed += 1
            errors.append((trip_id, str(error)))

            print("Status: FAILED")
            print("Error:", error)

    print("\n" + "=" * 50)
    print("BATCH HMM SUMMARY")
    print("=" * 50)
    print("Total candidate files:", len(candidate_files))
    print("Successful trips:", successful)
    print("Failed trips:", failed)
    print("Total selected GPS points:", total_selected)
    print("Output folder:", OUTPUT_DIR)

    if errors:
        print("\nFAILED TRIPS:")
        for trip_id, error in errors:
            print(trip_id, ":", error)

    print("\nBatch HMM processing completed.")


if __name__ == "__main__":
    main()
