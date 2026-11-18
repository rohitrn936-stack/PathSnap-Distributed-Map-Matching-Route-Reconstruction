
import csv
import os
import glob
from collections import deque

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train"
)

INPUT_DIR = os.path.join(
    BASE_DIR, "results", "batch_reconstructed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "batch_connected"
)

MAX_DEPTH = 200


def load_data(trip_id, input_file):

    data_dir = os.path.join(TRAIN_DIR, trip_id)

    arcs = np.loadtxt(
        os.path.join(data_dir, f"{trip_id}.arcs"),
        dtype=int,
        ndmin=2
    )

    selected = []

    with open(input_file, newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:
            selected.append(
                int(row["selected_arc_index"])
            )

    return arcs, selected


def build_graph(arcs):

    graph = {}

    for arc_index, (start, end) in enumerate(arcs):

        graph.setdefault(start, []).append(
            (end, arc_index)
        )

    return graph


def find_connecting_path(graph, start_node, target_node):

    if start_node == target_node:
        return []

    queue = deque([(start_node, 0)])

    visited = {start_node}
    parent = {}

    while queue:

        node, depth = queue.popleft()

        if depth >= MAX_DEPTH:
            continue

        for next_node, arc_index in graph.get(node, []):

            if next_node in visited:
                continue

            visited.add(next_node)
            parent[next_node] = (node, arc_index)

            if next_node == target_node:

                path = []
                current = target_node

                while current != start_node:

                    previous, used_arc = parent[current]

                    path.append(used_arc)
                    current = previous

                path.reverse()

                return path

            queue.append((next_node, depth + 1))

    return None


def process_trip(trip_id, input_file):

    arcs, selected = load_data(
        trip_id,
        input_file
    )

    if not selected:
        raise ValueError("No selected arcs found.")

    if any(
        arc < 0 or arc >= len(arcs)
        for arc in selected
    ):
        raise ValueError("Invalid selected arc index.")

    graph = build_graph(arcs)

    connected_route = []
    unresolved = []

    inserted_count = 0

    connected_route.append(
        (selected[0], False, 0)
    )

    for position in range(1, len(selected)):

        previous_arc = selected[position - 1]
        current_arc = selected[position]

        previous_end = arcs[previous_arc, 1]
        current_start = arcs[current_arc, 0]

        if previous_arc == current_arc:

            connected_route.append(
                (current_arc, False, position)
            )

            continue

        if previous_end == current_start:

            connected_route.append(
                (current_arc, False, position)
            )

            continue

        path = find_connecting_path(
            graph,
            previous_end,
            current_start
        )

        if path is None:

            unresolved.append({
                "trip_id": trip_id,
                "gps_transition": position - 1,
                "previous_arc": previous_arc,
                "current_arc": current_arc
            })

        else:

            for intermediate_arc in path:

                connected_route.append(
                    (intermediate_arc, True, position)
                )

                inserted_count += 1

        connected_route.append(
            (current_arc, False, position)
        )

    # Verify final route
    same_arc = 0
    connected = 0
    disconnected = 0

    for i in range(len(connected_route) - 1):

        first_arc = connected_route[i][0]
        second_arc = connected_route[i + 1][0]

        if first_arc == second_arc:

            same_arc += 1

        elif arcs[first_arc, 1] == arcs[second_arc, 0]:

            connected += 1

        else:

            disconnected += 1

    total_transitions = max(
        len(connected_route) - 1, 0
    )

    connectivity = (
        (same_arc + connected) / total_transitions * 100
        if total_transitions else 100.0
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"connected_route_{trip_id}.csv"
    )

    with open(output_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "route_position",
            "arc_index",
            "inserted_intermediate",
            "source_gps_transition"
        ])

        for route_position, item in enumerate(connected_route):

            arc_index, inserted, source_position = item

            writer.writerow([
                route_position,
                arc_index,
                inserted,
                source_position
            ])

    return {
        "trip_id": trip_id,
        "status": "success",
        "original_arcs": len(selected),
        "inserted_arcs": inserted_count,
        "final_arcs": len(connected_route),
        "same_arc_transitions": same_arc,
        "connected_transitions": connected,
        "disconnected_transitions": disconnected,
        "unresolved_gps_transitions": len(unresolved),
        "connectivity_percent": round(connectivity, 2),
        "output_file": output_file,
        "unresolved": unresolved
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "reconstructed_*.csv"
            )
        )
    )

    print("\nPATHSNAP - BATCH CONNECTED ROUTE PROCESSING")
    print("---------------------------------------------")
    print("Reconstructed files found:", len(input_files))

    results = []
    all_unresolved = []

    for input_file in input_files:

        filename = os.path.basename(input_file)

        trip_id = filename.replace(
            "reconstructed_", ""
        ).replace(".csv", "")

        print("\nProcessing trip:", trip_id)

        try:

            result = process_trip(
                trip_id,
                input_file
            )

            all_unresolved.extend(
                result["unresolved"]
            )

            summary = {
                key: value
                for key, value in result.items()
                if key not in ("output_file", "unresolved")
            }

            results.append(summary)

            print("Status: SUCCESS")
            print("Original arcs:", result["original_arcs"])
            print("Intermediate arcs:", result["inserted_arcs"])
            print("Final arcs:", result["final_arcs"])
            print("Connectivity:", result["connectivity_percent"], "%")
            print("Unresolved transitions:", result["unresolved_gps_transitions"])

        except Exception as error:

            print("Status: FAILED")
            print("Reason:", str(error))

            results.append({
                "trip_id": trip_id,
                "status": "failed",
                "original_arcs": 0,
                "inserted_arcs": 0,
                "final_arcs": 0,
                "same_arc_transitions": 0,
                "connected_transitions": 0,
                "disconnected_transitions": 0,
                "unresolved_gps_transitions": 0,
                "connectivity_percent": 0
            })

    summary_file = os.path.join(
        OUTPUT_DIR,
        "batch_connectivity_summary.csv"
    )

    fields = [
        "trip_id",
        "status",
        "original_arcs",
        "inserted_arcs",
        "final_arcs",
        "same_arc_transitions",
        "connected_transitions",
        "disconnected_transitions",
        "unresolved_gps_transitions",
        "connectivity_percent"
    ]

    with open(summary_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    unresolved_file = os.path.join(
        OUTPUT_DIR,
        "unresolved_transitions.csv"
    )

    with open(unresolved_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trip_id",
                "gps_transition",
                "previous_arc",
                "current_arc"
            ]
        )

        writer.writeheader()
        writer.writerows(all_unresolved)

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    failed = len(results) - successful

    print("\n---------------------------------------------")
    print("BATCH CONNECTION COMPLETED")
    print("Total trips:", len(results))
    print("Successful:", successful)
    print("Failed:", failed)
    print("Total unresolved transitions:", len(all_unresolved))
    print("Summary:", summary_file)
    print("Unresolved report:", unresolved_file)
    print("Output folder:", OUTPUT_DIR)
    print("---------------------------------------------")


if __name__ == "__main__":
    main()
