
import csv
import os
from collections import deque
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIP_ID = "00000001"

DATA_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train", TRIP_ID
)

INPUT_FILE = os.path.join(
    BASE_DIR, "results",
    f"reconstructed_{TRIP_ID}.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "results",
    f"connected_route_{TRIP_ID}.csv"
)

MAX_DEPTH = 200


def load_data():
    arcs = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.arcs"),
        dtype=int
    )

    selected = []

    with open(INPUT_FILE, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            selected.append(int(row["selected_arc_index"]))

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


def main():

    print("\nPATHSNAP - CONNECTED ROUTE RECONSTRUCTION")
    print("------------------------------------------")

    arcs, selected = load_data()
    graph = build_graph(arcs)

    print("Selected arcs:", len(selected))
    print("Road network arcs:", len(arcs))

    connected_route = []
    unresolved = []
    inserted_count = 0

    if not selected:
        raise ValueError("No selected arcs found.")

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

    # Verify final route connectivity
    connected = 0
    disconnected = 0

    for i in range(len(connected_route) - 1):

        first_arc = connected_route[i][0]
        second_arc = connected_route[i + 1][0]

        if arcs[first_arc, 1] == arcs[second_arc, 0]:
            connected += 1
        else:
            disconnected += 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as file:

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

    print("------------------------------------------")
    print("CONNECTED ROUTE PROCESSING COMPLETED")
    print("Original selected arcs:", len(selected))
    print("Intermediate arcs inserted:", inserted_count)
    print("Final route arcs:", len(connected_route))
    print("Connected transitions:", connected)
    print("Disconnected transitions:", disconnected)
    print("Unresolved GPS transitions:", len(unresolved))
    print("Output:", OUTPUT_FILE)

    if unresolved:
        print("\nFirst unresolved transitions:")

        for item in unresolved[:15]:
            print(item)

    print("------------------------------------------")


if __name__ == "__main__":
    main()
