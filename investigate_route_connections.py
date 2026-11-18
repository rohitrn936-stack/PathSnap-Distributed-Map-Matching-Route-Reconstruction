
from pathlib import Path
from collections import deque, defaultdict
from math import radians, sin, cos, asin, sqrt

TRIP_ID = "00000086"
DATASET = Path("data/Student folder/train") / TRIP_ID
MAX_DEPTH = 200
MAX_VISITED = 100000


def load_arcs(path):
    return [
        tuple(map(int, line.split()))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_nodes(path):
    return [
        tuple(map(float, line.split()))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_route(path):
    return [
        int(line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def distance_m(a, b):
    lon1, lat1 = a
    lon2, lat2 = b

    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)

    h = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    return 12742000 * asin(sqrt(min(1, h)))


def find_path(start, target, adjacency):
    queue = deque([start])
    parent = {start: None}
    parent_arc = {}
    depth = {start: 0}

    while queue and len(parent) < MAX_VISITED:
        node = queue.popleft()

        if depth[node] >= MAX_DEPTH:
            continue

        for neighbor, arc_index in adjacency.get(node, []):
            if neighbor in parent:
                continue

            parent[neighbor] = node
            parent_arc[neighbor] = arc_index
            depth[neighbor] = depth[node] + 1

            if neighbor == target:
                path = []
                current = target

                while parent[current] is not None:
                    path.append(parent_arc[current])
                    current = parent[current]

                return path[::-1]

            queue.append(neighbor)

    return None


def main():
    print("Investigating trip:", TRIP_ID)

    nodes = load_nodes(DATASET / f"{TRIP_ID}.nodes")
    arcs = load_arcs(DATASET / f"{TRIP_ID}.arcs")
    route = load_route(DATASET / f"{TRIP_ID}.route")

    # Find the disconnected transition
    for position in range(len(route) - 1):
        current_index = route[position]
        next_index = route[position + 1]

        current_arc = arcs[current_index]
        next_arc = arcs[next_index]

        start_node = current_arc[1]
        target_node = next_arc[0]

        if start_node == target_node:
            continue

        print("\nDisconnected transition found")
        print("Route positions:", position, "to", position + 1)
        print("Arc rows:", current_index, "to", next_index)
        print("Start node:", start_node)
        print("Target node:", target_node)

        print("\nEndpoint coordinates:")
        print("Start:", nodes[start_node])
        print("Target:", nodes[target_node])

        gap = distance_m(nodes[start_node], nodes[target_node])
        print(f"Straight-line gap: {gap:.2f} metres")

        # Build directed graph
        directed = defaultdict(list)

        # Build undirected graph
        undirected = defaultdict(list)

        for arc_index, (u, v) in enumerate(arcs):
            directed[u].append((v, arc_index))

            undirected[u].append((v, arc_index))
            undirected[v].append((u, arc_index))

        print("\n1. Directed graph search")
        directed_path = find_path(
            start_node, target_node, directed
        )

        if directed_path is not None:
            print("Directed path found:", directed_path)
            print("Number of arcs:", len(directed_path))
        else:
            print("No directed path found within search limits.")

        print("\n2. Undirected graph search")
        undirected_path = find_path(
            start_node, target_node, undirected
        )

        if undirected_path is not None:
            print("Undirected path found:", undirected_path)
            print("Number of arcs:", len(undirected_path))
        else:
            print("No undirected path found within search limits.")

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
