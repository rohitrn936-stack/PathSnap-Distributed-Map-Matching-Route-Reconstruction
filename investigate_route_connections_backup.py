
from pathlib import Path
from collections import deque

BASE = Path("data/Student folder/train")
MAX_DEPTH = 50


def find_path(start, target, adjacency):
    queue = deque([start])
    parent = {start: None}
    parent_arc = {}
    depth = {start: 0}

    while queue:
        node = queue.popleft()

        if node == target:
            path = []
            current = target

            while parent[current] is not None:
                path.append(parent_arc[current])
                current = parent[current]

            return path[::-1]

        if depth[node] >= MAX_DEPTH:
            continue

        for next_node, arc_index in adjacency.get(node, []):
            if next_node not in parent:
                parent[next_node] = node
                parent_arc[next_node] = arc_index
                depth[next_node] = depth[node] + 1
                queue.append(next_node)

    return None


print("Starting route connectivity investigation...")

for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue

    trip = folder.name

    nodes = [
        tuple(map(float, line.split()))
        for line in (folder / f"{trip}.nodes").read_text().splitlines()
    ]

    arcs = [
        tuple(map(int, line.split()))
        for line in (folder / f"{trip}.arcs").read_text().splitlines()
    ]

    route = [
        int(line)
        for line in (folder / f"{trip}.route").read_text().splitlines()
    ]

    adjacency = {}

    for arc_index, (u, v) in enumerate(arcs):
        adjacency.setdefault(u, []).append((v, arc_index))

    for i in range(len(route) - 1):
        current_end = arcs[route[i]][1]
        next_start = arcs[route[i + 1]][0]

        if current_end == next_start:
            continue

        path = find_path(current_end, next_start, adjacency)

        print("\n------------------------------")
        print("Trip:", trip)
        print("Route positions:", i + 1, "to", i + 2)
        print("Arc rows:", route[i], "to", route[i + 1])

        if path is None:
            print("Result: No connecting path found within search limit.")
        else:
            print("Result: Connecting path found.")
            print("Intermediate arc rows:", path)
            print("Number of intermediate arcs:", len(path))

print("\nInvestigation completed.")
