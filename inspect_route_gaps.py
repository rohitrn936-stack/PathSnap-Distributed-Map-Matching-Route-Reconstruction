
from pathlib import Path
from math import radians, sin, cos, asin, sqrt

BASE = Path("data/Student folder/train")

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

    for i in range(len(route) - 1):
        current_arc = arcs[route[i]]
        next_arc = arcs[route[i + 1]]

        end_node = current_arc[1]
        start_node = next_arc[0]

        if end_node != start_node:
            gap = distance_m(nodes[end_node], nodes[start_node])

            print(f"\nTrip: {trip}")
            print(f"Route positions: {i + 1} to {i + 2}")
            print(f"Arc rows: {route[i]} to {route[i + 1]}")
            print(f"Node IDs: {end_node} to {start_node}")
            print(f"Gap distance: {gap:.2f} metres")
