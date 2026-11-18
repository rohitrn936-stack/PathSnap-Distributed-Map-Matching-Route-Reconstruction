
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

def read_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None)

def validate_trip(folder):
    trip_id = folder.name

    nodes = read_data(folder / f"{trip_id}.nodes")
    arcs = read_data(folder / f"{trip_id}.arcs")
    route = read_data(folder / f"{trip_id}.route").iloc[:, 0]

    node_ids = set(range(len(nodes)))
    arc_ids = set(range(len(arcs)))

    arc_start = set(arcs.iloc[:, 0])
    arc_end = set(arcs.iloc[:, 1])

    invalid_node_references = (
        len((arc_start | arc_end) - node_ids)
    )

    route_ids = set(route)
    invalid_route_ids = len(route_ids - arc_ids)

    # Check whether consecutive route arcs connect
    route_values = route.tolist()
    disconnected = 0

    for i in range(len(route_values) - 1):
        current_arc = route_values[i]
        next_arc = route_values[i + 1]

        if current_arc not in arc_ids or next_arc not in arc_ids:
            continue

        current_end = arcs.iloc[current_arc, 1]
        next_start = arcs.iloc[next_arc, 0]

        if current_end != next_start:
            disconnected += 1

    return {
        "trip_id": trip_id,
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_arcs": len(route),
        "invalid_node_references": invalid_node_references,
        "invalid_route_ids": invalid_route_ids,
        "disconnected_pairs": disconnected
    }

def main():
    folders = sorted({
        p.parent for p in DATA_DIR.rglob("*.track")
    })

    results = []

    for folder in folders:
        results.append(validate_trip(folder))

    report = pd.DataFrame(results)
    output = RESULTS_DIR / "network_validation.csv"
    report.to_csv(output, index=False)

    print("\nNETWORK VALIDATION")
    print("=" * 45)
    print(report.to_string(index=False))

    print("\nTOTALS")
    print("Invalid node references:",
          report["invalid_node_references"].sum())
    print("Invalid route IDs:",
          report["invalid_route_ids"].sum())
    print("Disconnected route pairs:",
          report["disconnected_pairs"].sum())

    print("\nSaved:", output)

if __name__ == "__main__":
    main()
