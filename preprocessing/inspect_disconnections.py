
from pathlib import Path
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def read_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None)


def haversine(a, b):
    # Dataset coordinate order: longitude, latitude
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 6371000 * 2 * math.asin(math.sqrt(value))


def main():
    folders = sorted({
        p.parent for p in DATA_DIR.rglob("*.track")
    })

    total = 0

    for folder in folders:
        trip_id = folder.name

        nodes = read_data(folder / f"{trip_id}.nodes")
        arcs = read_data(folder / f"{trip_id}.arcs")
        route = read_data(folder / f"{trip_id}.route").iloc[:, 0].tolist()

        for i in range(len(route) - 1):
            current_id = route[i]
            next_id = route[i + 1]

            current = arcs.iloc[current_id].tolist()
            following = arcs.iloc[next_id].tolist()

            if current[1] == following[0]:
                continue

            total += 1

            current_end = nodes.iloc[int(current[1])].tolist()
            next_start = nodes.iloc[int(following[0])].tolist()

            distance = haversine(current_end, next_start)

            print("\n------------------------------")
            print("Trip:", trip_id)
            print("Route positions:", i, "to", i + 1)
            print("Arc IDs:", current_id, "to", next_id)
            print("Gap distance:", f"{distance:.2f} metres")

    print("\nTotal disconnected pairs:", total)


if __name__ == "__main__":
    main()
