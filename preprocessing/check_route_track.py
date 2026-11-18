
from pathlib import Path
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def read_data(path):
    return pd.read_csv(path, sep=r"\s+", header=None)


def distance(a, b):
    # Coordinates are stored as longitude, latitude.
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


def nearest_gps_point(track, coordinate):
    best_index = -1
    best_distance = float("inf")

    for index, point in enumerate(track.iloc[:, :2].values):
        current_distance = distance(coordinate, point)

        if current_distance < best_distance:
            best_distance = current_distance
            best_index = index

    timestamp = track.iloc[best_index, 2]

    return best_index, timestamp, best_distance


def main():
    folders = sorted({
        p.parent for p in DATA_DIR.rglob("*.track")
    })

    for folder in folders:
        trip_id = folder.name

        nodes = read_data(folder / f"{trip_id}.nodes")
        arcs = read_data(folder / f"{trip_id}.arcs")
        route = read_data(folder / f"{trip_id}.route").iloc[:, 0].tolist()
        track = read_data(folder / f"{trip_id}.track")

        for i in range(len(route) - 1):
            current = arcs.iloc[route[i]].tolist()
            following = arcs.iloc[route[i + 1]].tolist()

            if current[1] == following[0]:
                continue

            current_end = nodes.iloc[int(current[1])].tolist()
            next_start = nodes.iloc[int(following[0])].tolist()

            first = nearest_gps_point(track, current_end)
            second = nearest_gps_point(track, next_start)

            print("\n------------------------------")
            print("Trip:", trip_id)
            print("Route positions:", i, "to", i + 1)

            print(
                "Current endpoint nearest GPS:",
                f"index={first[0]}, time={first[1]},",
                f"distance={first[2]:.2f} m"
            )

            print(
                "Next start nearest GPS:",
                f"index={second[0]}, time={second[1]},",
                f"distance={second[2]:.2f} m"
            )


if __name__ == "__main__":
    main()
