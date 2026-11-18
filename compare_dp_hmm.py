
import csv
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRIP_ID = "00000001"

DATA_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train", TRIP_ID
)

DP_FILE = os.path.join(
    BASE_DIR, "results", f"reconstructed_{TRIP_ID}.csv"
)

HMM_FILE = os.path.join(
    BASE_DIR, "results", f"hmm_reconstructed_{TRIP_ID}.csv"
)


def load_result(path):
    result = {}

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            gps_index = int(row["gps_index"])
            arc_index = int(row["selected_arc_index"])
            distance = float(row["candidate_distance_metres"])

            result[gps_index] = (arc_index, distance)

    return result


def calculate_connectivity(result, arcs):
    ordered = sorted(result.keys())

    same_arc = 0
    connected = 0
    unresolved = 0

    for i in range(1, len(ordered)):
        previous_arc = result[ordered[i - 1]][0]
        current_arc = result[ordered[i]][0]

        if previous_arc == current_arc:
            same_arc += 1
        elif arcs[previous_arc][1] == arcs[current_arc][0]:
            connected += 1
        else:
            unresolved += 1

    total = same_arc + connected + unresolved

    percentage = (
        ((same_arc + connected) / total) * 100
        if total else 0
    )

    return same_arc, connected, unresolved, percentage


def main():
    print("\nPATHSNAP - DP VS HMM COMPARISON")
    print("=" * 45)

    arcs = np.loadtxt(
        os.path.join(DATA_DIR, f"{TRIP_ID}.arcs"),
        dtype=int,
        ndmin=2
    )

    dp = load_result(DP_FILE)
    hmm = load_result(HMM_FILE)

    common = sorted(set(dp) & set(hmm))

    if not common:
        print("No common GPS points found.")
        return

    dp_distances = [dp[i][1] for i in common]
    hmm_distances = [hmm[i][1] for i in common]

    identical = sum(
        dp[i][0] == hmm[i][0] for i in common
    )

    dp_same, dp_connected, dp_unresolved, dp_pct = (
        calculate_connectivity(dp, arcs)
    )

    hmm_same, hmm_connected, hmm_unresolved, hmm_pct = (
        calculate_connectivity(hmm, arcs)
    )

    print("Trip:", TRIP_ID)
    print("Common GPS points:", len(common))

    print("\nDP RESULTS")
    print("Selected points:", len(dp))
    print("Unique arcs:", len(set(v[0] for v in dp.values())))
    print("Average GPS-to-road distance:",
          round(np.mean(dp_distances), 3), "metres")
    print("Same-arc transitions:", dp_same)
    print("Connected transitions:", dp_connected)
    print("Unresolved transitions:", dp_unresolved)
    print("Connectivity:", round(dp_pct, 4), "%")

    print("\nHMM RESULTS")
    print("Selected points:", len(hmm))
    print("Unique arcs:", len(set(v[0] for v in hmm.values())))
    print("Average GPS-to-road distance:",
          round(np.mean(hmm_distances), 3), "metres")
    print("Same-arc transitions:", hmm_same)
    print("Connected transitions:", hmm_connected)
    print("Unresolved transitions:", hmm_unresolved)
    print("Connectivity:", round(hmm_pct, 4), "%")

    print("\nDIRECT COMPARISON")
    print("Identical selected arcs:", identical)
    print("Different selected arcs:", len(common) - identical)
    print("HMM average distance change:",
          round(np.mean(hmm_distances) - np.mean(dp_distances), 3),
          "metres")

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
