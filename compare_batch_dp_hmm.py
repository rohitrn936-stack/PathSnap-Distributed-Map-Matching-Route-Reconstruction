
import csv
import os
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "data", "Student folder", "train"
)

DP_DIR = os.path.join(
    BASE_DIR, "results", "batch_reconstructed"
)

HMM_DIR = os.path.join(
    BASE_DIR, "results", "batch_hmm"
)


def load_route(file_path):
    rows = []

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append({
                "gps": int(row["gps_index"]),
                "arc": int(row["selected_arc_index"]),
                "distance": float(row["candidate_distance_metres"])
            })

    return rows


def calculate_connectivity(rows, arcs):
    same = 0
    connected = 0
    unresolved = 0

    for i in range(1, len(rows)):
        previous = rows[i - 1]["arc"]
        current = rows[i]["arc"]

        if previous == current:
            same += 1

        elif arcs[previous][1] == arcs[current][0]:
            connected += 1

        else:
            unresolved += 1

    total = same + connected + unresolved

    percentage = (
        (same + connected) / total * 100
        if total else 0
    )

    return same, connected, unresolved, percentage


def main():
    print("\nPATHSNAP - BATCH DP VS HMM COMPARISON")
    print("=" * 55)

    dp_files = glob.glob(
        os.path.join(DP_DIR, "reconstructed_*.csv")
    )

    hmm_files = glob.glob(
        os.path.join(HMM_DIR, "hmm_reconstructed_*.csv")
    )

    dp_ids = {
        os.path.basename(f).replace(
            "reconstructed_", ""
        ).replace(".csv", "")
        for f in dp_files
    }

    hmm_ids = {
        os.path.basename(f).replace(
            "hmm_reconstructed_", ""
        ).replace(".csv", "")
        for f in hmm_files
    }

    common_ids = sorted(dp_ids & hmm_ids)

    print("DP files:", len(dp_ids))
    print("HMM files:", len(hmm_ids))
    print("Common trips:", len(common_ids))

    if not common_ids:
        print("ERROR: No matching trip files found.")
        return

    totals = {
        "dp_points": 0,
        "hmm_points": 0,
        "dp_distance": 0.0,
        "hmm_distance": 0.0,
        "dp_same": 0,
        "dp_connected": 0,
        "dp_unresolved": 0,
        "hmm_same": 0,
        "hmm_connected": 0,
        "hmm_unresolved": 0,
        "identical": 0,
        "different": 0,
        "common_points": 0
    }

    successful = 0
    failed = 0

    for trip_id in common_ids:
        try:
            dp_file = os.path.join(
                DP_DIR,
                f"reconstructed_{trip_id}.csv"
            )

            hmm_file = os.path.join(
                HMM_DIR,
                f"hmm_reconstructed_{trip_id}.csv"
            )

            arcs_file = os.path.join(
                TRAIN_DIR,
                trip_id,
                f"{trip_id}.arcs"
            )

            dp = load_route(dp_file)
            hmm = load_route(hmm_file)

            arcs = np.loadtxt(
                arcs_file,
                dtype=int,
                ndmin=2
            )

            dp_same, dp_conn, dp_unres, dp_pct = (
                calculate_connectivity(dp, arcs)
            )

            hmm_same, hmm_conn, hmm_unres, hmm_pct = (
                calculate_connectivity(hmm, arcs)
            )

            dp_map = {r["gps"]: r["arc"] for r in dp}
            hmm_map = {r["gps"]: r["arc"] for r in hmm}

            common_gps = set(dp_map) & set(hmm_map)

            identical = sum(
                dp_map[g] == hmm_map[g]
                for g in common_gps
            )

            different = len(common_gps) - identical

            totals["dp_points"] += len(dp)
            totals["hmm_points"] += len(hmm)

            totals["dp_distance"] += sum(
                r["distance"] for r in dp
            )

            totals["hmm_distance"] += sum(
                r["distance"] for r in hmm
            )

            totals["dp_same"] += dp_same
            totals["dp_connected"] += dp_conn
            totals["dp_unresolved"] += dp_unres

            totals["hmm_same"] += hmm_same
            totals["hmm_connected"] += hmm_conn
            totals["hmm_unresolved"] += hmm_unres

            totals["identical"] += identical
            totals["different"] += different
            totals["common_points"] += len(common_gps)

            successful += 1

            print(
                f"Trip {trip_id}: "
                f"DP {dp_pct:.2f}% | "
                f"HMM {hmm_pct:.2f}% | "
                f"Common GPS: {len(common_gps)}"
            )

        except Exception as error:
            failed += 1
            print(f"Trip {trip_id}: FAILED - {error}")

    def percentage(connected, unresolved):
        total = connected + unresolved
        return connected / total * 100 if total else 0

    print("\n" + "=" * 55)
    print("FINAL AGGREGATE RESULTS")
    print("=" * 55)

    print("Successfully compared trips:", successful)
    print("Failed comparisons:", failed)

    print("\nDP RESULTS")
    print("Selected points:", totals["dp_points"])
    print(
        "Average GPS-to-road distance:",
        round(
            totals["dp_distance"] / totals["dp_points"], 3
        ) if totals["dp_points"] else 0,
        "metres"
    )
    print("Same-arc transitions:", totals["dp_same"])
    print("Connected transitions:", totals["dp_connected"])
    print("Unresolved transitions:", totals["dp_unresolved"])
    print(
        "Direct connectivity:",
        round(
            percentage(
                totals["dp_same"] + totals["dp_connected"],
                totals["dp_unresolved"]
            ), 4
        ),
        "%"
    )

    print("\nHMM RESULTS")
    print("Selected points:", totals["hmm_points"])
    print(
        "Average GPS-to-road distance:",
        round(
            totals["hmm_distance"] / totals["hmm_points"], 3
        ) if totals["hmm_points"] else 0,
        "metres"
    )
    print("Same-arc transitions:", totals["hmm_same"])
    print("Connected transitions:", totals["hmm_connected"])
    print("Unresolved transitions:", totals["hmm_unresolved"])
    print(
        "Direct connectivity:",
        round(
            percentage(
                totals["hmm_same"] + totals["hmm_connected"],
                totals["hmm_unresolved"]
            ), 4
        ),
        "%"
    )

    print("\nDIRECT COMPARISON")
    print("Common GPS points:", totals["common_points"])
    print("Identical selected arcs:", totals["identical"])
    print("Different selected arcs:", totals["different"])

    print("\nBatch comparison completed.")


if __name__ == "__main__":
    main()
