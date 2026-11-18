
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_FILE = (
    BASE_DIR / "results" / "batch_connected"
    / "batch_connectivity_summary.csv"
)
UNRESOLVED_FILE = (
    BASE_DIR / "results" / "batch_connected"
    / "unresolved_transitions.csv"
)


def main():
    with open(SUMMARY_FILE, newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No summary data found.")
        return

    total_trips = len(rows)
    total_arcs = 0
    total_same = 0
    total_connected = 0
    total_disconnected = 0
    trips_perfect = 0

    print("\nPATHSNAP - CONNECTIVITY ANALYSIS")
    print("=" * 45)

    for row in rows:
        arcs = int(row["final_arcs"])
        same = int(row["same_arc_transitions"])
        connected = int(row["connected_transitions"])
        disconnected = int(row["disconnected_transitions"])

        total_arcs += arcs
        total_same += same
        total_connected += connected
        total_disconnected += disconnected

        if disconnected == 0:
            trips_perfect += 1

    total_transitions = (
        total_same + total_connected + total_disconnected
    )

    overall_connectivity = (
        (total_same + total_connected)
        / total_transitions * 100
        if total_transitions else 100
    )

    print("Total trips:", total_trips)
    print("Total route arcs:", total_arcs)
    print("Total transitions:", total_transitions)
    print("Same-arc transitions:", total_same)
    print("Connected transitions:", total_connected)
    print("Unresolved transitions:", total_disconnected)
    print("Trips with 100% connectivity:", trips_perfect)
    print("Trips with unresolved transitions:",
          total_trips - trips_perfect)

    print("\nOverall network connectivity:",
          round(overall_connectivity, 4), "%")

    print("\nTOP 10 TRIPS WITH MOST UNRESOLVED TRANSITIONS")
    print("-" * 45)

    sorted_rows = sorted(
        rows,
        key=lambda row: int(row["disconnected_transitions"]),
        reverse=True
    )

    for row in sorted_rows[:10]:
        print(
            "Trip:", row["trip_id"],
            "| Unresolved:",
            row["disconnected_transitions"],
            "| Connectivity:",
            row["connectivity_percent"], "%"
        )

    with open(UNRESOLVED_FILE, newline="") as file:
        unresolved_rows = list(csv.DictReader(file))

    print("\nUnresolved report rows:",
          len(unresolved_rows))

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
