
import csv
import os

FILE = os.path.join(
    "candidate_results",
    "candidate_validation_summary.csv"
)

with open(FILE, "r", newline="") as file:
    rows = list(csv.DictReader(file))

successful = [
    row for row in rows
    if row["status"] == "SUCCESS"
]

successful.sort(
    key=lambda row: float(row["candidate_recall_percent"])
)

print("\nLOWEST CANDIDATE RECALL TRIPS")
print("-" * 55)

for row in successful[:15]:
    print(
        f"Trip: {row['trip_id']} | "
        f"Recall: {row['candidate_recall_percent']}% | "
        f"Missing: {row['missing_reference_arcs']} | "
        f"Eligible: {row['eligible_points']}"
    )

print("\nTOTAL SUCCESSFUL TRIPS:", len(successful))
