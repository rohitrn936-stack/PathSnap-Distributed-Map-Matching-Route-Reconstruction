from pathlib import Path

trip_id = "00000086"
folder = Path("data/Student folder/train") / trip_id

arcs_file = folder / f"{trip_id}.arcs"
route_file = folder / f"{trip_id}.route"

arcs = [
    tuple(map(int, line.split()))
    for line in arcs_file.read_text().splitlines()
    if line.strip()
]

route = [
    int(line.strip())
    for line in route_file.read_text().splitlines()
    if line.strip()
]

arc_indexes = [230068, 130144, 96025]

print("TRIP:", trip_id)
print("\nInspecting the three arcs:\n")

for index in arc_indexes:
    start_node, end_node = arcs[index]

    print(f"Arc row: {index}")
    print(f"Start node: {start_node}")
    print(f"End node: {end_node}")
    print("-" * 30)

print("\nOriginal reference route:")
print(route[:5])