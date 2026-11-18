
from pathlib import Path
import pandas as pd
import numpy as np

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

EXTENSIONS = ["track", "nodes", "arcs", "route"]


def read_file(path):
    """Read a whitespace-separated dataset file."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python"
    )


def inspect_trip(folder):
    """Inspect one trip and return its statistics."""

    trip_id = folder.name
    files = {
        ext: folder / f"{trip_id}.{ext}"
        for ext in EXTENSIONS
    }

    missing = [
        ext for ext, path in files.items()
        if not path.exists()
    ]

    if missing:
        return {
            "trip_id": trip_id,
            "status": "INCOMPLETE",
            "missing_files": ", ".join(missing)
        }

    try:
        track = read_file(files["track"])
        nodes = read_file(files["nodes"])
        arcs = read_file(files["arcs"])
        route = read_file(files["route"])

        # Validate GPS coordinates
        valid_gps = (
            track.shape[1] >= 3
            and track.iloc[:, :3].notna().all().all()
            and np.isfinite(track.iloc[:, :3].to_numpy()).all()
        )

        # Validate node and arc structures
        valid_nodes = (
            nodes.shape[1] == 2
            and nodes.notna().all().all()
        )

        valid_arcs = (
            arcs.shape[1] == 2
            and arcs.notna().all().all()
        )

        valid_route = (
            route.shape[1] == 1
            and route.notna().all().all()
        )

        return {
            "trip_id": trip_id,
            "status": "READ",
            "gps_points": len(track),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "route_arcs": len(route),
            "valid_gps": valid_gps,
            "valid_nodes": valid_nodes,
            "valid_arcs": valid_arcs,
            "valid_route": valid_route,
            "missing_files": ""
        }

    except Exception as error:
        return {
            "trip_id": trip_id,
            "status": "ERROR",
            "error": str(error)
        }


def main():

    print("\nPATHSNAP DATASET INSPECTION")
    print("=" * 45)

    if not DATA_DIR.exists():
        print("Dataset folder not found:", DATA_DIR)
        return

    # Find folders containing .track files
    trip_folders = sorted({
        path.parent
        for path in DATA_DIR.rglob("*.track")
    })

    print("Trip folders found:", len(trip_folders))

    results = []

    for folder in trip_folders:
        result = inspect_trip(folder)
        results.append(result)

        print(
            f"Trip {result['trip_id']}: "
            f"{result['status']}"
        )

    report = pd.DataFrame(results)

    output_file = RESULTS_DIR / "trip_inspection.csv"
    report.to_csv(output_file, index=False)

    print("\nINSPECTION SUMMARY")
    print("=" * 45)
    print(report["status"].value_counts().to_string())

    if "gps_points" in report.columns:
        print("\nTotal GPS observations:",
              report["gps_points"].sum())

    if "road_nodes" in report.columns:
        print("Total road nodes:",
              report["road_nodes"].sum())

    if "road_arcs" in report.columns:
        print("Total road arcs:",
              report["road_arcs"].sum())

    print("\nSaved report:", output_file)


if __name__ == "__main__":
    main()
