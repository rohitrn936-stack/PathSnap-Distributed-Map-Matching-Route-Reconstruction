"""PathSnap web API and dashboard entry point.

Run from the repository root with:
    uvicorn backend.main:app --reload
"""

import csv
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PathSnap API", description="Distributed map-matching and route reconstruction dashboard", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "batch_connected"
DATA_DIR = BASE_DIR / "datasets" / "student_test"
CANDIDATES_DIR = BASE_DIR / "memberOneOutput" / "student_test"
FRONTEND_DIR = BASE_DIR / "frontend"


def _trip_ids() -> list[str]:
    """Return every dashboard-ready student-test trip available locally."""
    ids: set[str] = set()
    if DATA_DIR.is_dir():
        ids.update(path.name for path in DATA_DIR.iterdir() if path.is_dir() and path.name.isdigit())
    if RESULTS_DIR.is_dir():
        ids.update(path.stem.removeprefix("connected_route_") for path in RESULTS_DIR.glob("connected_route_*.csv"))
    return sorted(ids)


def _load_arc_indices(trip_id: str) -> tuple[list[int], str]:
    """Load a reconstruction or use the saved nearest-candidate baseline."""
    route_file = RESULTS_DIR / f"connected_route_{trip_id}.csv"
    if route_file.is_file():
        with route_file.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        try:
            arcs = [int(row["arc_index"]) for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid route CSV: {error}") from error
        if arcs:
            return arcs, "connected_route"

    candidates_file = CANDIDATES_DIR / trip_id / "candidateRecords.csv"
    if not candidates_file.is_file():
        raise FileNotFoundError("No reconstructed route or candidate baseline is available.")

    by_observation: dict[int, tuple[int, float, int]] = {}
    with candidates_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                observation = int(row["observationIndex"])
                candidate = (int(row["candidateRank"]), float(row["distanceMeters"]), int(row["candidateArcId"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid candidate CSV: {error}") from error
            if observation not in by_observation or candidate < by_observation[observation]:
                by_observation[observation] = candidate

    arcs = [item[2] for _, item in sorted(by_observation.items())]
    if not arcs:
        raise ValueError("Candidate baseline contains no route arcs.")
    return arcs, "nearest_candidate_baseline"


@app.get("/")
def home():
    return {"project": "PathSnap", "status": "running", "dashboard": "/dashboard/", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "available_trips": len(_trip_ids())}


@app.get("/trips")
def get_trips():
    trips = _trip_ids()
    return {"total_trips": len(trips), "trips": trips}


@app.get("/trips/{trip_id}/route")
def get_route(trip_id: str):
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(status_code=400, detail="Trip ID must be an eight-digit number.")

    trip_folder = DATA_DIR / trip_id
    arcs_file = trip_folder / f"{trip_id}.arcs"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    if not arcs_file.is_file() or not nodes_file.is_file():
        raise HTTPException(status_code=404, detail="Road network files not found for this trip.")

    try:
        arc_indices, route_source = _load_arc_indices(trip_id)
        arcs = np.loadtxt(arcs_file, dtype=int, ndmin=2)
        nodes = np.loadtxt(nodes_file, dtype=float, ndmin=2)
        coordinates: list[list[float]] = []
        valid_indices: list[int] = []
        for arc_index in arc_indices:
            if not 0 <= arc_index < len(arcs):
                continue
            start_node, end_node = arcs[arc_index, :2]
            if not (0 <= start_node < len(nodes) and 0 <= end_node < len(nodes)):
                continue
            start = [float(nodes[start_node, 0]), float(nodes[start_node, 1])]
            end = [float(nodes[end_node, 0]), float(nodes[end_node, 1])]
            if not coordinates:
                coordinates.append(start)
            coordinates.append(end)
            valid_indices.append(arc_index)
        if not valid_indices:
            raise ValueError("The route has no valid road arcs.")

        same_arc = connected = disconnected = 0
        for first, second in zip(valid_indices, valid_indices[1:]):
            if first == second:
                same_arc += 1
            elif arcs[first, 1] == arcs[second, 0]:
                connected += 1
            else:
                disconnected += 1
        transitions = max(len(valid_indices) - 1, 0)
        connectivity = 100.0 if not transitions else 100 * (same_arc + connected) / transitions
        return {
            "trip_id": trip_id, "status": "success", "route_source": route_source,
            "total_route_arcs": len(valid_indices), "total_coordinates": len(coordinates),
            "route_connectivity_percent": round(connectivity, 2),
            "same_arc_transitions": same_arc, "connected_transitions": connected,
            "disconnected_transitions": disconnected, "coordinates": coordinates,
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Route processing failed: {error}") from error


app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")
