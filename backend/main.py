
import csv
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction",
    version="1.0.0"
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "batch_connected"
DATA_DIR = BASE_DIR / "data" / "Student folder" / "train"
FRONTEND_DIR = BASE_DIR / "frontend"

# Dashboard connection
app.mount(
    "/dashboard",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="dashboard"
)


@app.get("/")
def home():
    return {
        "project": "PathSnap",
        "message": "Map-Matching and Route Reconstruction API",
        "status": "running",
        "dashboard": "/dashboard/"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/trips")
def get_trips():
    files = sorted(
        RESULTS_DIR.glob("connected_route_*.csv")
    )

    trips = [
        file.stem.replace("connected_route_", "")
        for file in files
    ]

    return {
        "total_trips": len(trips),
        "trips": trips
    }


@app.get("/trips/{trip_id}/route")
def get_route(trip_id: str):

    # Validate trip ID
    if not trip_id.isdigit() or len(trip_id) != 8:
        raise HTTPException(
            status_code=400,
            detail="Invalid trip ID."
        )

    route_file = (
        RESULTS_DIR / f"connected_route_{trip_id}.csv"
    )

    trip_folder = DATA_DIR / trip_id

    arcs_file = trip_folder / f"{trip_id}.arcs"
    nodes_file = trip_folder / f"{trip_id}.nodes"

    if not route_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Reconstructed route not found."
        )

    if not arcs_file.exists() or not nodes_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Road network files not found."
        )

    try:
        arcs = np.loadtxt(
            arcs_file, dtype=int, ndmin=2
        )

        nodes = np.loadtxt(
            nodes_file, ndmin=2
        )

        with open(route_file, newline="") as file:
            rows = list(csv.DictReader(file))

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="Route file is empty."
            )

        arc_indices = [
            int(row["arc_index"])
            for row in rows
        ]

        # Build route coordinates
        coordinates = []

        for arc_index in arc_indices:

            if arc_index < 0 or arc_index >= len(arcs):
                raise ValueError("Invalid arc index.")

            start_node, end_node = arcs[arc_index]

            if (
                start_node < 0 or end_node < 0
                or start_node >= len(nodes)
                or end_node >= len(nodes)
            ):
                raise ValueError("Invalid road node index.")

            start_coordinate = [
                float(nodes[start_node, 0]),
                float(nodes[start_node, 1])
            ]

            end_coordinate = [
                float(nodes[end_node, 0]),
                float(nodes[end_node, 1])
            ]

            if not coordinates:
                coordinates.append(start_coordinate)

            coordinates.append(end_coordinate)

        # Calculate network connectivity
        same_arc = 0
        connected = 0
        disconnected = 0

        for first, second in zip(
            arc_indices, arc_indices[1:]
        ):

            if first == second:
                same_arc += 1

            elif arcs[first, 1] == arcs[second, 0]:
                connected += 1

            else:
                disconnected += 1

        total_transitions = max(
            len(arc_indices) - 1, 0
        )

        connectivity = (
            (same_arc + connected) / total_transitions * 100
            if total_transitions else 100.0
        )

        return {
            "trip_id": trip_id,
            "status": "success",
            "total_route_arcs": len(arc_indices),
            "total_coordinates": len(coordinates),
            "route_connectivity_percent": round(
                connectivity, 2
            ),
            "same_arc_transitions": same_arc,
            "connected_transitions": connected,
            "disconnected_transitions": disconnected,
            "coordinates": coordinates
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Route processing failed: {str(error)}"
        )
