
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="PathSnap API",
    description="Backend for intelligent route reconstruction",
    version="1.0.0"
)

# Temporary trip IDs for testing.
# Later, we will load actual trip IDs from the dataset.
TRIPS = [
    "trip_01",
    "trip_02",
    "trip_03",
    "trip_04",
    "trip_05"
]


@app.get("/")
def home():
    return {
        "message": "Welcome to PathSnap API!",
        "status": "Running successfully"
    }


@app.get("/trips")
def get_trips():
    return {
        "total_trips": len(TRIPS),
        "trips": TRIPS
    }


@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str):

    if trip_id not in TRIPS:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )

    # Temporary response.
    # We will replace this with the real ML pipeline later.

    return {
        "trip_id": trip_id,
        "status": "Prediction request successful",
        "predicted_route": [],
        "metrics": {
            "accuracy": None,
            "route_continuity": None
        }
    }
