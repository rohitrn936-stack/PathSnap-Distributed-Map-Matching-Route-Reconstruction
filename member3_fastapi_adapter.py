"""Minimal FastAPI adapter for Member 3 Viterbi decoder.

Calls member3_match() from member3_api and exposes the result via HTTP.
Intended for Member 4 to integrate with via a simple GET endpoint.
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Ensure the project inner repo is on the Python path so member3_api is importable
_proj = os.path.join(os.path.abspath('.'), 'PathSnap-Distributed-Map-Matching-Route-Reconstruction')
if os.path.isdir(_proj):
    sys.path.insert(0, _proj)

from member3_api import member3_match  # type: ignore

app = FastAPI(title="Member 3 Viterbi Decoder API", version="1.0.0")


@app.get("/member3_match")
def api_member3_match(trip_id: str = "00000001", split: str = "train"):
    """Call member3_match and return the JSON-serializable result.

    Parameters
    ----------
    trip_id : str
        Trip identifier, e.g. '00000001'.
    split : str
        'train' or 'student_test' (default: 'train').

    Returns
    -------
    JSONResponse
        The member3_match result dict with observation-level arrays.
    """
    try:
        result = member3_match(trip_id, split)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate that observation-level lengths agree
    n_obs = result.get('n_observations')
    arc_ids = result.get('arc_ids', [])
    statuses = result.get('statuses', [])
    route_positions = result.get('route_positions', [])

    if n_obs is not None and not (n_obs == len(arc_ids) == len(statuses) == len(route_positions)):
        # Return what we have, but flag the mismatch
        return JSONResponse(
            status_code=200,
            content={**result, "length_mismatch_warning": "observation-level array lengths disagree"},
        )

    return JSONResponse(status_code=200, content=result)


@app.get("/member3_match/post")
def api_member3_match_post(trip_id: str = "00000001", split: str = "train"):
    """POST variant that accepts JSON body for trip_id and split."""
    try:
        result = member3_match(trip_id, split)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    n_obs = result.get('n_observations')
    arc_ids = result.get('arc_ids', [])
    statuses = result.get('statuses', [])
    route_positions = result.get('route_positions', [])

    if n_obs is not None and not (n_obs == len(arc_ids) == len(statuses) == len(route_positions)):
        return JSONResponse(
            status_code=200,
            content={**result, "length_mismatch_warning": "observation-level array lengths disagree"},
        )

    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)