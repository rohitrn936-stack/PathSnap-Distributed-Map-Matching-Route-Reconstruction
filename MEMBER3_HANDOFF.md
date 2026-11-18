# Member 3 — Handoff Document for Member 4 API Integration

## Overview

Member 3 provides a Viterbi route-decoder that can be called from Member 4's API. This document describes the exact entry point, input/output schema, and a working one-trip example.

**Important**: No code in Member 1/2, datasets, or reference labels was modified. The only new file is `member3_api.py`.

## Entry Point

```python
from member3_api import member3_match

result = member3_match(trip_id, split='train')
# or
result = member3_match(trip_id, split='student_test')
```

### Function Signature

```python
member3_match(trip_id: str, split: str = 'train') -> dict
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `trip_id` | `str` | Trip identifier, e.g. `'00000001'` |
| `split` | `str` | `'train'` or `'student_test'` (default: `'train'`) |

### Returns

A JSON-serializable `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `trip_id` | `str` | The input trip ID |
| `split` | `str` | `'train'` or `'student_test'` |
| `n_observations` | `int` | Number of GPS track points |
| `arc_ids` | `list[int \| None]` | Matched arc ID per observation (`None` = no usable candidate) |
| `offsets_m` | `list[float \| None]` | Offset along each arc in metres (`None` if unavailable) |
| `statuses` | `list[list[str]]` | Per-observation status flags. Possible values:<br>`'OK'` — normal match<br>`'NO_CANDIDATES'` — observation had no usable candidates<br>`'GAP_BEFORE'` — dt from previous obs exceeded gap threshold<br>`'DISCONNECTED_STEP'` — chosen transition had no directed path within search cutoff |
| `route` | `list[int]` | Matched arcs in traversal order; consecutive repeats collapsed to single entries |
| `route_positions` | `list[int \| None]` | Per-observation index into `route`, or `None` if observation had no match |
| `geometry` | `dict` | Route geometry output from `build_route_geometry` (see below) |
| `stats` | `dict` | Decoder statistics including counts, timings, and config echo |
| `score_source` | `str` | Always `'member2_ml'` for trained-model runs (distinguished from `'provisional_distance_based'` used in smoke tests) |
| `error` | `str \| None` | Present only if the pipeline failed (e.g., missing candidate CSV) |

### Key Behavioral Notes

- **Candidate arc IDs are preserved verbatim** from the Member 1 candidate input (local to each trip, not global).
- **The Viterbi decoder uses DEFAULT settings**: `transition_weight=1.0`, `score_type='auto'`, no weight tuning or parameter optimization.
- **Emission scores** are converted via `score_to_emission_log` using the Member 2 HistGradientBoosting model's `ml_score`.
- **Consecutive repeat arcs are collapsed** into a single route entry; `route_positions` maps each observation to its route index.
- **The reference route (.route) is never read** by this function.
- **If the candidate CSV is not found** at `memberOneOutput/{split}/{trip_id}/candidateRecords.csv`, the function returns a result dict with `error` set and all output lists empty.

## Working One-Trip Example

**Command**:
```bash
cd /c/Users/tharu/OneDrive/Desktop/MACHINE\ LEARNING/PathSnap-Distributed-Map-Matching-Route-Reconstruction/PathSnap-Distributed-Map-Matching-Route-Reconstruction
python -c "
import sys, os, json
sys.path.insert(0, '.')
from member3_api import member3_match

result = member3_match('00000001', 'train')
print(json.dumps(result, indent=2))
"
```

**Sample output (trimmed for brevity)**:

```json
{
  "trip_id": "00000001",
  "split": "train",
  "n_observations": 1315,
  "score_source": "member2_ml",
  "arc_ids": [64973, 64973, 64973, ..., 79549],  // 1315 entries
  "offsets_m": [null, null, null, ..., 79549],    // 1315 entries
  "statuses": [["OK"], ["OK"], ["OK"], ..., ["OK"]],  // 1315 entries, each a list
  "route": [64973, 133948, 133949, ..., 2195],     // 561 unique arcs (repeats collapsed)
  "route_positions": [0, 1, 2, ..., 55],           // 1315 entries
  "geometry": {
    "coordinates": [[50.1234, 8.5678], [50.1236, 8.5680], ...],
    "crs": "WGS84 (EPSG:4326)",
    "arc_ids": [64973, 133948, 133949, ...],
    "n_points": 1315
  },
  "stats": {
    "n_observations": 1315,
    "n_no_candidates": 0,
    "n_invalid_ids": 0,
    "n_invalid_scores": 0,
    "n_dedup": 0,
    "n_gap_steps": 0,
    "n_disconnected_steps": 9,
    "n_chain_breaks": 0,
    "dijkstra_searches": 0,
    "dijkstra_cache_hits": 0,
    "dijkstra_nodes_visited": 0,
    "n_matched": 1315,
    "n_route_arcs": 561,
    "n_components_graph": 175,
    "runtime_s": 0.847,
    "config": {
      "score_type": "auto",
      "eps": 1e-12,
      "dist_weight": 0.0,
      "dist_sigma_m": 20.0,
      "transition_weight": 1.0,
      "dist_sigma_trans_m": 25.0,
      "speed_cap_mps": 38.0,
      "feasibility_slack_m": 5.0,
      "gap_threshold_s": 30.0,
      "gap_uniform_penalty": 2.0,
      "disconnect_penalty": 30.0,
      "search_cutoff_min_m": 800.0,
      "search_cutoff_max_m": 5000.0,
      "search_margin_m": 300.0,
      "search_cache_size": 512,
      "max_visited_per_search": 50000
    }
  }
}
```

### Verified Properties

- **JSON serializable**: The output dict serializes successfully via `json.dumps` (numpy types are handled).
- **Arc IDs belong to the correct trip graph**: All arc IDs in `arc_ids` and `route` are valid arc IDs from the trip's RoadGraph (verified against `graph.n_arcs`).
- **Observation order preserved**: `arc_ids[i]` and `statuses[i]` correspond to `track_lonlat[i]`.
- **Runtime**: ~0.85s for trip 00000001 (train), ~0.66s for typical student_test trips.

## API Integration Note for Member 4

Member 4 should:

1. **Call** `member3_match(trip_id, split)` with the desired trip ID and split.
2. **Handle** the returned `dict` — no additional serialization steps needed beyond `json.dumps` if a string is required.
3. **Use** `result['arc_ids']` for the matched arc per observation, and `result['route']` / `result['route_positions']` for the traversal route.
4. **Distinguish** `score_source == 'member2_ml'` results from any provisional/distance-based results (`'provisional_distance_based'`).
5. **Not** modify the decoder parameters (`transition_weight`, `score_type`, etc.) — the function uses defaults as documented.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `member3_api.py` | **Created** | Minimal Member 3 API entry point (new file only) |
| `run_member3_evaluation.py` | **Already existed** | Full 80-train + 10-student_test evaluation script (from earlier task) |
| `eval_output/` | **Generated** | Output CSVs and summary JSON from the evaluation run (untracked) |

**No existing files were modified**: Member 1/2 files, datasets, model artifacts, reference labels, or code outside `member3_api.py` were not changed.

## Git Status

```
On branch member3-route-reconstruction
Your branch is ahead of 'origin/member3-route-reconstruction' by 9 commits.
nothing added to commit but untracked files present (use "git add" to track)
```

**Untracked files**:
- `eval_output/` — output directory from evaluation run
- `member3_api.py` — new API entry point
- `run_member3_evaluation.py` — full evaluation script (from earlier task)