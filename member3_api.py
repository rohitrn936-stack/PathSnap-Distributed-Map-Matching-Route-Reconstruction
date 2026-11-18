#!/usr/bin/env python3
"""
Member 3 API entry point for Member 4.

Provides one function: member3_match(trip_id)
- Loads the trip .track/.nodes/.arcs files
- Loads Member 1 candidateRecords.csv from memberOneOutput/{split}/{trip_id}/
- Scores candidates with the saved Member 2 HistGradientBoosting model
- Runs Viterbi decode with DEFAULT DecoderConfig() (transition_weight=1.0)
- Returns JSON-serializable dict with matched arc IDs, statuses, and route geometry

This is a standalone entry point; no code in Member 1/2, datasets, or reference labels
is modified. The function can be imported and called from Member 4's API.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

# Add project subdirectory to path so route_reconstruction is importable
_proj = os.path.join(os.path.abspath('.'), 'PathSnap-Distributed-Map-Matching-Route-Reconstruction')
if os.path.isdir(_proj):
    sys.path.insert(0, _proj)

from route_reconstruction.candidates import normalize_candidates, score_to_emission_log
from route_reconstruction.graph import RoadGraph
from route_reconstruction.evaluate import load_trip, compare_for_trip, EvalConfig
from route_reconstruction.viterbi import match_route as vr_match_route, DecoderConfig
from member2_ml.predict import predict_candidates


# Base paths - adjust for your environment
PROJECT_ROOT = r'C:\Users\tharu\OneDrive\Desktop\MACHINE LEARNING\PathSnap-Distributed-Map-Matching-Route-Reconstruction\PathSnap-Distributed-Map-Matching-Route-Reconstruction'
CANDIDATE_ROOT = os.path.join(PROJECT_ROOT, 'memberOneOutput')
TRIP_ROOTS = {
    'train': os.path.join(PROJECT_ROOT, 'datasets', 'train'),
    'student_test': os.path.join(PROJECT_ROOT, 'datasets', 'student_test'),
}


def member3_match(trip_id, split='train'):
    """
    Run the full Member 3 pipeline for one trip.

    Parameters
    ----------
    trip_id : str
        Trip identifier, e.g. '00000001'
    split : {'train', 'student_test'}
        Which split the trip belongs to.

    Returns
    -------
    dict
        JSON-serializable result with the following keys:
        - trip_id: the input trip ID
        - split: the split name
        - n_observations: int, number of GPS track points
        - arc_ids: list[int | None], matched arc ID per observation (None = no candidate)
        - offsets_m: list[float | None], offset along each arc, or None
        - statuses: list[list[str]], per-observation status flags
            (e.g. ['OK'], ['NO_CANDIDATES'], ['GAP_BEFORE'], ['DISCONNECTED_STEP', 'OK'])
        - route: list[int], matched arcs in traversal order (consecutive repeats collapsed)
        - route_positions: list[int | None], per-observation index into `route`, or None
        - geometry: dict, route geometry (build_route_geometry output)
        - stats: dict, decoder statistics (counts, timings, config echo)
        - score_source: str, always 'member2_ml' for trained-model runs
        - error: str | None, any error message if the pipeline failed

    Notes
    -----
    - Candidate arc IDs are preserved verbatim from the candidate input (local to each trip).
    - The Viterbi decoder uses DEFAULT settings: transition_weight=1.0, score_type='auto', etc.
    - Emission scores are converted via score_to_emission_log with the model's ml_score.
    - Consecutive repeat arcs are collapsed into a single route entry; route_positions maps
      each observation to its route index.
    - If the candidate CSV is not found at the expected path, the function returns an error result.
    """
    # Determine trip directory
    trip_dir = os.path.join(TRIP_ROOTS[split], trip_id)
    cand_dir = os.path.join(CANDIDATE_ROOT, split, trip_id)

    # Check that trip directory and candidate CSV exist
    if not os.path.isdir(trip_dir):
        return {
            'trip_id': trip_id,
            'split': split,
            'n_observations': None,
            'arc_ids': [],
            'offsets_m': [],
            'statuses': [],
            'route': [],
            'route_positions': [],
            'geometry': {},
            'stats': {'error': f'Trip directory not found: {trip_dir}'},
            'score_source': 'member2_ml',
        }

    if not os.path.exists(os.path.join(cand_dir, 'candidateRecords.csv')):
        return {
            'trip_id': trip_id,
            'split': split,
            'n_observations': None,
            'arc_ids': [],
            'offsets_m': [],
            'statuses': [],
            'route': [],
            'route_positions': [],
            'geometry': {},
            'stats': {'error': f'Candidate CSV not found: {cand_dir}/candidateRecords.csv'},
            'score_source': 'member2_ml',
        }

    # Load trip data
    trip = load_trip(trip_dir, trip_id)

    # Load road graph from .nodes and .arcs files
    graph = RoadGraph.from_trip_files(trip_dir, trip_id)

    # Load candidate records from Member 1 output
    cand_df = pd.read_csv(os.path.join(cand_dir, 'candidateRecords.csv'))

    # Rename columns to match expected names (same as the evaluator pipeline)
    rename_map = {
        'candidateRank': 'candidate_rank',
        'distanceMeters': 'distance_m',
        'segmentFraction': 'segment_fraction',
        'headingDifferenceDegrees': 'heading_diff_deg',
        'vehicleSpeedMetersPerSecond': 'gps_speed_mps',
        'timeGapSeconds': 'time_gap_s',
    }
    cand_renamed = cand_df.rename(columns=rename_map)
    cand_renamed['split'] = cand_renamed['partition']
    cand_renamed['trip_id'] = cand_renamed['tripId']
    cand_renamed['observation_idx'] = cand_renamed['observationIndex']
    cand_renamed['arc_id'] = cand_renamed['candidateArcId']

    cand_final = cand_renamed[
        ['split', 'trip_id', 'observation_idx', 'arc_id'] +
        list(rename_map.values())
    ]
    cand_filled = cand_final.fillna({
        'candidate_rank': 1,
        'distance_m': 0.0,
        'segment_fraction': 0.5,
        'heading_diff_deg': 0.0,
        'gps_speed_mps': 10.0,
        'time_gap_s': 1.0,
    })

    # Score candidates with Member 2 trained model
    model_result = predict_candidates(cand_filled)

    # Build candidate sets per observation: list of (arc_id, score) tuples
    n_obs = len(trip.track_lonlat)
    candidate_sets = [[] for _ in range(n_obs)]
    for _, row in model_result.iterrows():
        obs_idx = int(row['observation_idx'])
        arc_id = int(row['arc_id'])
        score = float(row['ml_score'])
        if 0 <= obs_idx < n_obs:
            candidate_sets[obs_idx].append((arc_id, score))

    # Normalize candidates (assigns offsets, deduplicates, etc.)
    norm = normalize_candidates(candidate_sets, graph.n_arcs, expect_obs=n_obs)

    # Viterbi decode with DEFAULT config (transition_weight=1.0, no tuning)
    cfg = DecoderConfig()  # w=1.0 default
    result = vr_match_route(graph, trip.track_lonlat, trip.ts, None, config=cfg, normalized=norm)

    # Convert MatchResult to JSON-serializable dict
    output = result.to_dict()
    output['trip_id'] = trip_id
    output['split'] = split
    output['n_observations'] = n_obs
    output['score_source'] = 'member2_ml'

    # Ensure arc_ids and offsets are correctly paired with observations
    # (to_dict already does this, but we double-check the types)
    output['arc_ids'] = [int(a) if a is not None else None for a in output['arc_ids']]
    output['offsets_m'] = [float(o) if o is not None else None for o in output['offsets_m']]
    # statuses are already list of lists of strings
    # route and route_positions are already correct

    # Ensure stats config echo uses finite values
    if 'stats' in output and 'config' in output['stats']:
        cfg_echo = output['stats']['config']
        for k in cfg_echo:
            if isinstance(cfg_echo[k], float) and not np.isfinite(cfg_echo[k]):
                cfg_echo[k] = 0.0
        output['stats']['config'] = cfg_echo

    return output


def quick_serialize(result_dict, indent=2):
    """Helper: serialize the result dict to a JSON string, handling numpy types."""
    import numpy as np

    class _JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    return json.dumps(result_dict, cls=_JSONEncoder, indent=indent)


if __name__ == '__main__':
    # Test: run on trip 00000001 from the train split
    import argparse
    parser = argparse.ArgumentParser(description='Member 3 single-trip API demo')
    parser.add_argument('trip_id', help='Trip ID, e.g. 00000001')
    parser.add_argument('--split', choices=['train', 'student_test'], default='train',
                        help='Trip split (default: train)')
    args = parser.parse_args()

    result = member3_match(args.trip_id, args.split)
    json_str = quick_serialize(result)
    print(json_str)