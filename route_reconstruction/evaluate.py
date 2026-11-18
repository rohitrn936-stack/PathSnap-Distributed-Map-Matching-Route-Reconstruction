"""Evaluation: classifier-only top-1 vs classifier + Viterbi (Member 3).

THIS IS THE ONLY MODULE IN route_reconstruction THAT READS THE REFERENCE
ROUTE (.route). The decoder (viterbi.py) and the candidate generator never
touch it; it is used exclusively here, for scoring.

REFERENCE ALIGNMENT (documented, deterministic)
-----------------------------------------------
The raw data does NOT align track rows with route rows (0/90 trips have
equal lengths), so observation-level reference labels must be derived. For
each GPS observation, in time order, we project it onto the reference-route
polyline inside a monotone sliding window of route arcs and take the
nearest route arc as the reference label:

* window: positions [r - backtrack, r + W] around the current route
  position r (monotone, never moves backwards); W grows adaptively after
  large time gaps (W >= dt * speed_cap / mean_route_arc_length).
* reliable label: snap distance <= snap_threshold_m (default 25 m).
  Otherwise the observation is UNRELIABLE and is EXCLUDED from accuracy
  (counted in coverage, never fabricated).
* soft advance: if snap distance <= 2x threshold the pointer still advances
  to the best position (keeps alignment monotone through noisy patches);
  beyond that the pointer stays.

METRICS (all computed only on reliable aligned observations)
------------------------------------------------------------
* top1_exact / top1_equiv : classifier-only top-scoring candidate equals
  the reference arc (exact arc-ID equality / or same directed endpoints,
  i.e. duplicate-arc twins count as equivalent -- the dataset contains many
  duplicated arc rows pointing to the same physical segment).
* vit_exact / vit_equiv   : same for the Viterbi-decoded arc.
* candidate_miss_rate     : reference arc absent from the candidate set
  (strict ID and endpoint-equivalent variants). Computed on the same
  candidate sets both systems consume.
* route continuity        : fraction of consecutive matched-arc pairs that
  are driveable (same arc, head-to-tail join, or a directed path within
  continuity_cutoff_m). Reported for classifier-only, Viterbi, and the
  reference route itself (baseline).
* sequence-level metric   : per trip, arc-length-weighted top-1 accuracy
  over reliable observations (weights = reference arc length in metres);
  aggregated as mean/median over trips with >= min_reliable reliable
  observations, plus the fraction of trips with a perfect sequence.
  Trips below the minimum are listed as excluded -- nothing is invented.

PROVISIONAL MODE: when no classifier scores exist yet, candidates come from
the documented distance-based generator; every output is tagged
score_source="provisional_distance_based" and must not be reported as an ML
result.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .candidates import (generate_candidates_provisional, ProvisionalCandidateConfig,
                         normalize_candidates)
from .geo import LocalMetric, haversine_m
from .graph import RoadGraph, SearchCache
from .viterbi import DecoderConfig, match_route

PROVISIONAL_TAG = "provisional_distance_based"


@dataclass
class EvalConfig:
    window_arcs: int = 50            # base alignment window (route positions)
    window_cap: int = 2000
    backtrack: int = 2               # how far back the pointer may look
    snap_threshold_m: float = 25.0   # reliability threshold for a reference label
    min_reliable_for_sequence: int = 20
    continuity_cutoff_m: float = 100.0
    speed_cap_mps: float = 38.0      # adaptive window growth after gaps


@dataclass
class TripData:
    trip_id: str
    track_lonlat: np.ndarray
    ts: np.ndarray
    route_ids: np.ndarray | None     # reference arc IDs (.route) -- EVALUATION ONLY


def load_trip(trip_dir: str, trip_id: str) -> TripData:
    """Adapter: load .track and .route for one trip from the raw dataset."""
    track = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.track"), ndmin=2)
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    route = np.atleast_1d(np.loadtxt(route_path, ndmin=1)) if os.path.isfile(route_path) else None
    route_ids = np.round(route).astype(np.int64) if route is not None and route.size else None
    return TripData(trip_id, track[:, :2], track[:, 2], route_ids)


def _segment_dist_many(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance + parameter t for many segments."""
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    safe = denom > 0
    t = np.zeros_like(abx)
    t[safe] = ((px - ax[safe]) * abx[safe] + (py - ay[safe]) * aby[safe]) / denom[safe]
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    d = np.hypot(px - cx, py - cy)
    return d, t


def align_reference(graph: RoadGraph, track_lonlat, ts, route_ids, cfg: EvalConfig):
    """Deterministic monotone projection of the track onto the reference route.

    Returns (ref_arc int64 array (-1 = unreliable), snap_m float array,
    reliable bool array). Uses .route content -- evaluation only.
    """
    n = len(track_lonlat)
    ref = np.full(n, -1, dtype=np.int64)
    snap = np.full(n, np.inf)
    if route_ids is None or len(route_ids) == 0 or n == 0:
        return ref, snap, np.zeros(n, dtype=bool)

    metric = LocalMetric(graph.nodes_lonlat)
    R = route_ids  # 1D array of arc IDs, evaluation only

    # Build directed segment representation using actual lon/lat coords
    # graph.nodes_lonlat cols: [0]=lon, [1]=lat
    u_lon = graph.nodes_lonlat[graph.arcs[R, 0], 0]
    u_lat = graph.nodes_lonlat[graph.arcs[R, 0], 1]
    v_lon = graph.nodes_lonlat[graph.arcs[R, 1], 0]
    v_lat = graph.nodes_lonlat[graph.arcs[R, 1], 1]

    ux, uy = metric.to_xy(u_lon, u_lat)
    vx, vy = metric.to_xy(v_lon, v_lat)

    mean_route_len = float(np.mean(graph.arc_len_m[R])) if len(R) else 1.0
    mean_route_len = max(mean_route_len, 1.0)

    tx, ty = metric.to_xy(track_lonlat[:, 0], track_lonlat[:, 1])
    r = 0  # current route position (pointer, monotone)
    dt_prev = 1.0
    for i in range(n):
        if i > 0:
            dt_prev = float(ts[i] - ts[i - 1])
        grow = int(np.ceil(dt_prev * cfg.speed_cap_mps / mean_route_len)) + 2
        W = int(min(max(cfg.window_arcs, grow), cfg.window_cap))
        lo = max(0, r - cfg.backtrack)
        # hi bound: ensure we don't exceed route length
        hi = min(len(R), r + W + 1)
        if lo >= hi:
            # window empty, pointer stays
            snap[i] = np.inf
            continue
        d, t = _segment_dist_many(float(tx[i]), float(ty[i]), ux[lo:hi], uy[lo:hi],
                                  vx[lo:hi], vy[lo:hi])
        best = int(np.argmin(d))
        bd = float(d[best])
        snap[i] = bd
        pos = lo + best
        if bd <= cfg.snap_threshold_m:
            ref[i] = int(R[pos])
            r = pos
        elif bd <= 2.0 * cfg.snap_threshold_m:
            ref[i] = int(R[pos])
            r = pos
        # else: pointer stays; ref[i] stays -1, reliable stays False
    reliable = snap <= cfg.snap_threshold_m
    return ref, snap, reliable


def _driveable_flags(graph: RoadGraph, arc_seq, cutoff_m: float, cache: SearchCache | None):
    """bool flags for consecutive pairs in `arc_seq` that are driveable."""
    flags = []
    seq = [int(a) for a in arc_seq if a is not None]
    for p, q in zip(seq[:-1], seq[1:]):
        if p == q:
            flags.append(True)
            continue
        eu = int(graph.arcs[p, 1])
        vs = int(graph.arcs[q, 0])
        if eu == vs:
            flags.append(True)
            continue
        if cache is not None:
            res = cache.get(eu, cutoff_m)
        else:
            res = graph.dijkstra(eu, cutoff_m)
        flags.append(vs in res.dist)
    n_pairs = max(0, len(seq) - 1)
    arr = np.zeros(n_pairs, dtype=bool)
    if flags:
        arr[: len(flags)] = flags
    return arr


def _top1_arcs(sets):
    """Classifier-only prediction: highest-scoring candidate per observation."""
    out = []
    for cs in sets:
        if len(cs) == 0:
            out.append(None)
        else:
            j = int(np.argmax(cs.scores))  # ties -> first = lowest arc ID
            out.append(int(cs.arc_ids[j]))
    return out


def _endpoint_eq(graph: RoadGraph, a: int, b: int) -> bool:
    return (int(graph.arcs[a, 0]) == int(graph.arcs[b, 0])
            and int(graph.arcs[a, 1]) == int(graph.arcs[b, 1]))


def compare_for_trip(graph: RoadGraph, trip: TripData, result, sets, cfg: EvalConfig) -> dict:
    """Metrics for one trip (classifier-only top-1 vs Viterbi, same candidates)."""
    n = len(trip.track_lonlat)
    ref, snap, reliable = align_reference(graph, trip.track_lonlat, trip.ts,
                                          trip.route_ids, cfg)
    top1 = _top1_arcs(sets)
    vit = result.arc_ids

    rel = reliable & (trip.route_ids is not None)
    n_reliable = int(rel.sum())
    m = {
        "trip_id": trip.trip_id,
        "n_obs": n,
        "n_route_refs": 0 if trip.route_ids is None else int(len(trip.route_ids)),
        "n_reliable": n_reliable,
        "coverage": (n_reliable / n) if n else None,
        "snap_median_m": float(np.median(snap[rel])) if n_reliable else None,
        "n_top1_missing": sum(1 for i in np.flatnonzero(rel) if top1[i] is None),
        "n_viterbi_skipped": sum(1 for i in np.flatnonzero(rel) if vit[i] is None),
    }

    if n_reliable == 0:
        for key in ("top1_exact", "top1_equiv", "vit_exact", "vit_equiv",
                    "miss_rate", "miss_rate_equiv", "seq_w_exact_top1",
                    "seq_w_exact_vit", "seq_w_equiv_top1", "seq_w_equiv_vit",
                    "cont_top1", "cont_vit", "cont_ref"):
            m[key] = None
        m["align_available"] = False
        return m

    idx = np.flatnonzero(rel)
    w_ref = graph.arc_len_m[ref[idx]]
    top1_exact = np.array([top1[i] is not None and top1[i] == int(ref[i]) for i in idx])
    top1_equiv = np.array([top1[i] is not None and (top1[i] == int(ref[i])
                           or _endpoint_eq(graph, top1[i], int(ref[i]))) for i in idx])
    vit_exact = np.array([vit[i] is not None and vit[i] == int(ref[i]) for i in idx])
    vit_equiv = np.array([vit[i] is not None and (vit[i] == int(ref[i])
                          or _endpoint_eq(graph, vit[i], int(ref[i]))) for i in idx])

    # candidate miss rate on reliable observations (same candidate sets for both)
    miss, miss_eq = [], []
    for i in idx:
        ids = sets[i].arc_ids
        r = int(ref[i])
        strict = bool(np.any(ids == r)) if len(ids) else False
        soft = strict or any(_endpoint_eq(graph, int(a), r) for a in ids)
        miss.append(not strict)
        miss_eq.append(not soft)

    m["top1_exact"] = float(top1_exact.mean())
    m["top1_equiv"] = float(top1_equiv.mean())
    m["vit_exact"] = float(vit_exact.mean())
    m["vit_equiv"] = float(vit_equiv.mean())
    m["miss_rate"] = float(np.mean(miss))
    m["miss_rate_equiv"] = float(np.mean(miss_eq))

    # sequence-level: arc-length-weighted accuracy per trip
    m["seq_w_exact_top1"] = float(np.sum(w_ref * top1_exact) / np.sum(w_ref))
    m["seq_w_exact_vit"] = float(np.sum(w_ref * vit_exact) / np.sum(w_ref))
    m["seq_w_equiv_top1"] = float(np.sum(w_ref * top1_equiv) / np.sum(w_ref))
    m["seq_w_equiv_vit"] = float(np.sum(w_ref * vit_equiv) / np.sum(w_ref))

    # route continuity (consecutive matched pairs driveable)
    cache = SearchCache(graph, 256)
    m["cont_top1"] = float(np.mean(_driveable_flags(graph, [top1[i] for i in idx],
                                                   cfg.continuity_cutoff_m, cache)))
    m["cont_vit"] = float(np.mean(_driveable_flags(graph, [vit[i] for i in idx],
                                                  cfg.continuity_cutoff_m, cache)))
    m["cont_ref"] = float(np.mean(_driveable_flags(graph, list(trip.route_ids),
                                                  cfg.continuity_cutoff_m, cache)))
    m["align_available"] = True
    m["snap_p95_m"] = float(np.percentile(snap[rel], 95))
    return m


def aggregate(per_trip: list, cfg: EvalConfig) -> dict:
    """Cross-trip summary. Sequence metric only over trips meeting the minimum."""
    seq_rows = [t for t in per_trip if t.get("align_available")
                and t["n_reliable"] >= cfg.min_reliable_for_sequence]
    excluded = [t["trip_id"] for t in per_trip
                if not (t.get("align_available")
                        and t["n_reliable"] >= cfg.min_reliable_for_sequence)]

    def col(key):
        vals = [t[key] for t in seq_rows if t.get(key) is not None]
        return vals

    def mstats(key):
        vals = col(key)
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}

    tot_rel = sum(t["n_reliable"] for t in seq_rows)

    def pooled(key_prefix, exact=True):
        num = den = 0.0
        for t in seq_rows:
            if t.get(f"seq_w_{key_prefix}") is None:
                continue
            num += t[f"seq_w_{key_prefix}"] * t["sum_w_ref"]
            den += t["sum_w_ref"]
        return (num / den) if den > 0 else None

    # total reference weight per trip (for pooling)
    return {
        "n_trips": len(per_trip),
        "n_trips_in_sequence": len(seq_rows),
        "n_trips_excluded": len(excluded),
        "excluded_trip_ids": excluded,
        "exclusion_rule": f"reliable observations < {cfg.min_reliable_for_sequence} "
                          "or no reference alignment",
        "top1_exact": mstats("top1_exact"),
        "top1_equiv": mstats("top1_equiv"),
        "vit_exact": mstats("vit_exact"),
        "vit_equiv": mstats("vit_equiv"),
        "miss_rate": mstats("miss_rate"),
        "miss_rate_equiv": mstats("miss_rate_equiv"),
        "cont_top1": mstats("cont_top1"),
        "cont_vit": mstats("cont_vit"),
        "cont_ref": mstats("cont_ref"),
        "seq_w_exact_top1": mstats("seq_w_exact_top1"),
        "seq_w_exact_vit": mstats("seq_w_exact_vit"),
        "seq_w_equiv_top1": mstats("seq_w_equiv_top1"),
        "seq_w_equiv_vit": mstats("seq_w_equiv_vit"),
        "perfect_sequence_rate_exact_vit": (
            float(np.mean([t["seq_w_exact_vit"] >= 0.999 for t in seq_rows]))
            if seq_rows else None),
        "perfect_sequence_rate_exact_top1": (
            float(np.mean([t["seq_w_exact_top1"] >= 0.999 for t in seq_rows]))
            if seq_rows else None),
    }


def run_trip(graph_dir: str, trip: TripData,
             decoder_cfg: DecoderConfig | None = None,
             cand_cfg: ProvisionalCandidateConfig | None = None,
             eval_cfg: EvalConfig | None = None,
             candidates=None,
             score_source: str = PROVISIONAL_TAG):
    """Full pipeline for one trip: candidates -> decode -> compare.

    `candidates=None` uses the documented PROVISIONAL distance-based
    generator (decoder testing only). Pass a normalised candidate list from
    the ML pipeline with `score_source` labelled appropriately for real runs.
    """
    decoder_cfg = decoder_cfg or DecoderConfig()
    cand_cfg = cand_cfg or ProvisionalCandidateConfig()
    eval_cfg = eval_cfg or EvalConfig()

    graph = RoadGraph.from_trip_files(graph_dir, trip.trip_id)
    t0 = time.perf_counter()
    gen_stats = {}
    if candidates is None:
        norm, gen_stats = generate_candidates_provisional(graph, trip.track_lonlat, cand_cfg)
        sets = norm.sets
    else:
        norm = normalize_candidates(candidates, graph.n_arcs, expect_obs=len(trip.ts))
        sets = norm.sets
    result = match_route(graph, trip.track_lonlat, trip.ts, None,
                         config=decoder_cfg, normalized=norm)
    cand_time = time.perf_counter() - t0

    m = compare_for_trip(graph, trip, result, sets, eval_cfg)
    m["score_source"] = score_source
    m["n_candidate_gen_fallback"] = gen_stats.get("n_fallback_nearest", 0)
    m["n_no_candidates"] = result.stats["n_no_candidates"]
    m["n_gap_steps"] = result.stats["n_gap_steps"]
    m["n_disconnected_steps"] = result.stats["n_disconnected_steps"]
    m["n_chain_breaks"] = result.stats["n_chain_breaks"]
    m["n_route_arcs_matched"] = result.stats["n_route_arcs"]
    m["decode_runtime_s"] = result.stats["runtime_s"]
    m["candgen_runtime_s"] = round(cand_time, 3)
    # reference weight sums (used for pooling across trips)
    _, _, reliable = align_reference(graph, trip.track_lonlat, trip.ts,
                                     trip.route_ids, eval_cfg)
    ref, _, _ = align_reference(graph, trip.track_lonlat, trip.ts,
                                trip.route_ids, eval_cfg)
    idx = np.flatnonzero(reliable)
    m["sum_w_ref"] = float(graph.arc_len_m[ref[idx]].sum()) if idx.size else 0.0
    return m, result, sets


# --------------------------------------------------------------------------- #
# batch runner + outputs
# --------------------------------------------------------------------------- #
def run_split(dataset_root: str, split: str, trip_ids=None,
              decoder_cfg=None, cand_cfg=None, eval_cfg=None,
              out_dir=None, score_source: str = PROVISIONAL_TAG):
    """Run the comparison over a whole split; writes CSV + JSON, returns summary."""
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    if trip_ids is None:
        trip_ids = sorted(d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d)))
    per_trip = []
    results = {}
    for tid in trip_ids:
        trip = load_trip(os.path.join(split_dir, tid), tid)
        m, result, _ = run_trip(os.path.join(split_dir, trip.trip_id), trip, decoder_cfg, cand_cfg, eval_cfg,
                                candidates=None, score_source=score_source)
        per_trip.append(m)
        results[tid] = result
        print(f"  {tid}: reliable {m['n_reliable']}/{m['n_obs']}  "
              f"top1_exact {m['top1_exact'] if m['top1_exact'] is not None else 'n/a'}  "
              f"vit_exact {m['vit_exact'] if m['vit_exact'] is not None else 'n/a'}  "
              f"cont_vit {m['cont_vit'] if m['cont_vit'] is not None else 'n/a'}  "
              f"({m['decode_runtime_s']}s + {m['candgen_runtime_s']}s)")
    agg = aggregate(per_trip, eval_cfg or EvalConfig())
    summary = {
        "score_source": score_source,
        "provisional_disclaimer": (
            "PROVISIONAL: candidate scores are distance-based stand-ins from "
            "route_reconstruction.candidates.generate_candidates_provisional, NOT "
            "trained classifier outputs. Metrics validate the decoder only and "
            "must not be reported as ML results.")
            if score_source == PROVISIONAL_TAG else None,
        "split": split,
        "dataset_root": dataset_root,
        "decoder_config": asdict(decoder_cfg or DecoderConfig()),
        "candidate_config": asdict(cand_cfg or ProvisionalCandidateConfig()),
        "eval_config": asdict(eval_cfg or EvalConfig()),
        "aggregate": agg,
        "per_trip": per_trip,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"compare_{split}.csv")
        keys = list(per_trip[0].keys()) if per_trip else []
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in per_trip:
                w.writerow(row)
        json_path = os.path.join(out_dir, f"compare_{split}_summary.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(summary), fh, indent=1)
        print(f"wrote {csv_path}")
        print(f"wrote {json_path}")
    return summary, results


def _json_safe(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj
