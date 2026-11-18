"""HMM / Viterbi route-reconstruction decoder (Member 3).

MODEL
-----
* Hidden state at GPS observation i  =  the road arc the vehicle is on.
  The state space at step i is exactly the candidate set of observation i
  (candidate arc IDs are preserved verbatim from the candidate input).
* Emission log-score of candidate (arc a) at observation i:
      log P(obs_i | on arc a)  =  converted classifier score
      (see candidates.score_to_emission_log; probabilities / logits /
      decision scores are converted as documented there).
* Transition log-score from arc a (observation i) to arc b (observation
  i+1) uses ONLY directed road connectivity + geometry + time:

      d_obs    = great-circle displacement between obs i and obs i+1 (m)
      dt       = elapsed time between the two observations (s)
      feasible = speed_cap_mps * dt + feasibility_slack_m
      road     = shortest directed road-travel distance from the position
                 reached on arc a to the position on arc b where obs i+1
                 sits, moving forward along arcs:
                     road = (len(a) - s_a)                remaining distance
                            + dijkstra(end(a) -> start(b)) between-arcs
                            + s_b                          travelled already
                 s_a / s_b come from candidate offsets when supplied;
                 otherwise documented fallbacks are used (half arc length
                 for different arcs; min(d_obs, len(a)) for the same arc).
      penalty  = transition_weight *
                   ( |road - d_obs| + max(0, road - feasible) ) / dist_sigma_m
                 (displacement consistency + speed-feasibility soft tail)
      gap      : dt > gap_threshold_s  ->  transition treated as
                 UNCONSTRAINED: every pair (including staying on the same
                 arc) gets the constant -gap_uniform_penalty; no road search
                 is run (a big gap can jump anywhere; emission decides).
      no path  : no directed path between end(a) and start(b) within the
                 bounded search cutoff -> -disconnect_penalty and the
                 DISCONNECTED_STEP flag on the chosen step (a soft, finite
                 penalty so Viterbi stays deterministic even when nothing
                 is reachable; it is a floor for unreachable pairs, while
                 within-cutoff penalties can grow beyond it -- both are
                 practically forbidden states).

* Same-arc stays are allowed at every step (a == b is scored like any pair).

DETERMINISM
-----------
All argmax ties resolve to the lowest candidate index (candidates are sorted
by arc ID at normalisation, so ties resolve to the lowest arc ID). No
randomness anywhere; identical inputs give identical outputs.

EDGE CASES (per-observation statuses; see MatchResult)
-----------------------------------------------------
* NO_CANDIDATES      observation has no usable candidate -> matched arc is
                     None and the Viterbi chain is BROKEN: the next
                     observation with candidates starts a fresh segment
                     (no transition term from before the break).
* GAP_BEFORE         dt from the previous observation exceeded the gap
                     threshold -> unconstrained transition for that step.
* DISCONNECTED_STEP  the chosen transition had no directed path within the
                     search cutoff (disconnect fallback was used).
* OK                 none of the above.

REFERENCE ROUTES (.route) ARE NOT TOUCHED ANYWHERE IN THIS MODULE.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np

from .candidates import (NormalizedCandidates, CandidateSet,
                         normalize_candidates, score_to_emission_log)
from .geo import haversine_m
from .graph import RoadGraph, SearchCache

STATUS_OK = "OK"
STATUS_NO_CAND = "NO_CANDIDATES"
STATUS_GAP = "GAP_BEFORE"
STATUS_DISCONNECTED = "DISCONNECTED_STEP"


@dataclass
class DecoderConfig:
    """All decoder knobs (documented in route_reconstruction/README.md)."""

    # --- emission ---------------------------------------------------------
    score_type: str = "auto"        # "probability" | "logit" | "decision" | "auto"
    eps: float = 1e-12              # probability floor before log (avoids log(0))
    dist_weight: float = 0.0        # optional geometric emission refinement (default OFF)
    dist_sigma_m: float = 20.0      # length scale for the optional refinement

    # --- transitions ------------------------------------------------------
    transition_weight: float = 1.0  # weight of the geometric transition penalty
    dist_sigma_trans_m: float = 25.0  # length scale of the transition penalty
    speed_cap_mps: float = 38.0     # plausibility speed cap (~137 km/h)
    feasibility_slack_m: float = 5.0
    gap_threshold_s: float = 30.0   # dt above which transitions are unconstrained
    gap_uniform_penalty: float = 2.0
    disconnect_penalty: float = 30.0  # nats; floor for unreachable pairs
    # bounded Dijkstra cutoff: min(cutoff_max, max(cutoff_min, feasible + margin))
    search_cutoff_min_m: float = 800.0
    search_cutoff_max_m: float = 5000.0
    search_margin_m: float = 300.0
    search_cache_size: int = 512
    max_visited_per_search: int = 50000


@dataclass
class MatchResult:
    """Decoder output (see README for the JSON schema)."""

    arc_ids: list                    # per observation: int arc ID or None
    offsets_m: list                  # per observation: float or None
    statuses: list                   # per observation: list of status strings
    route: list                      # matched arcs in traversal order,
                                     # consecutive repeats collapsed
    route_positions: list            # per observation: index into route, or None
    geometry: dict                   # see geometry.build_route_geometry
    stats: dict                      # counts, timings, configuration echo
    score_source: str = "unknown"    # e.g. "provisional_distance_based" or ML tag

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _transition_matrix(graph: RoadGraph, cache: SearchCache, cfg: DecoderConfig,
                       prev: CandidateSet, nxt: CandidateSet,
                       d_obs: float, dt: float):
    """Transition log-score matrix (prev states x next states) + disconnect map.

    Returns (T, disconnected, road_known) where T[k, j] is the log-score of
    moving from prev candidate k to next candidate j, `disconnected[k, j]`
    marks pairs that used the disconnect fallback, and `road_known[k, j]`
    marks pairs whose road distance was actually computed.
    """
    k1, k2 = len(prev), len(nxt)
    T = np.empty((k1, k2), dtype=np.float64)
    disconnected = np.zeros((k1, k2), dtype=bool)
    road_known = np.zeros((k1, k2), dtype=bool)

    if dt > cfg.gap_threshold_s:
        # large GPS gap: unconstrained transitions, no road search at all
        T[:] = -cfg.gap_uniform_penalty
        return T, disconnected, road_known

    feasible = cfg.speed_cap_mps * max(dt, 0.0) + cfg.feasibility_slack_m
    cutoff = min(cfg.search_cutoff_max_m,
                 max(cfg.search_cutoff_min_m, feasible + cfg.search_margin_m))
    sigma = cfg.dist_sigma_trans_m
    w = cfg.transition_weight

    # per-candidate cached quantities
    prev_end = [int(graph.arcs[a, 1]) for a in prev.arc_ids]
    nxt_start = [int(graph.arcs[a, 0]) for a in nxt.arc_ids]
    prev_off = prev.offsets_m
    nxt_off = nxt.offsets_m
    prev_len = graph.arc_len_m[prev.arc_ids]
    nxt_len = graph.arc_len_m[nxt.arc_ids]
    # remaining distance on the previous arc (fallback: half the arc)
    rem = np.where(np.isfinite(prev_off), prev_len - prev_off, prev_len * 0.5)
    # distance already travelled on the next arc (fallback: half the arc)
    trav = np.where(np.isfinite(nxt_off), nxt_off, nxt_len * 0.5)

    road = np.full((k1, k2), np.inf)
    for k in range(k1):
        a = int(prev.arc_ids[k])
        for j in range(k2):
            b = int(nxt.arc_ids[j])
            if a == b:
                if np.isfinite(prev_off[k]) and np.isfinite(nxt_off[j]):
                    road[k, j] = abs(float(nxt_off[j]) - float(prev_off[k]))
                else:
                    # fallback: the vehicle moved at most along the whole arc
                    road[k, j] = min(d_obs, float(prev_len[k]))
                road_known[k, j] = True
            elif prev_end[k] == nxt_start[j]:
                # arcs join head-to-tail: no intermediate search needed
                road[k, j] = float(rem[k]) + 0.0 + float(trav[j])
                road_known[k, j] = True
            else:
                res = cache.get(prev_end[k], cutoff)
                d = res.dist.get(nxt_start[j])
                if d is None:
                    disconnected[k, j] = True
                else:
                    road[k, j] = float(rem[k]) + float(d) + float(trav[j])
                    road_known[k, j] = True

    with np.errstate(invalid="ignore"):
        excess = np.where(np.isfinite(road), np.maximum(road - feasible, 0.0), 0.0)
        consist = np.where(np.isfinite(road), np.abs(road - d_obs), 0.0)
        pen = w * (consist + excess) / sigma
    T = -pen
    T[disconnected] = -cfg.disconnect_penalty
    return T, disconnected, road_known


def match_route(graph: RoadGraph,
                track_lonlat,
                ts,
                candidates,
                config: DecoderConfig | None = None,
                normalized: NormalizedCandidates | None = None) -> MatchResult:
    """Decode the matched road arc for every GPS observation via Viterbi.

    Parameters
    ----------
    graph        : RoadGraph for this trip (arc IDs preserved from .arcs).
    track_lonlat : (N, 2) array, columns (longitude, latitude) as in .track.
    ts           : (N,) array of timestamps in seconds (may be float).
    candidates   : per-observation candidate input, see candidates.py
                   (list of length N; each entry a list of (arc_id, score[, offset_m])
                   tuples or dicts; empty list / None = missing candidates).
    config       : DecoderConfig (defaults documented in README).
    normalized   : optionally pass a pre-normalised candidate structure to
                   avoid re-normalising (used by the evaluator pipeline).

    Returns
    -------
    MatchResult  (arc_ids / offsets_m / statuses / route / route_positions /
                  geometry / stats). The reference route (.route) is never
                  read by this function.
    """
    t_start = time.perf_counter()
    cfg = config or DecoderConfig()
    track_lonlat = np.asarray(track_lonlat, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    n = int(track_lonlat.shape[0])
    if ts.shape[0] != n:
        raise ValueError("ts must have one entry per track row")
    if normalized is None:
        normalized = normalize_candidates(candidates, graph.n_arcs, expect_obs=n)
    sets = normalized.sets
    emis = score_to_emission_log(sets, cfg.score_type, cfg.eps,
                                 cfg.dist_weight, cfg.dist_sigma_m)

    cache = SearchCache(graph, cfg.search_cache_size, cfg.max_visited_per_search)

    n_obs = n
    arc_ids: list = [None] * n
    offsets_out: list = [None] * n
    statuses: list = [[] for _ in range(n)]
    stats = {
        "n_observations": n_obs,
        "n_no_candidates": normalized.n_empty,
        "n_invalid_ids": normalized.n_invalid_ids,
        "n_invalid_scores": normalized.n_invalid_scores,
        "n_dedup": normalized.n_dedup,
        "n_gap_steps": 0,
        "n_disconnected_steps": 0,
        "n_chain_breaks": 0,
        "dijkstra_searches": cache.misses,
        "dijkstra_cache_hits": cache.hits,
        "dijkstra_nodes_visited": cache.n_visited_total,
    }

    # ---- forward pass (segment-wise: chain breaks at NO_CANDIDATES) --------
    # dp/bp stored per observation with candidates
    dp: dict = {}
    bp: dict = {}
    bdisc: dict = {}
    prev_idx = None  # index of previous observation WITH candidates, or None

    for i in range(n):
        cs = sets[i]
        if len(cs) == 0:
            statuses[i].append(STATUS_NO_CAND)
            if prev_idx is not None:
                stats["n_chain_breaks"] += 1
            prev_idx = None
            continue

        k = len(cs)
        e = emis[i]
        if prev_idx is None:
            dp[i] = e.copy()
            bp[i] = None
            bdisc[i] = None
        else:
            dt = float(ts[i] - ts[prev_idx])
            if dt < 0:
                dt = 0.0  # guard (dataset timestamps are strictly increasing)
            d_obs = float(haversine_m(track_lonlat[prev_idx, 0], track_lonlat[prev_idx, 1],
                                      track_lonlat[i, 0], track_lonlat[i, 1]))
            T, disc, _ = _transition_matrix(graph, cache, cfg, sets[prev_idx], cs,
                                            d_obs, dt)
            if dt > cfg.gap_threshold_s:
                statuses[i].append(STATUS_GAP)
                stats["n_gap_steps"] += 1
            tot = dp[prev_idx][:, None] + T
            bp_i = np.argmax(tot, axis=0)  # ties -> lowest prev candidate index
            dp[i] = tot[bp_i, np.arange(k)]
            bp[i] = bp_i
            bdisc[i] = disc[bp_i, np.arange(k)]
        prev_idx = i

    # ---- backtracking (segment-wise; chain breaks reset the chain) ---------
    last_valid = None
    for i in range(n - 1, -1, -1):
        if i in dp:
            last_valid = i
            break
    arc_ids = [None] * n
    offsets_out = [None] * n
    obs_states: list = [None] * n  # chosen candidate index per observation
    j = last_valid
    while j is not None and j >= 0:
        if j not in dp:
            j -= 1
            continue
        state = int(np.argmax(dp[j]))  # best final state; ties -> lowest arc ID
        while True:
            obs_states[j] = state
            b = bp[j]
            if b is None:
                break  # reached the start of this segment
            state = int(b[state])
            j -= 1
        # skip back over the NO_CANDIDATES observation(s) to the previous segment
        j -= 1
        while j >= 0 and j not in dp:
            j -= 1

    # materialise outputs + statuses (GAP flags were appended during the
    # forward pass; DISCONNECTED_STEP comes from the chosen backpointer)
    for i in range(n):
        s = obs_states[i]
        if s is None:
            continue
        cs = sets[i]
        arc_ids[i] = int(cs.arc_ids[s])
        off = float(cs.offsets_m[s])
        offsets_out[i] = off if math.isfinite(off) else None
        s_i = statuses[i]
        if bdisc[i] is not None and bool(bdisc[i][s]):
            s_i.append(STATUS_DISCONNECTED)
            stats["n_disconnected_steps"] += 1
        if not s_i:
            s_i.append(STATUS_OK)

    # ---- collapse consecutive repeats into the traversal route -------------
    route: list = []
    pos_of_arc: dict = {}
    route_positions: list = [None] * n
    last = None
    for i in range(n):
        a = arc_ids[i]
        if a is None:
            continue
        if a != last:
            route.append(a)
            last = a
        route_positions[i] = len(route) - 1

    stats["n_matched"] = sum(1 for a in arc_ids if a is not None)
    stats["n_route_arcs"] = len(route)
    stats["n_components_graph"] = graph.n_components()
    stats["runtime_s"] = round(time.perf_counter() - t_start, 3)
    stats["config"] = asdict(cfg)

    from .geometry import build_route_geometry
    geometry = build_route_geometry(graph, route)

    return MatchResult(
        arc_ids=arc_ids,
        offsets_m=offsets_out,
        statuses=statuses,
        route=route,
        route_positions=route_positions,
        geometry=geometry,
        stats=stats,
        score_source="unknown",
    )
