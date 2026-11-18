"""Candidate road arcs per GPS observation + emission log-scores (Member 3).

HIDDEN-STATE DEFINITION
-----------------------
The HMM hidden state at GPS observation i is *which road arc the vehicle is
on*. The candidate set of observation i is therefore the state space at step
i; candidate arc IDs are the raw zero-based arc row IDs from .arcs and are
NEVER renumbered. Every score stays paired with its original candidate ID.

ACCEPTED CANDIDATE INPUT (one python list of length N, one entry per
observation; N must equal the number of track rows)
-----------------------------------------------------------------------
Each per-observation entry may be an empty list/None (missing candidates) or
any iterable of items in one of these forms:

    (arc_id, score)                         tuple or 2-list
    (arc_id, score, offset_m)               3-list, offset along arc in metres
    {"arc_id": int, "score": float,
     "offset_m": float | None,              optional keys
     "dist_m": float | None}

    arc_id  : zero-based arc row ID (0 <= arc_id < graph.n_arcs)
    score   : classifier output for this candidate -- semantics MUST be
              documented by the producer; see score_to_emission_log() for the
              accepted semantics ("probability" | "logit" | "decision").
    offset_m: metres already travelled along that arc from its start node
              (0..arc length) at the observation -- optional; nan when absent.
              Used by the transition model ("distance already travelled on
              the next arc"); when absent a documented fallback is used.
    dist_m  : optional distance (metres) from the observation to the arc;
              used only by the optional geometric emission refinement.

SCORE -> EMISSION LOG-SCORE (score_to_emission_log)
---------------------------------------------------
Accepted semantics (explicit `score_type` argument, or "auto"):

  "probability" : scores are P(candidate | observation) in [0, 1] (they do
                  not need to sum to 1). Emission = log(clip(p, eps, 1)).
  "logit"       : scores are unbounded real-valued logits of one candidate
                  vs the others; emission = log softmax over the candidates
                  of the same observation (shift-invariant, numerically
                  stable).
  "decision"    : raw decision-function / margin values with unknown range;
                  emission = log(clip(minmax_norm_within_observation, eps, 1)).
  "auto"        : all finite scores in [0, 1]  -> "probability"
                  otherwise                    -> "decision"
                  (auto can NOT distinguish "logit" from "decision"; the ML
                  member should pass score_type explicitly.)

Validation: non-finite (NaN/inf) scores are dropped from the candidate set
(counted in stats); probabilities outside [0,1] are clipped; log(0) is
avoided by the eps floor (default 1e-12, i.e. at most -27.6 nats).

PROVISIONAL CANDIDATE GENERATOR (adapter -- clearly labelled, NOT an ML model)
-----------------------------------------------------------------------------
`generate_candidates_provisional()` builds candidate sets directly from
geometry (arcs within a radius, scored by exp(-distance/sigma)) so that the
decoder can be exercised end-to-end before Member 2's classifier exists.
Every output produced with it is marked `score_source =
"provisional_distance_based"` and must NOT be reported as an ML result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geo import LocalMetric, point_segment_projection

SCORE_TYPES = ("probability", "logit", "decision", "auto")


@dataclass
class CandidateSet:
    """Normalised candidate set for one observation (parallel arrays)."""

    arc_ids: np.ndarray      # int64   (k,) zero-based arc row IDs
    scores: np.ndarray       # float64 (k,) raw scores as provided
    offsets_m: np.ndarray    # float64 (k,) metres along arc, NaN when unknown
    dists_m: np.ndarray      # float64 (k,) obs-to-arc distance, NaN when unknown

    def __len__(self):
        return int(self.arc_ids.size)


@dataclass
class NormalizedCandidates:
    """All candidate sets + normalisation statistics."""

    sets: list
    n_invalid_ids: int = 0       # candidates with arc_id out of range -> dropped
    n_invalid_scores: int = 0    # candidates with NaN/inf scores -> dropped
    n_dedup: int = 0             # duplicate arc IDs within one observation -> best kept
    n_empty: int = 0             # observations with no usable candidate


def normalize_candidates(raw, n_arcs: int, expect_obs: int | None = None) -> NormalizedCandidates:
    """Validate + normalise any accepted candidate input form.

    * keeps every score paired with its original candidate ID
    * drops candidates whose arc_id is out of range (counted)
    * drops candidates whose score is not finite (counted)
    * duplicates of the same arc_id inside ONE observation: keeps the best
      (max) score and the first offset/distance seen (deterministic)
    * candidate list is sorted by arc_id ascending (deterministic tie-breaks
      downstream: np.argmax picks the first = lowest arc ID on ties)
    """
    if expect_obs is not None and len(raw) != expect_obs:
        raise ValueError(
            f"candidates has {len(raw)} entries but track has {expect_obs} observations"
        )
    sets: list = []
    n_bad_id = n_bad_score = n_dup = n_empty = 0
    for obs in raw:
        if obs is None:
            obs = []
        per: dict = {}
        order: list = []
        for item in obs:
            if isinstance(item, dict):
                aid = item.get("arc_id", None)
                sc = item.get("score", None)
                off = item.get("offset_m", None)
                dm = item.get("dist_m", None)
            else:
                seq = list(item)
                if len(seq) < 2:
                    continue
                aid, sc = seq[0], seq[1]
                off = seq[2] if len(seq) >= 3 else None
                dm = seq[3] if len(seq) >= 4 else None
            try:
                aid_i = int(aid)
            except (TypeError, ValueError):
                n_bad_id += 1
                continue
            try:
                sc_f = float(sc)
            except (TypeError, ValueError):
                n_bad_score += 1
                continue
            if aid_i < 0 or aid_i >= n_arcs:
                n_bad_id += 1
                continue
            if not math.isfinite(sc_f):
                n_bad_score += 1
                continue
            off_f = float(off) if off is not None and math.isfinite(float(off)) else float("nan")
            dm_f = float(dm) if dm is not None and math.isfinite(float(dm)) else float("nan")
            if aid_i in per:
                n_dup += 1
                prev = per[aid_i]
                # best (max) score wins; the FIRST valid offset / dist seen is kept
                new_off = prev[1] if not np.isnan(prev[1]) else off_f
                new_dm = prev[2] if not np.isnan(prev[2]) else dm_f
                if sc_f > prev[0]:
                    per[aid_i] = (sc_f, new_off, new_dm)
                else:
                    per[aid_i] = (prev[0], new_off, new_dm)
            else:
                per[aid_i] = (sc_f, off_f, dm_f)
                order.append(aid_i)
        if not per:
            n_empty += 1
            sets.append(CandidateSet(
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            ))
            continue
        order.sort()
        arr_a = np.array(order, dtype=np.int64)
        arr_s = np.array([per[a][0] for a in order], dtype=np.float64)
        arr_o = np.array([per[a][1] for a in order], dtype=np.float64)
        arr_d = np.array([per[a][2] for a in order], dtype=np.float64)
        sets.append(CandidateSet(arr_a, arr_s, arr_o, arr_d))
    return NormalizedCandidates(sets, n_bad_id, n_bad_score, n_dup, n_empty)


def detect_score_type(all_scores) -> str:
    """Documented auto-detection heuristic (see module docstring)."""
    vals = np.asarray([s for s in all_scores if s is not None and np.isfinite(s)],
                      dtype=np.float64)
    if vals.size == 0:
        return "probability"
    if vals.min() >= 0.0 and vals.max() <= 1.0:
        return "probability"
    return "decision"


def score_to_emission_log(sets, score_type: str = "auto", eps: float = 1e-12,
                          dist_weight: float = 0.0, dist_sigma_m: float = 20.0):
    """Convert raw candidate scores into emission log-scores (natural log).

    See module docstring for the exact semantics of each score_type. Returns
    a list of float64 arrays parallel to each CandidateSet. An optional
    geometric refinement can be enabled with dist_weight > 0 (requires
    dist_m on candidates; adds -dist_weight * dist/dist_sigma_m nats).
    """
    if score_type not in SCORE_TYPES:
        raise ValueError(f"score_type must be one of {SCORE_TYPES}, got {score_type!r}")
    if score_type == "auto":
        all_scores = []
        for cs in sets:
            all_scores.extend(cs.scores.tolist())
        score_type = detect_score_type(all_scores)

    out = []
    for cs in sets:
        k = len(cs)
        if k == 0:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        s = cs.scores.astype(np.float64)
        if score_type == "probability":
            s = np.clip(s, 0.0, 1.0)
            e = np.log(np.clip(s, eps, 1.0))
        elif score_type == "logit":
            z = s - s.max()  # shift-invariant, avoids overflow
            ez = np.exp(z)
            e = z - np.log(ez.sum())
        else:  # decision / margins -> min-max within the observation
            lo, hi = float(s.min()), float(s.max())
            if hi > lo:
                u = (s - lo) / (hi - lo)
            else:
                u = np.ones_like(s)  # all equal -> no discrimination
            e = np.log(np.clip(u, eps, 1.0))
        if dist_weight != 0.0:
            d = cs.dists_m
            known = np.isfinite(d)
            e = e - np.where(known, dist_weight * np.nan_to_num(d) / dist_sigma_m, 0.0)
        out.append(e)
    return out


# --------------------------------------------------------------------------- #
# PROVISIONAL candidate generator (adapter, used only until Member 2's model
# produces real candidate scores). Clearly labelled; never report as ML.
# --------------------------------------------------------------------------- #
@dataclass
class ProvisionalCandidateConfig:
    k: int = 5                 # max candidates per observation
    radius_m: float = 25.0     # search radius around the observation
    sigma_m: float = 20.0      # score scale: p = exp(-dist / sigma)
    fallback_nearest: bool = True  # if radius is empty, take nearest arc anyway
    cell_m: float = 25.0       # spatial index cell size


class ArcGridIndex:
    """Simple cell-grid spatial index over arcs (pure numpy, no scipy).

    Every arc is registered in the grid cells of points sampled along its
    segment (start, end, and interior points every ~cell_m metres), so lookups
    return all arcs whose *segment* passes near a query point.
    """

    def __init__(self, graph, metric: LocalMetric, cell_m: float = 25.0):
        self.graph = graph
        self.metric = metric
        self.cell = float(cell_m)
        self.buckets: dict = {}
        nodes = graph.nodes_lonlat
        nx, ny = metric.to_xy(nodes[:, 0], nodes[:, 1])
        self.node_x, self.node_y = nx, ny
        for a in range(graph.n_arcs):
            if not graph.valid_arc[a]:
                continue
            u, v = int(graph.arcs[a, 0]), int(graph.arcs[a, 1])
            ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
            seg_len = float(np.hypot(bx - ax, by - ay))
            n_samp = max(2, int(math.ceil(seg_len / max(cell_m, 1e-6))) + 1)
            for t in np.linspace(0.0, 1.0, n_samp):
                cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
                self._register(a, cx, cy)

    def _cell(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _register(self, arc_id, x, y):
        self.buckets.setdefault(self._cell(x, y), []).append(arc_id)

    def nearby(self, x, y) -> set:
        cx, cy = self._cell(x, y)
        out = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                b = self.buckets.get((cx + dx, cy + dy))
                if b:
                    out.update(b)
        return out

    def nearby_expanding(self, x, y, max_rings: int = 8) -> set:
        """Cells in the 3x3 neighbourhood, then ring 2, 3, ... until non-empty.

        Deterministic ring order (sorted by |dx|+|dy|, then dx, dy). Used as a
        fast 'nearest arc' fallback that never scans all arcs.
        """
        cx, cy = self._cell(x, y)
        out = self.nearby(x, y)
        if out:
            return out
        for r in range(2, max_rings + 2):
            ring = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                    if max(abs(dx), abs(dy)) == r]
            ring.sort(key=lambda t: (abs(t[0]) + abs(t[1]), t[0], t[1]))
            for dx, dy in ring:
                b = self.buckets.get((cx + dx, cy + dy))
                if b:
                    out.update(b)
            if out:
                return out
        return out


def generate_candidates_provisional(graph, track_lonlat, cfg: ProvisionalCandidateConfig | None = None):
    """PROVISIONAL distance-based candidate sets (NOT classifier scores).

    For each observation: all arcs with segment distance <= radius_m become
    candidates (up to k, nearest first; ties broken by lowest arc ID). If the
    radius is empty and fallback_nearest is on, the single nearest arc is
    added so the decoder is never left without candidates (flagged in stats).
    score = exp(-dist / sigma_m)  (probability-like; a stand-in for a real
    classifier). offset_m and dist_m are filled from the exact projection.

    Returns (NormalizedCandidates, stats_dict). Every result produced with
    these candidates must be labelled `score_source="provisional_distance_based"`.
    """
    cfg = cfg or ProvisionalCandidateConfig()
    metric = LocalMetric(track_lonlat)
    index = ArcGridIndex(graph, metric, cfg.cell_m)
    ox, oy = metric.to_xy(track_lonlat[:, 0], track_lonlat[:, 1])

    raw = []
    n_fallback = 0
    for i in range(len(ox)):
        px, py = float(ox[i]), float(oy[i])
        cand = index.nearby(px, py)
        scored = []
        for a in cand:
            u, v = int(graph.arcs[a, 0]), int(graph.arcs[a, 1])
            d, t = point_segment_projection(px, py, index.node_x[u], index.node_y[u],
                                            index.node_x[v], index.node_y[v])
            if d <= cfg.radius_m:
                scored.append((d, a, t))
        if not scored and cfg.fallback_nearest:
            # ring-expanding grid lookup: nearest arcs only, never a full scan
            all_arcs = index.nearby_expanding(px, py)
            for a in all_arcs:
                u, v = int(graph.arcs[a, 0]), int(graph.arcs[a, 1])
                d, t = point_segment_projection(px, py, index.node_x[u], index.node_y[u],
                                                index.node_x[v], index.node_y[v])
                scored.append((d, a, t))
            n_fallback += 1
        scored.sort(key=lambda x: (x[0], x[1]))  # nearest first, then lowest arc ID
        scored = scored[: max(1, int(cfg.k))]
        obs = []
        for d, a, t in scored:
            offset = float(np.clip(t, 0.0, 1.0)) * float(graph.arc_len_m[a])
            obs.append((a, math.exp(-d / max(cfg.sigma_m, 1e-6)), offset, d))
        raw.append(obs)
    norm = normalize_candidates(raw, graph.n_arcs, expect_obs=len(ox))
    stats = {
        "generator": "provisional_distance_based",
        "k": cfg.k, "radius_m": cfg.radius_m, "sigma_m": cfg.sigma_m,
        "n_fallback_nearest": n_fallback,
        "n_invalid_ids": norm.n_invalid_ids, "n_invalid_scores": norm.n_invalid_scores,
        "n_dedup": norm.n_dedup, "n_empty": norm.n_empty,
    }
    return norm, stats
