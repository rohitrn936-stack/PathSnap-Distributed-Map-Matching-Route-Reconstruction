#!/usr/bin/env python3
"""PathSnap dataset inventory (Member 1 - data pipeline).

Reads every trip folder (train / student_test) and produces a reproducible,
compact inventory covering:

  * file counts and row counts per file type (.track .nodes .arcs .route)
  * coordinate ranges (lon/lat min/max) and geographic clusters ("cities")
  * invalid / missing values (NaN, non-numeric lines, ragged rows, empty files)
  * timestamp order, duplicates, and gap statistics (track is nominally 1 Hz)
  * node and arc validity (ID ranges, self-loops, duplicate arcs, degrees)
  * graph connectivity (weak connected components, largest component share)
  * reference route structure (length, repeats, directed/undirected continuity,
    endpoint proximity to the track) -- no assumption that track rows and
    route rows align; the counts are compared explicitly.

Outputs (written next to this script):
  * per_trip_inventory.csv   - one row per trip
  * dataset_summary.json     - machine-readable aggregate + anomaly lists
  * DATASET_INVENTORY.md     - human-readable compact summary

Usage:
  python data/inspect_dataset.py [--dataset PATH] [--out DIR]

Dataset location resolution order: --dataset flag, $DATASET_PATH env var,
then a few known candidate directories on this machine. Raw data is NEVER
copied into the repo; only generated summaries are written to data/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

EXPECTED_FILES = ("track", "nodes", "arcs", "route")
SPLITS = ("train", "student_test")

# Column counts expected in each file type.
EXPECTED_COLS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}

EARTH_R = 6371000.0  # metres


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance in metres."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pct(arr, q):
    if arr is None or len(arr) == 0:
        return None
    return float(np.percentile(arr, q))


def fnum(x, nd=3):
    """Compact rounding for display; None-safe."""
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def load_numeric(path, kind):
    """Load a whitespace-delimited numeric table.

    Returns (array2d, problem_dict). `problem_dict` is non-empty when the file
    is empty, ragged, or contains non-numeric tokens; in that case a
    line-level diagnosis is attached.
    """
    problems = {}
    try:
        arr = np.loadtxt(path, ndmin=2)
    except Exception as exc:  # noqa: BLE001 - fall back to line diagnosis
        problems["parse_error"] = str(exc)
        bad_lines, blanks, ncols_seen = [], 0, Counter()
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                toks = line.split()
                if not toks:
                    blanks += 1
                    continue
                ncols_seen[len(toks)] += 1
                if len(bad_lines) < 5:
                    try:
                        [float(t) for t in toks]
                    except ValueError:
                        bad_lines.append({"line": i, "content": line.strip()[:80]})
        problems.update(
            {
                "blank_lines": blanks,
                "column_counts_seen": dict(ncols_seen),
                "non_numeric_sample": bad_lines,
            }
        )
        return None, problems

    if arr.size == 0:
        problems["empty_file"] = True
        return arr, problems
    if not np.isfinite(arr).all():
        nan_rows = np.unique(np.where(~np.isfinite(arr))[0])
        problems["non_finite_rows"] = nan_rows[:10].tolist()
        problems["non_finite_count"] = int(np.size(nan_rows))
    return arr, problems


def connected_components(n_nodes, ea, eb):
    """Weak connected components via label propagation + pointer jumping.

    Labels only ever decrease and every edge's endpoints end up sharing a
    label; convergence is verified by asserting no edge is left split.
    Returns (labels, n_components, largest_component_size).
    """
    if n_nodes == 0:
        return np.empty(0, dtype=np.int64), 0, 0
    lab = np.arange(n_nodes, dtype=np.int64)
    if ea.size:
        for _ in range(128):
            # hook current ROOT labels across edges (union-find style scatter-min
            # on the label holders, not on edge endpoints)
            ru, rv = lab[ea], lab[eb]
            m = np.minimum(ru, rv)
            np.minimum.at(lab, ru, m)
            np.minimum.at(lab, rv, m)
            # path-compression style pointer jump until stable
            while True:
                nxt = lab[lab]
                if np.array_equal(nxt, lab):
                    break
                lab = nxt
            if np.array_equal(lab[ea], lab[eb]):
                break
        assert np.array_equal(lab[ea], lab[eb]), "component labels failed to converge"

    roots, counts = np.unique(lab, return_counts=True)
    return lab, int(roots.size), int(counts.max()) if counts.size else 0


def find_dataset():
    """Resolve dataset root: flag > env var > known candidates."""
    candidates = [
        r"C:\Users\tharu\OneDrive\Desktop\MACHINE LEARNING\Ml HACKATHON"
        r"\Student folder\Student folder",
        r"C:\Users\tharu\OneDrive\Desktop\MACHINE LEARNING\Ml HACKATHON",
    ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "train")):
            return cand
    return None


# --------------------------------------------------------------------------- #
# per-trip inspection
# --------------------------------------------------------------------------- #
def inspect_trip(trip_dir, trip_id):
    """Inspect one trip folder; returns a dict of metrics (JSON-friendly)."""
    res = {"trip_id": trip_id}
    files_present, file_rows, problems = [], {}, {}
    data = {}

    for kind in EXPECTED_FILES:
        path = os.path.join(trip_dir, f"{trip_id}.{kind}")
        if not os.path.isfile(path):
            problems[f"{kind}_missing"] = True
            data[kind] = None
            continue
        files_present.append(kind)
        arr, probs = load_numeric(path, kind)
        data[kind] = arr
        file_rows[kind] = 0 if arr is None else int(arr.shape[0])
        if probs:
            problems[kind] = probs

    res["files_present"] = files_present
    res["row_counts"] = file_rows
    res["has_problems"] = bool(problems)
    res["problems"] = problems

    track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]

    # ---------------- track ----------------
    if track is not None and track.shape[0] > 0 and track.shape[1] == 3:
        lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
        res["track_points"] = int(track.shape[0])
        res["lon_min"], res["lon_max"] = float(lon.min()), float(lon.max())
        res["lat_min"], res["lat_max"] = float(lat.min()), float(lat.max())
        res["ts_min"], res["ts_max"] = float(ts.min()), float(ts.max())
        frac = float(np.max(np.abs(ts - np.round(ts)))) if ts.size else 0.0
        res["ts_non_integer_max_frac"] = frac
        res["ts_n_non_integer"] = int((np.abs(ts - np.round(ts)) > 1e-9).sum())
        if ts.size > 1:
            d = np.diff(ts)
            res["ts_strictly_increasing"] = bool((d > 0).all())
            res["ts_nondecreasing"] = bool((d >= 0).all())
            res["ts_gap_max"] = float(d.max())
            res["ts_gap_p95"] = pct(d, 95)
            res["ts_gap_median"] = float(np.median(d))
            res["ts_gap_min"] = float(d.min())
            res["ts_n_gaps_gt1"] = int((d > 1).sum())
            res["ts_n_gaps_gt60"] = int((d > 60).sum())
            res["ts_n_equal"] = int((d == 0).sum())
            res["ts_n_backward"] = int((d < 0).sum())
        seg = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        res["track_len_km"] = float(seg.sum() / 1000.0)
        if ts.size > 1:
            sp = seg / np.maximum(np.diff(ts), 1e-9)
            res["speed_mean_ms"] = float(sp.mean())
            res["speed_p95_ms"] = pct(sp, 95)
            res["speed_max_ms"] = float(sp.max())
            res["ts_n_speed_gt33"] = int((sp > 33).sum())
        else:
            res["speed_mean_ms"] = None
        res["centroid_lon"] = float(lon.mean())
        res["centroid_lat"] = float(lat.mean())
    else:
        res["track_points"] = 0

    # ---------------- nodes ----------------
    n_nodes = 0
    if nodes is not None and nodes.shape[0] > 0 and nodes.shape[1] == 2:
        n_nodes = int(nodes.shape[0])
        res["n_nodes"] = n_nodes
        res["node_lon_min"] = float(nodes[:, 0].min())
        res["node_lon_max"] = float(nodes[:, 0].max())
        res["node_lat_min"] = float(nodes[:, 1].min())
        res["node_lat_max"] = float(nodes[:, 1].max())
        uniq = np.unique(nodes, axis=0)
        res["n_duplicate_node_coords"] = n_nodes - int(uniq.shape[0])
    else:
        res["n_nodes"] = 0

    # ---------------- arcs ----------------
    n_arcs = 0
    if arcs is not None and arcs.shape[0] > 0 and arcs.shape[1] == 2:
        n_arcs = int(arcs.shape[0])
        res["n_arcs"] = n_arcs
        ids = arcs.astype(np.int64)
        bad = ((ids < 0) | (ids >= n_nodes)).any(axis=1)
        res["arc_invalid_count"] = int(bad.sum())
        res["arc_id_min"], res["arc_id_max"] = int(ids.min()), int(ids.max())
        self_loops = ids[:, 0] == ids[:, 1]
        res["arc_self_loops"] = int(self_loops.sum())
        # exact duplicate directed arcs (same from->to listed twice)
        uniq_directed = np.unique(ids, axis=0)
        res["arc_exact_duplicates"] = n_arcs - int(uniq_directed.shape[0])
        # reciprocal arcs: reverse direction (to->from) also present,
        # excluding self-loops (roads stored in both directions)
        nl = ids[~self_loops]
        keys = nl[:, 0] * n_nodes + nl[:, 1]
        rev = nl[:, 1] * n_nodes + nl[:, 0]
        ks = np.sort(keys)
        loc = np.clip(np.searchsorted(ks, rev), 0, ks.size - 1)
        res["arc_reciprocal_count"] = int((ks[loc] == rev).sum())
        deg = np.bincount(ids.ravel(), minlength=n_nodes) if n_nodes else None
        if deg is not None and n_nodes:
            res["n_isolated_nodes"] = int((deg == 0).sum())
            res["node_degree_max"] = int(deg.max())
            res["node_degree_mean"] = float(deg.mean())
        if not bad.any():
            lab, ncomp, largest = connected_components(n_nodes, ids[:, 0], ids[:, 1])
            res["n_components"] = ncomp
            res["largest_component_nodes"] = largest
            res["largest_component_frac"] = largest / n_nodes if n_nodes else None
            # count components of size 1
            _, cnts = np.unique(lab, return_counts=True)
            res["n_singleton_components"] = int((cnts == 1).sum())
        else:
            res["n_components"] = None  # skipped: invalid arc IDs
    else:
        res["n_arcs"] = 0

    # ---------------- route ----------------
    if route is not None and route.shape[0] > 0 and route.shape[1] == 1:
        r = route[:, 0]
        frac = float(np.max(np.abs(r - np.round(r)))) if r.size else 0.0
        res["route_non_integer_max_frac"] = frac
        rid = np.round(r).astype(np.int64)
        res["route_len"] = int(rid.size)
        res["route_n_unique"] = int(np.unique(rid).size)
        res["route_n_repeats"] = int(rid.size - np.unique(rid).size)
        res["route_invalid_count"] = int(((rid < 0) | (rid >= n_arcs)).sum()) if n_arcs else int(rid.size)
        # continuity between consecutive route arcs
        if n_arcs and not res["route_invalid_count"]:
            af, at = arcs[rid[:-1], 0], arcs[rid[:-1], 1]
            bf, bt = arcs[rid[1:], 0], arcs[rid[1:], 1]
            res["route_directed_continuous"] = int((at == bf).sum())
            undirected = ((at == bf) | (at == bt) | (af == bf) | (af == bt)).sum()
            res["route_undirected_continuous"] = int(undirected)
            res["route_breaks_directed"] = int(rid.size - 1 - res["route_directed_continuous"])
            res["route_breaks_undirected"] = int(rid.size - 1 - int(undirected))
        # endpoint proximity to the track (does the route start/end where the
        # track starts/ends?) -- no row-alignment assumed
        if track is not None and res.get("track_points", 0) > 0:
            first_arc = arcs[rid[0]].astype(np.int64)
            last_arc = arcs[rid[-1]].astype(np.int64)
            e0 = nodes[first_arc]  # endpoints of first route arc
            e1 = nodes[last_arc]
            res["route_start_track_dist_m"] = float(
                haversine_m(track[0, 0], track[0, 1], e0[:, 0], e0[:, 1]).min()
            )
            res["route_end_track_dist_m"] = float(
                haversine_m(track[-1, 0], track[-1, 1], e1[:, 0], e1[:, 1]).min()
            )
        # mean arc length (scale of the road graph)
        if n_arcs:
            ia = arcs.astype(np.int64)
            alen = haversine_m(
                nodes[ia[:, 0], 0], nodes[ia[:, 0], 1],
                nodes[ia[:, 1], 0], nodes[ia[:, 1], 1],
            )
            res["arc_len_mean_m"] = float(alen.mean())
            res["arc_len_max_m"] = float(alen.max())
    else:
        res["route_len"] = 0

    # explicit comparison track rows vs route rows (do NOT assume alignment)
    if "track_points" in res and "route_len" in res:
        res["track_eq_route_len"] = res["track_points"] == res["route_len"]

    return res


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(trips):
    """Aggregate per-trip dicts into split-level and global summaries."""
    agg = {}
    numeric_keys = [
        "track_points", "route_len", "n_nodes", "n_arcs",
        "lon_min", "lon_max", "lat_min", "lat_max",
        "ts_gap_max", "ts_gap_median", "ts_n_gaps_gt1", "ts_n_equal",
        "ts_n_backward", "speed_max_ms", "speed_p95_ms", "track_len_km",
        "n_duplicate_node_coords", "arc_invalid_count", "arc_self_loops",
        "arc_duplicate_pairs", "n_isolated_nodes", "n_components",
        "largest_component_frac", "n_singleton_components",
        "route_invalid_count", "route_n_repeats", "route_breaks_directed",
        "route_breaks_undirected", "route_start_track_dist_m",
        "route_end_track_dist_m", "arc_len_mean_m", "node_degree_max",
        "node_degree_mean", "ts_non_integer_max_frac",
    ]
    for key in numeric_keys:
        vals = [t[key] for t in trips if t.get(key) is not None]
        if not vals:
            agg[key] = None
        else:
            agg[key] = {
                "min": float(np.min(vals)),
                "median": float(np.median(vals)),
                "max": float(np.max(vals)),
                "sum": float(np.sum(vals)),
                "n_trips_with_value": len(vals),
            }
    agg["track_eq_route_len_n"] = sum(1 for t in trips if t.get("track_eq_route_len"))
    agg["ts_strictly_increasing_n"] = sum(1 for t in trips if t.get("ts_strictly_increasing"))
    agg["ts_nondecreasing_n"] = sum(1 for t in trips if t.get("ts_nondecreasing"))
    return agg


def cluster_cities(trips):
    """Group trips by rounded track centroid (multiple cities in one dataset)."""
    groups = defaultdict(list)
    for t in trips:
        key = (round(t.get("centroid_lon", 0.0), 1), round(t.get("centroid_lat", 0.0), 1))
        groups[key].append(t["trip_id"])
    return [
        {"centroid_lonlat": k, "n_trips": len(v), "example_trips": v[:5]}
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_csv(trips, path):
    keys = sorted({k for t in trips for k in t if k not in ("problems",)})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for t in trips:
            w.writerow(t)


def write_markdown(summary, path):
    s = summary
    g, tr, te = s["global"], s["train"], s["student_test"]
    lines = []
    a = lines.append
    a("# PathSnap dataset inventory (generated)")
    a("")
    a(f"Generated by `data/inspect_dataset.py` on {s['generated_utc']} (UTC). "
      "Do not edit by hand - rerun the script to refresh.")
    a("")
    a(f"- Dataset root: `{s['dataset_root']}` (kept outside Git)")
    a(f"- Splits: train = **{s['n_train']} trips**, student_test = **{s['n_test']} trips**")
    a(f"- Files found: track={s['files_found']['track']}, nodes={s['files_found']['nodes']}, "
      f"arcs={s['files_found']['arcs']}, route={s['files_found']['route']} "
      f"(expected 4 per trip, {s['expected_file_total']} total)")
    a(f"- Trips with problems (parse issues / missing files): "
      f"**{len(s['trips_with_problems'])}** -> see JSON `trips_with_problems`")
    a("")

    a("## Global aggregate")
    a("")
    a("| metric | min | median | max | sum |")
    a("|---|---|---|---|---|")

    def row(name, key, nd=1):
        v = g.get(key)
        if v is None:
            return
        fmt = lambda x: fnum(x, nd)
        a(f"| {name} | {fmt(v['min'])} | {fmt(v['median'])} | {fmt(v['max'])} | "
          f"{fmt(v['sum'])} |")

    row("track points / trip", "track_points", 0)
    row("route length / trip", "route_len", 0)
    row("nodes / trip", "n_nodes", 0)
    row("arcs / trip", "n_arcs", 0)
    row("track length (km)", "track_len_km")
    row("track ts gap median (s)", "ts_gap_median")
    row("track ts gap max (s)", "ts_gap_max", 0)
    row("ts gaps >1s / trip", "ts_n_gaps_gt1", 0)
    row("ts equal steps / trip", "ts_n_equal", 0)
    row("ts backward steps / trip", "ts_n_backward", 0)
    row("implied speed p95 (m/s)", "speed_p95_ms")
    row("implied speed max (m/s)", "speed_max_ms")
    row("dup node coords / trip", "n_duplicate_node_coords", 0)
    row("invalid arc rows / trip", "arc_invalid_count", 0)
    row("arc self-loops / trip", "arc_self_loops", 0)
    row("dup arc pairs / trip", "arc_duplicate_pairs", 0)
    row("isolated nodes / trip", "n_isolated_nodes", 0)
    row("weak components / trip", "n_components", 0)
    row("largest comp. fraction", "largest_component_frac", 3)
    row("singleton components", "n_singleton_components", 0)
    row("invalid route IDs / trip", "route_invalid_count", 0)
    row("repeated route arcs / trip", "route_n_repeats", 0)
    row("route breaks (directed)", "route_breaks_directed", 0)
    row("route breaks (undirected)", "route_breaks_undirected", 0)
    row("route start -> track start (m)", "route_start_track_dist_m", 0)
    row("route end -> track end (m)", "route_end_track_dist_m", 0)
    row("arc length mean (m)", "arc_len_mean_m", 0)
    row("node degree max", "node_degree_max", 0)
    a("")

    a("## Key checks")
    a("")
    a(f"- Timestamps integer-valued in every trip: **{g.get('ts_non_integer_max_frac', {}).get('max', 0) == 0}** "
      f"(max fractional part across all trips: {fnum((g.get('ts_non_integer_max_frac') or {}).get('max', 0), 6)})")
    a(f"- Trips with strictly increasing timestamps: **{g.get('ts_strictly_increasing_n')}** / "
      f"{s['n_total']}; non-decreasing: **{g.get('ts_nondecreasing_n')}** / {s['n_total']}")
    a(f"- Trips where len(track) == len(route): **{g.get('track_eq_route_len_n')}** / "
      f"{s['n_total']} -> track rows and route rows must NOT be assumed to align")
    a(f"- Trips with invalid arc endpoints or invalid route IDs: "
      f"**{(g.get('arc_invalid_count') or {}).get('sum', 0) + (g.get('route_invalid_count') or {}).get('sum', 0):.0f}** "
      "rows in total")
    a(f"- Routes: directed continuity breaks median "
      f"{fnum((g.get('route_breaks_directed') or {}).get('median', 0), 0)}, "
      f"undirected breaks median "
      f"{fnum((g.get('route_breaks_undirected') or {}).get('median', 0), 0)} "
      "(0 = every consecutive route arc pair shares an endpoint / is driveable)")
    a("")

    a("## Geographic clusters (rounded centroid lon/lat)")
    a("")
    a("| centroid (lon, lat) | trips | examples |")
    a("|---|---|---|")
    for c in s["city_clusters"][:15]:
        a(f"| {c['centroid_lonlat'][0]:.1f}, {c['centroid_lonlat'][1]:.1f} | "
          f"{c['n_trips']} | {', '.join(c['example_trips'][:3])} |")
    a("")

    a("## Split-level coordinate envelope")
    a("")
    a("| split | lon min | lon max | lat min | lat max |")
    a("|---|---|---|---|---|")
    for name, x in (("train", tr), ("student_test", te)):
        a(f"| {name} | {fnum((x.get('lon_min') or {}).get('min'))} | "
          f"{fnum((x.get('lon_max') or {}).get('max'))} | "
          f"{fnum((x.get('lat_min') or {}).get('min'))} | "
          f"{fnum((x.get('lat_max') or {}).get('max'))} |")
    a("")
    a("Full per-trip metrics: `per_trip_inventory.csv`; machine-readable aggregate: "
      "`dataset_summary.json`.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="PathSnap dataset inventory")
    ap.add_argument("--dataset", default=None, help="path to dataset root (contains train/, student_test/)")
    ap.add_argument("--out", default=None, help="output dir (defaults to dir of this script)")
    args = ap.parse_args()

    dataset = args.dataset or os.environ.get("DATASET_PATH") or find_dataset()
    if not dataset or not os.path.isdir(os.path.join(dataset, "train")):
        sys.exit("Dataset root not found. Pass --dataset /path/to/dataset (must contain train/).")
    out_dir = args.out or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    all_trips, per_split = [], {}
    files_found = {k: 0 for k in EXPECTED_FILES}
    trips_with_problems = []

    for split in SPLITS:
        split_dir = os.path.join(dataset, split)
        trip_ids = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        trips = []
        for tid in trip_ids:
            t = inspect_trip(os.path.join(split_dir, tid), tid)
            t["split"] = split
            trips.append(t)
            for k in EXPECTED_FILES:
                if k in t["files_present"]:
                    files_found[k] += 1
            if t["has_problems"]:
                trips_with_problems.append({"split": split, **{k: t[k] for k in ("trip_id", "problems")}})
        per_split[split] = trips
        all_trips.extend(trips)

    global_agg = aggregate(all_trips)
    split_aggs = {sp: aggregate(ts) for sp, ts in per_split.items()}

    summary = {
        "generated_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_root": dataset,
        "n_train": len(per_split["train"]),
        "n_test": len(per_split["student_test"]),
        "n_total": len(all_trips),
        "expected_file_total": len(all_trips) * len(EXPECTED_FILES),
        "files_found": files_found,
        "trips_with_problems": trips_with_problems,
        "city_clusters": cluster_cities(all_trips),
        "global": global_agg,
        "train": split_aggs["train"],
        "student_test": split_aggs["student_test"],
    }

    write_csv(all_trips, os.path.join(out_dir, "per_trip_inventory.csv"))
    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    write_markdown(summary, os.path.join(out_dir, "DATASET_INVENTORY.md"))

    # console digest
    print(f"trips inspected: {len(all_trips)} (train {summary['n_train']} / test {summary['n_test']})")
    print(f"files found: {files_found} of {summary['expected_file_total']}")
    print(f"trips with problems: {len(trips_with_problems)}")
    print(f"track==route len trips: {global_agg.get('track_eq_route_len_n')} / {len(all_trips)}")
    print(f"strictly increasing ts: {global_agg.get('ts_strictly_increasing_n')} / {len(all_trips)}")
    print(f"city clusters: {len(summary['city_clusters'])}")
    print(f"wrote: per_trip_inventory.csv, dataset_summary.json, DATASET_INVENTORY.md in {out_dir}")


if __name__ == "__main__":
    main()
