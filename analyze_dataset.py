#!/usr/bin/env python3
"""
PathSnap Stage 1: Dataset Exploration and Analysis Script

Analyzes the PathSnap distributed map-matching & route reconstruction datasets.
"""

import os
import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Constants
REPO_ROOT = Path(__file__).parent if Path(__file__).parent else Path(".")
DATASETS_DIR = REPO_ROOT / "datasets"
TRAIN_DIR = DATASETS_DIR / "train"
STUDENT_TEST_DIR = DATASETS_DIR / "student_test"
OUTPUT_DIR = REPO_ROOT / "stage_1_output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def discover_trips():
    """Discover all training and student-test trips."""
    training_trips = []
    student_trips = []

    # Training trips - look inside datasets/train/
    if TRAIN_DIR.is_dir():
        for trip_dir in sorted(TRAIN_DIR.iterdir()):
            if trip_dir.is_dir() and not trip_dir.name.startswith("."):
                trip_id = trip_dir.name
                # Check that all 4 file types exist
                has_all = all(
                    (trip_dir / f"{trip_id}.{ext}").exists()
                    for ext in ["track", "nodes", "arcs", "route"]
                )
                if has_all:
                    training_trips.append(trip_id)

    # Student-test trips - look inside datasets/student_test/
    if STUDENT_TEST_DIR.is_dir():
        for trip_dir in sorted(STUDENT_TEST_DIR.iterdir()):
            if trip_dir.is_dir() and not trip_dir.name.startswith("."):
                trip_id = trip_dir.name
                has_all = all(
                    (trip_dir / f"{trip_id}.{ext}").exists()
                    for ext in ["track", "nodes", "arcs", "route"]
                )
                if has_all:
                    student_trips.append(trip_id)

    return sorted(training_trips), sorted(student_trips)


def count_files_by_extension():
    """Count files of each extension in the datasets."""
    counts = defaultdict(int)

    # Training trips
    if TRAIN_DIR.is_dir():
        for trip_dir in TRAIN_DIR.iterdir():
            if trip_dir.is_dir():
                for f in trip_dir.iterdir():
                    if f.is_file():
                        ext = f.suffix.lstrip(".")
                        counts[ext] += 1

    # Student-test trips
    if STUDENT_TEST_DIR.is_dir():
        for trip_dir in STUDENT_TEST_DIR.iterdir():
            if trip_dir.is_dir():
                for f in trip_dir.iterdir():
                    if f.is_file():
                        ext = f.suffix.lstrip(".")
                        counts[ext] += 1

    return dict(counts)


def find_missing_files(training_trips, student_trips):
    """Find trips missing some file types."""
    missing = {"training": [], "student": []}

    # Check training trips
    for trip_id in training_trips:
        trip_dir = TRAIN_DIR / trip_id
        for ext in ["track", "nodes", "arcs", "route"]:
            if not (trip_dir / f"{trip_id}.{ext}").exists():
                missing["training"].append(f"{trip_id}.{ext}")

    # Check student trips
    for trip_id in student_trips:
        trip_dir = STUDENT_TEST_DIR / trip_id
        for ext in ["track", "nodes", "arcs", "route"]:
            if not (trip_dir / f"{trip_id}.{ext}").exists():
                missing["student"].append(f"{trip_id}.{ext}")

    return missing


def find_empty_files(training_trips, student_trips):
    """Find empty files in the datasets."""
    empty = {"training": [], "student": []}

    # Check training trips
    for trip_id in training_trips:
        trip_dir = TRAIN_DIR / trip_id
        for ext in ["track", "nodes", "arcs", "route"]:
            fpath = trip_dir / f"{trip_id}.{ext}"
            if fpath.exists() and fpath.stat().st_size == 0:
                empty["training"].append(f"{trip_id}.{ext}")

    # Check student trips
    for trip_id in student_trips:
        trip_dir = STUDENT_TEST_DIR / trip_id
        for ext in ["track", "nodes", "arcs", "route"]:
            fpath = trip_dir / f"{trip_id}.{ext}"
            if fpath.exists() and fpath.stat().st_size == 0:
                empty["student"].append(f"{trip_id}.{ext}")

    return empty


def parse_track_file(filepath):
    """Parse a .track file and return observations."""
    observations = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    obs_id = float(parts[2]) if parts[2] else None
                    observations.append({"lon": lon, "lat": lat, "obs_id": obs_id})
                except ValueError:
                    observations.append({"lon": None, "lat": None, "obs_id": None, "error": "invalid"})
    return observations


def parse_nodes_file(filepath):
    """Parse a .nodes file and return nodes."""
    nodes = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    nodes.append({"lon": lon, "lat": lat})
                except ValueError:
                    nodes.append({"lon": None, "lat": None, "error": "invalid"})
    return nodes


def parse_arcs_file(filepath):
    """Parse a .arcs file and return connections."""
    arcs = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    src = int(parts[0])
                    tgt = int(parts[1])
                    arcs.append({"source": src, "target": tgt})
                except ValueError:
                    arcs.append({"source": None, "target": None, "error": "invalid"})
    return arcs


def parse_route_file(filepath):
    """Parse a .route file and return route entries."""
    route = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                node_id = int(line.strip())
                route.append(node_id)
            except ValueError:
                route.append(None)
    return route


def analyze_trip(trip_id, is_student=False):
    """Analyze a single trip's data."""
    trip_dir = TRAIN_DIR / trip_id if not is_student else STUDENT_TEST_DIR / trip_id

    results = {
        "trip_id": trip_id,
        "is_student": is_student,
        "track": {},
        "nodes": {},
        "arcs": {},
        "route": {},
    }

    # Parse track
    track_path = trip_dir / f"{trip_id}.track"
    if track_path.exists():
        results["track"] = {
            "file_exists": True,
            "observations": len(parse_track_file(track_path)),
            "file_path": str(track_path),
        }
    else:
        results["track"] = {"file_exists": False, "observations": 0}

    # Parse nodes
    nodes_path = trip_dir / f"{trip_id}.nodes"
    if nodes_path.exists():
        results["nodes"] = {
            "file_exists": True,
            "nodes": len(parse_nodes_file(nodes_path)),
            "file_path": str(nodes_path),
        }
    else:
        results["nodes"] = {"file_exists": False, "nodes": 0}

    # Parse arcs
    arcs_path = trip_dir / f"{trip_id}.arcs"
    if arcs_path.exists():
        results["arcs"] = {
            "file_exists": True,
            "arcs": len(parse_arcs_file(arcs_path)),
            "file_path": str(arcs_path),
        }
    else:
        results["arcs"] = {"file_exists": False, "arcs": 0}

    # Parse route
    route_path = trip_dir / f"{trip_id}.route"
    if route_path.exists():
        route_data = parse_route_file(route_path)
        results["route"] = {
            "file_exists": True,
            "entries": len(route_data),
            "route": route_data,
            "file_path": str(route_path),
        }
    else:
        results["route"] = {"file_exists": False, "entries": 0}

    return results


def compute_coord_ranges(trip_stats, is_student=False):
    """Compute coordinate ranges across trips."""
    all_lons = []
    all_lats = []
    per_trip = {}

    for trip_stat in trip_stats:
        trip_id = trip_stat["trip_id"]

        # Track coordinates - parse from file
        track_obs = trip_stat["track_obs"]
        track_lons = []
        track_lats = []
        if track_obs > 0:
            obs = parse_track_file(TRAIN_DIR / trip_id / f"{trip_id}.track")
            track_lons = [o["lon"] for o in obs if o.get("lon") is not None]
            track_lats = [o["lat"] for o in obs if o.get("lat") is not None]

        # Nodes coordinates
        nodes_count = trip_stat["nodes"]
        node_lons = []
        node_lats = []
        if nodes_count > 0:
            nodes = parse_nodes_file(TRAIN_DIR / trip_id / f"{trip_id}.nodes")
            node_lons = [n["lon"] for n in nodes if n.get("lon") is not None]
            node_lats = [n["lat"] for n in nodes if n.get("lat") is not None]

        trip_ranges = {
            "track_lon_min": min(track_lons) if track_lons else None,
            "track_lon_max": max(track_lons) if track_lons else None,
            "track_lat_min": min(track_lats) if track_lats else None,
            "track_lat_max": max(track_lats) if track_lats else None,
            "node_lon_min": min(node_lons) if node_lons else None,
            "node_lon_max": max(node_lons) if node_lons else None,
            "node_lat_min": min(node_lats) if node_lats else None,
            "node_lat_max": max(node_lats) if node_lats else None,
        }
        per_trip[trip_id] = trip_ranges

        all_lons.extend(track_lons)
        all_lons.extend(node_lons)
        all_lats.extend(track_lats)
        all_lats.extend(node_lats)

    global_ranges = {
        "global_lon_min": min(all_lons) if all_lons else None,
        "global_lon_max": max(all_lons) if all_lons else None,
        "global_lat_min": min(all_lats) if all_lats else None,
        "global_lat_max": max(all_lats) if all_lats else None,
    }

    return global_ranges, per_trip


def compute_sampling_intervals(trips_data, is_student=False):
    """Compute sampling intervals for trips with timestamps."""
    intervals = {}

    for trip_info in trips_data:
        trip_id = trip_info["trip_id"]
        track_path = (
            TRAIN_DIR / trip_id / f"{trip_id}.track"
            if not is_student
            else STUDENT_TEST_DIR / trip_id / f"{trip_id}.track"
        )

        if not track_path.exists():
            continue

        obs = parse_track_file(track_path)
        valid_obs = [o for o in obs if o.get("obs_id") is not None and not o.get("error")]

        if len(valid_obs) < 2:
            intervals[trip_id] = {"intervals": [], "note": "Too few valid observations"}
            continue

        # obs_id appears to be sequential numbers (1, 2, 3, ...)
        obs_ids = [o["obs_id"] for o in valid_obs]
        min_id = min(obs_ids)
        max_id = max(obs_ids)
        n_intervals = max_id - min_id

        # Calculate intervals between consecutive observations
        time_diffs = []
        for i in range(1, len(valid_obs)):
            diff = valid_obs[i]["obs_id"] - valid_obs[i - 1]["obs_id"]
            time_diffs.append(diff)

        intervals[trip_id] = {
            "num_observations": len(valid_obs),
            "obs_id_range": [min_id, max_id],
            "num_intervals": n_intervals,
            "min_interval": min(time_diffs) if time_diffs else None,
            "max_interval": max(time_diffs) if time_diffs else None,
            "avg_interval": sum(time_diffs) / len(time_diffs) if time_diffs else None,
            "note": "obs_id appears to be sequential integer timestamps",
        }

    return intervals


def check_quality(trip_stats, is_student=False):
    """Check data quality issues using trip_stats."""
    quality = {
        "missing_coords": 0,
        "invalid_coords": 0,
        "duplicate_nodes": 0,
        "invalid_arc_refs": 0,
        "self_loops": 0,
        "duplicate_arcs": 0,
        "route_gaps": 0,
    }

    # trip_stats entries have: trip_id, track_obs, nodes, arcs, route_entries
    # We need to parse the actual files for quality checks
    for trip_stat in trip_stats:
        trip_id = trip_stat["trip_id"]

        # Check track for missing/invalid coords
        if trip_stat["track_obs"] > 0:
            obs = parse_track_file(TRAIN_DIR / trip_id / f"{trip_id}.track")
            for o in obs:
                if o.get("lon") is None or o.get("lat") is None:
                    quality["missing_coords"] += 1
                if o.get("error") == "invalid":
                    quality["invalid_coords"] += 1

        # Check nodes for duplicates/invalid
        nodes_count = trip_stat["nodes"]
        if nodes_count > 0:
            nodes = parse_nodes_file(TRAIN_DIR / trip_id / f"{trip_id}.nodes")
            # Check for duplicate coordinates
            coord_tuples = [(n["lon"], n["lat"]) for n in nodes if n.get("lon") is not None]
            if len(coord_tuples) != len(set(coord_tuples)):
                quality["duplicate_nodes"] += 1  # at least one dup

        # Check arcs for invalid refs, self-loops, duplicates
        arcs_count = trip_stat["arcs"]
        if arcs_count > 0:
            arcs = parse_arcs_file(TRAIN_DIR / trip_id / f"{trip_id}.arcs")
            refs = set()
            for a in arcs:
                if a.get("source") is not None and a.get("target") is not None:
                    if a["source"] == a["target"]:
                        quality["self_loops"] += 1
                    if a["source"] not in refs:
                        refs.add(a["source"])
                    else:
                        quality["duplicate_arcs"] += 1

        # route_gaps not counted at this level; handled in route integrity

    return quality


def check_route_integrity(trips_data, is_student=False):
    """Check route integrity against road network."""
    route_integrity = {
        "total_entries": 0,
        "entries_with_valid_ref": 0,
        "entries_with_invalid_ref": 0,
        "connected_paths": 0,
        "disconnected_entries": 0,
    }

    # Build node reference set for each trip
    for trip_info in trips_data:
        trip_id = trip_info["trip_id"]

        # Get valid node IDs from nodes file
        nodes_path = (
            TRAIN_DIR / trip_id / f"{trip_id}.nodes"
            if not is_student
            else STUDENT_TEST_DIR / trip_id / f"{trip_id}.nodes"
        )
        arcs_path = (
            TRAIN_DIR / trip_id / f"{trip_id}.arcs"
            if not is_student
            else STUDENT_TEST_DIR / trip_id / f"{trip_id}.arcs"
        )
        route_path = (
            TRAIN_DIR / trip_id / f"{trip_id}.route"
            if not is_student
            else STUDENT_TEST_DIR / trip_id / f"{trip_id}.route"
        )

        valid_node_ids = set()
        if nodes_path.exists():
            nodes = parse_nodes_file(nodes_path)
            # Node IDs appear to be 1-indexed based on line number
            valid_node_ids = {i + 1 for i in range(len(nodes))}

        arcs = []
        if arcs_path.exists():
            arcs = parse_arcs_file(arcs_path)

        route = []
        if route_path.exists():
            route = parse_route_file(route_path)

        connected = 0
        disconnected = 0

        for i, entry in enumerate(route):
            if entry is not None:
                route_integrity["total_entries"] += 1
                if entry in valid_node_ids:
                    route_integrity["entries_with_valid_ref"] += 1
                else:
                    route_integrity["entries_with_invalid_ref"] += 1

                # Check if consecutive entries are connected
                if i > 0:
                    prev_entry = route[i - 1]
                    if prev_entry is not None and entry is not None:
                        # Check if there's an arc between prev_entry and entry
                        arc_exists = any(
                            (a["source"] == prev_entry and a["target"] == entry)
                            or (a["source"] == entry and a["target"] == prev_entry)
                            for a in arcs
                        )
                        if arc_exists:
                            connected += 1
                        else:
                            disconnected += 1

        route_integrity["connected_paths"] = connected
        route_integrity["disconnected_entries"] = disconnected

    return route_integrity


def generate_visualizations(training_trips, student_trips):
    """Generate Folium maps for visualizations."""
    try:
        import folium
        from folium import plugins
        folium_available = True
    except ImportError:
        folium_available = False
        print("Folium not installed. Installing...")
        os.system("pip3 install folium matplotlib")
        try:
            import folium
            from folium import plugins
            folium_available = True
        except ImportError:
            folium_available = False

    visualizations = {"generated": [], "skipped": []}

    if not folium_available:
        visualizations["skipped"].append("Folium not available")
        return visualizations

    # Select representative training trips (first 5 with variety of sizes)
    selected_trips = []
    for trip_id in training_trips:
        trip_dir = TRAIN_DIR / trip_id
        track_path = trip_dir / f"{trip_id}.track"
        if track_path.exists():
            obs = len(parse_track_file(track_path))
            selected_trips.append((trip_id, obs))
            if len(selected_trips) >= 5:
                break
        if len(selected_trips) >= 5:
            break

    # Sort by observation count for variety
    selected_trips.sort(key=lambda x: x[1])

    for trip_id, obs_count in selected_trips:
        trip_dir = TRAIN_DIR / trip_id
        track_path = trip_dir / f"{trip_id}.track"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        route_path = trip_dir / f"{trip_id}.route"

        # Parse data
        obs = parse_track_file(track_path)
        nodes = parse_nodes_file(nodes_path)
        route = parse_route_file(route_path)

        # Create map centered on first GPS observation
        if obs and obs[0]["lon"] is not None and obs[0]["lat"] is not None:
            m = folium.Map(location=[obs[0]["lat"], obs[0]["lon"]], zoom_start=13)

            # Add GPS trajectory (in red)
            if obs:
                lon_coords = [o["lon"] for o in obs if o["lon"] is not None]
                lat_coords = [o["lat"] for o in obs if o["lat"] is not None]
                if lon_coords and lat_coords:
                    line_coords = [(lat, lon) for lat, lon in zip(lat_coords, lon_coords)]
                    folium.PolyLine(
                        line_coords,
                        color="red",
                        weight=2,
                        opacity=0.8,
                        label="GPS Trajectory",
                    ).add_to(m)

                    # Add first and last point as markers
                    if len(lon_coords) > 1:
                        folium.Marker(
                            [lat_coords[0], lon_coords[0]],
                            popup="Start",
                            icon=folium.Icon(color="green", icon="play"),
                        ).add_to(m)
                        folium.Marker(
                            [lat_coords[-1], lon_coords[-1]],
                            popup="End",
                            icon=folium.Icon(color="red", icon="stop"),
                        ).add_to(m)

            # Add road network nodes (small blue dots)
            if nodes:
                for node in nodes[:200]:  # Limit to first 200 nodes
                    if node["lon"] is not None and node["lat"] is not None:
                        folium.CircleMarker(
                            [node["lat"], node["lon"]],
                            radius=1,
                            color="blue",
                            fill=True,
                            fill_opacity=0.5,
                            label="Node",
                        ).add_to(m)

            # Add route entries (green markers for route nodes)
            if route:
                valid_route_nodes = [r for r in route if r is not None]
                # Map node IDs to coordinates - node IDs are 1-indexed
                for node_id in valid_route_nodes[:50]:  # Limit to first 50 route entries
                    node_idx = node_id - 1
                    if 0 <= node_idx < len(nodes):
                        node = nodes[node_idx]
                        if node["lon"] is not None and node["lat"] is not None:
                            folium.Marker(
                                [node["lat"], node["lon"]],
                                popup=f"Route node {node_id}",
                                icon=folium.Icon(color="green", icon="info-sign"),
                            ).add_to(m)

            viz_path = OUTPUT_DIR / f"trip_{trip_id}_combined.html"
            m.save(str(viz_path))
            visualizations["generated"].append(str(viz_path))
            visualizations["descriptions"] = visualizations.get("descriptions", [])
            visualizations["descriptions"].append(
                f"Trip {trip_id}: Combined view ( {obs_count} GPS obs, {len(nodes)} nodes, {len(route)} route entries )"
            )

    # Save trip statistics summary as JSON
    stats = []
    for trip_id in training_trips[:20]:  # First 20 trips
        trip_dir = TRAIN_DIR / trip_id
        track_path = trip_dir / f"{trip_id}.track"
        nodes_path = trip_dir / f"{trip_id}.nodes"
        route_path = trip_dir / f"{trip_id}.route"

        if track_path.exists():
            obs = len(parse_track_file(track_path))
        else:
            obs = 0

        if nodes_path.exists():
            nodes = len(parse_nodes_file(nodes_path))
        else:
            nodes = 0

        if route_path.exists():
            route = len(parse_route_file(route_path))
        else:
            route = 0

        stats.append({"trip": trip_id, "obs": obs, "nodes": nodes, "route": route})

    stats_path = OUTPUT_DIR / "trip_statistics_summary.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    visualizations["generated"].append(str(stats_path))

    return visualizations


def main():
    """Main analysis function."""
    print("=" * 60)
    print("PATHSPARK Stage 1: Dataset Exploration and Analysis")
    print("=" * 60)

    # 1. Discover trips
    print("\n[1] Discovering trips...")
    training_trips, student_trips = discover_trips()
    print(f"  Training trips: {len(training_trips)}")
    print(f"  Student-test trips: {len(student_trips)}")

    # 2. Count files by extension
    print("\n[2] Counting files by extension...")
    file_counts = count_files_by_extension()
    for ext, count in sorted(file_counts.items()):
        print(f"  {ext}: {count}")

    # 3. Find missing files
    print("\n[3] Finding missing files...")
    missing = find_missing_files(training_trips, student_trips)
    if missing["training"]:
        print(f"  Training missing: {missing['training'][:5]}...")
    if missing["student"]:
        print(f"  Student missing: {missing['student'][:5]}...")

    # 4. Find empty files
    print("\n[4] Finding empty files...")
    empty = find_empty_files(training_trips, student_trips)
    if empty["training"]:
        print(f"  Training empty: {len(empty['training'])} files")
    if empty["student"]:
        print(f"  Student empty: {len(empty['student'])} files")

    # 5. Per-trip statistics
    print("\n[5] Computing per-trip statistics...")
    trip_stats = []

    for trip_id in training_trips:
        data = analyze_trip(trip_id, is_student=False)
        trip_stats.append({
            "trip_id": trip_id,
            "track_obs": data["track"].get("observations", 0),
            "nodes": data["nodes"].get("nodes", 0),
            "arcs": data["arcs"].get("arcs", 0),
            "route_entries": data["route"].get("entries", 0),
        })

    # Compute min, max, avg
    track_counts = [s["track_obs"] for s in trip_stats]
    nodes_counts = [s["nodes"] for s in trip_stats]
    arcs_counts = [s["arcs"] for s in trip_stats]
    route_counts = [s["route_entries"] for s in trip_stats]

    if track_counts:
        print(f"  Track observations: min={min(track_counts)}, max={max(track_counts)}, avg={sum(track_counts)/len(track_counts):.1f}")
    else:
        print("  Track observations: no data")

    if nodes_counts:
        print(f"  Nodes: min={min(nodes_counts)}, max={max(nodes_counts)}, avg={sum(nodes_counts)/len(nodes_counts):.1f}")
    else:
        print("  Nodes: no data")

    if arcs_counts:
        print(f"  Arcs: min={min(arcs_counts)}, max={max(arcs_counts)}, avg={sum(arcs_counts)/len(arcs_counts):.1f}")
    else:
        print("  Arcs: no data")

    if route_counts:
        print(f"  Route entries: min={min(route_counts)}, max={max(route_counts)}, avg={sum(route_counts)/len(route_counts):.1f}")
    else:
        print("  Route entries: no data")

    # 6. Dataset quality report
    print("\n[6] Generating dataset quality report...")
    quality = check_quality(trip_stats, is_student=False)
    print(f"  Missing coordinates: {quality['missing_coords']}")
    print(f"  Invalid coordinates: {quality['invalid_coords']}")
    print(f"  Duplicate nodes: {quality['duplicate_nodes']}")
    print(f"  Self-loops: {quality['self_loops']}")
    print(f"  Duplicate arcs: {quality['duplicate_arcs']}")

    # 7. Coordinate ranges
    print("\n[7] Computing coordinate ranges...")
    global_ranges, per_trip_ranges = compute_coord_ranges(trip_stats, is_student=False)
    if global_ranges["global_lon_min"]:
        print(f"  Global lon range: {global_ranges['global_lon_min']:.6f} to {global_ranges['global_lon_max']:.6f}")
        print(f"  Global lat range: {global_ranges['global_lat_min']:.6f} to {global_ranges['global_lat_max']:.6f}")

    # 8. Sampling intervals
    print("\n[8] Computing sampling intervals...")
    intervals = compute_sampling_intervals(trip_stats, is_student=False)
    sampled_trips = [tid for tid, iv in intervals.items() if iv.get("min_interval") is not None]
    print(f"  Trips with sampling data: {len(sampled_trips)}")

    # 9. Route integrity
    print("\n[9] Checking route integrity...")
    integrity = check_route_integrity(trip_stats, is_student=False)
    print(f"  Total route entries: {integrity['total_entries']}")
    print(f"  Valid refs: {integrity['entries_with_valid_ref']}")
    print(f"  Invalid refs: {integrity['entries_with_invalid_ref']}")
    print(f"  Connected paths: {integrity['connected_paths']}")
    print(f"  Disconnected entries: {integrity['disconnected_entries']}")

    # 10. Generate reports
    print("\n[10] Generating reports...")

    # Dataset inventory report
    inventory = {
        "training_trips": len(training_trips),
        "student_trips": len(student_trips),
        "file_counts": file_counts,
        "missing_files": {k: len(v) for k, v in missing.items()},
        "empty_files": {k: len(v) for k, v in empty.items()},
    }
    with open(OUTPUT_DIR / "dataset_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    print("  -> dataset_inventory.json")

    # Per-trip statistics
    with open(OUTPUT_DIR / "per_trip_statistics.json", "w") as f:
        json.dump(trip_stats, f, indent=2)
    print("  -> per_trip_statistics.json")

    # Dataset quality report
    with open(OUTPUT_DIR / "dataset_quality.json", "w") as f:
        json.dump(quality, f, indent=2)
    print("  -> dataset_quality.json")

    # Coordinate range report
    with open(OUTPUT_DIR / "coordinate_ranges.json", "w") as f:
        json.dump({"global": global_ranges, "per_trip": per_trip_ranges}, f, indent=2)
    print("  -> coordinate_ranges.json")

    # Sampling interval report
    with open(OUTPUT_DIR / "sampling_intervals.json", "w") as f:
        json.dump(intervals, f, indent=2)
    print("  -> sampling_intervals.json")

    # Road network statistics
    if trip_stats:
        road_stats = {
            "total_nodes": sum(s["nodes"] for s in trip_stats),
            "total_arcs": sum(s["arcs"] for s in trip_stats),
            "avg_nodes_per_trip": sum(s["nodes"] for s in trip_stats) / len(trip_stats),
            "avg_arcs_per_trip": sum(s["arcs"] for s in trip_stats) / len(trip_stats),
        }
    else:
        road_stats = {"total_nodes": 0, "total_arcs": 0, "avg_nodes_per_trip": 0, "avg_arcs_per_trip": 0}
    with open(OUTPUT_DIR / "road_network_statistics.json", "w") as f:
        json.dump(road_stats, f, indent=2)
    print("  -> road_network_statistics.json")

    # Route integrity report
    with open(OUTPUT_DIR / "route_integrity.json", "w") as f:
        json.dump(integrity, f, indent=2)
    print("  -> route_integrity.json")

    # Overall Markdown summary
    markdown_parts = [
        "# PathSnap Stage 1: Dataset Exploration Summary",
        "",
        "## Dataset Inventory",
        f"- **Training trips**: {len(training_trips)}",
        f"- **Student-test trips**: {len(student_trips)}",
        f"- **File extension counts**: {json.dumps(file_counts)}",
        "",
        "## Per-Trip Statistics",
    ]

    if track_counts:
        markdown_parts.append(
            f"- **Track observations**: min={min(track_counts)}, max={max(track_counts)}, avg={sum(track_counts)/len(track_counts):.1f}"
        )
    else:
        markdown_parts.append("- **Track observations**: no data")

    if nodes_counts:
        markdown_parts.append(
            f"- **Road network nodes**: min={min(nodes_counts)}, max={max(nodes_counts)}, avg={sum(nodes_counts)/len(nodes_counts):.1f}"
        )
    else:
        markdown_parts.append("- **Road network nodes**: no data")

    if arcs_counts:
        markdown_parts.append(
            f"- **Road network arcs**: min={min(arcs_counts)}, max={max(arcs_counts)}, avg={sum(arcs_counts)/len(arcs_counts):.1f}"
        )
    else:
        markdown_parts.append("- **Road network arcs**: no data")

    if route_counts:
        markdown_parts.append(
            f"- **Route entries**: min={min(route_counts)}, max={max(route_counts)}, avg={sum(route_counts)/len(route_counts):.1f}"
        )
    else:
        markdown_parts.append("- **Route entries**: no data")

    markdown_parts.append("")
    markdown_parts.append("## Discoveries")
    
    glon_min = global_ranges.get("global_lon_min", "N/A")
    glon_max = global_ranges.get("global_lon_max", "N/A")
    glat_min = global_ranges.get("global_lat_min", "N/A")
    glat_max = global_ranges.get("global_lat_max", "N/A")
    markdown_parts.append(
        f"- Training trips span coordinates approximately {glon_min:.2f} to {glon_max:.2f} longitude and {glat_min:.2f} to {glat_max:.2f} latitude"
    )
    
    markdown_parts.append(
        f"- {len(sampled_trips)} trips have timestamp data with sampling intervals"
    )
    
    markdown_parts.append(
        f"- Route integrity: {integrity['connected_paths']} connected paths out of {integrity['total_entries']} total entries"
    )
    
    markdown_parts.append(
        f"- Missing coordinates found: {quality['missing_coords']}"
    )
    
    markdown_parts.append(
        f"- Invalid coordinates found: {quality['invalid_coords']}"
    )

    markdown_parts.append("")
    markdown_parts.append("## Data Quality Issues")
    markdown_parts.append(
        f"- Missing GPS coordinates: {quality['missing_coords']}"
    )
    markdown_parts.append(
        f"- Invalid GPS coordinates: {quality['invalid_coords']}"
    )
    markdown_parts.append(
        f"- Duplicate road network nodes: {quality['duplicate_nodes']}"
    )
    markdown_parts.append(
        f"- Self-loop arcs: {quality['self_loops']}"
    )
    markdown_parts.append(
        f"- Duplicate arcs: {quality['duplicate_arcs']}"
    )
    markdown_parts.append(
        f"- Route entries with invalid node references: {integrity['entries_with_invalid_ref']}"
    )

    markdown_parts.append("")
    markdown_parts.append(
        f"*Generated by PathSnap Stage 1 analysis on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    with open(OUTPUT_DIR / "SUMMARY.md", "w") as f:
        f.write("\n".join(markdown_parts))
    print("  -> SUMMARY.md")

    # 11. Generate visualizations
    print("\n[11] Generating visualizations...")
    viz = generate_visualizations(training_trips, student_trips)
    if "descriptions" in viz:
        for desc in viz["descriptions"]:
            print(f"  {desc}")

    print("\n" + "=" * 60)
    print("Stage 1 analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()