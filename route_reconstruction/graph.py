"""Directed road graph built from one trip's .nodes and .arcs files (Member 3).

File formats (confirmed against the hackathon's Dataset Exploration document
and the raw files themselves):

    .nodes : M x 2 whitespace-separated columns, no header
             column 0 = longitude, column 1 = latitude
             node_id = 0-based row index
    .arcs  : K x 2 integer columns, no header
             column 0 = from_node_id, column 1 = to_node_id  (DIRECTED)
             arc_id  = 0-based row index  <- preserved exactly, never renumbered

Arc "segment length" in metres = great-circle (haversine) distance between the
endpoint nodes, because the raw files carry no length column. Duplicate arc
rows (the dataset contains many exact duplicates) keep their own arc IDs;
only the adjacency used for shortest-path search is de-duplicated (parallel
edges collapse to the minimum length). Disconnected components are kept as
they are -- nothing is merged or renumbered; component labels are available
for diagnostics and for detecting impossible transitions.
"""
from __future__ import annotations

import heapq
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .geo import haversine_m


@dataclass
class DijkstraResult:
    dist: dict  # node_id -> distance (metres), only nodes within cutoff
    n_visited: int  # number of nodes expanded (runtime diagnostics)
    exhausted: bool  # True when the visited-cap stopped the search early


class RoadGraph:
    """Directed graph over one trip's road network, arc IDs preserved."""

    def __init__(self, nodes_lonlat, arcs, name: str = ""):
        nodes_lonlat = np.asarray(nodes_lonlat, dtype=np.float64)
        arcs = np.asarray(arcs, dtype=np.int64)
        if nodes_lonlat.ndim != 2 or nodes_lonlat.shape[1] != 2:
            raise ValueError("nodes_lonlat must be an (M, 2) array of (lon, lat)")
        if arcs.ndim != 2 or arcs.shape[1] != 2:
            raise ValueError("arcs must be a (K, 2) array of (from_node, to_node)")

        self.name = name
        self.nodes_lonlat = nodes_lonlat
        self.n_nodes = int(nodes_lonlat.shape[0])
        self.arcs = arcs
        self.n_arcs = int(arcs.shape[0])

        # --- validity -------------------------------------------------------
        in_range = (arcs[:, 0] >= 0) & (arcs[:, 0] < self.n_nodes) & \
                   (arcs[:, 1] >= 0) & (arcs[:, 1] < self.n_nodes)
        self.valid_arc = in_range
        self.invalid_arc_rows = np.flatnonzero(~in_range)

        # --- arc lengths (metres, great-circle between endpoints) -----------
        lon_f = nodes_lonlat[np.clip(arcs[:, 0], 0, self.n_nodes - 1), 0]
        lat_f = nodes_lonlat[np.clip(arcs[:, 0], 0, self.n_nodes - 1), 1]
        lon_t = nodes_lonlat[np.clip(arcs[:, 1], 0, self.n_nodes - 1), 0]
        lat_t = nodes_lonlat[np.clip(arcs[:, 1], 0, self.n_nodes - 1), 1]
        self.arc_len_m = np.where(
            in_range,
            haversine_m(lon_f, lat_f, lon_t, lat_t),
            np.inf,
        )

        # --- de-duplicated adjacency (CSR) ----------------------------------
        # Parallel arcs (same from->to listed multiple times) collapse to ONE
        # adjacency edge with the minimum length; every arc row keeps its ID.
        self._build_adjacency()

        # --- weakly connected components (undirected view) -------------------
        self.comp_label = self._components()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_adjacency(self):
        n = self.n_nodes
        arcs = self.arcs
        valid = self.valid_arc
        if valid.any():
            u = arcs[valid, 0]
            v = arcs[valid, 1]
            w = self.arc_len_m[valid]
            keys = u * n + v
            order = np.lexsort((w, keys))  # within a key, min weight first
            ks, vs, ws = keys[order], v[order], w[order]
            first = np.concatenate(([True], ks[1:] != ks[:-1]))
            kf = np.flatnonzero(first)
            uniq_keys = ks[kf]
            uniq_w = ws[kf]
            self.n_unique_edges = int(uniq_keys.size)
            edge_u = (uniq_keys // n).astype(np.int64)
            edge_v = (uniq_keys % n).astype(np.int64)
            # CSR grouping by source node
            indptr = np.searchsorted(edge_u, np.arange(n + 1), side="left")
            self.adj_indptr = indptr.astype(np.int64)
            self.adj_indices = edge_v
            self.adj_weights = uniq_w
        else:
            self.n_unique_edges = 0
            self.adj_indptr = np.zeros(n + 1, dtype=np.int64)
            self.adj_indices = np.zeros(0, dtype=np.int64)
            self.adj_weights = np.zeros(0, dtype=np.float64)

    def _components(self):
        """Weak connected component label per node (numpy label propagation)."""
        n = self.n_nodes
        lab = np.arange(n, dtype=np.int64)
        if self.n_unique_edges == 0 or n == 0:
            return lab
        # reconstruct undirected pairs from CSR
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.adj_indptr))
        cols = self.adj_indices
        both = np.concatenate((rows, cols))
        other = np.concatenate((cols, rows))
        for _ in range(128):
            rb, ob = lab[both], lab[other]
            m = np.minimum(rb, ob)
            np.minimum.at(lab, rb, m)
            np.minimum.at(lab, ob, m)
            while True:
                nxt = lab[lab]
                if np.array_equal(nxt, lab):
                    break
                lab = nxt
            if np.array_equal(lab[both], lab[other]):
                break
        return lab

    # ------------------------------------------------------------------ #
    # loaders (adapter for the raw student dataset)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_trip_files(cls, trip_dir: str, trip_id: str) -> "RoadGraph":
        """Load <trip_dir>/<trip_id>.nodes and .arcs (raw dataset format)."""
        nodes = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.nodes"), ndmin=2)
        arcs = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.arcs"), ndmin=2)
        return cls(nodes[:, :2], np.round(arcs[:, :2]).astype(np.int64), name=trip_id)

    # ------------------------------------------------------------------ #
    # convenient accessors
    # ------------------------------------------------------------------ #
    def arc_start(self, arc_id):
        return int(self.arcs[arc_id, 0])

    def arc_end(self, arc_id):
        return int(self.arcs[arc_id, 1])

    def node_lonlat(self, node_id):
        return float(self.nodes_lonlat[node_id, 0]), float(self.nodes_lonlat[node_id, 1])

    def clamp_offset(self, arc_id: int, offset_m: float) -> float:
        """Clamp a position along an arc into [0, arc_length]."""
        return float(min(max(offset_m, 0.0), float(self.arc_len_m[arc_id])))

    def n_components(self) -> int:
        if self.n_nodes == 0:
            return 0
        return int(np.unique(self.comp_label).size)

    # ------------------------------------------------------------------ #
    # bounded shortest-path search
    # ------------------------------------------------------------------ #
    def dijkstra(self, source: int, cutoff_m: float, max_visited: int = 50000) -> DijkstraResult:
        """Bounded Dijkstra from `source` up to `cutoff_m` metres (directed).

        Returns distances for every reachable node within the cutoff. The
        search never expands more than `max_visited` nodes; if the cap stops
        the search early, `exhausted=True` (distances are still valid upper
        bounds for the returned nodes). Zero-length edges (self-loops) are
        handled by the standard visited-set logic.
        """
        indptr, idx, w = self.adj_indptr, self.adj_indices, self.adj_weights
        best = {}
        if source < 0 or source >= self.n_nodes or cutoff_m <= 0:
            return DijkstraResult(best, 0, False)
        heap = [(0.0, int(source))]
        visited = 0
        exhausted = False
        while heap:
            d, u = heapq.heappop(heap)
            if u in best:
                continue
            best[u] = d
            visited += 1
            if visited >= max_visited:
                exhausted = True
                break
            if d >= cutoff_m:
                continue
            for j in range(indptr[u], indptr[u + 1]):
                nd = d + float(w[j])
                if nd > cutoff_m:
                    continue
                v = int(idx[j])
                if v not in best:
                    heapq.heappush(heap, (nd, v))
        return DijkstraResult(best, visited, exhausted)


class SearchCache:
    """LRU cache of bounded Dijkstra results, keyed by (source_node, cutoff).

    Keeps decoder runtime practical: consecutive observations frequently reuse
    the same source nodes. Deterministic eviction (oldest entry first).
    """

    def __init__(self, graph: RoadGraph, capacity: int = 512, max_visited: int = 50000):
        self.graph = graph
        self.capacity = max(1, int(capacity))
        self.max_visited = int(max_visited)
        self._store: "OrderedDict[tuple, DijkstraResult]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.n_visited_total = 0

    def get(self, source: int, cutoff_m: float) -> DijkstraResult:
        key = (int(source), round(float(cutoff_m), 3))
        res = self._store.get(key)
        if res is not None:
            self._store.move_to_end(key)
            self.hits += 1
            return res
        self.misses += 1
        res = self.graph.dijkstra(source, cutoff_m, self.max_visited)
        self.n_visited_total += res.n_visited
        self._store[key] = res
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return res
