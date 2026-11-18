"""Geodesic and planar helpers for the route-reconstruction module (Member 3).

Coordinate-order convention used across this whole module (and documented for
the other members):

    * Raw dataset files (.track, .nodes) store (longitude, latitude) pairs.
      Column 0 = longitude, column 1 = latitude, column 2 = timestamp (seconds).
    * Every coordinate list produced here is [(lon, lat), ...] -- LON FIRST --
      except `folium_latlon` lists in geometry.py which are explicitly
      converted to [(lat, lon), ...] because Folium expects that order.

Only NumPy + the standard library are required (no scipy).
"""
from __future__ import annotations

import numpy as np

EARTH_R = 6371000.0  # mean Earth radius in metres (same convention as data/ inventory)


def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance in metres between (lon, lat) points.

    Accepts scalars or arrays (broadcasting applies). Inputs and outputs are
    in degrees / metres respectively.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalMetric:
    """Equirectangular projection of one trip into local plane metres.

    x = R * cos(lat0) * radians(lon - lon0)
    y = R * radians(lat - lat0)

    Valid for the scale of a single trip (<= a few hundred km); distortion is
    far below the candidate radius used for matching. Used for point-to-arc
    distances, projection offsets and the reference-route alignment in the
    evaluator. Great-circle distances (haversine) are used for observation
    displacement and arc lengths; the two metrics agree to <1% at trip scale.
    """

    __slots__ = ("lon0", "lat0", "kx")

    def __init__(self, lonlat):
        arr = np.asarray(lonlat, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            # degenerate: fall back to a fixed origin
            self.lon0, self.lat0, self.kx = 0.0, 0.0, 1.0
            return
        self.lon0 = float(np.mean(arr[:, 0]))
        self.lat0 = float(np.mean(arr[:, 1]))
        self.kx = EARTH_R * np.cos(np.radians(self.lat0)) * np.pi / 180.0

    def to_xy(self, lon, lat):
        """Project (lon, lat) scalars/arrays to local metres (x, y)."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        x = (lon - self.lon0) * self.kx
        y = (lat - self.lat0) * (EARTH_R * np.pi / 180.0)
        return x, y


def point_segment_projection(px, py, ax, ay, bx, by):
    """Closest point of segment A->B to point P, in local plane metres.

    Returns (dist, t) where t in [0, 1] is the fractional position along A->B
    (clamped to the segment) and dist is the perpendicular/endpoint distance.
    All inputs may be scalars.
    """
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        # degenerate segment (identical endpoints, e.g. a self-loop arc)
        d = float(np.hypot(px - ax, py - ay))
        return d, 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = min(1.0, max(0.0, float(t)))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(np.hypot(px - cx, py - cy)), t
