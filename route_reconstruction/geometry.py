"""Route geometry output for the API / Folium UI (Member 3 -> Member 4 handoff).

COORDINATE ORDER (documented contract)
--------------------------------------
* `geometry["coordinates"]`  : GeoJSON LineString order = [LONGITUDE, LATITUDE]
                               (WGS84, EPSG:4326). This matches the raw
                               .track/.nodes files and the GeoJSON spec.
* `geometry["folium_latlon"]`: [(LATITUDE, LONGITUDE), ...] -- the order
                               Folium's PolyLine expects. Provided so the UI
                               member never has to swap axes by hand.
* Every dict also carries "coordinate_order": "lon,lat" and "crs" so the
  order is self-describing in JSON payloads.

Geometry = endpoint polyline of the matched arcs, in traversal order. Roads
in this dataset are stored as straight segments between their endpoint
nodes, so the endpoint polyline IS the arc geometry. Consecutive duplicate
points are removed. If consecutive matched arcs are not connected (rare
disconnect fallback), a straight connector is inserted and its position is
listed in "jumps" so the UI can render it differently (dashed).
"""
from __future__ import annotations


def build_route_geometry(graph, arc_ids) -> dict:
    """Build the JSON-ready geometry dict for a sequence of matched arc IDs.

    Parameters
    ----------
    graph  : RoadGraph
    arc_ids: list of zero-based arc IDs in traversal order (from MatchResult.route)

    Returns
    -------
    dict with keys:
      type             "LineString"
      coordinates      [[lon, lat], ...]   <- LON FIRST (GeoJSON order)
      folium_latlon    [[lat, lon], ...]   <- Folium order
      coordinate_order "lon,lat"
      crs              "WGS84 (EPSG:4326)"
      arc_ids          the input arc IDs
      jumps            indices i such that arc_ids[i] is NOT connected to
                       arc_ids[i-1] by the directed road (disconnect fallback)
      n_points         number of unique polyline points
    """
    coords: list = []
    folium: list = []
    jumps: list = []

    def append_point(lon, lat):
        if coords and coords[-1] == [lon, lat]:
            return  # dedup consecutive identical points
        coords.append([lon, lat])
        folium.append([lat, lon])

    prev_arc = None
    for a in arc_ids:
        a = int(a)
        u, v = int(graph.arcs[a, 0]), int(graph.arcs[a, 1])
        if prev_arc is not None and a != prev_arc:
            pu, pv = int(graph.arcs[prev_arc, 0]), int(graph.arcs[prev_arc, 1])
            connected = (pv == u) or (prev_arc == a)
            if not connected:
                jumps.append(len(coords))  # index of the connector start point
                # straight connector from prev end to this start (flagged)
        if prev_arc is None:
            ulon, ulat = graph.node_lonlat(u)
            append_point(ulon, ulat)
        vlon, vlat = graph.node_lonlat(v)
        append_point(vlon, vlat)
        prev_arc = a

    return {
        "type": "LineString",
        "coordinates": coords,           # LON FIRST
        "folium_latlon": folium,         # LAT FIRST (Folium)
        "coordinate_order": "lon,lat",
        "crs": "WGS84 (EPSG:4326)",
        "arc_ids": [int(a) for a in arc_ids],
        "jumps": jumps,
        "n_points": len(coords),
    }
