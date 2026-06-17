"""
Event-Driven Congestion — Dynamic Alternate Route Calculator
Koramangala, Bengaluru Street Network

Problem: When a junction is blocked (accident, procession, breakdown),
calculate the optimal diversion route in real time.

Tech: osmnx + networkx
Note: If osmnx cannot reach OpenStreetMap APIs (network restrictions),
      the script falls back to a built-in synthetic Koramangala graph.
"""

import math
import json
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("[!] osmnx not installed. Run: pip install osmnx networkx")
    print("    Continuing with synthetic graph.\n")


# ─────────────────────────────────────────────
# 1.  GRAPH — real OSM or synthetic fallback
# ─────────────────────────────────────────────

def load_osm_graph() -> nx.DiGraph | None:
    if not OSMNX_AVAILABLE:
        return None
    try:
        # Koramangala centroid, 2km radius
        print("[*] Downloading OSM graph for: Koramangala, Bengaluru (point + radius)")
        G = ox.graph_from_point(
            (12.9352, 77.6245),   # Koramangala 1st Block — centroid
            dist=2000,             # 2 km radius — small enough to not crash RAM
            network_type="drive"
        )
        print(f"[+] OSM graph loaded — {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    except Exception as e:
        print(f"[!] OSM download failed ({e}). Using synthetic graph.")
        return None


def build_synthetic_graph() -> nx.DiGraph:
    """
    Synthetic but geographically accurate Koramangala graph.
    Nodes = real junction coordinates, edges = road segments
    with Haversine distances as weights.
    """
    JUNCTIONS = {
        1:  (12.9352, 77.6245, "Koramangala 1st Block Signal"),
        2:  (12.9340, 77.6270, "Sony World Junction"),
        3:  (12.9310, 77.6260, "Koramangala 4th Block"),
        4:  (12.9290, 77.6280, "Koramangala 5th Block"),
        5:  (12.9320, 77.6300, "BDA Complex Junction"),
        6:  (12.9350, 77.6310, "Forum Mall Signal"),
        7:  (12.9370, 77.6285, "Koramangala 3rd Block"),
        8:  (12.9380, 77.6250, "Old Airport Road Junction"),
        9:  (12.9300, 77.6240, "Intermediate Ring Road Cross"),
        10: (12.9330, 77.6220, "Koramangala 2nd Block"),
        11: (12.9360, 77.6330, "Ejipura Signal"),
        12: (12.9280, 77.6310, "Koramangala 6th Block"),
        13: (12.9260, 77.6290, "Koramangala 8th Block"),
        14: (12.9250, 77.6260, "HSR Layout Connector"),
        15: (12.9340, 77.6350, "Hosur Road Junction"),
    }

    ROADS = [
        (1, 2), (1, 8), (1, 10),
        (2, 3), (2, 7), (2, 6),
        (3, 4), (3, 9),
        (4, 5), (4, 12),
        (5, 6), (5, 12),
        (6, 7), (6, 11),
        (7, 8),
        (8, 10),
        (9, 10), (9, 14),
        (10, 1),
        (11, 15),
        (12, 13),
        (13, 14),
        (14, 4),
        (15, 6),
    ]

    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    G = nx.DiGraph()
    for nid, (lat, lon, name) in JUNCTIONS.items():
        G.add_node(nid, y=lat, x=lon, name=name)

    for u, v in ROADS:
        lat1, lon1, _ = JUNCTIONS[u]
        lat2, lon2, _ = JUNCTIONS[v]
        dist = haversine_m(lat1, lon1, lat2, lon2)
        G.add_edge(u, v, length=dist)
        G.add_edge(v, u, length=dist)

    print(f"[+] Synthetic graph built — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ─────────────────────────────────────────────
# 2.  CORE ALGORITHM
# ─────────────────────────────────────────────

def nearest_node(G: nx.DiGraph, lat: float, lon: float) -> int:
    """Find the graph node closest to the given coordinates."""
    if OSMNX_AVAILABLE and hasattr(ox, "distance"):
        try:
            return ox.distance.nearest_nodes(G, lon, lat)
        except Exception:
            pass
    # Pure-Python fallback (Euclidean on lat/lon is fine for small areas)
    best, best_dist = None, float("inf")
    for n, d in G.nodes(data=True):
        dist = math.hypot(d["y"] - lat, d["x"] - lon)
        if dist < best_dist:
            best, best_dist = n, dist
    return best


def get_diversion_by_node(G, incident_node, source_node, dest_node):
    """Same logic as get_diversion but takes node IDs directly — no coordinate snapping error."""
    result = {
        "incident_node": incident_node,
        "incident_name": G.nodes[incident_node].get("name", str(incident_node)),
        "source_node":   source_node,
        "source_name":   G.nodes[source_node].get("name", str(source_node)),
        "dest_node":     dest_node,
        "dest_name":     G.nodes[dest_node].get("name", str(dest_node)),
    }

    # Original route
    try:
        orig_path = nx.shortest_path(G, source_node, dest_node, weight="length")
        orig_len  = nx.shortest_path_length(G, source_node, dest_node, weight="length")
        result["original_route"] = {
            "path":     orig_path,
            "coords":   [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in orig_path],
            "length_m": round(orig_len, 1),
            "passes_through_incident": incident_node in orig_path,
        }
    except nx.NetworkXNoPath:
        result["original_route"] = None

    # Block incident node and reroute
    G_div = G.copy()
    G_div.remove_node(incident_node)

    try:
        div_path = nx.shortest_path(G_div, source_node, dest_node, weight="length")
        div_len  = nx.shortest_path_length(G_div, source_node, dest_node, weight="length")
        result["diverted_route"] = {
            "path":     div_path,
            "coords":   [(G_div.nodes[n]["y"], G_div.nodes[n]["x"]) for n in div_path],
            "length_m": round(div_len, 1),
        }
        orig_val = result["original_route"]["length_m"] if result["original_route"] else 0
        result["detour_extra_m"] = round(div_len - orig_val, 1)
    except nx.NetworkXNoPath:
        result["diverted_route"] = None
        result["error"] = "No alternate route found."

    return result


# ─────────────────────────────────────────────
# 3.  PRETTY PRINT HELPER
# ─────────────────────────────────────────────

def print_result(result: dict, G: nx.DiGraph) -> None:
    sep = "─" * 60

    def label(node):
        # OSM nodes rarely have a 'name'; fall back to the node ID
        return G.nodes[node].get("name") or str(node)

    print(sep)
    print(f"  INCIDENT : {result['incident_name']} (node {result['incident_node']})")
    print(f"  FROM     : {result['source_name']}")
    print(f"  TO       : {result['dest_name']}")
    print(sep)

    if result.get("original_route"):
        r = result["original_route"]
        names = " → ".join(label(n) for n in r["path"])
        print(f"\n  ORIGINAL ROUTE  ({r['length_m']} m)")
        print(f"  Path  : {names}")
        print(f"  Passes through incident: {r['passes_through_incident']}")
    else:
        print("\n  No original route found.")

    if result.get("diverted_route"):
        d = result["diverted_route"]
        names = " → ".join(label(n) for n in d["path"])
        print(f"\n  DIVERTED ROUTE  ({d['length_m']} m)  [+{result['detour_extra_m']} m extra]")
        print(f"  Path  : {names}")
    elif result.get("error"):
        print(f"\n  ERROR: {result['error']}")

    print()


# ─────────────────────────────────────────────
# 4.  DEMO — three real Koramangala scenarios
# ─────────────────────────────────────────────

SCENARIOS = [
    {
        "title":         "Vehicle Breakdown — Forum Mall Signal blocked",
        "incident_node": 3145158631,
        "source_node":   6528488886,
        "dest_node":     3870651467,
        # orig=634m → diverted=638m  (+4m — tight urban reroute)
    },
    {
        "title":         "Procession — Sony World Junction blocked",
        "incident_node": 305092008,
        "source_node":   11017447792,
        "dest_node":     1274328906,
        # orig=738m → diverted=1021m  (+283m detour)
    },
    {
        "title":         "Accident — Hosur Road Corridor blocked",
        "incident_node": 1272830640,
        "source_node":   6528482654,
        "dest_node":     443224565,
        # orig=677m → diverted=1368m  (+691m detour — most dramatic)
    },
]

if __name__ == "__main__":
    G = load_osm_graph() or build_synthetic_graph()

    print("\n" + "═" * 60)
    print("  KORAMANGALA DYNAMIC DIVERSION ENGINE")
    print("═" * 60 + "\n")

    for i, s in enumerate(SCENARIOS, 1):
        print(f"  Scenario {i}: {s['title']}\n")
        result = get_diversion_by_node(
            G,
            incident_node=s["incident_node"],
            source_node=s["source_node"],
            dest_node=s["dest_node"],
        )
        print_result(result, G)

    # ── Export last result as JSON (handy for API / front-end) ───
    with open("diversion_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  Last result saved to diversion_result.json")
