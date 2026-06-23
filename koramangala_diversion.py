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
import re
import difflib
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
            (12.9347, 77.6245),   # Koramangala — centroid
            dist=2000,             # 2 km radius — small enough to not crash RAM
            network_type="drive"
        )
        print(f"[+] OSM graph loaded — {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    except Exception as e:
        print(f"[!] OSM download failed ({e}). Using synthetic graph.")
        return None


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in metres."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# A few well-known real-world landmarks that sit just outside the 15-node
# synthetic graph but are commonly referenced in field reports (e.g. "Silk
# Board blocked"). Kept separate from JUNCTIONS so the routing graph itself
# stays small and well-connected; nearest_node() still snaps these to the
# closest in-graph junction for actual route computation.
EXTRA_LANDMARKS = {
    # Major junctions / chokepoints around Koramangala
    "silk board junction":        (12.9166, 77.6234),
    "silk board":                  (12.9166, 77.6234),
    "central silk board":          (12.9166, 77.6234),
    "agara junction":             (12.9242, 77.6504),
    "agara":                      (12.9242, 77.6504),
    "ibblur junction":            (12.9213, 77.6631),
    "iblur junction":             (12.9213, 77.6631),
    "iblur":                      (12.9213, 77.6631),
    "wipro junction":             (12.9132, 77.6843),
    "ecospace junction":          (12.9275, 77.6815),
    "kudlu gate":                 (12.8910, 77.6400),
    "jakkasandra junction":       (12.9248, 77.6382),

    # Roads
    "hosur road":                 (12.9295, 77.6152),
    "outer ring road":            (12.9245, 77.6510),
    "orr":                        (12.9245, 77.6510),
    "intermediate ring road":     (12.9355, 77.6248),
    "80 feet road":               (12.9355, 77.6275),
    "100 feet road":              (12.9341, 77.6234),
    "sarjapur road":              (12.9109, 77.6844),
    "old airport road":           (12.9380, 77.6250),
    "st johns road":              (12.9335, 77.6195),
    "jyoti nivas college road":   (12.9337, 77.6162),
    "koramangala industrial layout": (12.9350, 77.6360),
    "ejipura road":               (12.9431, 77.6284),
    "bannerghatta road":          (12.8928, 77.5989),

    # Areas / layouts (resolve to a representative point in that area)
    "koramangala":                (12.9347, 77.6245),
    "koramangala 1st block":      (12.9253, 77.6367),
    "koramangala 2nd block":      (12.9247, 77.6207),
    "koramangala 3rd block":      (12.9286, 77.6291),
    "koramangala 4th block":      (12.9339, 77.6302),
    "koramangala 5th block":      (12.9352, 77.6200),
    "koramangala 6th block":      (12.9397, 77.6222),
    "koramangala 7th block":      (12.9366, 77.6134),
    "koramangala 8th block":      (12.9411, 77.6174),
    "madiwala":                   (12.9226, 77.6174),
    "btm layout":                 (12.9100, 77.6100),
    "btm":                        (12.9100, 77.6100),
    "hsr layout":                 (12.9121, 77.6446),
    "hsr":                        (12.9121, 77.6446),
    "adugodi":                    (12.9420, 77.6083),
    "ejipura":                    (12.9431, 77.6284),
    "viveknagar":                 (12.9532, 77.6204),
    "jakkasandra":                (12.9248, 77.6382),
    "domlur":                     (12.9610, 77.6387),
    "indiranagar":                (12.9784, 77.6408),
    "bommanahalli":               (12.9030, 77.6242),
    "marathahalli":               (12.9569, 77.7011),
    "ecocity":                    (12.9275, 77.6815),

    # Named landmarks commonly used in field reports
    "forum mall":                 (12.9347, 77.6111),
    "sony world":                 (12.9342, 77.6234),
    "bda complex":                (12.9308, 77.6227),
    "jyoti nivas college":        (12.9337, 77.6162),
    "christ university":          (12.9344, 77.6059),
    "national games village":     (12.9454, 77.6223),
    "ngv":                        (12.9454, 77.6223),
    "raheja arcade":              (12.9346, 77.6133),
    "wipro corporate office":     (12.9132, 77.6843),
}

# Bengaluru civic-addressing pattern: "<N>(st|nd|rd|th) (Block|Cross|Main|Phase|Stage)"
# of a known area. Field reports very often phrase incidents this way
# ("7th Cross Koramangala", "4th Block, Koramangala") in combinations too
# numerous to hand-enumerate. When the curated dictionary above misses,
# we detect this pattern and resolve to the named area's anchor point —
# still meaningfully more accurate than the generic centroid fallback
# within a ~1-2km layout, and the caller clearly labels it as such.
_AREA_ANCHORS = {
    "koramangala": (12.9347, 77.6245),
    "btm":         (12.9100, 77.6100),
    "btm layout":  (12.9100, 77.6100),
    "hsr":         (12.9121, 77.6446),
    "hsr layout":  (12.9121, 77.6446),
    "indiranagar": (12.9784, 77.6408),
    "jayanagar":   (12.9308, 77.5838),
}
_ORDINAL_UNIT_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)\s+(block|cross|main|phase|stage)\b", re.IGNORECASE
)


def build_synthetic_graph() -> nx.DiGraph:
    """
    Synthetic but geographically accurate Koramangala graph.
    Nodes = real junction coordinates, edges = road segments
    with Haversine distances as weights.
    """
    JUNCTIONS = {
        1:  (12.9352, 77.6245, "Koramangala 1st Block Signal"),
        2:  (12.9342, 77.6234, "Sony World Junction"),
        3:  (12.9339, 77.6302, "Koramangala 4th Block"),
        4:  (12.9352, 77.6200, "Koramangala 5th Block"),
        5:  (12.9308, 77.6227, "BDA Complex Junction"),
        6:  (12.9347, 77.6111, "Forum Mall Signal"),
        7:  (12.9286, 77.6291, "Koramangala 3rd Block"),
        8:  (12.9380, 77.6250, "Old Airport Road Junction"),
        9:  (12.9355, 77.6248, "Intermediate Ring Road Cross"),
        10: (12.9247, 77.6207, "Koramangala 2nd Block"),
        11: (12.9384, 77.6326, "Ejipura Signal"),
        12: (12.9397, 77.6222, "Koramangala 6th Block"),
        13: (12.9411, 77.6174, "Koramangala 8th Block"),
        14: (12.9121, 77.6446, "HSR Layout Connector"),
        15: (12.9295, 77.6152, "Hosur Road Junction"),
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


def build_landmark_index(G: nx.DiGraph) -> dict:
    """
    Build a {lowercased landmark name -> (lat, lon)} lookup so free-text
    incident reports can be geocoded without a paid geocoding API.

    Two sources, since synthetic and real-OSM graphs name things
    differently:
      - Synthetic graph nodes carry a 'name' attribute (junction names) —
        a couple of short aliases are generated per node (e.g. "Forum Mall
        Signal" -> also matches "forum mall").
      - Real OSM graphs rarely name nodes but DO carry street names on
        edges ('name' tag, sometimes a list for multi-name segments). Those
        are indexed too, mapped to one of the edge's endpoint coordinates,
        so a real downloaded graph doesn't end up with an empty index.

    EXTRA_LANDMARKS / curated aliases are layered on top either way.
    """
    index: dict[str, tuple[float, float]] = {}
    STRIP_SUFFIXES = ("signal", "junction", "cross", "connector")

    for _, data in G.nodes(data=True):
        name = data.get("name")
        if not name:
            continue
        lat, lon = data.get("y"), data.get("x")
        if lat is None or lon is None:
            continue
        full = name.lower().strip()
        if len(full) >= 4:
            index[full] = (lat, lon)
        words = full.split()
        if len(words) > 1 and words[-1] in STRIP_SUFFIXES:
            short = " ".join(words[:-1])
            if len(short) >= 4:
                index.setdefault(short, (lat, lon))

    # Real-OSM fallback: edges carry street names even when nodes don't.
    for u, v, data in G.edges(data=True):
        raw = data.get("name")
        if not raw:
            continue
        names = raw if isinstance(raw, list) else [raw]
        u_data = G.nodes.get(u, {})
        lat, lon = u_data.get("y"), u_data.get("x")
        if lat is None or lon is None:
            continue
        for nm in names:
            full = str(nm).lower().strip()
            if len(full) >= 4:
                index.setdefault(full, (lat, lon))

    # Layer in well-known landmarks just outside the graph's own junctions.
    for alias, coords in EXTRA_LANDMARKS.items():
        index.setdefault(alias, coords)

    return index


def geocode_from_text(text: str, landmark_index: dict, default: tuple[float, float] = (12.9347, 77.6245)):
    """
    Resolve free-text (e.g. a field report) to coordinates using three
    passes, each one only run if the previous one found nothing:

      1. Exact substring match against `landmark_index` (curated names +
         EXTRA_LANDMARKS). Picks the LONGEST matching alias so a specific
         name ("koramangala 4th block") wins over a generic substring
         ("block").
      2. Bengaluru civic-addressing pattern — "<N>th Block/Cross/Main/Phase"
         next to (or combined with) a known area name — resolved to that
         area's anchor point. Covers the huge number of real addresses
         that can't be hand-enumerated in a fixed dictionary.
      3. Fuzzy match — catches typos / spacing / minor phrasing differences
         against every known alias using sequence-similarity, accepted only
         above a conservative threshold so it doesn't fire on unrelated text.

    Returns (lat, lon, matched_label) — matched_label is None only when ALL
    three passes fail, in which case (lat, lon) is the supplied default
    (Koramangala area centroid) and the caller should flag that clearly in
    the UI rather than silently presenting it as a confident detection.
    """
    text_l = (text or "").lower()

    # ── Pass 1: exact substring, longest alias wins ──────────────────────
    # Word-boundary aware: a naive `alias in text_l` check lets short
    # aliases like "orr" or "btm" false-match inside unrelated words
    # (e.g. "orr" inside "lorry"), which is exactly the kind of silent
    # wrong-answer this tool can't afford for a live police dispatch feed.
    best_alias = None
    for alias in landmark_index:
        if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text_l):
            if best_alias is None or len(alias) > len(best_alias):
                best_alias = alias

    # ── Pass 2: "<N>th Block/Cross/Main/Phase" + known area name ────────
    # Checked even when Pass 1 already matched, because a bare area name
    # ("koramangala") is less specific than an ordinal+area combo
    # ("5th block koramangala") even if the words appear in reverse order
    # and therefore don't form one contiguous substring.
    ordinal_match = _ORDINAL_UNIT_RE.search(text_l)
    if ordinal_match and (best_alias is None or best_alias in _AREA_ANCHORS):
        num, unit = ordinal_match.groups()
        for area_name, anchor in _AREA_ANCHORS.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(area_name) + r"(?![a-z0-9])", text_l):
                # If the exact "<area> <num><suffix> <unit>" string is itself
                # a curated entry (e.g. "koramangala 5th block"), that's more
                # precise than the bare area anchor — use it.
                specific_key = f"{area_name} {num}{_ordinal_suffix(int(num))} {unit}"
                if specific_key in landmark_index:
                    lat, lon = landmark_index[specific_key]
                    return lat, lon, specific_key.title()
                lat, lon = anchor
                label = f"{num}{_ordinal_suffix(int(num))} {unit.title()}, {area_name.title()} (area match)"
                return lat, lon, label
        if best_alias is None:
            # No explicit area named, but an ordinal-unit address pattern
            # was found — anchor to Koramangala since this tool's landmark
            # index is scoped to that beat, rather than the generic centroid.
            lat, lon = _AREA_ANCHORS["koramangala"]
            return lat, lon, f"{num}{_ordinal_suffix(int(num))} {unit.title()} (Koramangala area match)"

    if best_alias:
        lat, lon = landmark_index[best_alias]
        return lat, lon, best_alias.title()

    # ── Pass 3: fuzzy match against every known alias ────────────────────
    # Covers typos, abbreviations ("jn" for "junction"), and concatenated
    # spelling ("silkboard" for "silk board") that substring matching misses.
    words = re.findall(r"[a-z0-9]+", text_l)
    candidates = set(landmark_index.keys()) | set(_AREA_ANCHORS.keys())
    # Space-stripped lookup so "silkboard" can match "silk board".
    stripped_map = {c.replace(" ", ""): c for c in candidates}

    best_ratio, best_fuzzy_alias = 0.0, None
    for n in (1, 2, 3, 4):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            phrase_stripped = phrase.replace(" ", "")

            match = difflib.get_close_matches(phrase, candidates, n=1, cutoff=0.84)
            if match:
                ratio = difflib.SequenceMatcher(None, phrase, match[0]).ratio()
                if ratio > best_ratio:
                    best_ratio, best_fuzzy_alias = ratio, match[0]

            stripped_match = difflib.get_close_matches(phrase_stripped, stripped_map.keys(), n=1, cutoff=0.84)
            if stripped_match:
                ratio = difflib.SequenceMatcher(None, phrase_stripped, stripped_match[0]).ratio()
                if ratio > best_ratio:
                    best_ratio, best_fuzzy_alias = ratio, stripped_map[stripped_match[0]]

    if best_fuzzy_alias:
        coords = landmark_index.get(best_fuzzy_alias) or _AREA_ANCHORS.get(best_fuzzy_alias)
        if coords:
            return coords[0], coords[1], f"{best_fuzzy_alias.title()} (fuzzy match)"

    return default[0], default[1], None


def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def pick_through_route(G: nx.DiGraph, incident_node):
    """
    Pick a (source, dest) pair such that the *original* shortest path
    naturally runs through `incident_node` — i.e. simulate two ends of the
    road that the blocked junction connects, rather than requiring the
    caller to already know two arbitrary far-apart nodes.

    Ranks candidate neighbour pairs by three criteria, in priority order:
      1. The original (undisturbed) shortest path between them actually
         passes through incident_node — otherwise "diverting around it"
         is meaningless (some low-degree junctions have a direct shortcut
         edge between their two neighbours that bypasses them entirely).
      2. An alternate path still exists once incident_node is removed —
         this is the genuinely useful diversion case.
      3. Among ties, prefer the pair that's geographically farthest apart
         (more realistic "through traffic" than a short local loop).

    If no pair satisfies criterion 1, the junction isn't really on anyone's
    direct path between its own neighbours (a rare, degenerate case) and
    the best-effort farthest pair is returned. If criterion 1 is met but
    criterion 2 isn't, the junction is a genuine chokepoint (cut vertex)
    for that stretch of road — callers should treat "no alternate route"
    as a real, meaningful finding rather than a failure.

    Falls back to a 2-hop neighbour if the junction only has one connecting
    road. Returns (None, None) if there's nowhere sensible to route
    from/to at all (e.g. a fully isolated node).
    """
    neighbors = set(G.successors(incident_node)) | set(G.predecessors(incident_node))
    neighbors.discard(incident_node)

    if len(neighbors) == 0:
        return None, None

    if len(neighbors) == 1:
        n = next(iter(neighbors))
        further = (set(G.successors(n)) | set(G.predecessors(n))) - {incident_node, n}
        if not further:
            return None, None
        return n, next(iter(further))

    neighbors = list(neighbors)
    G_without = G.copy()
    G_without.remove_node(incident_node)
    G_undirected = G_without.to_undirected(as_view=True)

    candidates = []
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            a, b = neighbors[i], neighbors[j]
            dist_apart = haversine_m(G.nodes[a]["y"], G.nodes[a]["x"], G.nodes[b]["y"], G.nodes[b]["x"])
            try:
                orig_path = nx.shortest_path(G, a, b, weight="length")
                passes_through = incident_node in orig_path
            except nx.NetworkXNoPath:
                passes_through = False
            has_alternate = nx.has_path(G_undirected, a, b)
            candidates.append((passes_through, has_alternate, dist_apart, a, b))

    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    _, _, _, a, b = candidates[0]
    return a, b


def diversion_for_point(G: nx.DiGraph, lat: float, lon: float) -> dict:
    """
    One-call entry point for the app: snap (lat, lon) to the nearest graph
    junction, pick a sensible through-route around it, and compute the
    real diversion. This is what makes the diversion genuinely react to
    *where* the incident actually is, instead of always routing between two
    fixed demo nodes.
    """
    incident_node = nearest_node(G, lat, lon)
    source_node, dest_node = pick_through_route(G, incident_node)

    if source_node is None or dest_node is None:
        return {
            "incident_node": incident_node,
            "incident_name": G.nodes[incident_node].get("name", str(incident_node)),
            "error": "Incident junction has too few connecting roads to compute a through-route diversion.",
        }

    return get_diversion_by_node(G, incident_node, source_node, dest_node)


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