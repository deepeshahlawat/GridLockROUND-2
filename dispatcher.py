"""
=============================================================================
  PHASE 3 — THE PRESCRIPTIVE DISPATCHER (Manpower & Barricade Engine)
  Event-Driven Congestion — Koramangala, Bengaluru

  Inputs   : incident location + S_risk (NLP score) + P(Long delay) from
             xgb_clearance_classifier.pkl
  Output   : officer count, dispatch station, distance, barricade protocol

  This is the Zero-Shot Prescriptive MOBLP described in the project PDF
  (p.11): since there is no historical `assigned_to_police_id` column to
  learn from, manpower is PRESCRIBED with a small heuristic matrix instead
  of a trained model. It is deliberately simple — a 6-day hackathon needs a
  defensible rule set, not a genetic algorithm.

  Pipeline position:
    1. Incident reported           (lat/lon, text)
    2. NLP                          -> S_risk
    3. xgb_clearance_classifier.pkl -> P(Long delay)
    4. koramangala_diversion.py     -> alternate route + detour length
    5. dispatcher.py  (THIS FILE)   -> "Send N officers from Station X,
                                        with barricade protocol Y"
=============================================================================
"""

from __future__ import annotations

import json
from math import radians, cos, sin, asin, sqrt
from typing import Optional

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("[!] joblib not installed. Run: pip install joblib")
    print("    The dispatcher will still run using a supplied probability.\n")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Reuse the router engine if it's sitting next to this file — lets the
# detour length from Stream B influence the barricade decision in Step 4.
try:
    from koramangala_diversion import build_synthetic_graph, get_diversion_by_node
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False


# ─────────────────────────────────────────────
# 1.  POLICE STATION REGISTRY  (Ω_p — jurisdictional constraint)
# ─────────────────────────────────────────────
# Small built-in default so this file runs standalone with no data file.
# For real deployments, call build_station_registry(ml_ready_df) instead —
# it derives the actual station roster (53 stations, not 5) straight from
# the `police_station` column already in ml_ready.csv.
STATIONS: dict[str, tuple[float, float]] = {
    "Koramangala PS": (12.9411, 77.6210),
    "Madiwala PS":    (12.9210, 77.6207),
    "HSR Layout PS":  (12.9202, 77.6513),
    "Adugodi PS":     (12.9416, 77.6087),
    "Viveknagar PS":  (12.9519, 77.6224),
}

# Soft cap on simultaneously-deployed officers per station before the
# dispatcher looks at the next-nearest one. Tune per actual station strength.
STATION_CAPACITY = 25


def build_station_registry(
    df,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    station_col: str = "police_station",
    exclude: tuple[str, ...] = ("No Police Station",),
) -> dict[str, tuple[float, float]]:
    """
    Build the REAL station registry from ml_ready.csv, replacing the small
    hardcoded STATIONS demo dict above.

    ml_ready.csv has no station-house coordinates of its own — only the
    incident's lat/lon and the jurisdiction it was filed under. This takes
    the centroid of all incidents reported under each station as a stand-in
    for that station's location. It's an approximation (true station-house
    coordinates would be tighter), but it's grounded in real jurisdiction
    boundaries rather than 5 guessed coordinates, and it scales to every
    station in the dataset instead of just the Koramangala-adjacent ones.
    """
    valid = df[~df[station_col].isin(exclude)]
    centroids = valid.groupby(station_col)[[lat_col, lon_col]].mean()
    return {name: (row[lat_col], row[lon_col]) for name, row in centroids.iterrows()}


# ─────────────────────────────────────────────
# 2.  DISTANCE — Haversine
# ─────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    R = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return R * (2 * asin(sqrt(a)))


def find_nearest_station(
    lat: float,
    lon: float,
    stations: dict[str, tuple[float, float]] = STATIONS,
    station_load: Optional[dict[str, int]] = None,
    requested_officers: int = 1,
) -> tuple[str, float]:
    """
    Jurisdictional constraint (Ω_p): pick the closest station.
    If `station_load` is supplied and the nearest station is already at/near
    capacity, fall back to the next-nearest rather than over-drawing one
    station for every nearby incident.
    """
    station_load = station_load or {}
    ranked = sorted(
        ((name, haversine_km(lat, lon, s_lat, s_lon)) for name, (s_lat, s_lon) in stations.items()),
        key=lambda x: x[1],
    )
    for name, dist in ranked:
        current = station_load.get(name, 0)
        if current + requested_officers <= STATION_CAPACITY:
            return name, dist
    # Every station is saturated — fall back to the nearest one anyway and
    # let a human dispatcher know it's over capacity.
    return ranked[0]


# ─────────────────────────────────────────────
# 3.  HEURISTIC OPTIMIZATION MATRIX  (PDF p.11, rules 1-4)
# ─────────────────────────────────────────────
def get_deployment_plan(
    incident_lat: float,
    incident_lon: float,
    event_cause: str,
    s_risk: int,
    prob_long_delay: float,
    stations: dict[str, tuple[float, float]] = STATIONS,
    station_load: Optional[dict[str, int]] = None,
) -> dict:
    """
    Zero-Shot Prescriptive Manpower Optimization.

    Rule 1 — Base demand (D_min):        every acute incident needs >=1 officer
    Rule 2 — Risk multiplier:            S_risk == 3  -> +2 officers
                                          (S_risk == 2 -> +1, a smoother
                                           extension of the spec so a
                                           medium-risk incident isn't treated
                                           identically to a routine one)
    Rule 3 — Probabilistic multiplier:   P(Long delay) > 0.60 -> +2 officers
    Rule 4 — Jurisdictional constraint:  dispatch from the nearest station
                                          (capacity-aware, see Ω_p above)
    """
    officers_needed = 1  # Rule 1

    if s_risk == 3:
        officers_needed += 2
    elif s_risk == 2:
        officers_needed += 1

    if prob_long_delay > 0.60:
        officers_needed += 2

    officers_needed = min(officers_needed, 6)  # don't drain a station for one incident

    station, dist_km = find_nearest_station(
        incident_lat, incident_lon, stations, station_load, officers_needed
    )

    if s_risk == 3:
        barricade = "Type III MUTCD + Scene Cordon"
    elif prob_long_delay > 0.60:
        barricade = "Type III MUTCD"
    else:
        barricade = "Standard Cones"

    return {
        "event_cause": event_cause,
        "s_risk": s_risk,
        "prob_long_delay": round(prob_long_delay, 3),
        "recommended_officers": officers_needed,
        "dispatch_station": station,
        "distance_to_incident_km": round(dist_km, 2),
        "barricade_protocol": barricade,
        "over_capacity_warning": (station_load or {}).get(station, 0) + officers_needed > STATION_CAPACITY,
    }


# ─────────────────────────────────────────────
# 4.  ML INTEGRATION — xgb_clearance_classifier.pkl
# ─────────────────────────────────────────────
def within_graph_bounds(graph, lat: float, lon: float, buffer_deg: float = 0.01) -> bool:
    """
    True if (lat, lon) falls inside the diversion graph's own coverage area
    (+ a small buffer). nearest_node() will always return SOME node no
    matter how far away the query point is, so without this check an
    incident in, say, Yelahanka would silently get "rerouted" around a
    Koramangala junction 20km away — wrong, but no exception is raised.
    """
    lats = [d["y"] for _, d in graph.nodes(data=True)]
    lons = [d["x"] for _, d in graph.nodes(data=True)]
    return (min(lats) - buffer_deg <= lat <= max(lats) + buffer_deg) and \
           (min(lons) - buffer_deg <= lon <= max(lons) + buffer_deg)


def event_cause_from_row(row, prefix: str = "event_cause_") -> str:
    """ml_ready.csv stores event_cause one-hot encoded; recover the label."""
    for col in row.index:
        if col.startswith(prefix) and row[col] == 1:
            return col[len(prefix):]
    return "unknown"


def load_classifier(model_path: str = "xgb_clearance_classifier.pkl"):
    """Load the trained model from train_model.py. Returns None if unavailable."""
    if not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"[!] {model_path} not found — run train_model.py first, "
              f"or pass prob_long_delay directly to dispatch_for_incident().")
        return None


def build_feature_row(raw_rows, routing_cols=("latitude", "longitude", "police_station"), target_col="Target_Duration_Mins"):
    """
    Reproduces the exact preprocessing in train_model.py's Section 2 so a
    raw row (or rows) pulled straight from ml_ready.csv can be fed into
    model.predict_proba() without the columns drifting out of sync with how
    the model was trained. `raw_rows` is a DataFrame slice with the ORIGINAL
    ml_ready.csv columns still intact (don't pre-drop anything).
    """
    drop_cols = [c for c in ([target_col] + list(routing_cols)) if c in raw_rows.columns]
    X = raw_rows.drop(columns=drop_cols)
    X.columns = [
        c.replace(" ", "_").replace("/", "_").replace("-", "_").replace("(", "").replace(")", "")
        for c in X.columns
    ]
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X.fillna(0, inplace=True)
    return X


def predict_long_delay_probability(model, feature_row) -> float:
    """
    feature_row: a single-row DataFrame with the SAME columns/order the
    model was trained on in train_model.py (i.e. X.columns after the
    sanitisation step there). Class index 2 == "Long (>120 min)".
    """
    proba = model.predict_proba(feature_row)
    return float(proba[0][2])


# ─────────────────────────────────────────────
# 5.  END-TO-END HELPER — ties Steps 3 + 5 (and optionally 4) together
# ─────────────────────────────────────────────
def dispatch_for_incident(
    incident: dict,
    model=None,
    feature_row=None,
    diversion_graph=None,
    station_load: Optional[dict[str, int]] = None,
    stations: dict[str, tuple[float, float]] = STATIONS,
) -> dict:
    """
    incident: {
        "lat": float, "lon": float, "event_cause": str, "s_risk": int,
        "prob_long_delay": float (optional, used if model/feature_row absent),
        "incident_node": int (optional, only if diversion_graph is supplied),
        "source_node": int, "dest_node": int (optional, for routing context)
    }
    `stations` defaults to the small built-in STATIONS dict — pass the output
    of build_station_registry(ml_ready_df) to dispatch against the real
    station roster instead.
    """
    # Step 3: get P(Long) from the trained classifier if one was supplied,
    # otherwise trust whatever probability the caller already computed.
    if model is not None and feature_row is not None:
        prob_long_delay = predict_long_delay_probability(model, feature_row)
    else:
        prob_long_delay = incident.get("prob_long_delay", 0.0)

    plan = get_deployment_plan(
        incident_lat=incident["lat"],
        incident_lon=incident["lon"],
        event_cause=incident["event_cause"],
        s_risk=incident["s_risk"],
        prob_long_delay=prob_long_delay,
        stations=stations,
        station_load=station_load,
    )

    # Step 4 (optional): if the router is available and node IDs were
    # supplied, fold the detour length into the plan as extra context —
    # a long forced detour means upstream barricading matters even more.
    if (
        ROUTING_AVAILABLE
        and diversion_graph is not None
        and all(k in incident for k in ("incident_node", "source_node", "dest_node"))
    ):
        diversion = get_diversion_by_node(
            diversion_graph,
            incident["incident_node"],
            incident["source_node"],
            incident["dest_node"],
        )
        plan["diversion"] = {
            "detour_extra_m": diversion.get("detour_extra_m"),
            "diverted_route_length_m": (diversion.get("diverted_route") or {}).get("length_m"),
        }
        if (diversion.get("detour_extra_m") or 0) > 250:
            plan["barricade_protocol"] += " + Extended Upstream Barricading"

    plan["incident_location"] = {"lat": incident["lat"], "lon": incident["lon"]}
    return plan


# ─────────────────────────────────────────────
# 6.  PRETTY PRINT
# ─────────────────────────────────────────────
def print_plan(title: str, plan: dict) -> None:
    sep = "─" * 60
    print(sep)
    print(f"  INCIDENT : {title}")
    print(f"  CAUSE    : {plan['event_cause']}  (S_risk={plan['s_risk']})")
    print(f"  P(Long)  : {plan['prob_long_delay']:.0%}")
    print(sep)
    print(f"  → Deploy {plan['recommended_officers']} officer(s) from {plan['dispatch_station']}"
          f"  ({plan['distance_to_incident_km']} km away)")
    print(f"  → Barricade protocol: {plan['barricade_protocol']}")
    if plan.get("over_capacity_warning"):
        print("  ⚠  Nearest eligible station is at/near capacity — consider mutual aid.")
    if plan.get("diversion"):
        d = plan["diversion"]
        print(f"  → Forced detour: +{d['detour_extra_m']} m "
              f"(diverted route {d['diverted_route_length_m']} m)")
    print()


# ─────────────────────────────────────────────
# 7.  DEMO  vs  REAL RUN
# ─────────────────────────────────────────────
# If ml_ready.csv + xgb_clearance_classifier.pkl are sitting next to this
# file, run against the REAL model and REAL station roster. Otherwise fall
# back to the small synthetic demo so the script still runs standalone.
ML_READY_CSV = "ml_ready.csv"
MODEL_PKL    = "xgb_clearance_classifier.pkl"

DEMO_INCIDENTS = [
    {
        "title":           "Vehicle Breakdown — Forum Mall Signal",
        "lat": 12.9350, "lon": 77.6310,
        "event_cause":     "vehicle_breakdown",
        "s_risk":          1,
        "prob_long_delay": 0.18,
    },
    {
        "title":           "Procession — Sony World Junction",
        "lat": 12.9340, "lon": 77.6270,
        "event_cause":     "procession",
        "s_risk":          2,
        "prob_long_delay": 0.55,
    },
    {
        "title":           "Accident — Hosur Road Corridor",
        "lat": 12.9340, "lon": 77.6350,
        "event_cause":     "accident",
        "s_risk":          3,
        "prob_long_delay": 0.72,
    },
]


def run_demo() -> None:
    print("  [i] ml_ready.csv / xgb_clearance_classifier.pkl not found next to")
    print("      this script — running the synthetic demo instead.\n")
    station_load: dict[str, int] = {}
    all_plans = []
    for incident in DEMO_INCIDENTS:
        plan = dispatch_for_incident(incident, station_load=station_load)
        print_plan(incident["title"], plan)
        all_plans.append({"title": incident["title"], **plan})
        station_load[plan["dispatch_station"]] = (
            station_load.get(plan["dispatch_station"], 0) + plan["recommended_officers"]
        )
    with open("dispatch_plan.json", "w") as f:
        json.dump(all_plans, f, indent=2)
    print("  All plans saved to dispatch_plan.json")
    if not ROUTING_AVAILABLE:
        print("\n  [i] koramangala_diversion.py not found alongside this file —")
        print("      running without Step 4 (diversion-aware barricading).")


def run_real(n_samples: int = 5) -> None:
    """
    Real end-to-end run: real model.predict_proba(), real station roster
    built from ml_ready.csv, real (or synthetic-fallback) diversion routing
    for incidents that fall inside koramangala_diversion.py's coverage area.
    """
    if not PANDAS_AVAILABLE:
        print("  [!] pandas required for run_real(). pip install pandas.")
        return

    df = pd.read_csv(ML_READY_CSV, low_memory=False)
    model = load_classifier(MODEL_PKL)
    if model is None:
        run_demo()
        return

    stations = build_station_registry(df)
    print(f"  Loaded {len(df)} incidents, {len(stations)} real stations from {ML_READY_CSV}\n")

    diversion_graph = None
    if ROUTING_AVAILABLE:
        from koramangala_diversion import load_osm_graph, build_synthetic_graph, nearest_node
        diversion_graph = load_osm_graph() or build_synthetic_graph()

    # Sample a spread of real incidents: prioritise variety in S_risk so the
    # heuristic matrix's branches all actually get exercised in one run.
    # (Deliberately not using groupby().apply() here — pandas 3.x silently
    # drops the grouping column from the result unless include_groups=True
    # is passed, which doesn't exist on older pandas. Plain concat avoids
    # the version trap entirely.)
    per_group = max(1, n_samples // 3)
    sample = pd.concat(
        [group.sample(min(len(group), per_group), random_state=42) for _, group in df.groupby("S_risk")]
    ).reset_index(drop=True)

    station_load: dict[str, int] = {}
    all_plans = []

    for i, row in sample.iterrows():
        feat = build_feature_row(pd.DataFrame([row]))
        incident = {
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "event_cause": event_cause_from_row(row),
            "s_risk": int(row["S_risk"]),
        }

        # Only wire in routing if this incident actually falls inside the
        # diversion engine's known graph — otherwise node IDs are meaningless.
        if diversion_graph is not None and diversion_graph.number_of_nodes() > 2:
            inc_node = nearest_node(diversion_graph, incident["lat"], incident["lon"])
            others = [n for n in diversion_graph.nodes if n != inc_node]
            if len(others) >= 2:
                incident["incident_node"] = inc_node
                incident["source_node"] = others[0]
                incident["dest_node"] = others[-1]

        plan = dispatch_for_incident(
            incident, model=model, feature_row=feat,
            diversion_graph=diversion_graph,
            station_load=station_load, stations=stations,
        )
        title = f"Incident #{i} — {incident['event_cause']} (CSV jurisdiction: {row['police_station']})"
        print_plan(title, plan)
        all_plans.append({"title": title, **plan})
        station_load[plan["dispatch_station"]] = (
            station_load.get(plan["dispatch_station"], 0) + plan["recommended_officers"]
        )

    with open("dispatch_plan.json", "w") as f:
        json.dump(all_plans, f, indent=2)
    print(f"  {len(all_plans)} real plans saved to dispatch_plan.json")


if __name__ == "__main__":
    import os
    print("\n" + "═" * 60)
    print("  KORAMANGALA PRESCRIPTIVE DISPATCHER")
    print("═" * 60 + "\n")

    if os.path.exists(ML_READY_CSV) and os.path.exists(MODEL_PKL):
        run_real()
    else:
        run_demo()