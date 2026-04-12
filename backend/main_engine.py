import os
import sys
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "c101.txt")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from load_data import load_solomon_data, calculate_distance_matrix  # noqa: E402
from aco_algorithm import ACO_Colony  # noqa: E402

# Real-world map regions (WGS84) — Solomon X/Y are scaled into these bounds.
SCENARIOS: Dict[str, Dict[str, float]] = {
    # Strictly Land-locked: Central Park / Midtown East (Zero Water)
    "nyc": {"south": 40.760, "west": -73.980, "north": 40.800, "east": -73.950},
    # Strictly Land-locked: Hyde Park / Mayfair
    "london": {"south": 51.500, "west": -0.160, "north": 51.520, "east": -0.120},
    "bangalore": {"south": 12.898, "west": 77.548, "north": 13.048, "east": 77.688},
    "sf": {"south": 37.740, "west": -122.440, "north": 37.800, "east": -122.380},
    "pittsburgh": {"south": 40.435, "west": -79.965, "north": 40.455, "east": -79.915},
    "denver": {"south": 39.720, "west": -105.020, "north": 39.770, "east": -104.970},
    "kansas_city": {"south": 39.080, "west": -94.600, "north": 39.120, "east": -94.560},
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "num_depots": 3,
    "vehicle_capacity": 200,
    "num_ants": 20,
    "alpha": 1.0,
    "beta": 2.0,
    "evaporation": 0.5,
    "iterations": 50,
    "vehicle_penalty": 100.0,
    "scenario": "nyc",
    "max_customers": 200,
    "synthetic_extra": 0,
    "random_seed": 42,
    "dataset_path": None,
    "exploration_bias": 0.5,     # 0 to 1, mixes alpha/beta ratios
    "failure_mode": False,       # forces strictly suboptimal parameters
    "use_local_search": True,    # 2-opt toggle
    "incremental": False,        # if true, try warm-starting pheromones
    "showcase": False            # educational walkthrough mode
}

PHEROMONE_CACHE: Dict[str, List[np.ndarray]] = {}

def _merge_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    if params:
        for key, value in params.items():
            if value is not None and key in DEFAULT_PARAMS:
                merged[key] = value
    return merged

def _xy_bounds(customers: pd.DataFrame) -> Tuple[float, float, float, float]:
    xs = customers["XCOORD"].astype(float)
    ys = customers["YCOORD"].astype(float)
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())

def project_xy_to_geo(
    x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float, bbox: Dict[str, float]
) -> Tuple[float, float]:
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    u = (float(x) - xmin) / dx
    v = (float(y) - ymin) / dy
    # 15% Safety Margin Inset (Node centering logic)
    margin = 0.15
    u_safe = margin + u * (1.0 - 2.0 * margin)
    v_safe = margin + v * (1.0 - 2.0 * margin)
    lat = bbox["south"] + u_safe * (bbox["north"] - bbox["south"])
    lng = bbox["west"] + v_safe * (bbox["east"] - bbox["west"])
    return lat, lng

def _augment_synthetic(customers: pd.DataFrame, n_extra: int, rng: np.random.Generator) -> pd.DataFrame:
    if n_extra <= 0:
        return customers
    xmin, xmax, ymin, ymax = _xy_bounds(customers)
    rows = []
    base_id = int(customers["CUST_NO"].max()) + 1
    for i in range(n_extra):
        rows.append({
            "CUST_NO": base_id + i,
            "XCOORD": float(rng.uniform(xmin, xmax)),
            "YCOORD": float(rng.uniform(ymin, ymax)),
            "DEMAND": float(rng.integers(10, 41)),
            "READY_TIME": 0.0,
            "DUE_DATE": 1e6,
            "SERVICE_TIME": 90.0,
        })
    extra = pd.DataFrame(rows)
    return pd.concat([customers, extra], ignore_index=True)


def adjust_boundaries(K: int, labels: np.ndarray, coords: np.ndarray, depots_coords: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Implement Dynamic Boundary Adjustment weighing multiple objectives: Geometry + Regional Demand Density"""
    new_labels = labels.copy()
    
    # Calculate initial demand density per cluster
    cluster_demands = {d: 0.0 for d in range(K)}
    for i in range(len(demands)):
        cluster_demands[new_labels[i]] += demands[i]
        
    for i in range(len(coords)):
        curr_depot = new_labels[i]
        curr_dist = np.linalg.norm(coords[i] - depots_coords[curr_depot])
        
        for d in range(K):
            if d != curr_depot:
                # Multi-objective transition probability:
                # 1. Must be physically closer (saving global travel)
                # 2. Target depot shouldn't be critically overloaded compared to source
                dist_shift = np.linalg.norm(coords[i] - depots_coords[d])
                if dist_shift < curr_dist * 0.85:
                    if cluster_demands[d] + demands[i] < cluster_demands[curr_depot] * 1.5:
                        cluster_demands[curr_depot] -= demands[i]
                        cluster_demands[d] += demands[i]
                        new_labels[i] = d
                        curr_dist = dist_shift
                        curr_depot = d
    return new_labels

def run_full_optimization(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = _merge_params(params)
    
    if p.get("failure_mode", False):
        p["alpha"] = 0.01
        p["beta"] = 0.01
        p["evaporation"] = 0.99
        p["use_local_search"] = False
    else:
        # adjust alpha/beta based on exploration bias (0 to 1) 
        # higher exploration = higher alpha, lower beta
        # exploitation = lower alpha, higher beta
        bias = float(p.get("exploration_bias", 0.5))
        p["alpha"] = p["alpha"] * (0.5 + bias)
        p["beta"] = p["beta"] * (1.5 - bias)

    if p.get("showcase", False):
        p["num_depots"] = 1
        p["max_customers"] = 12
        p["iterations"] = 30 # Keep it short for educational speed
        p["scenario"] = "kansas_city"


    dataset_path = p["dataset_path"] or DATASET_PATH
    scenario_id = str(p["scenario"]).lower()
    if scenario_id not in SCENARIOS:
        scenario_id = "nyc"
    bbox = SCENARIOS[scenario_id]

    rng = np.random.default_rng(int(p["random_seed"]))

    depot, customers = load_solomon_data(dataset_path)
    if depot is None:
        return {"error": "Dataset not loaded"}

    customers = customers.copy()
    if len(customers) == 0:
        return {"error": "No customers in dataset"}

    max_c = max(1, min(int(p["max_customers"]), len(customers)))
    order = rng.permutation(len(customers))[:max_c]
    customers = customers.iloc[order].reset_index(drop=True)
    customers = _augment_synthetic(customers, int(p["synthetic_extra"]), rng)

    xmin, xmax, ymin, ymax = _xy_bounds(customers)
    K = max(1, min(int(p["num_depots"]), len(customers)))

    coords = customers[["XCOORD", "YCOORD"]].values
    kmeans = KMeans(n_clusters=K, random_state=int(p["random_seed"]), n_init=10)
    labels = kmeans.fit_predict(coords)
    depot_coords = kmeans.cluster_centers_

    # Dynamic boundary adjustment with multi-objective demand checking
    labels = adjust_boundaries(K, labels, coords, depot_coords, customers["DEMAND"].values)

    depots = pd.DataFrame(depot_coords, columns=["XCOORD", "YCOORD"])
    depots["DEMAND"] = 0
    depots["READY_TIME"] = 0
    depots["DUE_DATE"] = 10000.0
    depots["SERVICE_TIME"] = 0

    customers_reset = customers.reset_index(drop=True)
    assignments: Dict[int, List[int]] = {i: [] for i in range(K)}
    for i, cust in customers_reset.iterrows():
        assignments[int(labels[i])].append(int(i))

    results: List[Dict[str, Any]] = []
    convergence_per_depot: List[List[float]] = []
    timeline_playback: List[List[Dict[str, Any]]] = [] # iterations -> depots -> {best_route, pheromones}
    
    total_distance = 0.0
    total_vehicles = 0
    iterations = max(1, int(p["iterations"]))
    
    # State hash for incremental caching
    hash_params = {k: p[k] for k in ["num_depots", "max_customers", "scenario"]}
    cache_key = hashlib.md5(json.dumps(hash_params, sort_keys=True).encode()).hexdigest()
    
    using_warm_start = False
    new_pheromones_for_cache = []

    for depot_id in range(K):
        cust_list = assignments[depot_id]
        if len(cust_list) == 0:
            convergence_per_depot.append([])
            new_pheromones_for_cache.append(None)
            continue

        cust_df = customers_reset.iloc[cust_list].copy().reset_index(drop=True)
        distance_matrix, demand_array, ready_time, due_time, service_time = calculate_distance_matrix(
            depots.iloc[depot_id], cust_df
        )

        # Sync Physics with Urban Geography for Showcase
        if p.get("showcase", False):
            # 1.4x circuity accounts for Pittsburgh's non-Euclidean street grid & elevation
            distance_matrix *= 1.4

        colony = ACO_Colony(
            distance_matrix=distance_matrix,
            demand_array=demand_array,
            ready_time=ready_time,
            due_time=due_time,
            service_time=service_time,
            vehicle_capacity=int(p["vehicle_capacity"]),
            num_ants=int(p["num_ants"]),
            alpha=float(p["alpha"]),
            beta=float(p["beta"]),
            evaporation=float(p["evaporation"]),
            vehicle_penalty=float(p["vehicle_penalty"]),
            use_local_search=p["use_local_search"]
        )

        # Warm start
        if p["incremental"] and cache_key in PHEROMONE_CACHE:
            if PHEROMONE_CACHE[cache_key][depot_id] is not None:
                # Need to resize/truncate safely if customer count changed perfectly, but exact cache handling for different grid sizes is tricky.
                # Assuming same nodes for simplicity of demo, otherwise let it pass.
                cached_p = PHEROMONE_CACHE[cache_key][depot_id]
                if cached_p.shape == colony.pheromone_matrix.shape:
                    colony.pheromone_matrix = cached_p.copy()
                    using_warm_start = True

        best_ant = None
        best_distance = float("inf")
        conv: List[float] = []

        depot_timeline = []

        for it in range(iterations):
            ant, entropy, phase, savings, swarm = colony.run_one_iteration()
            
            # Showcase data: capture explorer and risky paths for animation
            swarm_json = {}
            if p.get("showcase", False):
                for key, sample_ant in swarm.items():
                    swarm_json[key] = {
                        "route": [int(x) for x in sample_ant.route],
                        "rationales": list(sample_ant.route_rationales)
                    }

            # Aha moment trap demonstration
            if p.get("local_optimum_trap", False) and it == 8:
                # Force maximum pheromone lockdown everywhere and blind the heuristic
                colony.pheromone_matrix.fill(1000.0)
                colony.evaporation = 0.0
                colony.beta = 0.0 # Ants can no longer 'see' distance!
                phase = "Trapped in Local Optimum"

            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_ant = ant.clone()
            conv.append(float(best_distance))

            if best_ant:
                # Split the combined ant route into individual vehicle sub-routes
                fleet_routes = []
                current_v = [0]
                for node in best_ant.route[1:]:
                    if node == 0:
                        current_v.append(0)
                        fleet_routes.append(current_v)
                        current_v = [0]
                    else:
                        current_v.append(int(node))
                if len(current_v) > 1:
                    current_v.append(0)
                    fleet_routes.append(current_v)

                phero_copy = colony.pheromone_matrix.copy()
                phero_json = []
                threshold = phero_copy.mean() + phero_copy.std()
                for i_n in range(len(phero_copy)):
                    for j_n in range(i_n + 1, len(phero_copy)):
                        val = phero_copy[i_n][j_n]
                        if val > threshold:
                            phero_json.append({"edge": [i_n, j_n], "intensity": float(val)})
                            
                from collections import Counter
                r_counts = Counter(best_ant.route_rationales)
                top_rationale = r_counts.most_common(1)[0][0] if r_counts else "Stable Trajectory"

                depot_timeline.append({
                    "iteration": it,
                    "distance": float(best_distance) if best_distance != float('inf') else -1.0,
                    "route": [int(x) for x in best_ant.route],
                    "route_rationales": list(best_ant.route_rationales),
                    "first_decision_probs": getattr(best_ant, "first_decision_probs", None),
                    "swarm_samples": swarm_json,
                    "fleet": fleet_routes,
                    "pheromones": phero_json[:50],
                    "entropy": float(entropy) if entropy != float('inf') else 0.0,
                    "phase": phase,
                    "dominant_rationale": top_rationale,
                    "hybrid_savings": float(savings) if savings != float('inf') else 0.0
                })

        convergence_per_depot.append(conv)
        total_distance += float(best_distance)
        new_pheromones_for_cache.append(colony.pheromone_matrix.copy())

        if best_ant is None:
            continue

        vehicles: List[List[int]] = []
        current_route = [0]
        for node in best_ant.route[1:]:
            if node == 0:
                current_route.append(0)
                vehicles.append(current_route)
                current_route = [0]
            else:
                current_route.append(int(node))

        if len(current_route) > 1:
            current_route.append(0)
            vehicles.append(current_route)

        total_vehicles += len(vehicles)

        route_coords: List[Dict[str, Any]] = []
        for vehicle in vehicles:
            x_path: List[float] = []
            y_path: List[float] = []
            lat_path: List[float] = []
            lng_path: List[float] = []
            for node in vehicle:
                if node == 0:
                    dx = float(depots.iloc[depot_id]["XCOORD"])
                    dy = float(depots.iloc[depot_id]["YCOORD"])
                else:
                    row = cust_df.iloc[node - 1]
                    dx = float(row["XCOORD"])
                    dy = float(row["YCOORD"])
                lat, lng = project_xy_to_geo(dx, dy, xmin, xmax, ymin, ymax, bbox)
                x_path.append(dx)
                y_path.append(dy)
                lat_path.append(lat)
                lng_path.append(lng)

            route_coords.append({"x": x_path, "y": y_path, "lat": lat_path, "lng": lng_path})

        dlat, dlng = project_xy_to_geo(
            float(depots.iloc[depot_id]["XCOORD"]),
            float(depots.iloc[depot_id]["YCOORD"]),
            xmin, xmax, ymin, ymax, bbox,
        )

        results.append({
            "depot_id": int(depot_id),
            "depot": {"x": float(depots.iloc[depot_id]["XCOORD"]), "y": float(depots.iloc[depot_id]["YCOORD"]), "lat": dlat, "lng": dlng},
            "routes": route_coords,
            "best_distance": float(best_distance),
            "timeline": depot_timeline,
            "customer_mapping": cust_list # Local node ID -> Global Customer index
        })

    # Cache update
    PHEROMONE_CACHE[cache_key] = new_pheromones_for_cache

    customers_geo: List[Dict[str, Any]] = []
    for _, row in customers_reset.iterrows():
        lat, lng = project_xy_to_geo(float(row["XCOORD"]), float(row["YCOORD"]), xmin, xmax, ymin, ymax, bbox)
        customers_geo.append({"id": int(row["CUST_NO"]), "lat": lat, "lng": lng, "demand": float(row["DEMAND"])})

    max_conv = max((len(c) for c in convergence_per_depot), default=0)
    combined_convergence: List[float] = []
    for t in range(max_conv):
        s = sum(series[t] for series in convergence_per_depot if len(series) > t)
        combined_convergence.append(s)

    # Sync Technical Rationale with Physics Type
    phys_desc = "Road-Aware Manhattan (1.4x circuity)" if p.get("showcase", False) else "Euclidean"
    return {
        "status": "ok",
        "scenario": {"id": scenario_id, "bbox": bbox},
        "params": {k: p[k] for k in DEFAULT_PARAMS.keys()},
        "stats": {
            "total_distance": total_distance,
            "total_vehicles": total_vehicles,
            "iterations": iterations,
            "customer_count": len(customers_reset),
            "warm_started": using_warm_start
        },
        "convergence": {"per_depot": convergence_per_depot, "combined": combined_convergence},
        "customers_geo": customers_geo,
        "results": results,
        "technical_rationale": (
            f"Showcase Mode: {phys_desc} physics synced with Heartland road-snapping (Kansas City)."
        ) if p.get("showcase", False) else (
            f"MDVRP Simulation Active. Solomon Node 0 splintered into {K} virtual hubs via K-Means. "
            f"Synthetic coordinates linearly projected to {scenario_id.upper()} land-locked grid (with 15% margin)."
        )
    }
