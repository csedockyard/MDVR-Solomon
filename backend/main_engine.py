import os
import sys
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
    "nyc": {"south": 40.698, "west": -74.028, "north": 40.822, "east": -73.918},
    "london": {"south": 51.472, "west": -0.172, "north": 51.532, "east": -0.048},
    "bangalore": {"south": 12.898, "west": 77.548, "north": 13.048, "east": 77.688},
    "sf": {"south": 37.738, "west": -122.448, "north": 37.802, "east": -122.378},
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "num_depots": 3,
    "vehicle_capacity": 200,
    "num_ants": 20,
    "alpha": 1.0,
    "beta": 2.0,
    "evaporation": 0.5,
    "iterations": 30,
    "vehicle_penalty": 100.0,
    "scenario": "nyc",
    "max_customers": 100,
    "synthetic_extra": 0,
    "random_seed": 42,
    "dataset_path": None,
}


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
    x: float,
    y: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    bbox: Dict[str, float],
) -> Tuple[float, float]:
    """Affine map from problem coordinates to WGS84 inside bbox."""
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    u = (float(x) - xmin) / dx
    v = (float(y) - ymin) / dy
    lat = bbox["south"] + u * (bbox["north"] - bbox["south"])
    lng = bbox["west"] + v * (bbox["east"] - bbox["west"])
    return lat, lng


def _augment_synthetic(customers: pd.DataFrame, n_extra: int, rng: np.random.Generator) -> pd.DataFrame:
    if n_extra <= 0:
        return customers
    xmin, xmax, ymin, ymax = _xy_bounds(customers)
    rows = []
    base_id = int(customers["CUST_NO"].max()) + 1
    for i in range(n_extra):
        rows.append(
            {
                "CUST_NO": base_id + i,
                "XCOORD": float(rng.uniform(xmin, xmax)),
                "YCOORD": float(rng.uniform(ymin, ymax)),
                "DEMAND": float(rng.integers(10, 41)),
                "READY_TIME": 0.0,
                "DUE_DATE": 1e6,
                "SERVICE_TIME": 90.0,
            }
        )
    extra = pd.DataFrame(rows)
    return pd.concat([customers, extra], ignore_index=True)


def run_full_optimization(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = _merge_params(params)
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

    K = max(1, int(p["num_depots"]))
    K = min(K, len(customers))

    coords = customers[["XCOORD", "YCOORD"]].values
    kmeans = KMeans(n_clusters=K, random_state=int(p["random_seed"]), n_init=10)
    labels = kmeans.fit_predict(coords)

    depot_coords = kmeans.cluster_centers_
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
    total_distance = 0.0
    total_vehicles = 0

    iterations = max(1, int(p["iterations"]))

    for depot_id, cust_list in assignments.items():
        if len(cust_list) == 0:
            convergence_per_depot.append([])
            continue

        cust_df = customers_reset.iloc[cust_list].copy().reset_index(drop=True)

        distance_matrix, demand_array, ready_time, due_time, service_time = calculate_distance_matrix(
            depots.iloc[depot_id],
            cust_df,
        )

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
        )

        best_ant = None
        best_distance = float("inf")
        conv: List[float] = []

        for _ in range(iterations):
            ant = colony.run_one_iteration()
            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_ant = ant
            conv.append(float(best_distance))

        convergence_per_depot.append(conv)
        total_distance += float(best_distance)

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
            xmin,
            xmax,
            ymin,
            ymax,
            bbox,
        )

        results.append(
            {
                "depot_id": int(depot_id),
                "depot": {
                    "x": float(depots.iloc[depot_id]["XCOORD"]),
                    "y": float(depots.iloc[depot_id]["YCOORD"]),
                    "lat": dlat,
                    "lng": dlng,
                },
                "routes": route_coords,
                "best_distance": float(best_distance),
            }
        )

    customers_geo: List[Dict[str, Any]] = []
    for _, row in customers_reset.iterrows():
        lat, lng = project_xy_to_geo(
            float(row["XCOORD"]),
            float(row["YCOORD"]),
            xmin,
            xmax,
            ymin,
            ymax,
            bbox,
        )
        customers_geo.append(
            {
                "id": int(row["CUST_NO"]),
                "lat": lat,
                "lng": lng,
                "demand": float(row["DEMAND"]),
            }
        )

    max_conv = max((len(c) for c in convergence_per_depot), default=0)
    combined_convergence: List[float] = []
    for t in range(max_conv):
        s = 0.0
        for series in convergence_per_depot:
            if len(series) > t:
                s += series[t]
        combined_convergence.append(s)

    return {
        "status": "ok",
        "scenario": {"id": scenario_id, "bbox": bbox},
        "params": {k: p[k] for k in DEFAULT_PARAMS.keys()},
        "stats": {
            "total_distance": total_distance,
            "total_vehicles": total_vehicles,
            "iterations": iterations,
            "customer_count": len(customers_reset),
        },
        "convergence": {
            "per_depot": convergence_per_depot,
            "combined": combined_convergence,
        },
        "customers_geo": customers_geo,
        "results": results,
    }
