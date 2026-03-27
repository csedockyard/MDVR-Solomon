import numpy as np
import time
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

from load_data import load_solomon_data, calculate_distance_matrix
from aco_algorithm import ACO_Colony

np.random.seed(42)


# -------------------------------
# Split route into vehicles
# -------------------------------
def split_routes(route):
    vehicles = []
    current = []

    for node in route:
        current.append(node)

        if node == 0 and len(current) > 1:
            vehicles.append(current)
            current = [0]

    return vehicles


# -------------------------------
# ACO Runner
# -------------------------------
def run_aco_with_params(distance_matrix, demand_array,
                        ready_time, due_time, service_time,
                        params):

    colony = ACO_Colony(
    distance_matrix=distance_matrix,
    demand_array=demand_array,
    ready_time=ready_time,
    due_time=due_time,
    service_time=service_time,
    vehicle_capacity=200,
    num_ants=params["ants"],
    alpha=params["alpha"],
    beta=params["beta"],
    evaporation=params["evap"]
)

    best_distance = float('inf')
    best_route = []
    history = []

    for _ in range(30):
        ant = colony.run_one_iteration()

        if ant.total_distance < best_distance:
            best_distance = ant.total_distance
            best_route = ant.route

        history.append(best_distance)

    return best_distance, best_route, history


# -------------------------------
# Adaptive Tuning
# -------------------------------
def adaptive_tuning(distance_matrix, demand_array,
                    ready_time, due_time, service_time,
                    iterations=10):

    current = {"alpha": 1.0, "beta": 2.0, "evap": 0.5, "ants": 25}

    best_dist, best_route, best_history = run_aco_with_params(
        distance_matrix, demand_array,
        ready_time, due_time, service_time,
        current
    )

    for _ in range(iterations):

        candidate = {
            "alpha": np.clip(current["alpha"] + np.random.uniform(-0.5, 0.5), 0.5, 3),
            "beta": np.clip(current["beta"] + np.random.uniform(-0.5, 0.5), 1, 5),
            "evap": np.clip(current["evap"] + np.random.uniform(-0.1, 0.1), 0.1, 0.9),
            "ants": int(np.clip(current["ants"] + np.random.randint(-5, 6), 10, 50))
        }

        dist, route, history = run_aco_with_params(
            distance_matrix, demand_array,
            ready_time, due_time, service_time,
            candidate
        )

        if dist < best_dist:
            best_dist = dist
            current = candidate
            best_route = route
            best_history = history

    return best_dist, current, best_route, best_history


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("=====================================================")
    print("  TATA MOTORS: MULTI-DEPOT VEHICLE ROUTING AI ENGINE ")
    print("=====================================================\n")

    print("[1/3] Ingesting Solomon C101 Dataset...")

    base_dir = Path(__file__).parent
    dataset_path = base_dir.parent / "data" / "c101.txt"

    depot, customers = load_solomon_data(dataset_path)

    if depot is None:
        return

    # IMPORTANT: preserve ALL columns
    customers = customers.copy()
    customers_reset = customers.reset_index(drop=True)

    print("[2/3] Generating depots using K-Means clustering...")

    K = 3
    coords = customers_reset[['XCOORD', 'YCOORD']].values

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    depots = pd.DataFrame(kmeans.cluster_centers_, columns=['XCOORD', 'YCOORD'])

# Add ALL required fields
    depots['DEMAND'] = 0
    depots['READY_TIME'] = 0
    depots['DUE_DATE'] = 999999   # large value = always feasible
    depots['SERVICE_TIME'] = 0

    assignments = {i: [] for i in range(K)}

    for i in range(len(customers_reset)):
        assignments[labels[i]].append(i)

    print("[3/3] Running Adaptive ACO Tuning...\n")
    time.sleep(1)

    total_distance = 0
    all_results = []

    for depot_id, cust_list in assignments.items():

        if not cust_list:
            continue

        print(f"\n--- Depot {depot_id} ---")

        cust_df = customers_reset.iloc[cust_list].copy().reset_index(drop=True)

        # DEBUG (can remove later)
        print("Columns:", cust_df.columns)

        distance_matrix, demand_array, ready_time, due_time, service_time = calculate_distance_matrix(
            depots.iloc[depot_id], cust_df
        )

        best_dist, best_params, best_route, best_history = adaptive_tuning(
            distance_matrix, demand_array,
            ready_time, due_time, service_time
        )

        vehicles = split_routes(best_route)

        print(f"Best Distance: {best_dist:.2f}")
        print({
            "alpha": round(float(best_params["alpha"]), 2),
            "beta": round(float(best_params["beta"]), 2),
            "evap": round(float(best_params["evap"]), 2),
            "ants": best_params["ants"]
        })
        print(f"Vehicles used: {len(vehicles)}")

        total_distance += best_dist

        all_results.append({
            "depot": depots.iloc[depot_id],
            "customers": cust_df,
            "vehicles": vehicles
        })

    print("\n=====================================================")
    print("         MULTI-DEPOT OPTIMIZATION COMPLETE           ")
    print("=====================================================")
    print(f"Total Optimized Distance: {total_distance:.2f} units")

    print("DEBUG: Calling final visualization...")

    from visualize import plot_multi_depot_routes
    plot_multi_depot_routes(all_results)


if __name__ == "__main__":
    main()