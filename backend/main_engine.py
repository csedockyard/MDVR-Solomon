import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys
import os

sys.path.append(os.path.abspath("../src"))

from load_data import load_solomon_data, calculate_distance_matrix
from aco_algorithm import ACO_Colony

import load_data
print(load_data.__file__)

def run_full_optimization(params=None):

    # -------------------------------
    # Load dataset
    # -------------------------------
    dataset_path = "../data/c101.txt"
    depot, customers = load_solomon_data(dataset_path)

    if depot is None:
        return {"error": "Dataset not loaded"}

    # -------------------------------
    # K-Means depots
    # -------------------------------
    K = 3
    coords = customers[['XCOORD', 'YCOORD']].values

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    depot_coords = kmeans.cluster_centers_
    depots = pd.DataFrame(depot_coords, columns=['XCOORD', 'YCOORD'])

    depots['DEMAND'] = 0

    # 🔥 ADD THESE (CRITICAL)
    depots['READY_TIME'] = 0
    depots['DUE_DATE'] = 10000   # large window
    depots['SERVICE_TIME'] = 0

    # -------------------------------
    # Assign customers
    # -------------------------------
    customers_reset = customers.reset_index(drop=True)
    assignments = {i: [] for i in range(K)}

    for i, cust in customers_reset.iterrows():
        assignments[labels[i]].append(i)

    # -------------------------------
    # Run ACO per depot
    # -------------------------------
    results = []

    for depot_id, cust_list in assignments.items():

        if len(cust_list) == 0:
            continue

        cust_df = customers_reset.iloc[cust_list].copy()

        distance_matrix, demand_array, ready_time, due_time, service_time = calculate_distance_matrix(
            depots.iloc[depot_id],
            cust_df
        )

        colony = ACO_Colony(
            distance_matrix=distance_matrix,
            demand_array=demand_array,
            ready_time=ready_time,
            due_time=due_time,
            service_time=service_time,
            vehicle_capacity=200,
            num_ants=20
        )

        best_ant = None
        best_distance = float('inf')

        for _ in range(30):
            ant = colony.run_one_iteration()
            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_ant = ant

        # -------------------------------
        # Convert route → coordinates
        # -------------------------------
        routes = []
        vehicles = []
        current_route = [0]

        for node in best_ant.route[1:]:  # skip first 0

            if node == 0:
                current_route.append(0)
                vehicles.append(current_route)
                current_route = [0]  # ALWAYS restart from depot
            else:
                current_route.append(int(node))

# handle last route
        if len(current_route) > 1:
            current_route.append(0)
            vehicles.append(current_route)
        # Build coordinate routes
        route_coords = []

        for vehicle in vehicles:
            x, y = [], []

            for node in vehicle:
                if node == 0:
                    x.append(float(depots.iloc[depot_id]['XCOORD']))
                    y.append(float(depots.iloc[depot_id]['YCOORD']))
                else:
                    cust = cust_df.iloc[node - 1]
                    x.append(float(cust['XCOORD']))
                    y.append(float(cust['YCOORD']))

            route_coords.append({"x": x, "y": y})

    
        print("VEHICLES:", vehicles)



        results.append({
            "depot": {
                "x": float(depots.iloc[depot_id]['XCOORD']),
                "y": float(depots.iloc[depot_id]['YCOORD'])
            },
            "routes": route_coords
        })

    return {
        "status": "ok",
        "results": results
    }