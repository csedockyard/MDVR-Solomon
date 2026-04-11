import pandas as pd
import numpy as np
import math
import os

def load_solomon_data(filepath):
    import pandas as pd

    with open(filepath, "r") as f:
        lines = f.readlines()

    # -------- FIND CUSTOMER SECTION --------
    start = 0
    for i, line in enumerate(lines):
        if "CUSTOMER" in line:
            start = i + 2   # skip header line
            break

    # -------- READ DATA --------
    data = []
    for line in lines[start:]:
        if line.strip() == "":
            continue

        parts = line.split()

        if len(parts) < 7:
            continue

        data.append([
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6])
        ])

    df = pd.DataFrame(data, columns=[
        'CUST_NO',
        'XCOORD',
        'YCOORD',
        'DEMAND',
        'READY_TIME',
        'DUE_DATE',
        'SERVICE_TIME'
    ])

    depot = df.iloc[0]
    customers = df.iloc[1:].copy()

    return depot, customers

def calculate_distance_matrix(depot, customers):
    """
    Builds distance + demand + time arrays
    """

    all_nodes = [depot.to_dict()] + [row.to_dict() for _, row in customers.iterrows()]
    num_nodes = len(all_nodes)

    distance_matrix = np.zeros((num_nodes, num_nodes))
    demand_array = np.zeros(num_nodes)

    # TIME ARRAYS
    ready_time = np.zeros(num_nodes)
    due_time = np.zeros(num_nodes)
    service_time = np.zeros(num_nodes)

    for i in range(num_nodes):
        node = all_nodes[i]

        demand_array[i] = node['DEMAND']
        ready_time[i] = node['READY_TIME']
        due_time[i] = node['DUE_DATE']
        service_time[i] = node['SERVICE_TIME']

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = all_nodes[i]['XCOORD'], all_nodes[i]['YCOORD']
                x2, y2 = all_nodes[j]['XCOORD'], all_nodes[j]['YCOORD']
                distance_matrix[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance_matrix, demand_array, ready_time, due_time, service_time