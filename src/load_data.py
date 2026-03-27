import pandas as pd
import numpy as np
import math
import os

def load_solomon_data(filepath):
    """
    Parses Solomon dataset safely (no column loss).
    """

    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return None, None

    # Read raw (no column assumptions)
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=9,
        header=None
    )

    # Keep only first 7 columns (important!)
    df = df.iloc[:, :7]

    # Assign correct column names
    df.columns = [
        'CUST_NO',
        'XCOORD',
        'YCOORD',
        'DEMAND',
        'READY_TIME',
        'DUE_DATE',
        'SERVICE_TIME'
    ]

    # Extract depot and customers
    depot = df.iloc[0]
    customers = df.iloc[1:].copy()

    return depot, customers


def calculate_distance_matrix(depot, customers):
    """
    Builds distance + demand + time arrays
    """

    all_nodes = [depot] + [row for _, row in customers.iterrows()]
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