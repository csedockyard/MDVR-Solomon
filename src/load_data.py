import pandas as pd
import numpy as np
import math
import os

def load_solomon_data(filepath):
    """
    Parses the standard Solomon dataset file (e.g., c101.txt).
    """
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return None, None

    column_names = ['CUST_NO', 'XCOORD', 'YCOORD', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME']
    
    # Read the text file, skipping the header lines
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=9, header=None, names=column_names)
    
    # Node 0 is the Tata Depot
    depot = df.iloc[0]
    # The rest are dealerships
    customers = df.iloc[1:]
    
    return depot, customers

def calculate_distance_matrix(depot, customers):
    """
    Translates physical X/Y coordinates into the mathematical distance matrix the AI needs.
    """
    # Combine depot and customers into one list to create a unified matrix
    all_nodes = [depot] + [row for _, row in customers.iterrows()]
    num_nodes = len(all_nodes)
    
    distance_matrix = np.zeros((num_nodes, num_nodes))
    demand_array = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        demand_array[i] = all_nodes[i]['DEMAND']
        
        for j in range(num_nodes):
            if i != j:
                x1, y1 = all_nodes[i]['XCOORD'], all_nodes[i]['YCOORD']
                x2, y2 = all_nodes[j]['XCOORD'], all_nodes[j]['YCOORD']
                
                # Calculate straight-line distance between nodes
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distance_matrix[i][j] = dist
                
    return distance_matrix, demand_array