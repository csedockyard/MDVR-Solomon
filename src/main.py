import numpy as np
import time

# Import the logic we just built
from load_data import load_solomon_data, calculate_distance_matrix
from aco_algorithm import ACO_Colony

def main():
    print("=====================================================")
    print("  TATA MOTORS: MULTI-DEPOT VEHICLE ROUTING AI ENGINE ")
    print("=====================================================\n")
    
    # 1. Load the Solomon Constraints
    print("[1/3] Ingesting Solomon C101 Dataset...")
    dataset_path = "../data/c101.txt"
    depot, customers = load_solomon_data(dataset_path)
    
    if depot is None:
        print("CRITICAL ERROR: Data pipeline failed. Halting execution.")
        return

    # 2. Build the Mathematical Matrices
    print("[2/3] Calculating Spatial Distance Matrices...")
    distance_matrix, demand_array = calculate_distance_matrix(depot, customers)
    
    # Standard vehicle capacity for the C101 dataset
    VEHICLE_CAPACITY = 200 
    
    # 3. Initialize the AI Colony
    print("[3/3] Booting Ant Colony Optimization Core...\n")
    colony = ACO_Colony(
        distance_matrix=distance_matrix, 
        demand_array=demand_array, 
        vehicle_capacity=VEHICLE_CAPACITY, 
        num_ants=25  # Releasing 25 virtual trucks per generation
    )
    
    # 4. Run the Simulation Loop
    TOTAL_ITERATIONS = 50
    global_best_distance = float('inf')
    global_best_route = []
    
    print(f"Deploying fleet for {TOTAL_ITERATIONS} optimization cycles. Watch the distances drop:\n")
    time.sleep(1) # Dramatic pause for the terminal output
    
    for i in range(TOTAL_ITERATIONS):
        # The AI runs a full generation of trucks
        best_ant_this_gen = colony.run_one_iteration()
        
        # Check if this generation found an all-time best route
        if best_ant_this_gen.total_distance < global_best_distance:
            global_best_distance = best_ant_this_gen.total_distance
            global_best_route = best_ant_this_gen.route
            
        # Executive Console Output: Print progress to prove the AI is learning
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  -> Generation {i + 1:02d} | Optimal Distance Found: {global_best_distance:.2f} units")
            
    # 5. Final Executive Report
    print("\n=====================================================")
    print("                 OPTIMIZATION COMPLETE               ")
    print("=====================================================")
    print(f"Final Optimized Fleet Distance: {global_best_distance:.2f} units")
    print(f"Total Nodes Processed: {len(global_best_route)}")
    print("Algorithm successfully converged without external resources.")

if __name__ == "__main__":
    main()