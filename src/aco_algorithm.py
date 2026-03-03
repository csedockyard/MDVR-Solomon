import numpy as np

class Ant:
    """
    Represents a single Tata truck finding a route.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_load = 0
        self.route = []          # List of customer IDs visited
        self.total_distance = 0.0
        self.current_node = 0    # Always starts at 0 (The Depot)

    def visit_node(self, node_id, demand, distance):
        """
        Updates the truck's state. If it visits the Depot (0), it unloads.
        """
        self.route.append(node_id)
        self.total_distance += distance
        self.current_node = node_id
        
        if node_id == 0:
            self.current_load = 0  # Reload the truck at the depot
        else:
            self.current_load += demand

    def can_visit(self, demand):
        """
        Checks if the truck has enough space for the next dealership's demand.
        """
        return (self.current_load + demand) <= self.capacity

class ACO_Colony:
    """
    The central AI that manages the fleet and the pheromone trails.
    """
    def __init__(self, distance_matrix, demand_array, vehicle_capacity, num_ants=20):
        self.distance_matrix = distance_matrix
        self.demand_array = demand_array
        self.vehicle_capacity = vehicle_capacity
        self.num_ants = num_ants
        
        self.num_nodes = len(distance_matrix)
        
        # Initialize pheromones: A matrix where every path starts with a base scent of 1.0
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))
        
        # ACO Hyperparameters (The knobs you will tune to get the best result)
        self.alpha = 1.0       # How much the ant cares about the pheromone trail
        self.beta = 2.0        # How much the ant cares about the physical distance (closer is better)
        self.evaporation = 0.5 # How fast bad trails disappear (0.0 to 1.0)

    def run_one_iteration(self):
        """
        Releases the fleet, builds the routes, and updates the map's intelligence.
        """
        # 1. Spawn a new generation of ants
        ants = [Ant(self.vehicle_capacity) for _ in range(self.num_ants)]
        
        # 2. Tour Construction: Let the ants drive until all dealerships are visited
        for ant in ants:
            # Keep driving while there are still unvisited dealerships
            # (We check against num_nodes - 1 because Node 0 is the depot)
            while len(set(ant.route) - {0}) < (self.num_nodes - 1):
                
                next_node = self.choose_next_node(ant)
                distance = self.distance_matrix[ant.current_node][next_node]
                demand = self.demand_array[next_node]
                
                ant.visit_node(next_node, demand, distance)
                
            # Once all dealerships are visited, the truck MUST return to the depot
            if ant.current_node != 0:
                final_distance = self.distance_matrix[ant.current_node][0]
                ant.visit_node(0, 0, final_distance)
                
        # 3. Pheromone Update: The AI learns from this generation
        self.update_pheromones(ants)
        
        # 4. Find the best ant of this generation to report to the executives
        best_ant = min(ants, key=lambda a: a.total_distance)
        return best_ant

    def update_pheromones(self, ants):
        """
        Evaporates old trails and reinforces the best new routes.
        """
        # Evaporation: Multiply all existing scents by a fraction to fade them
        self.pheromone_matrix *= (1.0 - self.evaporation)
        
        # Deposit: Add new scent based on how good the ant's total route was
        for ant in ants:
            # The shorter the total distance, the stronger the pheromone dropped
            pheromone_to_drop = 100.0 / ant.total_distance 
            
            # Drop this scent on every road the ant traveled
            for i in range(len(ant.route) - 1):
                from_node = ant.route[i]
                to_node = ant.route[i+1]
                
                self.pheromone_matrix[from_node][to_node] += pheromone_to_drop
                self.pheromone_matrix[to_node][from_node] += pheromone_to_drop # Assuming roads are two-way

    def choose_next_node(self, ant):
        """
        Determines the next dealership the truck should visit using ACO probability.
        """
        current = ant.current_node
        
        # 1. Filter: Where can the truck legally go?
        # Must be unvisited AND the truck must have enough empty space for the demand
        # Start the range at 1 to exclude the depot from the dealership list
        unvisited = [node for node in range(1, self.num_nodes) if node not in ant.route]
        feasible_nodes = [node for node in unvisited if ant.can_visit(self.demand_array[node])]
        
        # If the truck is full or all dealerships are visited, it must return to Depot (Node 0)
        if not feasible_nodes:
            return 0
            
        # 2. Calculate the Desirability of each legal move
        probabilities = np.zeros(len(feasible_nodes))
        
        for index, next_node in enumerate(feasible_nodes):
            # Grab the historical scent (Tau)
            pheromone = self.pheromone_matrix[current][next_node]
            
            # Grab the physical distance. Avoid division by zero.
            distance = self.distance_matrix[current][next_node]
            visibility = 1.0 / distance if distance > 0 else 0.0001
            
            # The core ACO Math
            probabilities[index] = (pheromone ** self.alpha) * (visibility ** self.beta)
            
        # 3. Convert the raw scores into actual percentages (0.0 to 1.0)
        total_prob = np.sum(probabilities)
        if total_prob == 0:
            # Fallback in case of extreme parameter tuning: pick randomly
            probabilities = np.ones(len(feasible_nodes)) / len(feasible_nodes)
        else:
            probabilities = probabilities / total_prob
        
        # 4. Spin the roulette wheel! 
        # The AI chooses the next node based on the weighted probabilities.
        chosen_node = np.random.choice(feasible_nodes, p=probabilities)
        
        return chosen_node