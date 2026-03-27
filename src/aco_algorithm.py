import numpy as np


# -------------------------------
# ANT CLASS (Agent)
# -------------------------------
class Ant:
    def __init__(self, vehicle_capacity, ready_time, due_time, service_time):
        self.capacity = vehicle_capacity
        self.current_time = 0
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.current_load = 0
        self.vehicle_capacity = vehicle_capacity
        self.route = [0]
        self.total_distance = 0
        self.current_node = 0
        self.reset()

    def reset(self):
        self.route = [0]  # start from depot
        self.current_node = 0
        self.current_load = 0
        self.total_distance = 0
        

    def visit_node(self, node_id, demand, distance):
    
        # 1. Travel time (distance = time assumption)
        travel_time = distance

        # 2. Compute arrival
        arrival_time = self.current_time + travel_time

        # 3. Apply waiting if early
        if arrival_time < self.ready_time[node_id]:
            arrival_time = self.ready_time[node_id]

        # 4. Update time AFTER service
        self.current_time = arrival_time + self.service_time[node_id]

        # 5. Standard updates
        self.route.append(node_id)
        self.total_distance += distance
        self.current_node = node_id

        if node_id == 0:
            # Return to depot → reset load + time
            self.current_load = 0
            self.current_time = 0
        else:
            self.current_load += demand

    def can_visit(self, demand):
            return (self.current_load + demand) <= self.capacity


# -------------------------------
# ACO COLONY
# -------------------------------
class ACO_Colony:
    def __init__(self, distance_matrix, demand_array,
             ready_time, due_time, service_time,
             vehicle_capacity,
             num_ants=20, alpha=1.0, beta=2.0, evaporation=0.5):

        self.distance_matrix = distance_matrix
        self.demand_array = demand_array
        self.vehicle_capacity = vehicle_capacity
        self.num_ants = num_ants

        self.num_nodes = len(distance_matrix)

        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))

        # Tunable parameters
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation

        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

    # -------------------------------
    def run_one_iteration(self):
        ants = [
    Ant(self.vehicle_capacity,
        self.ready_time,
        self.due_time,
        self.service_time)
    for _ in range(self.num_ants)
]

        for ant in ants:

            while len(set(ant.route) - {0}) < (self.num_nodes - 1):

                next_node = self.choose_next_node(ant)
                distance = self.distance_matrix[ant.current_node][next_node]
                demand = self.demand_array[next_node]

                ant.visit_node(next_node, demand, distance)

            # return to depot
            if ant.current_node != 0:
                final_distance = self.distance_matrix[ant.current_node][0]
                ant.visit_node(0, 0, final_distance)

        self.update_pheromones(ants)

        best_ant = min(ants, key=lambda a: a.total_distance)
        return best_ant

    # -------------------------------
    def update_pheromones(self, ants):

        self.pheromone_matrix *= (1.0 - self.evaporation)

        for ant in ants:
            pheromone_to_drop = 100.0 / ant.total_distance

            for i in range(len(ant.route) - 1):
                a = ant.route[i]
                b = ant.route[i + 1]

                self.pheromone_matrix[a][b] += pheromone_to_drop
                self.pheromone_matrix[b][a] += pheromone_to_drop

    # -------------------------------
    def choose_next_node(self, ant):

        current = ant.current_node

        unvisited = [n for n in range(1, self.num_nodes) if n not in ant.route]
        feasible = [n for n in unvisited if ant.can_visit(self.demand_array[n])]

        if not feasible:
            return 0

        probs = np.zeros(len(feasible))

        for i, nxt in enumerate(feasible):
            pheromone = self.pheromone_matrix[current][nxt]
            distance = self.distance_matrix[current][nxt]

            visibility = 1.0 / distance if distance > 0 else 1e-4

            probs[i] = (pheromone ** self.alpha) * (visibility ** self.beta)

        total = np.sum(probs)

        if total == 0:
            probs = np.ones(len(feasible)) / len(feasible)
        else:
            probs /= total

        return np.random.choice(feasible, p=probs)