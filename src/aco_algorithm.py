import numpy as np
import concurrent.futures
import copy
import math

def calculate_diversity_entropy(pheromone_matrix):
    """
    Calculates the Shannon Entropy of the pheromone distribution to measure global diversity.
    Low entropy = Colony has converged to a local optimum (exploitation trap).
    High entropy = System is actively exploring the combinatorial space.
    """
    total = np.sum(pheromone_matrix)
    if total <= 0: return 0.0
    probabilities = pheromone_matrix / total
    # mask out zeros to avoid log(0)
    p = probabilities[probabilities > 1e-12]
    return -np.sum(p * np.log2(p))

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
        self.vehicle_count = 1
        self.visited_customers = set()
        self.route_rationales = []
        self.reset()

    def reset(self):
        self.route = [0]
        self.current_node = 0
        self.current_load = 0
        self.total_distance = 0
        self.current_time = 0
        self.vehicle_count = 1
        self.visited_customers.clear()
        self.route_rationales.clear()

    def clone(self):
        new_ant = Ant(self.capacity, self.ready_time, self.due_time, self.service_time)
        new_ant.route = list(self.route)
        new_ant.total_distance = self.total_distance
        new_ant.current_node = self.current_node
        new_ant.vehicle_count = self.vehicle_count
        new_ant.visited_customers = set(self.visited_customers)
        new_ant.current_time = self.current_time
        new_ant.current_load = self.current_load
        new_ant.route_rationales = list(self.route_rationales)
        return new_ant

    def visit_node(self, node_id, demand, distance, rationale="Start"):
        travel_time = distance
        arrival_time = self.current_time + travel_time

        if arrival_time < self.ready_time[node_id]:
            arrival_time = self.ready_time[node_id]

        self.current_time = arrival_time + self.service_time[node_id]
        self.route.append(node_id)
        self.route_rationales.append(rationale)
        self.total_distance += distance
        self.current_node = node_id

        if node_id == 0:
            self.current_load = 0
            self.current_time = 0
            self.vehicle_count += 1
        else:
            self.current_load += demand
            self.visited_customers.add(int(node_id))

    def can_visit(self, demand):
        return (self.current_load + demand) <= self.capacity


# -------------------------------
# ACO COLONY
# -------------------------------
class ACO_Colony:
    def __init__(self, distance_matrix, demand_array,
             ready_time, due_time, service_time,
             vehicle_capacity,
             num_ants=20, alpha=1.0, beta=2.0, evaporation=0.5,
             vehicle_penalty=100.0, use_local_search=True, use_multiprocessing=False):

        self.distance_matrix = distance_matrix
        self.demand_array = demand_array
        self.vehicle_capacity = vehicle_capacity
        self.num_ants = num_ants
        self.vehicle_penalty = float(vehicle_penalty)

        self.num_nodes = len(distance_matrix)
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))

        # Core/Tunable parameters
        self.base_alpha = alpha
        self.base_beta = beta
        self.base_evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation

        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time

        self.use_local_search = use_local_search
        
        """
        Academic Hardware Topology Toggle:
        - True Multiprocessing bypasses the Python GIL (Global Interpreter Lock), allowing true parallel CPU core utilization for 
          ant simulation batches. However, due to process-forking/pickling overheads (Inter-Process Communication IPC) over large matrices
          like the distance & pheromone layers, multi-threading in numpy-heavy code often provides superior real-world latency profiles.
        - Defaults to Threading to preserve low latency, but multiprocessing hooks exist for HPC distribution capability.
        """
        self.use_multiprocessing = use_multiprocessing

        # Adaptive parameters tracking
        self.stagnation_counter = 0
        self.best_global_distance = float('inf')
        self.current_phase = "Initialization"

    def evaluate_route(self, route_nodes):
        """Re-evaluates a list of nodes and returns (distance, vehicle_count, is_valid)"""
        ant = Ant(self.vehicle_capacity, self.ready_time, self.due_time, self.service_time)
        is_valid = True
        
        for i in range(1, len(route_nodes)):
            prev = route_nodes[i-1]
            curr = route_nodes[i]
            dist = self.distance_matrix[prev][curr]
            dem = self.demand_array[curr]
            
            # Check constraints capacity
            if curr != 0 and not ant.can_visit(dem):
                return float('inf'), 0, False
                
            # Travel
            travel_time = dist
            arrival_time = ant.current_time + travel_time
            if curr != 0 and arrival_time > self.due_time[curr]:
                return float('inf'), 0, False
                
            ant.visit_node(curr, dem, dist, "Evaluate")
            
        return ant.total_distance, ant.vehicle_count, True

    def apply_local_search(self, ant):
        """Applies 2-opt purely within individual vehicle routes to ensure capacity is conserved."""
        if not self.use_local_search:
            return ant, 0.0

        sub_routes = []
        current_sub = []
        for n in ant.route:
            current_sub.append(n)
            if n == 0 and len(current_sub) > 1:
                sub_routes.append(current_sub)
                current_sub = [0]
        if len(current_sub) > 1:
            if current_sub[-1] != 0: current_sub.append(0)
            sub_routes.append(current_sub)

        improved_total = False
        new_full_route = [0]
        total_2opt_savings = 0.0

        for sub in sub_routes:
            if len(sub) < 5: 
                new_full_route.extend(sub[1:])
                continue

            improved = True
            current_sub_route = list(sub)
            while improved:
                improved = False
                for i in range(1, len(current_sub_route) - 2):
                    for j in range(i + 1, len(current_sub_route) - 1):
                        new_sub = current_sub_route[:i] + current_sub_route[i:j+1][::-1] + current_sub_route[j+1:]
                        dist_old, _, val_old = self.evaluate_route(current_sub_route)
                        dist_new, _, val_new = self.evaluate_route(new_sub)
                        
                        if val_new and dist_new < dist_old:
                            if dist_old != float('inf'):
                                total_2opt_savings += (dist_old - dist_new)
                            current_sub_route = new_sub
                            improved = True
                            improved_total = True
                            break
                    if improved:
                        break
            new_full_route.extend(current_sub_route[1:])

        if improved_total:
            dist, v_count, valid = self.evaluate_route(new_full_route)
            if valid and dist < ant.total_distance:
                new_ant = Ant(self.vehicle_capacity, self.ready_time, self.due_time, self.service_time)
                for i in range(1, len(new_full_route)):
                    nxt = new_full_route[i]
                    d = self.distance_matrix[new_full_route[i-1]][nxt]
                    new_ant.visit_node(nxt, self.demand_array[nxt], d, "2-opt Hybrid Optimize")
                return new_ant, total_2opt_savings
        
        return ant, total_2opt_savings

    def _simulate_single_ant(self, ant_id):
        ant = Ant(
            self.vehicle_capacity,
            self.ready_time,
            self.due_time,
            self.service_time,
        )
        steps = 0
        max_steps = max(100, self.num_nodes * 8)
        stalled_at_depot = 0

        while len(ant.visited_customers) < (self.num_nodes - 1):
            next_node, rationale, prob_payload = self.choose_next_node(ant)
            
            if prob_payload and not hasattr(ant, 'first_decision_probs'):
                ant.first_decision_probs = prob_payload

            if next_node == 0 and ant.current_node == 0:
                stalled_at_depot += 1
            else:
                stalled_at_depot = 0

            if stalled_at_depot >= self.num_nodes or steps >= max_steps:
                break

            distance = self.distance_matrix[ant.current_node][next_node]
            demand = self.demand_array[next_node]

            ant.visit_node(next_node, demand, distance, rationale)
            steps += 1

        if ant.current_node != 0:
            final_distance = self.distance_matrix[ant.current_node][0]
            ant.visit_node(0, 0, final_distance, "Return to Depot")

        final_ant, savings = self.apply_local_search(ant)
        return final_ant, savings

    def run_one_iteration(self):
        ants = []
        total_iteration_savings = 0.0
        
        ExecutorCls = concurrent.futures.ProcessPoolExecutor if self.use_multiprocessing else concurrent.futures.ThreadPoolExecutor
        
        with ExecutorCls(max_workers=min(16, self.num_ants)) as executor:
            futures = [executor.submit(self._simulate_single_ant, i) for i in range(self.num_ants)]
            for f in concurrent.futures.as_completed(futures):
                ant_obj, savings = f.result()
                ants.append(ant_obj)
                total_iteration_savings += savings

        self.update_pheromones(ants)
        
        # Swarm Sampling for Showcase: Champion, Explorer, Stochastic Fail
        champion = min(
            ants,
            key=lambda a: (
                a.total_distance
                + (a.vehicle_count * self.vehicle_penalty)
                + ((self.num_nodes - 1 - len(a.visited_customers)) * 1e6)
            ),
        )

        # Explorer: Ant that made the most 'Probabilistic Diversity Jumps'
        explorer = max(ants, key=lambda a: a.route_rationales.count("Probabilistic Diversity Jump"))
        
        # Failed/Risky: A random ant that isn't the champion
        risky = next((a for a in ants if a != champion), champion)

        swarm_samples = {
            "champion": champion.clone(),
            "explorer": explorer.clone(),
            "risky": risky.clone()
        }

        entropy = calculate_diversity_entropy(self.pheromone_matrix)

        if champion.total_distance < self.best_global_distance - 1e-4:
            self.best_global_distance = champion.total_distance
            self.stagnation_counter = 0
            self.alpha = self.base_alpha
            self.evaporation = self.base_evaporation
            self.current_phase = "Converging (Exploiting Pheromones)"
        else:
            self.stagnation_counter += 1

        # Multi-Metric Intelligent Tuning System
        # We don't just trigger blindly. If entropy crashed (too ordered) AND we are stagnating, we explore.
        # If entropy is high, the system is globally searching but failing to find better paths.
        
        if self.stagnation_counter > 5:
            if entropy < 5.0: # Arbitrary threshold mapping diversity crash
                self.evaporation = min(0.9, self.evaporation + 0.15)
                self.alpha = max(0.2, self.alpha - 0.2)
                self.current_phase = "Diversity Crash Triggered -> Exploration Burst"
            else:
                self.current_phase = "Stagnation (Navigating Complex Optima)"
            
            if self.stagnation_counter > 10:
                self.stagnation_counter = 0
        
        elif self.stagnation_counter <= 2:
            self.current_phase = "Convergence Phase (Entropy Stabilizing)"

        return champion, entropy, self.current_phase, total_iteration_savings, swarm_samples

    def update_pheromones(self, ants):
        self.pheromone_matrix *= (1.0 - self.evaporation)

        for ant in ants:
            if ant.total_distance <= 0:
                continue
            pheromone_to_drop = 100.0 / ant.total_distance

            for i in range(len(ant.route) - 1):
                a = ant.route[i]
                b = ant.route[i + 1]

                self.pheromone_matrix[a][b] += pheromone_to_drop
                self.pheromone_matrix[b][a] += pheromone_to_drop

    def choose_next_node(self, ant):
        current = ant.current_node
        unvisited = [n for n in range(1, self.num_nodes) if n not in ant.visited_customers]
        if not unvisited:
            return 0, "All Visited", None

        capacity_feasible = [n for n in unvisited if ant.can_visit(self.demand_array[n])]
        if not capacity_feasible:
            return 0, "Capacity Exceeded", None

        time_feasible = []
        for n in capacity_feasible:
            travel_time = self.distance_matrix[ant.current_node][n]
            arrival_time = ant.current_time + travel_time
            if arrival_time < self.ready_time[n]:
                arrival_time = self.ready_time[n]
            if arrival_time > self.due_time[n]:
                continue
            time_feasible.append(n)

        feasible = time_feasible if time_feasible else capacity_feasible
        if current != 0:
            feasible.append(0)

        probs = np.zeros(len(feasible))
        phero_factors = np.zeros(len(feasible))
        heur_factors = np.zeros(len(feasible))

        for i, nxt in enumerate(feasible):
            pheromone = self.pheromone_matrix[current][nxt]
            phero_factors[i] = pheromone
            if nxt == 0:
                visibility = 1.0 / (1 + self.vehicle_penalty)
            else:
                distance = self.distance_matrix[current][nxt]
                visibility = 1.0 / distance if distance > 0 else 1e-4

            heur_factors[i] = visibility
            probs[i] = (pheromone ** self.alpha) * (visibility ** self.beta)

        total = np.sum(probs)
        if total == 0:
            probs = np.ones(len(feasible)) / len(feasible)
            selected_idx = np.random.choice(len(feasible))
            rationale = "Fallback Uniform Pick (Null Probs)"
        else:
            probs /= total
            selected_idx = np.random.choice(len(feasible), p=probs)
            
        # Formulate mathematical explanation
        if feasible[selected_idx] == 0:
            rationale = "Time/Route Depleted -> Returned"
        else:
            p_dominance = phero_factors[selected_idx] / (np.max(phero_factors) + 1e-9)
            h_dominance = heur_factors[selected_idx] / (np.max(heur_factors) + 1e-9)
            
            if p_dominance > 0.8 and h_dominance < 0.5:
                rationale = "Strong Pheromone Trail Followed"
            elif h_dominance > 0.8 and p_dominance < 0.5:
                rationale = "Heuristic Save (Optimized Close Node)"
            elif p_dominance > 0.7 and h_dominance > 0.7:
                rationale = "Optimal Blend Choice"
            else:
                rationale = "Probabilistic Diversity Jump"

        # Export probabilities for first decision (Showcase HUD)
        prob_payload = None
        if current == 0 and len(ant.visited_customers) == 0:
            prob_payload = {
                "nodes": [int(n) for n in feasible],
                "probs": [float(p) for p in probs]
            }

        return feasible[selected_idx], rationale, prob_payload
