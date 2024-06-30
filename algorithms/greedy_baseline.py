import random
import copy
import numpy as np
import networkx as nx
from itertools import combinations
from python_tsp.exact import solve_tsp_dynamic_programming

def calculate_rewards_and_penalties(neighborhoods, P):
    
    rewards_and_penalties = {}

    for neighborhood in neighborhoods:
        # Extract rewards and penalties for all nodes in the neighborhood
        rewards_penalties = [(node, data['reward'], data['penalty']) for node, data in neighborhood.nodes(data=True)]
        
        # Sort nodes by rewards in descending order and take the top P nodes
        top_rewards_penalties = sorted(rewards_penalties, key=lambda x: (x[1], x[2]), reverse=True)[:P]
        
        # Calculate the sum of rewards for the top P nodes
        total_rewards = sum([reward for _, reward, _ in top_rewards_penalties])
        
        # Calculate the absolute sum of penalties for the top P nodes
        total_penalty = sum([abs(penalty) for _, _, penalty in top_rewards_penalties])
        
        rewards_and_penalties[neighborhood] = {"reward_sum": total_rewards, "penalty_sum": total_penalty}
        
    return rewards_and_penalties

def allocate_defense_drones(D, rewards_and_penalties):
    allocation = {neighborhood: [0, 0] for neighborhood in rewards_and_penalties}  # Initialize allocation with 0 drones for each neighborhood

    # Extract penalties
    penalties = {neighborhood: abs(data['penalty_sum']) for neighborhood, data in rewards_and_penalties.items()}  # Absolute sum of penalties
    
    # Calculate defender_utility_gain based on the absolute sum of penalties
    defender_utility_gain = penalties

    # Calculate the neighborhood coverage for the defender based on utility gain
    total_defender_utility = sum(defender_utility_gain.values())
    neighborhood_coverage = {neighborhood: (D * gain) / total_defender_utility for neighborhood, gain in defender_utility_gain.items()}
    return neighborhood_coverage

def allocate_attack_drones(A, rewards_and_penalties, neighborhood_coverage):
    
    attack_allocation = {neighborhood: 0 for neighborhood in rewards_and_penalties}  # Initialize allocation with 0 drones for each neighborhood

    # Extract rewards and penalties
    rewards = {neighborhood: data['reward_sum'] for neighborhood, data in rewards_and_penalties.items()}

    # Utility calculation for the attacker
    attacker_utility = {neighborhood: (1 - neighborhood_coverage[neighborhood]) * rewards[neighborhood] for neighborhood in rewards}
    
    # Attacker allocation
    neighborhoods_sorted_by_utility = sorted(attacker_utility, key=attacker_utility.get, reverse=True)
    counter = 0
    for neighborhood in neighborhoods_sorted_by_utility:
        attack_allocation[neighborhood] = 1
        counter += 1
        if counter == A:
            break
            
    return attack_allocation

def tsp_solver_wrapper(G, nodes, B):
    """Solve the Travelling Salesman Problem using python-tsp library."""
    # Create a distance matrix
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    
    for i, u in enumerate(nodes):
        distances = nx.single_source_dijkstra_path_length(G, u)
        for j, v in enumerate(nodes):
            if u != v:
                if v in distances and distances[v] <= B:
                    distance_matrix[i][j] = distances[v]
                else:
                    distance_matrix[i][j] = 1e9  # Set to a large number if no path or path exceeds B

    # Solve TSP using dynamic programming
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    
    # Return the optimal path based on the permutation

    return [nodes[i] for i in permutation], distance

def tsp_solver(G, nodes, B):
    curr_P = len(nodes)
    while True:
        curr_nodes_combinations = {tuple(curr_nodes): sum([G.nodes[node]['reward'] for node in curr_nodes])
                                   for curr_nodes in combinations(nodes, curr_P)}
        for curr_nodes in sorted(curr_nodes_combinations, reverse=True):
            path, distance = tsp_solver_wrapper(G, curr_nodes, B)
            if distance <= B+1:
                return path, distance
        curr_P -= 1

def select_top_reward_nodes(path, P, rewards):
    """
    Given a path, select the top P nodes to attack for maximum reward.
    """
    # Sort nodes in path by rewards
    sorted_path_by_reward = sorted(path, key=lambda x: rewards.get(x, 0), reverse=True)
    
    # Return top P nodes
    return sorted_path_by_reward[:P]

def decide_attacker_strategy_tsp(neighborhood, B, P, overall_graph, attacker_occupied_nodes):
    # Extract rewards and penalties for all nodes in the neighborhood
    rewards_penalties = [(node, data['reward'], data['penalty']) for node, data in neighborhood.nodes(data=True) if node not in attacker_occupied_nodes]
        
    # Sort nodes by rewards in descending order and take the top P nodes
    top_rewards_penalties = sorted(rewards_penalties, key=lambda x: (x[1], x[2]), reverse=True)[:P]

    # Get the rewards for each node in the neighborhood
    top_P_nodes = [(node, reward) for node, reward, _ in top_rewards_penalties]
    top_P_nodes_list = [node[0] for node in top_P_nodes]
    
    # Limit the number of nodes to B+1 for the TSP solver
    nodes_for_tsp = top_P_nodes_list[:B+1]
    
    # Solve TSP for the nodes
    path, _ = tsp_solver(neighborhood, nodes_for_tsp, B)
    
    rewards = {node: data['reward'] for node, data in neighborhood.nodes(data=True) if node not in attacker_occupied_nodes}
    # Determine which nodes to attack for maximum reward
    attacked_nodes = select_top_reward_nodes(path, P, rewards)
    
    # Create the list of tuples (node, timestep, attack status) for the nodes
    path_with_timestep_and_attack_with_bool = [(node, i+1, node in attacked_nodes) for i, node in enumerate(path)]
    # path_with_timestep_and_attack = [(node, i+1) for i, node in enumerate(path) if node in attacked_nodes]
    
    return [path_with_timestep_and_attack_with_bool, path]

def defender_path_in_response_to_attacker_wrong(neighborhood, attacker_path, current_defender_node, delta=1):
    # Initialize the defender's path
    defender_path = [current_defender_node]
    
    # Store the previously calculated path for the defender
    previous_path = [current_defender_node]
    
    for i, (node, _, attacked) in enumerate(attacker_path):

        if attacked:
            neighborhood.nodes[node]['reward'] = 0 #make node value to 0
            # Find single-hop neighbors of the attacked node
            neighbors = list(neighborhood.neighbors(node))
            
            # Filter neighbors based on rewards greater than delta
            neighbor_rewards = dict(neighborhood.nodes(data='reward'))
            eligible_neighbors = [n for n in neighbors if neighbor_rewards.get(n, 0) > delta]
            
            # If no eligible neighbors, continue to next iteration
            if not eligible_neighbors:
                continue
            
            # Find the closest neighbor among the eligible ones based on shortest path length
            distances = {}
            for target_node in eligible_neighbors:
                try:
                    distance = len(nx.dijkstra_path(neighborhood, node, target_node)) - 1  # subtract 1 to exclude starting node
                    distances[target_node] = distance
                except nx.NetworkXNoPath:
                    distances[target_node] = float('inf')
                    
            target_node = min(distances, key=distances.get)
            
            # Calculate shortest path for the defender to the target node
            try:
                shortest_path = nx.dijkstra_path(neighborhood, current_defender_node, target_node)
                
                #If the attacker need X hops to attack the closest most rewarding node for the attacker, 
                #and you can get there in Y<X steps, That's when you want to start going towards there.
                
                #NOTE: "the penalty on that target for the defender is significant." is not added yet
                
                # if len(shortest_path) < distances[target_node]:
                    
                previous_path = shortest_path
                
            except nx.NetworkXNoPath:
                print(f"Defender: No path between nodes {current_defender_node} and {target_node}")
                continue
        
        # Defender moves one step based on the previously calculated path
        if len(previous_path) > 1:
            current_defender_node = previous_path[1]
            defender_path.append(current_defender_node)
            previous_path = [current_defender_node] + previous_path[2:]
        else:
            # Defender stays at the current node if no further move is possible
            defender_path.append(current_defender_node)
        
        # Check if the defender and attacker cross the same edge
        if i < len(attacker_path) - 1:
            next_attacker_node = attacker_path[i + 1][0]
            if (current_defender_node, next_attacker_node) in neighborhood.edges or (next_attacker_node, current_defender_node) in neighborhood.edges:
                break
    
    return defender_path

def defender_path_in_response_to_attacker(neighborhood, attacker_path, current_defender_node, delta=1):
    # Initialize the defender's path    
    defender_path = [current_defender_node]
    
    if not attacker_path:
        return defender_path, "no attacker is present"

    largest_cc = max(nx.connected_components(neighborhood), key=len)
    assert current_defender_node in largest_cc, "defender starts at largest connected component in baseline"
    
    #building the dumb baseline
    # if current_defender_node not in largest_cc:
    #     return defender_path, "defender doesn't start at largest CC"
    
    if attacker_path[0][0] not in largest_cc:
        return defender_path, "cannot catch the attacker, it is in a different connected component"
    
    subgraph = neighborhood.subgraph(largest_cc)

    # Possible targeted nodes for the attacker, that are also worth defending
    targeted_nodes = {node for node in subgraph.nodes if subgraph.nodes[node]['reward'] > delta and subgraph.nodes[node]['penalty'] < -delta}
    
    # if len(targeted_nodes) == 0:
    #     return defender_path, "no targeted nodes"

    for i, (node, _, attacked) in enumerate(attacker_path):
        
        if attacked:
            targeted_nodes.discard(node)

        attack_distances = {target_node: len(nx.dijkstra_path(subgraph, node, target_node)) - 1 for target_node in targeted_nodes}
        defense_distances = {target_node: len(nx.dijkstra_path(subgraph, defender_path[-1], target_node)) - 1 for target_node in targeted_nodes}

        # Find first node feasible to catch
        for target_node in attack_distances:
            if defense_distances[target_node] <= attack_distances[target_node]:
                break
        
        if len(targeted_nodes) == 0:
            return defender_path, "no targeted nodes left to defend"
        
        shortest_path = nx.dijkstra_path(subgraph, defender_path[-1], target_node)
        
        if len(shortest_path) > 1:
            defender_path.append(shortest_path[1])
    
    return defender_path, ""


def decide_defender_strategy(neighborhood, attacker_history, overall_graph, occupied_nodes, delta):
    # Defender starts at the center of the neighborhood
    # current_defender_node = nx.center(neighborhood)[0]
    
    largest_cc = max(nx.connected_components(neighborhood), key=len)
    subgraph = neighborhood.subgraph(largest_cc)
    current_defender_node = nx.center(subgraph)[0] #the highest connected component
    
    # Select a random node from the largest connected component as the starting point
    current_defender_node = random.choice(list(subgraph.nodes))
    
    # current_defender_node = random.choice(list(neighborhood.nodes))
    #dumb baseline
    
    if attacker_history == []:
        return [current_defender_node], "no attacker is present"

    # Apply Dijkstra's algorithm to find the shortest path to the last attacker position
    return defender_path_in_response_to_attacker(neighborhood, attacker_history[0], current_defender_node, delta)

def defense_drone_allocation(neighborhoods, P, D):
     _rewards_and_penalties = calculate_rewards_and_penalties(neighborhoods, P)
     return allocate_defense_drones(D, _rewards_and_penalties)

def strategy(neighborhoods, overall_graph, A, D, B, P, delta):
    
    _rewards_and_penalties = calculate_rewards_and_penalties(neighborhoods, P)
    defender_presence_probability = allocate_defense_drones(D, _rewards_and_penalties)
    attacker_allocation = allocate_attack_drones(A, _rewards_and_penalties, defender_presence_probability)
    
    attacker_strategies = {}
    defender_strategies = {}
    
    attacker_occupied_nodes = set()
    defender_occupied_nodes = set()
    
    # _allocation = copy.deepcopy(allocation)
    
    for _neighborhood in attacker_allocation:
        attackers = attacker_allocation[_neighborhood]
        defenders = defender_presence_probability[_neighborhood]
        
        neighborhood = copy.deepcopy(_neighborhood)
        
        if attackers > 0:
            attacker_strategy = decide_attacker_strategy_tsp(neighborhood, B, P, overall_graph, attacker_occupied_nodes)
        else:
            attacker_strategy = [[],[]]
            
        attacker_strategies[_neighborhood] = attacker_strategy
        attacker_occupied_nodes.update(node[0] for node in attacker_strategy[0])
        
        if defenders > 0:  # Defender calculates a strategy only if present
            
            defender_strategy, _ = decide_defender_strategy(neighborhood, attacker_strategies.get(_neighborhood, []), overall_graph, defender_occupied_nodes, delta)
            
            if defender_strategy:
                defender_strategy = [defender_strategy] if isinstance(defender_strategy, int) else defender_strategy

            else:
                defender_strategy = []  # No valid defender strategy
        
        else:
            defender_strategy = []  # No defender to protect
            
        defender_strategies[_neighborhood] = defender_strategy
        defender_occupied_nodes.update(defender_strategy)        
    
    return attacker_allocation, defender_presence_probability, attacker_strategies, defender_strategies
    """
    Calculate the expected utility for the attacker based on the attacker and defender's strategies across all neighborhoods.
    
    Args:
    - attacker_allocation: Attacker allocation of drones in neighborhoods.
    - defender_presence_probability: Dictionary of Probability of a defender protecting a neighborhood.
    - attacker_strategies: Dictionary of attacker strategies for each neighborhood.
    - defender_strategies: Dictionary of defender strategies for each neighborhood.
    
    Returns:
    - Total expected utility for the attacker across all neighborhoods.
    """
    
    total_expected_utility = 0
    
    for _neighborhood in attacker_allocation:
        # Flag if attack is in neighborhood
        attackers = attacker_allocation[_neighborhood]
        # Probability that a defender is present in the neighborhood
        presence_probability = defender_presence_probability[_neighborhood]

        neighborhood = copy.deepcopy(_neighborhood)
        
        # Utility when the defender is absent
        utility_absent = 0
        for node, _, attacked in attacker_strategies[_neighborhood]:
            if attacked:
                utility_absent += neighborhood.nodes[node]['penalty']
                
        # Utility when the defender is present
        utility_present = 0
        for i, (node, _, attacked) in enumerate(attacker_strategies[_neighborhood]):
            # If the defender and attacker cross the same edge or meet at the same node
            if (i < len(defender_strategies[_neighborhood]) and 
                (node == defender_strategies[_neighborhood][i] or 
                 (i > 0 and (node, defender_strategies[_neighborhood][i]) in neighborhood.edges) or 
                 (i > 0 and (defender_strategies[_neighborhood][i], node) in neighborhood.edges))):
                break

            # If the attacker successfully attacked a node
            if attacked:
                utility_present += neighborhood.nodes[node]['penalty']
                neighborhood.nodes[node]['penalty'] = 0  # nullify the reward once attacked
        
        # Compute the expected utility for this neighborhood
        expected_utility = (1 - presence_probability) * utility_absent + presence_probability * utility_present
        
        total_expected_utility += expected_utility
    
    return total_expected_utility