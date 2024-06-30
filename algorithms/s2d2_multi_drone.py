import random
import networkx as nx
import numpy as np
from gurobipy import Model, GRB
from algorithms.s2d2_single_drone import greedy_attacker_strategies, ScanDefenseStrategies
from algorithms.s2d2_single_drone import CompactTreeToPureStrategySet, optimize_defender_with_trees, flatten

def sample_percentage(items, percentage):
    sample_size = int(len(items) * percentage)
    sample_size = max(2, sample_size)  # Ensure at least 2 samples
    return random.sample(items, min(sample_size, len(items)))  # Ensure the sample size doesn't exceed the list size

def get_all_strategies_for_battery_range(G, start_battery, end_battery, P, lambda_val, attacker_start_positions, num_strategies=None):
    # Create an empty set to store the unique strategies with paths
    all_strategies = set()

    # Iterate over the battery values
    for attacker_battery in range(start_battery, end_battery + 1):
        # Get the strategies for the current battery value
        current_strategies = greedy_attacker_strategies(G, B=attacker_battery, P=P, lambda_val=lambda_val, 
                                                        start_nodes = attacker_start_positions, num_strategies=num_strategies)
       
        # Randomly sample x% of the strategies for this start position
        percent_val = 0.5 #0.1 #TODO: update later depending on the machine specs
        sampled_strategies = sample_percentage(current_strategies, percent_val)
        
        # Add each sampled strategy with its path to the set
        for strategy, path in sampled_strategies:
            all_strategies.add((tuple(strategy), tuple(path)))

    return [(list(strategy), list(path)) for strategy, path in all_strategies]

# Utility functions
def utilities(t, lambda_, info_dict):
    G = info_dict["neighborhoods"][t]

    attacker_battery = info_dict["attacker_battery"]
    attacker_payload = info_dict["attacker_payload"]
    _delta = info_dict["_delta"]
    rewards = nx.get_node_attributes(G, 'reward')

    attacker_start_positions = sorted(rewards.keys(), key=lambda node: rewards[node], reverse=True)[:2*attacker_payload]
    # attacker_start_positions.append(central_node) TODO: add this
    
    num_random_positions = 10 #TODO: update later depending on the machine specs
    
    if len(G.nodes) < num_random_positions: #Handling an edge case if the graph is too small
        num_random_positions = len(G.nodes)
        
    attacker_start_positions += list(np.random.choice(np.array(G.nodes), size=num_random_positions, replace=False))
    attacker_start_positions = list(set(attacker_start_positions))
    
    S_a = get_all_strategies_for_battery_range(G=G, start_battery=4, end_battery=attacker_battery, 
                    P=attacker_payload, lambda_val=lambda_, attacker_start_positions=attacker_start_positions)
    
    if S_a == []: #if no S_a is found (very unlikely, then return the bad utility)
        
        return {
        
        "utilities": np.array([-np.inf, 0]),
        "pure_attacker_strategy": [],
        "defender_mixed_strategies": [],
        "defender_mixed_strategies_probab_dist": [],
        "all_attacker_strategies":[]
    }
    
    assert all([s_a[1][0] in attacker_start_positions for s_a in S_a])

    defender_start_positions = attacker_start_positions
    compact_trees = [ScanDefenseStrategies(G, def_start_node, defender_start_positions, S_a, attacker_battery, 
                    attacker_payload, _delta) for def_start_node in defender_start_positions] # Compute strategies
    
    sample_size = 20 #TODO: update later depending on the machine specs
    sub_trees = [subtree for subtree in [CompactTreeToPureStrategySet(node=compact_tree, 
                                            num_samples=sample_size) for compact_tree in compact_trees]]
    sub_trees = flatten(sub_trees)
    output = optimize_defender_with_trees(G, S_a, sub_trees, lambda_) #(defender_utility, attacker_utility)
    return {
        "utilities": np.array(output[0]),
        "pure_attacker_strategy": output[2],
        "defender_mixed_strategies": sub_trees,
        "defender_mixed_strategies_probab_dist": output[1],
        "all_attacker_strategies": S_a,
    }

# Placeholder function for piecewise linear approximation of u_a
def sample_utilities(t, num_samples, info_dict):
    """
    Samples the u_a function for a given t over the interval [0,1].
    
    Args:
    - t (int): target index.
    - num_samples (int): number of points to sample.
    
    Returns:
    - list: List of tuples where each tuple represents a segment in the form (slope, intercept).
    """
    lambda_values = np.linspace(0,1,num_samples+1)
    
    all_utilities = []
    pure_attacker_strategies = []
    defender_mixed_strategies = []
    defender_mixed_strategies_probab_dist = []
    all_attacker_strategies = []
    
    for lambda_val in lambda_values:
        utility_and_samples = utilities(t, lambda_val, info_dict)
        all_utilities.append(np.array(utility_and_samples["utilities"]))
        pure_attacker_strategies.append(utility_and_samples["pure_attacker_strategy"])
        defender_mixed_strategies.append(utility_and_samples["defender_mixed_strategies"])
        defender_mixed_strategies_probab_dist.append(utility_and_samples["defender_mixed_strategies_probab_dist"])
        all_attacker_strategies.append(utility_and_samples["all_attacker_strategies"])
    
    return np.array(all_utilities), {
        "pure_attacker_strategies": pure_attacker_strategies,
        "defender_mixed_strategies": defender_mixed_strategies,
        "defender_mixed_strategies_probab_dist": defender_mixed_strategies_probab_dist,
        "all_attacker_strategies": all_attacker_strategies
    }

def refine_utility_samples(utility_samples):
    # Find the lowest negative number in the array (excluding '-inf' values)
    negative_values_excluding_inf = utility_samples[np.logical_and(utility_samples < 0, np.isfinite(utility_samples))]
    lowest_negative = np.min(negative_values_excluding_inf)
    
    # Replace '-inf' with the lowest negative number
    utility_samples[np.isneginf(utility_samples)] = 100 * lowest_negative
    
    return utility_samples

def optimize_multi_drone_solution_piecewise(num_samples, num_targets, num_attack_resources, num_defense_resources, 
                                            max_attacker_rewards, info_dict, utility_samples = None, strategies = None):
    # Sample u_a, u_d for each target and lambda to get segments for piecewise linear approximation
    lambda_values = np.linspace(0,1,num_samples+1)
    
    if info_dict["optimization"] == False:
        
        meta_output_results = [sample_utilities(t, num_samples, info_dict) for t in range(num_targets)]
        
        utility_samples, strategies = [], []
        
        for _data in meta_output_results:
            utility_samples.append(_data[0])
            strategies.append(_data[1])
        
        utility_samples = refine_utility_samples(np.array(utility_samples)) #-inf are replaced with 2*min negative value
        
        return {
            "utility_samples": utility_samples,
            "strategies": strategies
        }
    
    else:                
        # Create the MIP model
        m = Model("SSE_MIP_Piecewise")

        # Decision variables
        x_a = m.addVars(num_targets, vtype=GRB.BINARY, name="x_a")              # Attacker strategy
        theta_a = m.addVar(name="theta_a")                                      # Attacker threshold

        x_d = m.addVars(num_targets, lb=0, ub=1, name="x_d")                    # Defender strategy
        t_d = m.addVars(num_targets, num_samples+1, lb=0, ub=1, name="t_d")     # piecewise defender utility function to IP
        y_d = m.addVars(num_targets, num_samples, vtype=GRB.BINARY, name="y_d") # piecewise defender utility function to IP

        # Constraints
        m.addConstr(x_a.sum() == num_attack_resources, "attacker_resource_constraint")
        m.addConstr(x_d.sum() == num_defense_resources, "defender_resource_constraint")
        
        for t in range(num_targets):
            m.addConstr(t_d.sum(t,'*') == 1, "piecewise_constraint")
            m.addConstr(y_d.sum(t,'*') == 1, "piecewise_constraint")
        
        for t in range(num_targets):
            m.addConstr(sum(t_d[t,i]* lambda_values[i] for i in range(num_samples+1)) == x_d[t])
            m.addConstr(t_d[t,0] <= y_d[t,0])
            m.addConstr(t_d[t,num_samples] <= y_d[t,num_samples-1])
            for i in range(1, num_samples):
                m.addConstr(t_d[t,i] <= y_d[t,i-1] + y_d[t,i])

        # Add constraints for the attacker to best respond
        for t in range(num_targets):
            m.addConstr(sum(t_d[t, i] * utility_samples[t,i,1] for i in range(num_samples+1)) >= x_a[t] * theta_a, f"piecewise_lower_bound_{t}_{i}")
            m.addConstr(sum(t_d[t, i] * utility_samples[t,i,1] for i in range(num_samples+1)) <= (1-x_a[t]) * theta_a + x_a[t] * max_attacker_rewards[t], f"piecewise_upper_bound_{t}_{i}")

        # Objective: maximize the defender's utility
        m.setObjective(sum(x_a[t] * sum(t_d[t,i] * utility_samples[t,i,0] for i in range(num_samples+1)) for t in range(num_targets)), GRB.MAXIMIZE)

        m.setParam(GRB.Param.Threads, 0) #Using all possible threads
        m.setParam(GRB.Param.TimeLimit, 600) #10 minutes max to optimize
        # Optimize the model
        m.optimize()

        # Extract the results
        x_a_values = np.array([x_a[t].X for t in range(num_targets)]) #attacker allocation
        x_d_values = np.array([x_d[t].X for t in range(num_targets)]) #defender presence probability
        theta_a_value = theta_a.X
        t_d_values = np.array([[t_d[t,i].X for i in range(num_samples+1)] for t in range(num_targets)])
        y_d_values = np.array([[y_d[t,i].X for i in range(num_samples)] for t in range(num_targets)])

        total_defender_utility = sum(x_a[t].X * sum(t_d[t,i].X * utility_samples[t,i,0] for i in range(num_samples+1)) for t in range(num_targets))

        # total_attacker_utility = sum(x_d[t].X * sum(t_d[t,i].X * utility_samples[t,i,1] for i in range(num_samples+1)) for t in range(num_targets))
        total_attacker_utility = sum(x_a[t].X * sum(t_d[t,i].X * utility_samples[t,i,1] for i in range(num_samples+1)) for t in range(num_targets))

        return {
            "total_attacker_utility": total_attacker_utility,
            "total_defender_utility": total_defender_utility,
            "x_d_values": x_d_values,
            "x_a_values": x_a_values,
            "utility_samples": utility_samples,
            "strategies": strategies,
            "theta_a_value": theta_a_value,
            "t_d_values": t_d_values,
            "y_d_values": y_d_values,
            "lambda_values": lambda_values
        }