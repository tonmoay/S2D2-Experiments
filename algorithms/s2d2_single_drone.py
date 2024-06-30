'''
This is the main algorithm and will be used for evaluating the pipeline
'''
import networkx as nx
import itertools
import random
import numpy as np
from collections import defaultdict
from heapq import nlargest
from gurobipy import Model, GRB, quicksum

########### Helper Functions #############
def visualize_tree(node, graph=None):
    import pygraphviz as pgv
    if graph is None:
        graph = pgv.AGraph(directed=True, strict=True, rankdir='LR')

    if isinstance(node, LeafNode):
        graph.add_node(node.result, shape='ellipse', color='green')
        return node.result

    parent_label = f"Def Node: {node.v_d}"
    graph.add_node(parent_label, shape='box')

    for action, child in node.children:
        child_label = visualize_tree(child, graph)
        graph.add_edge(parent_label, child_label, label=f"Att Node: {action}")

    return parent_label

def save_tree_image(node, filename="tree.png"):
    graph = pgv.AGraph(directed=True, strict=True, rankdir='LR')
    visualize_tree(node, graph)
    graph.layout(prog="dot")  # using dot layout
    graph.draw(filename)


def print_tree(node, indent="", prefix="Root:"):
    if isinstance(node, LeafNode):
        print(f"{indent}{prefix} [Result: {node.result}]")
        return

    print(f"{indent}{prefix} [Def Node: {node.v_d}. Battery: {node.B}]")

    for action, child in node.children:
        print_tree(child, indent + "    ", f"Att Position: {action} leads to ->")

def string_tree(node, indent="", prefix="Root:"):
    if isinstance(node, LeafNode):
        return f"{indent}{prefix} [Result: {node.result}]"

    res = ""
    res += f"{indent}{prefix} [Def Node: {node.v_d}. Battery: {node.B}]\n"

    for action, child in node.children:
        res += string_tree(child, indent + "    ", f"Att Position: {action} leads to ->")
    return res
########## Helper Functions ############

def strategies_at_lambda(strategies, lambda_val, P):
    """
    Filter strategies based on the utility quantile for a given lambda value.
    """
    
    if strategies == []:
        return []
    
    if lambda_val == 1:
        return strategies  # Return all strategies if lambda is 1

    # Extract the utility values (total_reward) from the strategies
    utilities = np.array([strategy[0] for strategy in strategies])

    # Calculate the quantile value
    quantile_value = np.quantile(utilities, 1 - lambda_val)

    # Filter strategies that have utility >= the quantile value
    return [strategy for strategy in strategies if strategy[0] >= quantile_value]

def is_dominated(strategy1, strategy2, lambda_val):
    # Check if strategy1 is dominated by strategy2 based on the provided formula.
    return strategy1[0] < (1 - lambda_val) * strategy2[0] #this index stores the accumulated reward for attacker pure strategy

def greedy_attacker_strategies(G, B, P, lambda_val, start_nodes = None, num_strategies=None):
    '''
    This function performs a search on a graph G for multiple strategies.
    It makes 'P' attacks. 
    It considers all paths of length 'B' from all possible start nodes to all possible end nodes.
    It then returns the top 'num_strategies' strategies based on the total reward, 
    after filtering out dominated strategies.
    '''
    
    # Extract the top P crucial nodes
    crucial_nodes = set(nlargest(P, G.nodes(), key=lambda x: G.nodes[x]["reward"]))
    #TODO: we should take all nodes greater than delta reward instead
    
    start_node_set = G.nodes() if start_nodes is None else start_nodes
    
    # List to store all viable strategies
    # strategies = [[G.nodes[s_n]["reward"],[(s_n, 0)], [s_n]] for s_n in start_node_set]
    strategies = []

    # Iterate over all nodes in the graph as potential start nodes
    for start_node in start_node_set: #attacker can start from anywhere/or set of nodes to start
        # Iterate over all nodes in the graph as potential end nodes
        for end_node in crucial_nodes: #paths end at the valuable nod #G.nodes(): #can end up anywhere
            # Find all simple paths from start_node to end_node with length up to B
            for path in nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=B):
                # Find the 'P' most rewarding nodes in the path
                most_rewarding_nodes = nlargest(P, path, key=lambda x: G.nodes[x]["reward"])
                # Calculate the total reward for these nodes
                total_reward = sum(G.nodes[node]["reward"] for node in most_rewarding_nodes)
                # Create the strategy (most rewarding nodes with corresponding timesteps) 
                strategy = [(node, path.index(node)) for node in most_rewarding_nodes]
                # Sort the strategy based on timesteps
                strategy.sort(key=lambda x: x[1])
                # Add the strategy to the strategies list
                strategies.append((total_reward, strategy, path))
    
    strategies = strategies_at_lambda(strategies, lambda_val, P)
    
    # Filter out dominated strategies
    dominating_strategies = []
    for strat1 in strategies:
        dominated = False
        for strat2 in strategies:
            if strat1 != strat2 and is_dominated(strat1, strat2, lambda_val):
                dominated = True
                break
        if not dominated:
            dominating_strategies.append(strat1)
    
    # Find the top 'num_strategies' strategies based on total reward
    if num_strategies is not None:
        top_strategies = nlargest(num_strategies, dominating_strategies)
    else:
        top_strategies = dominating_strategies
    
    # Return the top strategies without the total rewards
    return [(strategy, path) for _, strategy, path in top_strategies]

def catch(G, v_prime_d, A_B ,B, s_prime_a, delta):
    '''
    This function calculates the shortest paths from node 'v_d' to each node in 'S_A' in the graph 'G'.
    '''
    
    # Initialize the path to catch the attacker as empty
    path_to_catch = []
    attacker_utility = 0
    defender_penalty = 0
        
    # For each node in s_prime_a (assuming it's sorted by timesteps)
    for i, (node, attack_time) in enumerate(s_prime_a):
        if G.nodes[node]['penalty'] > -delta:
            attacker_utility += G.nodes[node]['reward']
            defender_penalty += G.nodes[node]['penalty']
            continue        

        # Calculate the shortest path from v_d to the current node
        try:
            path = nx.shortest_path(G, v_prime_d, node)
        except Exception as e:
            return [(-1,), -1, attacker_utility, defender_penalty]
        
        # Check if the defender can reach the node before the attacker
        if len(path) - 1 <= attack_time - (A_B-B):
                # If yes, update the path to catch the attacker and break the loop
                path_to_catch = path
                break
            
        attacker_utility += G.nodes[node]['reward']
        defender_penalty += G.nodes[node]['penalty']
    
    # If no node is found where the defender can catch the attacker, return None
    if not path_to_catch:
        return [(-1,), -1, attacker_utility, defender_penalty] #no path

    # Return the path to catch the attacker
    return [tuple(path_to_catch), i, attacker_utility, defender_penalty] #return the first attack time (index between 0 to P) node.

def dict_to_matrix(dict_input):
		n = max(key[0] for key in dict_input) + 1
		m = max(key[1] for key in dict_input) + 1
		matrix = np.full((n, m), -1)  # Default value is -1
		for key, value in dict_input.items():
				matrix[key[0]][key[1]] = value
		return matrix

def build_undominated_rows(dict_input):
    matrix = dict_to_matrix(dict_input)
    
    n, m = matrix.shape
    undominated_rows = []
    # undominated_rows_pruned = []
    
    for i in range(n):
        is_undominated_row = True
        
        for j in range(n):
            
            if i != j:
                a = np.where(matrix[i] != -1, matrix[i], np.inf)
                b = np.where(matrix[j] != -1, matrix[j], np.inf)
                
                # Check if row j dominates row i
                if np.all(b <= a) and np.any(b < a):
                    is_undominated_row = False
                    break
                
                if np.all(b == a) and i>j:
                    is_undominated_row = False
                    break
        
        if is_undominated_row:
            undominated_rows.append((i, matrix[i]))
    
    return undominated_rows

def find_path_indices(paths, v_a, v_b):
    
    indices = []

    for i, (attack, path) in enumerate(paths):
        pos_va = path.index(v_a) if v_a in path else -1
        
        # Check if v_a is not the last element and if v_b immediately follows v_a
        if 0 <= pos_va < len(path) - 1 and path[pos_va + 1] == v_b:
            indices.append(i)

    # If no paths were found, return -1
    if not indices:
        return -1

    return indices

class TreeNode:
    def __init__(self, v_d, B):
        self.v_d = v_d  # defender's position
        self.B = B
        self.children = []  # list of (s_prime_a, next_node) tuples

    def add_child(self, s_prime_a, child_node):
        self.children.append((s_prime_a, child_node))

class LeafNode:
    def __init__(self, result):
        self.result = result  # end result when defender catches the attacker
        		
def ScanDefenseStrategies(G, v_d, selected_va, S_A, B, P, delta):

    def scan_strategy(G, v_d, v_a, S_A, A_B, B, P, delta, current_node):
        if v_d == v_a or len(S_A) == 1:
            result = catch(G, v_d, A_B, B+1, S_A[0][0], delta)
            leaf_node = LeafNode({"last_def_node": result[0][-1], "att_utility": result[2], "def_utility": result[3], "battery_capacity": B})
            current_node.add_child(v_a, leaf_node)
            return

        defender_times = {}
        
        def_neighbors = list(G.neighbors(v_d)) + [v_d]
        att_neighbors = list(G.neighbors(v_a)) + [v_a]

        for i, v_prime_d in enumerate(def_neighbors):
            for j, (s_prime_a, s_prime_a_path) in enumerate(S_A):
                defender_times[i,j] = catch(G, v_prime_d, A_B, B+1, s_prime_a, delta)[1]

        undominated = build_undominated_rows(defender_times)

        for v_d_next, times in undominated:
            
            next_v_d_id = def_neighbors[v_d_next]
            v_d_next_node = TreeNode(next_v_d_id, B)
            
            for v_prime_a in att_neighbors:
                updated_S_A_indices = find_path_indices(S_A, v_a, v_prime_a) #the update(S^a) function
                if updated_S_A_indices == -1:
                    continue
                updated_S_A = [S_A[i] for i in updated_S_A_indices]
                
                scan_strategy(G, next_v_d_id, v_prime_a, updated_S_A, A_B, B-1, P, delta, v_d_next_node)
            
            current_node.add_child(v_a, v_d_next_node)

    root = TreeNode(v_d, B)
    
    for v_a in selected_va:
        refined_S_A = [s_a for s_a in S_A if s_a[1][0] == v_a]
        if refined_S_A == []:
            continue
        scan_strategy(G, v_d, v_a, refined_S_A, B, B-1, P, delta, root)
    return root


def CompactTreeToExtensivePureStrategySet(node):
    
    """
    Generate multiple strategy trees from the main tree based on different defender moves for each attacker move.

    Parameters:
    - node: The main strategy tree.

    Returns:
    - A list of strategy trees.
    """

    # If it's a leaf node, return it as is.
    if isinstance(node, LeafNode):
        return [node]

    # Group child nodes by attacker move
    grouped_children = defaultdict(list)
    for s_prime_a, child in node.children:
        for subtree in CompactTreeToExtensivePureStrategySet(child):
            grouped_children[s_prime_a].append(subtree)

    # Generate combinations of trees for each set of attacker moves
    combinations = list(itertools.product(*grouped_children.values()))
    
    all_trees = []
    for combination in combinations:
        new_tree = TreeNode(node.v_d, node.B)
        for s_prime_a, child in zip(grouped_children.keys(), combination):
            new_tree.add_child(s_prime_a, child)
        all_trees.append(new_tree)

    return all_trees


def SamplePureStrategyFromCompactTree(node):
    """
    Sample a random pure strategy from the compact tree representation.

    Parameters:
    - node: The main strategy tree.

    Returns:
    - A pure strategy tree.
    """

    # If it's a leaf node, return it as is.
    if isinstance(node, LeafNode):
        return node

    # Group child nodes by attacker move
    grouped_children = defaultdict(list)
    for s_prime_a, child in node.children:
        grouped_children[s_prime_a].append(child)

    # Sample one tree for each attacker action s_prime_a
    new_tree = TreeNode(node.v_d, node.B)
    for s_prime_a in grouped_children:
        # Sample a random compact tree child
        compact_child = random.choice(grouped_children[s_prime_a])
        # Use recursion to sample a pure strategy
        pure_child = SamplePureStrategyFromCompactTree(compact_child)
        # Add pure child to the sampled tree
        new_tree.add_child(s_prime_a, pure_child)

    return new_tree

def unique_values(lst):
    seen = set()  # A set to keep track of seen values
    unique_lst = []  # The result list with unique values

    for value in lst:
        real_value = string_tree(value)
        if real_value not in seen:
            seen.add(real_value)
            unique_lst.append(value)

    return unique_lst

def CompactTreeToPureStrategySet(node, num_samples):
    """
    Sample a random pure strategies from the compact tree representation.

    Parameters:
    - node: The main strategy tree.
    - num_samples: Number of sampled pure strategies.

    Returns:
    - A list of strategy trees.
    """
    if num_samples is None:
        return CompactTreeToExtensivePureStrategySet(node)
    sampled_trees = []
    for i in range(1,num_samples+1):
        random.seed(i)
        sampled_trees.append(SamplePureStrategyFromCompactTree(node))
    # TODO: remove duplicates, you can check if two trees have the same string_tree() output.
    return unique_values(sampled_trees)


def compute_utility_for_tree(G, tree, attacker_strategy, target_paths, lambda_):
    """
    Compute the utility of an attacker strategy given a defender strategy tree.

    Parameters:
    - tree: Defender strategy tree.
    - attacker_strategy: The fixed attacker strategy. It contains a tuple with nodes to attack and all the patha
    - lambda_: Probability defender is present.

    Returns:
    - Utility for the attacker.
    """

    target_nodes, full_path = attacker_strategy
    defender_utility = sum(G.nodes[node]['penalty'] for node, _ in target_nodes)
    attacker_utility = sum(G.nodes[node]['reward'] for node, _ in target_nodes)

    if isinstance(tree, LeafNode):
        # Base case: If it's a leaf node, compute utility based on whether attacker is caught
        node_caught = tree.result['last_def_node']
        if node_caught == -1:  # attacker not caught
            return defender_utility, attacker_utility
        else:
            def_utility = (1 - lambda_) * defender_utility + lambda_ * tree.result['def_utility']
            att_utility = (1 - lambda_) * attacker_utility + lambda_ * tree.result['att_utility']
            return def_utility, att_utility
    
    if len(full_path) == 0: # Attacker is out of payload, and wasn't caught
         return defender_utility, attacker_utility

    attacker_node = full_path[0]
    if attacker_node in [s_prime_a for s_prime_a, _ in tree.children]:
        for s_prime_a, child in tree.children:
            if s_prime_a == attacker_node:
                return compute_utility_for_tree(G, child, (target_nodes, full_path[1:]),target_paths, lambda_)
    else:
        raise("Error: Attacker didn't best respond")

def optimize_defender_with_trees_inner(G, S_a, S_d_trees, lambda_, s_a):
    """
    Solve the LP to optimize the defender's strategy given a set of attacker strategies and defender strategy trees.

    Parameters:
    - S_a: List of attacker strategies.
    - S_d_trees: List of defender strategy trees.
    - lambda_: Probability defender is present.
    - Attacker assumed best reponse

    Returns:
    - Optimal defender strategy and the corresponding utility.
    """
    model = Model("defender_LP_with_trees")

    # Gurobi variable for each defender strategy tree
    x_d = model.addVars(len(S_d_trees), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x_d")

    # Objective is to maximize attacker payoff 
    model.setObjective(
        quicksum(compute_utility_for_tree(G, S_d_trees[i], s_a, s_a[1], lambda_)[0] * x_d[i] for i in range(len(S_d_trees))),
        GRB.MAXIMIZE
    )

    # Constraints ensure each s_a is the best response
    for s_a_prime in S_a:
        model.addConstr(
            quicksum(compute_utility_for_tree(G, S_d_trees[i], s_a, s_a[1], lambda_)[1] * x_d[i] for i in range(len(S_d_trees))) \
            >= quicksum(compute_utility_for_tree(G, S_d_trees[i], s_a_prime, s_a_prime[1], lambda_)[1] * x_d[i] for i in range(len(S_d_trees))),
            "Attacker best response constraint"
        )

    # Defender probabilities sum to 1
    model.addConstr(quicksum(x_d[i] for i in range(len(S_d_trees))) == 1, "defender constraint")
    
    model.setParam('OutputFlag', 0)

    model.optimize()

    try:
        # Get optimal defender strategy
        optimal_x_d = [x_d[i].X for i in range(len(S_d_trees))]
        
        # Compute defender and attacker utility given the optimal strategy
        defender_utility = model.ObjVal
        attacker_utility = sum(compute_utility_for_tree(G, S_d_trees[i], s_a, s_a[1], lambda_)[1] * optimal_x_d[i] for i in range(len(S_d_trees)))
        
        return (defender_utility, attacker_utility), optimal_x_d, s_a
    except Exception as e:
        return (-np.inf, -np.inf), []

def optimize_defender_with_trees(G, S_a, S_d_trees, lambda_):
     results = [optimize_defender_with_trees_inner(G, S_a, S_d_trees, lambda_, s_a) for s_a in S_a]
          
     # Sort results by highest defender utility and then by lowest attacker utility
     results.sort(key=lambda x: (-x[0][0], x[0][1]))
    
     return results[0]
          
def flatten(lst):
    """Flatten a nested list."""
    flat_list = []
    
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
            
    return flat_list