import os
import time
import copy
import yaml
import numpy as np
from tqdm import tqdm
from algorithms.greedy_baseline import strategy as baseline_strategy, decide_defender_strategy, defense_drone_allocation
from algorithms.s2d2_single_drone import compute_utility_for_tree
from utils import load_results, save_results, compute_num_neighborhoods, load_graph_data, compute_delta
from utils import chunker, update_recorded_results, get_city_list

def add_boolean_indexing(data):
    if data == []:
        return data
    list_without_flags, id_list = data
    attacked_nodes = [id for (id, _) in list_without_flags]
    list_with_flags = [(id, index + 1, id in attacked_nodes) for index, id in enumerate(id_list)]
    return [list_with_flags, id_list]

def remove_boolean_indexing(data):
    list_with_flags, id_list = data
    attacked_nodes = [id for (id, _, attacked) in list_with_flags if attacked==True]
    path_with_timestep_and_attack = [(node, i+1) for i, node in enumerate(id_list) if node in attacked_nodes]
    return [path_with_timestep_and_attack, id_list]

def sample_defense_strategy(neighborhood_coverage):
    D = round(sum(neighborhood_coverage.values()))
    # return npr.choice(list(neighborhood_coverage.keys()), size=D, p=list(neighborhood_coverage.values()), replace=False)
    # Step 1: Assign identifiers to graphs
    graph_mapping = {i: graph for i, graph in enumerate(neighborhood_coverage.keys())}
    inverse_mapping = {graph: i for i, graph in graph_mapping.items()}

    # Prepare your probabilities
    total_sum = sum(neighborhood_coverage.values())
    normalized_probabilities = [value / total_sum for value in neighborhood_coverage.values()]

    # Step 2: Select identifiers
    selected_indices = np.random.choice(
        list(graph_mapping.keys()), 
        size=D, 
        p=normalized_probabilities, 
        replace=False
    )

    # Step 3: Retrieve selected graphs
    selected_graphs = [graph_mapping[index] for index in selected_indices]
    
    return selected_graphs

def is_node_in_list(node_to_check, updated_list):
    for node, _ in updated_list:
        if node == node_to_check:
            return True
    return False

def calculate_sampled_attacker_utility(neighborhoods, attacker_allocation, defended_neighborhoods, attacker_strategies, 
                                       defender_strategies):
    
    all_attacker_utilities = []
    
    for i, _neighborhood in enumerate(neighborhoods):
        
        if attacker_allocation[i] == 1: #calculate utilities where the attacker is allocated

            # Probability that a defender is present in the neighborhood
            defender_presence = defended_neighborhoods[i]

            neighborhood = copy.deepcopy(_neighborhood)
            
            # Utility when the defender is absent
            utility_absent = 0
            for node, _ in attacker_strategies[i][0]:
                utility_absent += neighborhood.nodes[node]['reward']
                    
            # Utility when the defender is present
            utility_present = 0
            for j, node in enumerate(attacker_strategies[i][1]):
                # If the defender and attacker cross the same edge or meet at the same node
                if (j < len(defender_strategies[i]) and 
                    (node == defender_strategies[i][j] or 
                    (j > 0 and (node, defender_strategies[i][j]) in neighborhood.edges) or 
                    (j > 0 and (defender_strategies[i][j], node) in neighborhood.edges))):
                    break

                # If the attacker successfully attacked a node
                if is_node_in_list(node, attacker_strategies[i][0]):
                    utility_present += neighborhood.nodes[node]['reward']
                    neighborhood.nodes[node]['reward'] = 0  # nullify the reward once attacked
            
            # Compute the expected utility for this neighborhood
            expected_utility = (1 - defender_presence) * utility_absent + defender_presence * utility_present
            
            all_attacker_utilities.append(expected_utility)
    
    return sum(all_attacker_utilities)

def calculate_sampled_defender_utility(neighborhoods, attacker_allocation, defended_neighborhoods, attacker_strategies, 
                                       defender_strategies):
    
    all_def_utilities = []
    
    for i, _neighborhood in enumerate(neighborhoods):
        
        if attacker_allocation[i] == 1: #calculate utilities where the attacker is allocated

            # Probability that a defender is present in the neighborhood
            defender_presence = defended_neighborhoods[i]

            neighborhood = copy.deepcopy(_neighborhood)
            
            # Utility when the defender is absent
            utility_absent = 0
            for node, _ in attacker_strategies[i][0]:
                utility_absent += neighborhood.nodes[node]['penalty']
                    
            # Utility when the defender is present
            utility_present = 0
            for j, node in enumerate(attacker_strategies[i][1]):
                # If the defender and attacker cross the same edge or meet at the same node
                if (j < len(defender_strategies[i]) and 
                    (node == defender_strategies[i][j] or 
                    (j > 0 and (node, defender_strategies[i][j]) in neighborhood.edges) or 
                    (j > 0 and (defender_strategies[i][j], node) in neighborhood.edges))):
                    break

                # If the attacker successfully attacked a node
                if is_node_in_list(node, attacker_strategies[i][0]):
                    utility_present += neighborhood.nodes[node]['penalty']
                    neighborhood.nodes[node]['penalty'] = 0  # nullify the reward once attacked
            
            # Compute the expected utility for this neighborhood
            expected_utility = (1 - defender_presence) * utility_absent + defender_presence * utility_present
            
            all_def_utilities.append(expected_utility)
    
    return sum(all_def_utilities)

def calculate_sampled_utility(neighborhoods, attacker_allocation, defended_neighborhoods, attacker_strategies, 
                                       defender_strategies):
    sampled_def_utility = calculate_sampled_defender_utility(neighborhoods, attacker_allocation, 
                defended_neighborhoods, attacker_strategies, defender_strategies)
    sampled_att_utility = calculate_sampled_attacker_utility(neighborhoods, attacker_allocation, 
                defended_neighborhoods, attacker_strategies, defender_strategies)
    
    return sampled_def_utility, sampled_att_utility

def S2D2_calculate_sampled_utility(neighborhoods, attacker_allocation, pure_attacker_strategies, 
                                            mixed_defender_strategies, lambda_):
    #given sampled pure attacker and mixed defender strategies along with sampled allocation, 
    #the code will return the attacker and defender utility sum
    
    all_att_utilities = []
    all_def_utilities = []
    
    for i, neighborhohood in enumerate(neighborhoods):
        if attacker_allocation[i] == 1: #calculate utilities where the attacker is allocated 
            utility_list = compute_utility_for_tree(neighborhohood, mixed_defender_strategies[i], 
                pure_attacker_strategies[i], pure_attacker_strategies[i][1], lambda_[i]) 
            
            all_def_utilities.append(utility_list[0])
            all_att_utilities.append(utility_list[1])
        
    return sum(all_def_utilities), sum(all_att_utilities)

def calculate_attacker_defender_utility_samples(neighborhoods, overall_graph, A, D, B, P, delta, num_samples=1):
    attacker_allocation, defender_presence_probability, attacker_strategies, defender_strategies = baseline_strategy(neighborhoods, overall_graph, A, D, B, P, delta)

    attacker_utilities = []
    defender_utilities = []

    for _ in range(num_samples):
        defended_neighborhoods = sample_defense_strategy(defender_presence_probability)
        attacker_utilities.append(calculate_sampled_attacker_utility(attacker_allocation, defended_neighborhoods, attacker_strategies, defender_strategies))
        defender_utilities.append(calculate_sampled_defender_utility(attacker_allocation, defended_neighborhoods, attacker_strategies, defender_strategies))

    return np.array(attacker_utilities), np.array(defender_utilities)

def bin_probabilities(probabilities):
    # Define the reference points (0, 0.5, 1)
    reference_points = np.array([0, 0.5, 1])

    # Calculate the absolute differences between each probability and the reference points
    differences = np.abs(probabilities[:, None] - reference_points[None, :])

    # Find the index of the minimum difference for each probability
    indices = np.argmin(differences, axis=1)

    return indices

def sample_defender_allocations(defender_presence_probability):
    # Check if probability values are nonnegative
    
    D = round(sum(defender_presence_probability))
    
    if any(prob < 0 for prob in defender_presence_probability):
        defender_presence_probability = np.array([abs(prob) for prob in defender_presence_probability])
    
    _mapping = {i: value for i, value in enumerate(defender_presence_probability)}
    
    # Prepare your probabilities
    total_sum = sum(defender_presence_probability)
    normalized_probabilities = [value / total_sum for value in defender_presence_probability]
    
    if any(prob < 0 for prob in normalized_probabilities):
        raise ValueError("Probability values must be nonnegative.")
    
    selected_indices = np.random.choice(list(_mapping.keys()), size=D, p=normalized_probabilities, replace=False)
    
    final_indices = np.zeros(defender_presence_probability.shape[0])
    
    final_indices[selected_indices] = 1
    
    return final_indices

def S2D2_sample_mixed_defense_strategies(S2D2_mixed_defender_strategies, S2D2_mixed_defender_strategies_probabs):

    '''
    For each neighborhood, there are multiple mixed strategies. Take the probability for each neighborhood and sample one    
    '''
    
    sampled_strategies = []
    
    num_samples = 1
    
    for i, _strategy in enumerate(S2D2_mixed_defender_strategies):
        
        if _strategy == []:
            sampled_strategies.append([])
            continue
        
        sampled_probab = np.array(S2D2_mixed_defender_strategies_probabs[i])
        
        if any(prob < 0 for prob in sampled_probab):
            sampled_probab = np.array([abs(prob) for prob in sampled_probab])
        
        selected_indices = np.random.choice(_strategy, size=num_samples, p=sampled_probab, replace=False)
        
        sampled_strategies.append(selected_indices[0])
        
    return sampled_strategies

# def compute_utilities(neighborhood, attacker_strategy, defender_strategy):
#     # Utility when the defender is absent
#     utility_absent = 0
#     for node, _ in attacker_strategy[0]:
#         utility_absent += neighborhood.nodes[node]['reward']

#     # Utility when the defender is present
#     utility_present = 0
#     for j, node in enumerate(attacker_strategy[1]):
#         # If the defender and attacker cross the same edge or meet at the same node
#         if (j < len(defender_strategy) and 
#             (node == defender_strategy[j] or 
#             (j > 0 and (node, defender_strategy[j]) in neighborhood.edges) or 
#             (j > 0 and (defender_strategy[j], node) in neighborhood.edges))):
#             break

#         # If the attacker successfully attacked a node
#         if is_node_in_list(node, attacker_strategy[0]):
#             utility_present += neighborhood.nodes[node]['reward']
#             neighborhood.nodes[node]['reward'] = 0  # nullify the reward once attacked
            
#     return np.array([utility_absent, utility_present])
 
def compute_utilities(neighborhood, attacker_strategy, defender_strategy, _lambda):
    # Utility when the defender is absent
    utility_absent = 0
    for node, _ in attacker_strategy[0]:
        utility_absent += neighborhood.nodes[node]['reward']

    # Utility when the defender is present
    utility_present = 0
    
    utility_list = compute_utility_for_tree(neighborhood, defender_strategy, attacker_strategy, attacker_strategy[1], _lambda)
    utility_present += utility_list[1]
            
    return np.array([utility_absent, utility_present])
    
def S2D2_attacker_best_response_to_baseline_defender(neighborhoods, num_attackers, S2D2_all_attacker_strategies, 
                                 baseline_defender_presence_probability, delta):
    
    baseline_num_runs = 10
    #NOTE: As the baseline defender starts from random node, we need to sample multiple strategies for each 
    #neighborhood and want the S2D2 attacker best respond to it.
    
    attacker_best_strategies_per_neighborhood = []
    attacker_best_utilities_per_neighborhood = []
    defender_strategies = [[]]*len(neighborhoods)
    
    for i, _neighborhood in enumerate(neighborhoods):    
        best_attacker_strategy = []
        best_attacker_strategy_utility = 0 #we want to maximize that
    
        # Probability that a defender is present in the neighborhood
        defender_presence = baseline_defender_presence_probability[i]
        
        all_attacker_strategies = S2D2_all_attacker_strategies[i]
        
        if all_attacker_strategies == []:
            defender_strategies[i], _ = decide_defender_strategy(_neighborhood, [[],[]], [], [], delta)
        
        for attacker_strategy in all_attacker_strategies:
            
            # defender_strategy, _ = decide_defender_strategy(_neighborhood, add_boolean_indexing(attacker_strategy), 
            #                                                 [], [], delta)
            
            _defender_strategies = [decide_defender_strategy(_neighborhood, 
                    add_boolean_indexing(attacker_strategy), [], [], delta)[0] for _ in range(baseline_num_runs)]

            neighborhood = copy.deepcopy(_neighborhood)
            
            utility_absent, utility_present = list(np.array([compute_utilities(neighborhood, 
                        attacker_strategy, defender_strategy) for defender_strategy in _defender_strategies]).mean(axis=0))
            
            # Compute the expected utility for this neighborhood
            expected_utility = (1 - defender_presence) * utility_absent + defender_presence * utility_present
            
            if expected_utility > best_attacker_strategy_utility:
                best_attacker_strategy_utility = expected_utility
                best_attacker_strategy = attacker_strategy
                # defender_strategies[i] = random.choice(_defender_strategies)

        attacker_best_strategies_per_neighborhood.append(best_attacker_strategy)
        attacker_best_utilities_per_neighborhood.append(best_attacker_strategy_utility)
    
    attacker_best_utilities_per_neighborhood = np.array(attacker_best_utilities_per_neighborhood)
    # Finding indices of the top k values
    indices = np.argsort(attacker_best_utilities_per_neighborhood)[-num_attackers:]
    new_S2D2_attacker_allocations = np.zeros_like(attacker_best_utilities_per_neighborhood)
    new_S2D2_attacker_allocations[indices] = 1
    
    return new_S2D2_attacker_allocations, attacker_best_strategies_per_neighborhood #, defender_strategies

def S2D2_attacker_best_response_to_S2D2_defender(neighborhoods, num_attackers, S2D2_all_attacker_strategies, 
        s2d2_defender_presence_probability, S2D2_mixed_defender_strategies, S2D2_mixed_defender_strategies_probabs):
    
    attacker_best_strategies_per_neighborhood = []
    attacker_best_utilities_per_neighborhood = []
    
    for i, _neighborhood in enumerate(neighborhoods):    
        best_attacker_strategy = []
        best_attacker_strategy_utility = 0 #we want to maximize that
    
        # Probability that a defender is present in the neighborhood
        defender_presence = s2d2_defender_presence_probability[i]
        
        all_attacker_strategies = S2D2_all_attacker_strategies[i]
        
        for attacker_strategy in all_attacker_strategies:

            _defender_strategies = S2D2_mixed_defender_strategies[i]
            _defender_probabs = S2D2_mixed_defender_strategies_probabs[i]
            
            neighborhood = copy.deepcopy(_neighborhood)
            
            utility_absent, utility_present = list(np.array([compute_utilities(neighborhood, 
            attacker_strategy, defender_strategy, _lambda) for defender_strategy, _lambda 
                                        in zip(_defender_strategies, _defender_probabs)]).mean(axis=0))
            
            # utility_absent, utility_present = compute_utilities(neighborhood, attacker_strategy, defender_strategy)
            
            # Compute the expected utility for this neighborhood
            expected_utility = (1 - defender_presence) * utility_absent + defender_presence * utility_present
            
            if expected_utility > best_attacker_strategy_utility:
                best_attacker_strategy_utility = expected_utility
                best_attacker_strategy = attacker_strategy

        attacker_best_strategies_per_neighborhood.append(best_attacker_strategy)
        attacker_best_utilities_per_neighborhood.append(best_attacker_strategy_utility)
    
    attacker_best_utilities_per_neighborhood = np.array(attacker_best_utilities_per_neighborhood)
    # Finding indices of the top k values
    indices = np.argsort(attacker_best_utilities_per_neighborhood)[-num_attackers:]
    new_S2D2_attacker_allocations = np.zeros_like(attacker_best_utilities_per_neighborhood)
    new_S2D2_attacker_allocations[indices] = 1
    
    return new_S2D2_attacker_allocations, attacker_best_strategies_per_neighborhood
    
def cross_compare_methods(city_name, s2d2_file_name, neighborhoods, overall_graph, A, D, B, P, 
                          delta, num_samples=1, perturb_rewards=0):
    
    S2D2_all_results = load_results(filename=s2d2_file_name)
    
    S2D2_recorded_results = S2D2_all_results[city_name][B]["result_dict"]
    
    S2D2_attacker_allocation = S2D2_recorded_results["x_a_values"]
    num_attackers = round(sum(S2D2_attacker_allocation))
    
    S2D2_defender_presence_probability = S2D2_recorded_results["x_d_values"]
    
    defender_presence_prob_mapped_to_indices = bin_probabilities(S2D2_defender_presence_probability)
    #this one is used for picking up pure attacker strategies and mixed defender strategies from lambda split
    
    S2D2_attacker_strategies = [s["pure_attacker_strategies"][defender_presence_prob_mapped_to_indices[i]] for i, s in enumerate(S2D2_recorded_results["strategies"])]
    S2D2_mixed_defender_strategies = [s["defender_mixed_strategies"][defender_presence_prob_mapped_to_indices[i]] for i, s in enumerate(S2D2_recorded_results["strategies"])]
    S2D2_mixed_defender_strategies_probabs = [s["defender_mixed_strategies_probab_dist"][defender_presence_prob_mapped_to_indices[i]] for i, s in enumerate(S2D2_recorded_results["strategies"])]
    S2D2_all_attacker_strategies = [s["all_attacker_strategies"][defender_presence_prob_mapped_to_indices[i]] for i, s in enumerate(S2D2_recorded_results["strategies"])]

    new_S2D2_attacker_allocations, new_S2D2_attacker_best_response_strategies \
        = S2D2_attacker_best_response_to_S2D2_defender(neighborhoods, num_attackers, 
            S2D2_all_attacker_strategies, S2D2_defender_presence_probability, S2D2_mixed_defender_strategies, 
            S2D2_mixed_defender_strategies_probabs)
    
    pass    
    #############
    # new_S2D2_attacker_allocations, new_S2D2_attacker_best_response_strategies \
    #     = S2D2_attacker_best_response_to_baseline_defender(neighborhoods, num_attackers, S2D2_all_attacker_strategies, 
    #                                 baseline_defender_presence_probability, delta)
    # s2d2_att_to_baseline_def_time = time.time()
    
    # baseline_defender_presence_probability = defense_drone_allocation(neighborhoods, P, D)
    # baseline_defender_presence_probability = np.array(list(baseline_defender_presence_probability.values()))
    
    # if perturb_rewards == 0:
    # new_S2D2_attacker_allocations, new_S2D2_attacker_best_response_strategies \
    #     = S2D2_attacker_best_response_to_baseline_defender(neighborhoods, num_attackers, 
    #         S2D2_all_attacker_strategies, baseline_defender_presence_probability, delta)
        # s2d2_att_to_baseline_def_time = time.time() - s2d2_att_to_baseline_def_time
        #############
    
    # time_3 = time.time()
    # baseline_sampled_defender_strategies = [decide_defender_strategy(_neighborhood, 
    #         add_boolean_indexing(new_S2D2_attacker_best_response_strategies[i]), [], [], delta)[0]
    #             for i, _neighborhood in enumerate(neighborhoods)]
    # time_3 = time.time() - time_3

    # baseline_defender_utilities_baseline_attacker = []
    # S2D2_defender_utilities_baseline_attacker = []

    # baseline_defender_utilities_S2D2_attacker = []
    S2D2_defender_utilities_S2D2_attacker = []

    for _ in range(num_samples):
        
        # time_1  = time.time()
        # baseline_defended_neighborhoods = sample_defender_allocations(baseline_defender_presence_probability)
        # time_1 = time.time() - time_1
        # time_1 = round(time_1, 2)
        # time_3 += time_1 #adding the defense strategies + sampling time
        
        time_4 = time.time()
        S2D2_defended_neighborhoods = sample_defender_allocations(S2D2_defender_presence_probability)
        
        S2D2_sampled_defender_strategies = S2D2_sample_mixed_defense_strategies(S2D2_mixed_defender_strategies, 
                                                                          S2D2_mixed_defender_strategies_probabs)
        time_4 = time.time() - time_4
        # time_4 = round(time_4, 2)
         
        # PLOT 1:
        # baseline_def_baseline_att_def_utility = []
        # baseline_def_baseline_att_def_utility = calculate_sampled_utility(neighborhoods,
            # baseline_attacker_allocation, baseline_defended_neighborhoods, 
            # [remove_boolean_indexing(x) for x in baseline_attacker_strategies], baseline_defender_strategies)
        
        # baseline_defender_utilities_baseline_attacker.append([time_1] + list(baseline_def_baseline_att_def_utility))
        # S2D2_defender_utilities_baseline_attacker.append(S2D2_calculate_sampled_defender_utility(neighborhoods,
        #     baseline_attacker_allocation, baseline_attacker_strategies, 
        #     S2D2_sampled_defender_strategies, S2D2_defended_neighborhoods))

        # PLOT 2:
        # baseline_defender_utilities_S2D2_attacker.append(calculate_sampled_defender_utility(neighborhoods,
        #     S2D2_attacker_allocation, baseline_defended_neighborhoods, S2D2_attacker_strategies, 
        #     baseline_defender_strategies))
        
        # if perturb_rewards > 0:
        #     new_S2D2_attacker_allocations, new_S2D2_attacker_best_response_strategies \
        #     = S2D2_attacker_best_response_to_baseline_defender(neighborhoods, num_attackers, 
        #             S2D2_all_attacker_strategies, baseline_defender_presence_probability, delta)
        
        # baseline_def_s2d2_att_def_utility = calculate_sampled_utility(neighborhoods,
        #     new_S2D2_attacker_allocations, baseline_defended_neighborhoods, new_S2D2_attacker_best_response_strategies, 
        #     baseline_sampled_defender_strategies)
        
        # baseline_defender_utilities_S2D2_attacker.append([time_1] + list(baseline_def_s2d2_att_def_utility))
        #the baseline defense strategies are run based on S2D2 attackers
        
        s2d2_to_s2d2_utilities = S2D2_calculate_sampled_utility(neighborhoods, 
            new_S2D2_attacker_allocations, new_S2D2_attacker_best_response_strategies, 
            S2D2_sampled_defender_strategies, S2D2_defended_neighborhoods)
        
        # s2d2_to_s2d2_utilities = S2D2_calculate_sampled_utility(neighborhoods, 
        #     S2D2_attacker_allocation, S2D2_attacker_strategies, S2D2_sampled_defender_strategies, S2D2_defended_neighborhoods)
        
        S2D2_defender_utilities_S2D2_attacker.append([time_4] + list(s2d2_to_s2d2_utilities))
        
    return [
            [0, np.array([])], 
            [0, np.array([])]
        ],[
            [0, np.array([])],
            [S2D2_all_results[city_name][B]["runtime"], np.array(S2D2_defender_utilities_S2D2_attacker)]
        ]

def run_cross_compare_single_city(city_name, s2d2_file_name_list, num_defenders, adr_ratio, attacker_payload, attacker_batteries,
                                  num_samples, load_dir=None, perturb_rewards=0):
    
    params = {
        "num_neighborhoods": compute_num_neighborhoods(num_defenders),
        "P": attacker_payload
    }
            
    overall_graph, neighborhoods = load_graph_data(city_name, params, load_dir=load_dir, perturb_rewards=0)
    
    delta = compute_delta(overall_graph, neighborhoods, attacker_payload)

    A = int(num_defenders * adr_ratio)
    
    for B in attacker_batteries:
            
        sampled_baseline_attacker_utility, sampled_s2d2_attacker_utility = cross_compare_methods(
            city_name, s2d2_file_name_list[load_dir], neighborhoods, overall_graph, A, 
            num_defenders, B, attacker_payload, delta, num_samples=num_samples, perturb_rewards=perturb_rewards)
        
        return {
            "city_name": city_name,
            "battery": B,
            "num_nodes": overall_graph.number_of_nodes(),
            "sampled_baseline_attacker_utility": sampled_baseline_attacker_utility,
            #here, 0: baseline attacker vs. baseline defender. 1: baseline attacker vs. S2D2 defender
            "sampled_s2d2_attacker_utility": sampled_s2d2_attacker_utility
            #here, 0: S2D2 attacker vs. baseline defender. 1: S2D2 attacker vs. S2D2 defender
            
            #inside each variable, there are two lists. 
            # The first one is the runtime and the second one is the utility
        }

if __name__ == "__main__":
    
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    cities = get_city_list(config['cities'])
    num_defenders = config['num_defenders']
    adr_ratio = config['adr_ratio']
    attacker_payload = config['attacker_payload']
    attacker_batteries = config['attacker_batteries']
    num_samples = config['num_samples']
    parallelize = config['parallelize']
    perturb_rewards = config['perturb_rewards']
    
    experiment_name = config['experiment_name']
    results_directory = os.path.join('results', experiment_name)
    os.makedirs(results_directory, exist_ok=True)
    
    # Decide filename to save on rewards distribution
    comparison_results_filename = {
        # 'cities_manual_annotation': f"results/{experiment_name}/cross_compare_manual_adr_{adr_ratio}_perturb_{perturb_rewards}.pkl",
        # 'cities_lognormal_dist': f"results/{experiment_name}/cross_compare_lognormal_adr_{adr_ratio}_perturb_{perturb_rewards}.pkl",
        'cities_zipf_dist': f"results/{experiment_name}/cross_compare_zipf_adr_{adr_ratio}_perturb_{perturb_rewards}.pkl"
    }
    
    s2d2_file_name_list = {
        # 'cities_manual_annotation': f"results/{experiment_name}/s2d2_recorded_manual_adr_{adr_ratio}.pkl",
        # 'cities_lognormal_dist': f"results/{experiment_name}/s2d2_recorded_lognormal_adr_{adr_ratio}.pkl",
        'cities_zipf_dist': f"results/{experiment_name}/s2d2_recorded_zipf_adr_{adr_ratio}.pkl"
    }
    
    for load_dir, file_name in comparison_results_filename.items():
        
        if os.path.exists(file_name):
            print(f"File '{file_name}' already exists. Skipping processing.")
            continue
        
        recorded_results = load_results(filename=file_name)
        
        if parallelize:
            num_chunks = 27 #NOTE: This is the number of chunks based on the machine configuration
            print(f"Parallelization Enabled. Processing Chunk of {num_chunks} cities at a time.")
            
            chunked_cities = list(chunker(cities, num_chunks))
            processed_results = []
            
            from joblib import Parallel, delayed
            for city_chunk in tqdm(chunked_cities):
                processed_results.append(Parallel(n_jobs=-1)(delayed(run_cross_compare_single_city)(city_name, 
                    s2d2_file_name_list, num_defenders, adr_ratio, attacker_payload, attacker_batteries,
                    num_samples, load_dir, perturb_rewards) for city_name in city_chunk))

            processed_results = [item for sublist in processed_results for item in sublist]

        else:
            print("Parallelization Disabled")
            processed_results = []
            for city_name in tqdm(cities):
                processed_results.append(run_cross_compare_single_city(city_name, s2d2_file_name_list, num_defenders, 
                        adr_ratio, attacker_payload, attacker_batteries, num_samples, load_dir, perturb_rewards))
            
        update_recorded_results(recorded_results, processed_results)
        save_results(recorded_results, filename=file_name)