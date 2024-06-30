import os
import time
import yaml
import numpy as np
import networkx as nx
from tqdm import tqdm
from algorithms.s2d2_multi_drone import optimize_multi_drone_solution_piecewise
from utils import compute_delta, compute_num_neighborhoods
from utils import load_results, save_results, load_graph_data, chunker, update_recorded_results
from utils import get_city_list

def run_strategy(neighborhoods, num_attack_drones, num_defense_drones, attacker_battery, 
                 attacker_payload, _delta, utility_samples=None, strategies=None):

    info_dict = {
        "neighborhoods": neighborhoods,
        "attacker_battery": attacker_battery,
        "attacker_payload": attacker_payload,
        "_delta": _delta,
        "optimization": utility_samples is not None,
    }

    max_attacker_rewards = [0]*len(neighborhoods)
    for i, _G in enumerate(neighborhoods):        
        rewards = np.array(list(nx.get_node_attributes(_G, 'reward').values()))
        if len(rewards) < attacker_payload:
            max_attacker_rewards[i] = rewards.sum() #NOTE: edge case when the number of nodes is less than the payload
            print(f"WARNING: Number of nodes in neighborhood {i} is less than the payload.")
        else:
            max_attacker_rewards[i] = np.partition(rewards, -attacker_payload)[-attacker_payload:].sum()
    
    num_samples = 2
    
    result_dict = optimize_multi_drone_solution_piecewise(num_samples=num_samples, num_targets=len(neighborhoods), 
                    num_attack_resources=num_attack_drones, num_defense_resources=num_defense_drones, 
                    max_attacker_rewards=max_attacker_rewards, info_dict=info_dict, 
                    utility_samples=utility_samples, strategies=strategies)
    
    return result_dict #result_dict["total_attacker_utility"], result_dict["total_defender_utility"]

def run_s2d2_single_city(city_name, num_defenders, adr_ratio, attacker_payload, attacker_batteries, recorded_results, load_dir=None):
    
    params = {
        "num_neighborhoods": compute_num_neighborhoods(num_defenders),
        "P": attacker_payload
    }
            
    overall_graph, neighborhoods = load_graph_data(city_name, params, load_dir=load_dir)
    
    delta = compute_delta(overall_graph, neighborhoods, attacker_payload)

    A = int(num_defenders * adr_ratio)
    
    for B in attacker_batteries:
        
        if city_name in recorded_results and B in recorded_results[city_name]:
            _utility_samples = recorded_results[city_name][B]["result_dict"]["utility_samples"]
            _strategies = recorded_results[city_name][B]["result_dict"]["strategies"]
        else:
            _utility_samples = None
            _strategies = None
        
        start = time.time()
        result = run_strategy(neighborhoods, num_attack_drones=A, num_defense_drones=num_defenders, 
            attacker_battery=B, attacker_payload=attacker_payload, _delta=delta, 
            utility_samples=_utility_samples, strategies=_strategies)
        req_time = time.time() - start

        if city_name not in recorded_results:
            recorded_results[city_name] = {}
        
        if B not in recorded_results[city_name]:
            recorded_results[city_name][B] = {}
        
        updated_res = {
            "city_name": city_name,
            "result_dict": result,
            "num_nodes": overall_graph.number_of_nodes(),
            "num_edges": overall_graph.number_of_edges(),
            "num_attackers": A,
            "num_defenders": num_defenders,
            "adr_ratio": adr_ratio,
            "battery": B
        }
        
        if city_name in recorded_results and B in recorded_results[city_name] and "runtime" in recorded_results[city_name][B]:
            updated_res["runtime"] = req_time + recorded_results[city_name][B]["runtime"]
        else:
            updated_res["runtime"] = req_time

        # recorded_results[city_name][B] = updated_res
        return updated_res

def run_s2d2_experiment(cities, file_name_list, num_defenders, adr_ratio, attacker_payload, attacker_batteries, parallelize=False):
    
    for load_dir, file_name in file_name_list.items():
        
        if os.path.exists(file_name):
            print(f"File '{file_name}' already exists. Hence, utilities are computed. Setting the parallel to False for optimization algorithm to work.")
            parallelize = False #force set to False if utilities are already computed.
        
        recorded_results = load_results(filename=file_name)
        
        if parallelize:
            num_chunks = 20 #NOTE: This is the number of chunks based on the machine configuration
            
            print(f"Parallelization Enabled. Processing Chunk of {num_chunks} cities at a time.")
            
            chunked_cities = list(chunker(cities, num_chunks))
            processed_results = []
                        
            from joblib import Parallel, delayed
            for city_chunk in tqdm(chunked_cities):  
                processed_results.append(Parallel(n_jobs=-1)(delayed(run_s2d2_single_city)(city_name, num_defenders, adr_ratio, attacker_payload, 
                                    attacker_batteries, recorded_results, load_dir) for city_name in city_chunk))
            processed_results = [item for sublist in processed_results for item in sublist]
        else:
            print("Parallelization Disabled")
            processed_results = []
            for city_name in tqdm(cities):
                processed_results.append(run_s2d2_single_city(city_name, num_defenders, adr_ratio, attacker_payload, 
                            attacker_batteries, recorded_results, load_dir))
        
        update_recorded_results(recorded_results, processed_results)
        # Save the results after processing all cities
        save_results(recorded_results, filename=file_name)
        
        print(f"Finished Processing for {load_dir}")

if __name__ == "__main__":

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    cities = get_city_list(config['cities'])
    num_defenders = config['num_defenders']
    adr_ratio = config['adr_ratio']
    attacker_payload = config['attacker_payload']
    attacker_batteries = config['attacker_batteries']
    parallelize = config['parallelize']
    
    experiment_name = config['experiment_name']
    results_directory = os.path.join('results', experiment_name)
    os.makedirs(results_directory, exist_ok=True)
    
    # Decide filename to save on rewards distribution
    file_name_list = {
        # 'cities_manual_annotation': f"results/{experiment_name}/s2d2_recorded_manual_adr_{adr_ratio}.pkl",
        'cities_lognormal_dist': f"results/{experiment_name}/s2d2_recorded_lognormal_adr_{adr_ratio}.pkl",
        'cities_zipf_dist': f"results/{experiment_name}/s2d2_recorded_zipf_adr_{adr_ratio}.pkl"
    }
    
    run_s2d2_experiment(cities, file_name_list, num_defenders, adr_ratio, attacker_payload, attacker_batteries, parallelize)