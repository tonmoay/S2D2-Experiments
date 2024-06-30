import os
import yaml
import json
import pickle
import random
import networkx as nx
import numpy as np
    
def get_test_city_list():
    return ['City 1', 'City 2', 'City 3', 'City 4', 'City 5', 'City 6', 'City 7', 'City 8', 'City 9', 'City 10', 'City 11',
            'City 12', 'City 13', 'City 14', 'City 15', 'City 16', 'City 17', 'City 18', 'City 19', 'City 20', 'City 21', 'City 22', 
            'City 23', 'City 24', 'City 25', 'City 26', 'City 27', 'City 28', 'City 29', 'City 30', 'City 31', 'City 32', 'City 33', 
            'City 34', 'City 35', 'City 36', 'City 37', 'City 38', 'City 39', 'City 40', 'City 41', 'City 42', 'City 43', 'City 44', 
            'City 45', 'City 46', 'City 47', 'City 48', 'City 49', 'City 50', 'City 51', 'City 52', 'City 53', 'City 54', 'City 55',
            'City 56', 'City 57', 'City 58', 'City 59', 'City 60', 'City 61', 'City 62', 'City 63', 'City 64', 'City 65', 'City 66', 
            'City 67', 'City 68', 'City 69', 'City 70', 'City 71', 'City 72', 'City 73', 'City 74', 'City 75', 'City 76', 'City 77',
            'City 78', 'City 79', 'City 80'] #You may edit the city names, e.g., remove to run the code faster


def get_city_list(type=None):
    
    assert type is not None, "Please specify a type."
    
    function_map = {
        'test_city_list': get_test_city_list
    }
    
    if type not in function_map:
        raise ValueError(f"Invalid type: {type}")
    
    return function_map[type]()

def load_city(city_name):
    """
    Load a single city.
    """
    city_name = city_name.replace(',', '').replace(' ', '_')
    # Define the directory and file names
    directory = f"data/raw_city_data/{city_name}"
    nodes_file = os.path.join(directory, "nodes.geojson")
    edges_file = os.path.join(directory, "edges.geojson")
    
    with open(nodes_file, 'r') as f:
        nodes_gdf = json.load(f)
        
    with open(edges_file, 'r') as f:
        edges_gdf = json.load(f)
    
    return nodes_gdf, edges_gdf

def scale_reward(reward, scale_factor):
    assert scale_factor > 0, "Scale factor must be positive"
    
    if scale_factor == 1:
        return reward
    elif scale_factor == 2:
        return 2 ** reward
    elif scale_factor == 3:
        return 10 ** reward

    ValueError(f"Invalid scale factor: {scale_factor}")

def load_manual_rewards(city_name):
    
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    reward_scale_factor = config['reward_scale_factor']
    
    city_name = city_name.replace(',', '').replace(' ', '_')
    directory = f"data/raw_city_data/{city_name}"
    with open(f'{directory}/shapes.json', 'r') as f:
        data = json.load(f)

    rewards = {}
    
    assert len(data['annotations']) > 0, "No annotations found."
    
    for annotation in data['annotations']:
        osmids = annotation['nodes']
        
        if len(osmids) == 0:
            continue
        
        reward = float(annotation['value'])
        # Randomly sample one osmid
        random_osmid = random.choice(osmids)
        
        #assign reward to the random osmid
        rewards[random_osmid] = scale_reward(reward, reward_scale_factor)
        
        # for osmid in osmids:
        #     rewards[osmid] = reward   
    
    return rewards

def get_folder_name(num_neighborhoods, P):
    """
    Generate the folder name based on the parameters.
    """
    # return f"nbh_{num_neighborhoods}_P_{P}"
    return f"nbh_{32}_P_{4}" #for manual annotation suppl.

def save_graph_data(city_name, G, neighborhoods, params, save_dir=None):
    
    assert save_dir is not None, "Please specify a save directory."
    
    """
    Save the overall graph and neighborhoods to the specified folder structure.
    """
    
    city_dir = city_name.replace(',', '').replace(' ', '_')
    params_dir = get_folder_name(**params)

    save_dir = os.path.join('data', save_dir, city_dir, params_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Package everything into a dictionary
    data = {
        "params": params,
        "graph": G,
        "neighborhoods": neighborhoods
    }
    
    # Serialize and save the data using pickle
    with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

def perturb_graph_data(G, neighborhoods, perturb_rewards_percent=0):    
    # assert perturb_rewards_percent > 0, "Perturb rewards percent must be a positive number."
    
    _G = G.copy()
    _neighborhoods = [neighborhood.copy() for neighborhood in neighborhoods]
    
    # Assign rewards to each node in the neighborhood
    for i, neighborhood in enumerate(_neighborhoods):
        
        for node in neighborhood:
            
            if _G.nodes[node]['reward'] == 0:
                continue
            
            current_reward = _G.nodes[node]['reward']
            current_penalty = _G.nodes[node]['penalty']
            # perturbed_reward = current_reward * (1 + np.random.uniform(-perturb_rewards_percent, perturb_rewards_percent))
            perturbed_reward = current_reward * (1 + np.random.normal(0, perturb_rewards_percent))
            perturbed_penalty = current_penalty #* (1 + np.random.normal(0, perturb_rewards_percent))
            # perturbed_penalty = current_penalty #* (1 + np.random.uniform(-perturb_rewards_percent, perturb_rewards_percent))
            
            _G.nodes[node]['reward'] = perturbed_reward
            _G.nodes[node]['penalty'] = perturbed_penalty
            
            neighborhood.nodes[node]['reward'] = perturbed_reward
            neighborhood.nodes[node]['penalty'] = perturbed_penalty
    
    return _G, _neighborhoods

def load_graph_data(city_name, params, load_dir=None, perturb_rewards=0):
            
    assert load_dir is not None, "Please specify a load directory."
    
    """
    Load the overall graph and neighborhoods from the specified folder structure.
    """
    
    city_dir = city_name.replace(',', '').replace(' ', '_')
    params_dir = get_folder_name(**params)

    load_dir = os.path.join('data', load_dir, city_dir, params_dir)
    
    # Check if the data file exists
    data_file = os.path.join(load_dir, "data.pkl")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found in {load_dir}")
    
    # Load the data using pickle
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    # Extract the graph and neighborhoods from the loaded data
    G = data["graph"]
    neighborhoods = data["neighborhoods"]
    
    # Perturb the rewards
    if perturb_rewards > 0:
        return perturb_graph_data(G, neighborhoods, perturb_rewards)
    
    return G, neighborhoods

def compute_num_neighborhoods(num_defenders):
    
    assert isinstance(num_defenders, int), "num_defenders must be an integer"
    
    return 8 * num_defenders #NOTE: The integer value is hardcoded here

def compute_delta(overall_graph, neighborhoods, att_payload):
    
    node_rewards = nx.get_node_attributes(overall_graph, 'reward').values()
    num_neighborhoods = len(neighborhoods)
    
    node_rewards = np.sort(np.array(list(node_rewards)))
    
    value = node_rewards[-num_neighborhoods*att_payload-1]
    
    # Check if the value is 0 and if there is a non-zero value
    if value == 0:
        # Find the first value in the sorted array that is greater than 0
        non_zero_values = node_rewards[node_rewards > 0]
        if non_zero_values.size > 0:
            value = non_zero_values[0]  # Take the first non-zero value
        else:
            value = 0  # If there are no non-zero values, return 0

    return value
    
    # return np.quantile(np.array(list(node_rewards)), (1 - 1 / num_neighborhoods / att_payload))

def save_results(data, filename=None):
    assert filename is not None, "Please specify a filename."
    """
    Save the results data to a file.
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
def load_results(filename=None):
    assert filename is not None, "Please specify a filename."
    """
    Load the results data from a file.
    """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return {}

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def update_recorded_results(recorded_results, processed_results):
    for city_result in processed_results:
        if city_result is not None:
            city_name = city_result["city_name"]
            B = city_result["battery"]
            if city_name not in recorded_results:
                recorded_results[city_name] = {}
            if B not in recorded_results[city_name]:
                recorded_results[city_name][B] = {}
            recorded_results.update({city_name: {B: city_result}})
