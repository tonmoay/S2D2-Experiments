import networkx as nx
from matplotlib import colors as mcolors
from functools import partial
import numpy as np
import random
from utils import compute_num_neighborhoods, load_manual_rewards

zipf_distribution = partial(np.random.zipf, a=2)
lognormal_distribution = partial(np.random.lognormal, mean=0, sigma=4)

def manual_reward_assignment(city_name, nodes_gdf):
    
    rewards_dict = load_manual_rewards(city_name)
    
    node_rewards_dict = {}
    
    for feature in nodes_gdf['features']:
        osmid = feature['properties']['osmid']
        if osmid in rewards_dict:
            feature['properties']['reward'] = rewards_dict[osmid]
            penalty_percentage = random.uniform(0.7, 0.9)
            feature['properties']['penalty'] = -int(penalty_percentage * rewards_dict[osmid])
        else:
            feature['properties']['reward'] = 0
            feature['properties']['penalty'] = 0
        
        node_rewards_dict[feature['properties']['osmid']] = (feature['properties']['reward'],feature['properties']['penalty'])
    
    missing_nodes = len([osmid for osmid in rewards_dict if osmid not in node_rewards_dict])
    if missing_nodes > 0:
        print(f"Number of missing nodes: {missing_nodes} out of {len(rewards_dict)}")
    
    return nodes_gdf, node_rewards_dict

def is_red(color):
    r, g, b = mcolors.hex2color(color)
    h, s, v = mcolors.rgb_to_hsv((r, g, b))
    return 0 <= h <= 0.1 or 0.9 <= h <= 1

def assign_rewards_and_penalties(nodes_gdf, num_important, distribution):
    num_nodes = len(nodes_gdf['features'])
    
    # Generate Zipf distributed values
    values = distribution(size=num_nodes)
    important_values = values >= np.sort(values)[-num_important]
    
    node_rewards_dict = {}
    
    # Assign the rewards and penalties to the neighborhoods
    for i, feature in enumerate(nodes_gdf['features']):
        feature['properties']['reward'] = int(values[i])
        feature['properties']['important'] = bool(important_values[i])
        penalty_percentage = random.uniform(0.7, 0.9)  # TODO: can be updated later
        feature['properties']['penalty'] = -int(penalty_percentage * values[i])
        
        node_rewards_dict[feature['properties']['osmid']] = (feature['properties']['reward'],feature['properties']['penalty'])
    
    return nodes_gdf, node_rewards_dict

def assign_colors_to_neighborhoods(neighborhoods, nodes_gdf):
    num_neighborhoods = len(neighborhoods)
    colors = []
    while len(colors) < num_neighborhoods:
        color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        if not is_red(color):
            colors.append(color)
        
    # Create a mapping from neighborhoods to colors
    neighborhood_colors = {i: colors[i] for i in range(num_neighborhoods)}
    # Create a mapping from nodes (osmids) to colors
    osmid_colors = {}
    # osmid_values = {}
    for i, neighborhood in enumerate(neighborhoods):
        for node in neighborhood:
            osmid_colors[node] = neighborhood_colors[i]

    # Add color information to GeoJSON data
    for feature in nodes_gdf['features']:
        osmid = feature['properties']['osmid']
        if osmid in osmid_colors:
            feature['properties']['color'] = osmid_colors[osmid]
            # feature['properties']['value'] = osmid_values[osmid]
                
    return nodes_gdf

def city_graph_update_pipeline(city_name, nodes_gdf, edges_gdf, rewards_distribution_func, 
            city_graph_partition_func, num_defenders, payload):
    
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    for row in nodes_gdf['features']:
        G.add_node(row['properties']['osmid'])
         
    # Add edges to the graph
    for row in edges_gdf['features']:
        G.add_edge(row['properties']['from'], row['properties']['to'])

    num_neighborhoods = compute_num_neighborhoods(num_defenders)
    num_important = num_neighborhoods * payload
    
    if rewards_distribution_func == None:
        nodes_gdf, reward_penalty_list = manual_reward_assignment(city_name, nodes_gdf)
    else:
        nodes_gdf, reward_penalty_list = assign_rewards_and_penalties(nodes_gdf, num_important, rewards_distribution_func)
        
    _, partitions = city_graph_partition_func(G, num_neighborhoods, nodes_gdf)

    # Convert the partitions list to a list of subgraphs
    neighborhoods = [G.subgraph(nodes) for nodes in partitions]

    # Assign rewards to each node in the neighborhood
    for i, neighborhood in enumerate(neighborhoods):
        
        for node in neighborhood:
            G.nodes[node]['reward'] = reward_penalty_list[node][0]
            G.nodes[node]['penalty'] = reward_penalty_list[node][1]
    
    return G, neighborhoods
