import yaml
import pandas as pd
from tqdm import tqdm
from algorithms.graph_coarsening import zipf_distribution, lognormal_distribution, city_graph_update_pipeline
from utils import load_city, save_graph_data, get_city_list
from sklearn.cluster import KMeans

def parse_gdf_format(nodes_gdf) -> pd.DataFrame:
    data = pd.DataFrame(index=[node['properties']['osmid'] for node in nodes_gdf['features']],
                        data={'x': [node['properties']['x'] for node in nodes_gdf['features']],
                              'y': [node['properties']['y'] for node in nodes_gdf['features']],
                              'reward': [node['properties']['reward'] for node in nodes_gdf['features']],
                              'penalty': [node['properties']['penalty'] for node in nodes_gdf['features']]})
    return data

def city_graph_partition(G, num_neighborhoods, nodes_gdf):
    # Apply METIS for graph partitioning
    # nxmetis.partition(G, num_neighborhoods)

    data = parse_gdf_format(nodes_gdf)
    # model = NearestNeighbors(n_neighbors=num_neighborhoods)
    model = KMeans(n_clusters=num_neighborhoods, random_state=0, n_init="auto")
    model.fit(data[['x', 'y']], sample_weight=data['reward']**2)
    data['neighborhood'] = model.labels_

    my_neighborhoods = [[] for _ in range(num_neighborhoods)]
    for node in G.nodes():
        my_neighborhoods[data.loc[node, 'neighborhood']].append(node)
    return None, my_neighborhoods

if __name__ == "__main__":
    
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    cities = get_city_list(config['cities'])
    num_defenders = config['num_defenders']
    attacker_payload = config['attacker_payload']
    
    # Decide on rewards distribution
    rewards_distribution_func = {
        # 'cities_manual_annotation': None,
        'cities_lognormal_dist': lognormal_distribution,
        'cities_zipf_dist': zipf_distribution
    }
    
    for save_dir, distribution_func in rewards_distribution_func.items():

        for city in tqdm(cities):
            nodes_data, edges_data = load_city(city)
            # Apply the graph generation function
            G, neighborhoods = city_graph_update_pipeline(city, nodes_data, edges_data, distribution_func, 
                    city_graph_partition, num_defenders=num_defenders, payload=attacker_payload)
            
            params = {
                "num_neighborhoods": len(neighborhoods),
                "P": attacker_payload
            }
            
            save_graph_data(city, G, neighborhoods, params, save_dir)
    
"""
############# Code for loading the data ###############
with open("path_to_data.pkl", "rb") as f:
    data = pickle.load(f)

G = data["graph"]
neighborhoods = data["neighborhoods"]
"""