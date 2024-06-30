import os
import geopandas as gpd
import osmnx as ox
from utils import get_test_city_list

"""
The job of this code is to download all geopkg data of the cities from OSM.
After that, it extracts the geojson node and edges and saves them locally
"""

def download_city_network(city_name):
    """
    Download the street network for the given city and save it as GPKG and GEOJSON (nodes & edges).
    """
    # Create a directory for the city
    directory = f"./data/raw_city_data/{city_name.replace(',', '').replace(' ', '_')}"
    os.makedirs(directory, exist_ok=True)

    # Download the network and save as GPKG
    G = ox.graph_from_place(city_name, network_type="drive")
    gpkg_path = os.path.join(directory, f"{city_name.replace(',', '').replace(' ', '_')}.gpkg")
    ox.save_graph_geopackage(G, filepath=gpkg_path)

    # Extract nodes & edges and save as GEOJSON
    nodes = gpd.read_file(gpkg_path, layer='nodes')
    edges = gpd.read_file(gpkg_path, layer='edges')
    nodes.to_file(os.path.join(directory, "nodes.geojson"), driver='GeoJSON')
    edges.to_file(os.path.join(directory, "edges.geojson"), driver='GeoJSON')

if __name__ == "__main__":
    # Download networks for all cities
    # cities = get_all_city_list()
    cities = get_test_city_list()
    for city in cities:
        download_city_network(city)
        print(f'Downloaded and saved {city}')