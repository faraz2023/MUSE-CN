import torch
import wandb
import sys
import random
import numpy as np
import torch.nn as nn
import logging
import os
import pandas as pd
import networkx as nx
from collections import deque
import json


def find_best_model_path(exp_path):
    levels_dir = os.path.join(exp_path, "levels")
    best_model_path = None
    best_vc = float('inf')

    for level_dir in os.listdir(levels_dir):
        level_log_path = os.path.join(levels_dir, level_dir, "level_log.json")
        with open(level_log_path, "r") as f:
            level_log = json.load(f)

        for entry in level_log['checkpoint_vcs']:
            if entry['vc'] < best_vc:
                best_vc = entry['vc']
                best_model_path = os.path.join(levels_dir, level_dir, "models", f"model_iter_{entry['iteration']}.ckpt")

    return best_model_path


def calc_pairwise_connectivity(G, normalize=True, total_nodes=None):
    connected_pairs = 0
    if(total_nodes == None):
        total_nodes = G.number_of_nodes()

    for component in nx.connected_components(G):
        if len(component) == 1:
            continue
        size = len(component)
        connected_pairs += size * (size - 1) // 2

    # print("Connected pairs: ", connected_pairs)

    if normalize:
        total_possible_pairs = total_nodes * (total_nodes - 1) // 2
        connected_pairs = connected_pairs / total_possible_pairs


    return connected_pairs


def calc_ANC_score(G, sol, conn_type='pairwise_connectivity', return_scores=False, normalize=True, step_size=1):
    # remove nodes in sol from G one by one and calculate the connectivity score at each step
    # then get the area under the curve of the score as a function of the number of nodes removed
    conn_funct = None
    G = G.copy()
    n = G.number_of_nodes()
    if conn_type == 'pairwise_connectivity':
        conn_funct = lambda G: calc_pairwise_connectivity(G, total_nodes=n, normalize=normalize)
    else:
        raise ValueError(f"Unsupported connectivity type: {conn_type}")

    scores = [conn_funct(G)]
    for i in range(0, len(sol), step_size):
        step_end = min(i+step_size, len(sol))
        nodes_to_remove = sol[i:step_end]
        G.remove_nodes_from(nodes_to_remove)
        score = conn_funct(G)
        scores.append(score)
    
    # integrate the scores to get the area under the curve
    ANC_score = np.trapz(scores, dx=step_size)

    if normalize:
        ANC_score = ANC_score / len(sol)

    if return_scores:
        return ANC_score, scores
    else:
        return ANC_score
        


def load_real_world_graph(edge_list_path, node_attributes_path=None):
    # Load the graph from the edge list
    G = nx.read_edgelist(edge_list_path, nodetype=int)
    
    # If node attributes are provided, load and add them to the graph
    node_attrs_df = None
    if node_attributes_path:
        node_attrs_df = pd.read_csv(node_attributes_path, index_col=0) ### NOTE THAT FIRST COLUMN IS INDEX (node id)
        # nx.set_node_attributes(G, node_attrs.to_dict('index'))
    
    return G, node_attrs_df


def get_activation_function(act_name):
    if(act_name.lower() == 'relu'):
        return nn.ReLU()
    elif(act_name.lower() == 'tanh'):
        return nn.Tanh()
    elif(act_name.lower() == 'prelu'):
        return nn.PReLU()
    else:
        raise NotImplementedError("Activation function not implemented not supported!")



def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    if wandb.run is not None:
        wandb.finish()
    sys.exit(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("cpu") #MPS Is very slow for some reason # mps also not supported for some pyg helpers
    else:
        return torch.device("cpu")
    



def bfs_subgraph(G: nx.Graph, v, part_size, node_attributes=None, return_anchor_node_id=False) -> (nx.Graph, np.ndarray):
    # If the part_size is less than 1 or exceeds the size of the graph, raise an error.
    if not (1 <= part_size <= len(G)):
        raise ValueError("part_size must be between 1 and the size of the graph")

    visited = set()  # Nodes that have been visited
    queue = deque([v])  # Queue for BFS
    subgraph_nodes = set()  # Nodes to be added to the subgraph

    while queue and len(subgraph_nodes) < part_size:
        vertex = queue.popleft()

        if vertex not in visited:
            visited.add(vertex)
            subgraph_nodes.add(vertex)

            # Add neighboring nodes to the queue for BFS
            neighbors = list(G.neighbors(vertex))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)

    # Create subgraph using the selected nodes
    H = G.subgraph(subgraph_nodes).copy()

    # Create a mapping to re-label nodes from 0 to part_size-1
    mapping = {node: idx for idx, node in enumerate(H.nodes())}
    H = nx.relabel_nodes(H, mapping)
    

    subgraph_node_features = None
    if(node_attributes):
        node_indices, node_features = node_attributes[0], node_attributes[1]
        # Convert the mapping dictionary to two numpy arrays
        old_indices, new_ordering = np.array(list(mapping.keys())), np.array(list(mapping.values()))
        # Get the positions of old indices in node_indices
        positions = np.nonzero(np.isin(node_indices, old_indices))[0]
        # Extract and reorder the node features using numpy indexing
        subgraph_node_features = node_features[positions][np.argsort(new_ordering)]
    
        #subgraph_node_features to float32
        subgraph_node_features = subgraph_node_features.astype(np.double)

        if(return_anchor_node_id):
            return H, subgraph_node_features, mapping[v]
        else:
            return H, subgraph_node_features

    else:
        if(return_anchor_node_id):
            return H, mapping[v]
        else:
            return H