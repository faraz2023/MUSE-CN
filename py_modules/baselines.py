import networkx as nx
from collections import defaultdict
import random
import numpy as np
import torch

def agile_HDA(G, B):

    degrees = dict(G.degree())
    degree_buckets = defaultdict(set)
    max_degree = 0
    remaining_nodes = set(G.nodes)

    for vertex, degree in degrees.items():
        degree_buckets[degree].add(vertex)
        max_degree = max(max_degree, degree)

    # Initialize the ordering list
    ordering = []
    for _ in range(B):
        # Find the largest non-empty degree bucket
        for degree in range(max_degree, -1, -1):
            if degree_buckets[degree]:
                break
            else:
                degree_buckets[degree] = None
        if(degree == 0):
            break

        max_degree_vertex = degree_buckets[degree].pop()
        ordering.append(max_degree_vertex)

        degrees[max_degree_vertex] = -1
        # Update the degree of neighboring vertices
        for neighbor in list(G.neighbors(max_degree_vertex)):
            #if neighbour still exists in the network
            if degrees[neighbor] > 0:
                old_degree = degrees[neighbor]
                new_degree = old_degree - 1
                degrees[neighbor] = new_degree

                degree_buckets[old_degree].remove(neighbor)
                degree_buckets[new_degree].add(neighbor)

        # Remove the vertex from the graph
        remaining_nodes.remove(max_degree_vertex)
        G = G.subgraph(remaining_nodes)

    if(len(ordering) < B):
        ordering.extend(list(G.nodes())[:B - len(ordering)])
        
    return ordering



def collective_influence(graph, radius=2):
    """
    Calculate the Collective Influence (CI) of each node in the graph for a given radius.

    Parameters:
    graph (nx.Graph): The input NetworkX graph.
    radius (int): The radius for the CI computation.

    Returns:
    dict: A dictionary where keys are nodes and values are their CI scores.
    """
    degrees = dict(graph.degree())
    
    ci_scores = defaultdict(float)
    
    for node in graph.nodes():
        ball = nx.single_source_shortest_path_length(graph, node, cutoff=radius)
        boundary_nodes = [n for n, dist in ball.items() if dist == radius]
        
        ci_scores[node] = (degrees[node] - 1) * sum(degrees[neighbor] - 1 for neighbor in boundary_nodes)
    
    return dict(ci_scores)

def matrix_collective_influence(graph, radius=2):
    """
    Calculate the Collective Influence (CI) using PyTorch matrix operations with CUDA support.
    This implementation uses adjacency matrix powers to efficiently compute CI scores.
    
    Parameters:
    graph (nx.Graph): The input NetworkX graph
    radius (int): The radius for the CI computation
    
    Returns:
    dict: A dictionary where keys are nodes and values are their CI scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    A = nx.adjacency_matrix(graph).todense()
    A = torch.tensor(A, dtype=torch.float32, device=device)
    
    degrees = torch.sum(A, dim=1)
    n = len(degrees)
    
    A_r = torch.matrix_power(A, radius)
    k_minus_1 = degrees - 1
    
    ci_scores = {}
    nodes = list(graph.nodes())
    
    # Process in batches to avoid memory issues on large graphs
    batch_size = 1024
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        # Get boundary nodes for current batch
        boundary_mask = A_r[i:batch_end] > 0
        # Calculate sum of (k_j - 1) for boundary nodes
        boundary_sum = torch.mm(boundary_mask.float(), k_minus_1.unsqueeze(1))
        # Calculate CI scores for batch
        batch_scores = k_minus_1[i:batch_end] * boundary_sum.squeeze()
        
        # Store results
        for idx, score in enumerate(batch_scores.cpu().numpy()):
            ci_scores[nodes[i + idx]] = float(score)
    
    return ci_scores

def get_baseline_sol(G, remove_ratio, baseline):
    if isinstance(remove_ratio, int):
        budget = remove_ratio
    else:
        budget = int(G.number_of_nodes() * remove_ratio)
   
    if baseline == 'random':
        removed_nodes = random.sample(list(G.nodes), budget)
    elif baseline == 'HDA':
        removed_nodes = agile_HDA(G, budget)
    elif baseline == 'degree':
        removed_nodes = sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)[:budget]
    elif baseline == 'CI':
        ci_scores = matrix_collective_influence(G)  # Using the new matrix-based implementation
        removed_nodes = sorted(G.nodes, key=lambda x: ci_scores[x], reverse=True)[:budget]
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    return removed_nodes

