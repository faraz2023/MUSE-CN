import networkx as nx




def prep_G_for_tensor(path):
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    # print(f"G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    # remove all nodes with degree 0
    G.remove_nodes_from([node for node in G.nodes() if G.degree(node) == 0])
    # print(f"After removing nodes with degree 0, G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    # relable nodes such that they are 0, n-1 with no gaps
    G = nx.convert_node_labels_to_integers(G)
    # print(f"After relabeling, G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G
