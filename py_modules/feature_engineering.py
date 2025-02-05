import numpy as np
import networkx as nx
# from community import community_louvain
import os 
import pandas as pd
# encoding=utf8
import numpy as np
import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
import time
import sys
from collections import defaultdict


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def _get_pagerank_scores( G, args={}):
    pagerank = nx.pagerank(G)
    scores = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes():
        scores[node] = pagerank[node]

    if(args.get('normalize', False)):
        scores = (scores - scores.mean()) / (scores.std() + 1e-10)
    return scores

def _get_degree_scores(G, args={}):
    degrees = dict(G.degree())
    scores = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes():
        scores[node] = degrees[node]

    if(args.get('normalize', False)):
        scores = (scores - scores.mean()) / (scores.std() + 1e-10)
    return scores

def _get_katz_scores(G, args={}):
    katz = nx.katz_centrality(G, **args)
    scores = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes():
        scores[node] = katz[node]
    return scores

def _get_betweenness_scores(G, args={}):
    betweenness = nx.betweenness_centrality(G, **args)
    scores = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes():
        scores[node] = betweenness[node]
    return scores


def _get_collective_influence_scores(G, args={}):
    radius = args.get('radius', 2)
    ci_scores = collective_influence(G, radius)
    scores = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes():
        scores[node] = ci_scores[node]


    if args.get('normalize', False):
        scores = (scores - scores.mean()) / (scores.std() + 1e-10)
    return scores

def collective_influence(graph, radius):
    """
    Calculate the Collective Influence (CI) of each node in the graph for a given radius.

    Parameters:
    graph (nx.Graph): The input NetworkX graph.
    radius (int): The radius for the CI computation.

    Returns:
    dict: A dictionary where keys are nodes and values are their CI scores.
    """
    # Precompute degrees of all nodes
    degrees = dict(graph.degree())
    
    # Dictionary to store collective influence
    ci_scores = defaultdict(float)
    
    for node in graph.nodes():
        # BFS to find nodes at distance `radius`
        ball = nx.single_source_shortest_path_length(graph, node, cutoff=radius)
        boundary_nodes = [n for n, dist in ball.items() if dist == radius]
        
        # Compute CI using the formula
        ci_scores[node] = (degrees[node] - 1) * sum(degrees[neighbor] - 1 for neighbor in boundary_nodes)
    
    return dict(ci_scores)

def get_node_centrality_nx(G, list_of_features=['degree',  'pagerank'], feature_args={}):
    """Get node features for graph G.
    Args:
        G: networkx graph
        list_of_features: list of features to compute
    Returns:
        node_features: numpy array of shape (num_nodes, num_features)
    """
    list_of_features = [f.lower() for f in list_of_features]
    node_features = []
    nodes = list(G.nodes)

    feature_methods = {
        'degree': _get_degree_scores,
        'katz': _get_katz_scores,
        'pagerank': _get_pagerank_scores,
        'betweenness': _get_betweenness_scores,
        'collective_influence': _get_collective_influence_scores,
    }

    for feature in list_of_features:
        feature_args = feature_args.get(feature, {})
        method = feature_methods.get(feature)
        values = method(G, feature_args)


        node_features.append(values)

    #node_features.append(nodes)
    #make copy so not to modify original list

    list_of_features = list_of_features[:]
    #list_of_features.append('node_id')

    attribute_matrix = np.array(node_features).T

    return attribute_matrix, list_of_features





class ProNE():
    def __init__(self, G, dimension):
        self.G = G
        self.dimension = dimension
        
        self.node_ids = [i for i in range(self.G.number_of_nodes())]
        self.G = self.G.to_undirected()
        self.node_number = self.G.number_of_nodes()
        
        '''
        adj_matrix = nx.adjacency_matrix(self.G)  # Get adjacency matrix in CSR format
        adj_matrix_lil = adj_matrix.tolil()  # Convert to LIL format for efficient modifications
        adj_matrix_lil.setdiag(0)  # Remove self-edges
        adj_matrix = adj_matrix_lil.tocsr() # Remove zero entries

        
        self.matrix0 = adj_matrix
        '''

        matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))

        for e in self.G.edges():
            if e[0] != e[1]:
                matrix0[e[0], e[1]] = 1
                matrix0[e[1], e[0]] = 1
        self.matrix0 = scipy.sparse.csr_matrix(matrix0)
        #print(self.matrix0.shape)

		
    def get_embedding_rand(self, matrix):
		# Sparse randomized tSVD for fast embedding
        t1 = time.time()
        l = matrix.shape[0]
        smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
        #print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)

        U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        #print('sparsesvd time', time.time() - t1)
        return U

    def get_embedding_dense(self, matrix, dimension):
        # get dense embedding via SVD
        t1 = time.time()
        U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
        U = np.array(U)
        U = U[:, :dimension]
        s = s[:dimension]
        s = np.sqrt(s)
        U = U * s
        U = preprocessing.normalize(U, "l2")
        #print('densesvd time', time.time() - t1)
        return U

    def pre_factorization(self, tran, mask):
        # Network Embedding as Sparse Matrix Factorization
        t1 = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, "l1")
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / (neg.sum() + 1e-10)

        neg = scipy.sparse.diags(neg, format="csr")
        neg = mask.dot(neg)
        #print("neg", time.time() - t1)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1
        features_matrix = self.get_embedding_rand(F)
        return features_matrix

    def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
        # NE Enhancement via Spectral Propagation
        #print('Chebyshev Series -----------------')
        t1 = time.time()

        if order == 1:
            return a

        A = sp.eye(self.node_number) + A
        DA = preprocessing.normalize(A, norm='l1')
        L = sp.eye(self.node_number) - DA

        M = L - mu * sp.eye(self.node_number)

        Lx0 = a
        Lx1 = M.dot(a)
        Lx1 = 0.5 * M.dot(Lx1) - a

        conv = iv(0, s) * Lx0
        conv -= 2 * iv(1, s) * Lx1
        for i in range(2, order):
            Lx2 = M.dot(Lx1)
            Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
            #         Lx2 = 2*L.dot(Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, s) * Lx2
            else:
                conv -= 2 * iv(i, s) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
            del Lx2
            #print('Bessell time', i, time.time() - t1)
        mm = A.dot(a - conv)
        emb = self.get_embedding_dense(mm, self.dimension)
        return emb


def get_ProNE_emb(G, emb_size = 32,prone_step = 10, prone_theta = 0.5, prone_mu = 0.2):
    try:
        with HiddenPrints():
            prone_model = ProNE(G, emb_size)
            features_matrix = prone_model.pre_factorization(prone_model.matrix0, prone_model.matrix0)
            embeddings_matrix = prone_model.chebyshev_gaussian(prone_model.matrix0, features_matrix, prone_step, prone_mu, prone_theta)
            node_ids = prone_model.node_ids
            return embeddings_matrix, node_ids
    except Exception as e:
        print('Error in ProNE:', e)
        # print G stats
        print("G_nodes:", G.number_of_nodes())
        print("G_edges:", G.number_of_edges())
        print("G_density:", nx.density(G))
        # count number of isolated nodes
        print("Number of isolated nodes:", len(list(nx.isolates(G))))
        # print the isolated nodes
        print("Isolated nodes:", list(nx.isolates(G)))
        # print the connected components
        print("Connected components:", nx.number_connected_components(G))
        # print the largest connected component
        print("Largest connected component:", len(max(nx.connected_components(G), key=len)))

        # write the graph to a file with full information (inc. isolated nodes) as gml
        nx.write_gml(G, 'Z_G_error_causing.gml')
        nx.write_edgelist(G, 'Z_G_error_causing.edgelist')




        exit()


def get_ones_attrs(G, num_features=1):
    return np.ones((G.number_of_nodes(), num_features), dtype=np.double)
	



