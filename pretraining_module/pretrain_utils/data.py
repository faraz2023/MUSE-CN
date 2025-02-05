from pickle import HIGHEST_PROTOCOL
import torch
import numpy as np
import torch
import dgl
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
def preprocess(graph, no_self_loop=False):

    feat = graph.ndata["feat"]
    # labels = graph.ndata["label"]
    # node_assignment = graph.ndata["node_assignment"]
    # this line gets rid of the node_assignment added in the data loader function. (see how feat is added back to the graph)
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat
    # graph.ndata["label"] = labels
    # graph.ndata["node_assignment"] = node_assignment
    if not no_self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    else:
        graph = graph.remove_self_loop()
    graph.create_formats_()
    return graph

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def cross_validation_gen(y, k_fold=10):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_fold)
    train_splits = []
    val_splits = []
    test_splits = []

    for larger_group, smaller_group in skf.split(y, y):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=k_fold)
        train_split, val_split = next(iter(sub_skf.split(train_y, train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))
    
    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)

def load_data(data_path, no_self_loop=False):

    # read pt file

    g = torch.load(data_path)
    if isinstance(g, Data):
        x = g.x  
        edge_index = g.edge_index  
        src, dst = edge_index
        g = dgl.graph((src, dst), num_nodes=x.size(0))

        g.ndata['feat'] = x
        print("Converted pyg instance to dgl instance")



    g = preprocess(g, no_self_loop)
    # normalize graphs with discrete features
    # norm = StandardScaler()
    # norm.fit(g.ndata['feat'])
    # g.ndata['feat'] = torch.tensor(norm.transform(g.ndata['feat'])).float()
    g.ndata['feat'] = g.ndata['feat'].float()
    
    return g

