import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch
class CustomSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, add_self_loops=False):
        super(CustomSAGEConv, self).__init__(aggr='sum')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        # Shared Transformation for both central node and neighbors
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
    def forward(self, x, edge_index):
        # ADDITIVE opt-in: memory-scalable aggregation for very large graphs.
        # Default (use_spmm_aggr=False) keeps the original PyG propagate path
        # byte-for-byte. The spmm path computes the identical sum-aggregation
        # (out[i] = sum_{j: j->i} x[j]) via a sparse-dense matmul, avoiding the
        # [num_edges, dim] message tensor that OOMs on million-edge graphs.
        if getattr(self, 'use_spmm_aggr', False):
            return self._forward_spmm(x, edge_index)

        # Add self-loops to edge_index
        if(self.add_self_loops):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute the normalization similar to the GCN implementation. Commenting out to only mimic FINDER layers for now
        #row, col = edge_index
        #deg = degree(row, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Aggregate neighbor features
        out = self.propagate(edge_index, x=x)
        #pyg.MessagePassing.aggregation('add', x, edge_index, dim_size=x.size(0))

        # Apply the shared transformation
        out = self.lin(out)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def _forward_spmm(self, x, edge_index):
        """Sum-aggregation via sparse-dense matmul (no [E, dim] materialization).

        Equivalent to propagate(aggr='sum', message=x_j): with PyG default flow,
        message x_j is the source feature, aggregated at the target node
        (edge_index[1]). So out = A @ x where A[target, source] = 1.
        """
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        idx = torch.stack([dst, src], dim=0)                       # rows=target, cols=source
        val = torch.ones(idx.size(1), dtype=x.dtype, device=x.device)
        A = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        out = torch.sparse.mm(A, x)                                # [n, dim], true SpMM
        out = self.lin(out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
        


class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6, add_self_loops=True):
        super(CustomGATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Define the weights matrix for the linear transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Define the weights matrix for the attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # Optional: Add self-loops to the adjacency matrix
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transformation to compute features for each node
        x = self.lin(x)

        # Start propagating messages
        return self.propagate(edge_index, x=x, size=None)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_pair = torch.cat([x_i, x_j], dim=-1)
        
        # Apply a single linear transformation to every edge
        alpha = (x_pair * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out
