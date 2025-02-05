import torch
import torch.nn as nn
import torch.nn.functional as F
from py_modules.layers import CustomSAGEConv
from py_modules.utils import get_activation_function
from scipy.stats import truncnorm
import time
import numpy as np
from py_modules.utils import set_seed
import json
from copy import deepcopy

import torch_sparse

def custom_global_add_pool(x, batch, batch_size):
    out = torch.zeros(batch_size, x.size(1)).to(x.device)
    return out.scatter_add_(0, batch.unsqueeze(-1).repeat(1, x.size(1)), x)


class TransformerCrossAttentionBlock(nn.Module):
    """
    A single Transformer block (pre-LN style) with (self-)attention + feed-forward.
    You can adapt the code to do 'cross' attention by passing separate Q, K, V.
    """
    def __init__(self, 
                 embedding_size: int, 
                 num_heads: int, 
                 hidden_size: int, 
                 dropout: float, 
                 use_layer_norm: bool = True):
        super().__init__()

        # If you're strictly doing cross-attention to another sequence,
        # you’d pass those as K, V. If not, it acts like self-attention.
        self.attn = nn.MultiheadAttention(embed_dim=embedding_size, 
                                          num_heads=num_heads, 
                                          dropout=dropout, 
                                          batch_first=False)

        # Feed-forward layers
        # Often hidden_size is 2x or 4x the embedding_size (e.g. 4*embedding_size).
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),        # or GELU, etc.
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
            nn.Dropout(dropout),
        )

        # LayerNorm layers (pre-norm approach)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(embedding_size)
            self.ln2 = nn.LayerNorm(embedding_size)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, x, key_value=None):
        """
        x is [E, N, C] (num_encoders, num_nodes, channels) if batch_first=False
        (PyTorch MHA default).
        
        If you're truly doing cross-attention with a different sequence, pass in:
          key_value = (kv_tensor_of_shape_TKVxBxC)
        Otherwise, by default it uses x itself for K, V (self-attn).
        """
        # 1) Pre-LN then attention sub-layer
        x_ln = self.ln1(x)
        if key_value is None:
            # self-attention
            attn_output, _ = self.attn(x_ln, x_ln, x_ln)  
        else:
            # cross-attention to a different sequence
            attn_output, _ = self.attn(query=x_ln, 
                                       key=key_value, 
                                       value=key_value)

        x = x + attn_output

        x_ln = self.ln2(x)
        ff_output = self.ffn(x_ln)

        x = x + ff_output
        return x

class CrossAttentionModule(nn.Module):
    """
    Stacks multiple CrossAttentionBlocks. Each block can do self-attention
    or cross-attention if you pass in separate K,V.
    """
    def __init__(self, 
                 num_heads: int, 
                 dropout: float, 
                 use_layer_norm: bool, 
                 num_transformer_blocks: int, 
                 embedding_size: int, 
                 hidden_size: int):
        super().__init__()
        self.num_transformer_blocks = num_transformer_blocks

        # Build N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerCrossAttentionBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
                hidden_size=hidden_size,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_transformer_blocks)
        ])

    def forward(self, x):
        """
        x: [E, N, E] => [num_encoders, batch_size, embedding_size].
        Pass through N Transformer blocks, then return mean-pooled over T.
        """
        for block in self.blocks:
            x = block(x)  # (If truly cross-attending, pass key_value here)

        # Pool over the tokens dimension to get a single vector [B, E]
        # for each batch example.
        output = x.mean(dim=0)  # shape => [N, E]
        return output



# class CrossAttentionModule(nn.Module):
#     def __init__(self, num_heads, dropout, use_layer_norm, num_transformer_blocks, embedding_size):
#         super(CrossAttentionModule, self).__init__()
#         self.num_transformer_blocks = num_transformer_blocks

#         self.attention_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout)
#             for _ in range(num_transformer_blocks)
#         ])
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(embedding_size) if use_layer_norm else nn.Identity()
#             for _ in range(num_transformer_blocks)
#         ])

#     def forward(self, x):
#         # x is expected to be of shape [num_encoders, num_items, embedding_size]
#         for i in range(self.num_transformer_blocks):
#             attn_output, _ = self.attention_layers[i](x, x, x)
#             x = self.layer_norms[i](x + attn_output)  # Apply residual connection and layer norm
#         return x.mean(dim=0)  # Aggregate across encoders to get [num_items, embedding_size]


class MEGA_Encoder(nn.Module):
    def __init__(self, encoders_list, distilation_module_args, embedding_size, num_node_features=None):
        super(MEGA_Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_node_features = num_node_features
        
        self.encoders = nn.ModuleDict()
        total_encoder_embedding_size = 0
        for name, encoder_config in encoders_list.items():
            # Load config
            with open(encoder_config['config_file_path'], 'r') as f:
                curr_encoder_config = json.load(f)
            
            curr_encoder_type = curr_encoder_config['encoder_type']
            curr_encoder_args = curr_encoder_config['encoder_args']

            total_encoder_embedding_size += curr_encoder_args['embedding_size']
            if curr_encoder_type == 'FINDER_encoder_PyG':
                encoder = GraphSAGE_Encoder_PyG_FINDER_v2(**curr_encoder_args)
            
            checkpoint = torch.load(encoder_config['model_file_path'])
            encoder.load_state_dict(checkpoint)
            print("Successfully loaded pretrained encoder from: \n", encoder_config['model_file_path'])
            
            load_mode = encoder_config.get('load_mode', 'finetune')
            if load_mode == 'reset':
                encoder._initialize_weights()
                print("Successfully random initialized weights for encoder: ", name)
                for param in encoder.parameters():
                    param.requires_grad = True
            else:
                for param in encoder.parameters():
                    param.requires_grad = (load_mode == 'finetune')
            print("Encoder load mode: ", load_mode)
            print("--------------------------------")
            
            self.encoders[name] = encoder

        # Initialize distillation module
        distilation_type = distilation_module_args.get('distilation_type', 'mlp')
        self.distilation_type = distilation_type
        if distilation_type == 'mlp':
            layers = []
            input_size = total_encoder_embedding_size
            layer_sizes = [input_size] + distilation_module_args['distilation_layers'] + [embedding_size]
            activation = get_activation_function(distilation_module_args['activation'])

            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:  # No activation after last layer
                    layers.append(activation)

            self.distilation_module = nn.Sequential(*layers)
        elif distilation_type == 'cross_attention':
            distilation_args = deepcopy(distilation_module_args)
            distilation_args.pop('distilation_type')

            self.distilation_module = CrossAttentionModule(
                embedding_size=embedding_size,
                **distilation_args
            )
        else:
            raise ValueError(f"Invalid distilation type: {distilation_type}")
    def forward(self, data):
        embeddings = []
        global_projs = []
        
        for encoder in self.encoders.values():
            node_emb, global_proj = encoder(data)
            embeddings.append(node_emb)
            global_projs.append(global_proj)
        
        if self.distilation_type == 'mlp':
            embeddings = torch.cat(embeddings, dim=1)
            global_projs = torch.cat(global_projs, dim=1)
        elif self.distilation_type == 'cross_attention':
            embeddings = torch.stack(embeddings, dim=0)
            global_projs = torch.stack(global_projs, dim=0)
        else:
            raise ValueError(f"Invalid distilation type: {self.distilation_type}")
        
        # Apply distillation
        final_embeddings = self.distilation_module(embeddings)
        final_global = self.distilation_module(global_projs)

        
        # Normalize outputs
        final_embeddings = F.normalize(final_embeddings, p=2, dim=1)
        final_global = F.normalize(final_global, p=2, dim=1)
        
        return final_embeddings, final_global
    

class GraphSAGE_Encoder_PyG_FINDER_v2(nn.Module):
    def __init__(self, embedding_size=64, w_initialization_std=1, num_node_features=2, activation='relu', max_bp_iter=3, 
                    weight_init=False, inner_conv_normalize=False, seed=42):
        super(GraphSAGE_Encoder_PyG_FINDER_v2, self).__init__()

        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.act = get_activation_function(activation)
        self.num_node_features = num_node_features
        self.max_bp_iter = max_bp_iter
        self.weight_init = weight_init
        self.seed = seed
        self.inner_conv_normalize = inner_conv_normalize
        set_seed(seed)
        
        # Initialize layers
        self.lin = nn.Linear(self.num_node_features, self.embedding_size, bias=False)
        self.conv = CustomSAGEConv(self.embedding_size, self.embedding_size, normalize=self.inner_conv_normalize)
        
        # message passing linear layers used in FINDER
        self.mp_lin1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.mp_lin2 = nn.Linear(2*self.embedding_size, self.embedding_size, bias=False)
        
        if(self.weight_init):
            #print("Starting WI...")
            #start_time = time.time()
            self._initialize_weights()
            #print("WARNING: WEIGHT INIIT CAN LEAD TO NUMERICAL PROBLEMS!")
            #print("Time taken for weight initialization: ", time.time() - start_time)
            print("Weight initialization done!")
        else:
            print("No weight initialization for GraphSAGE_Encoder_PyG_FINDER_v2 encoder!")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.fmod(torch.normal(0, self.w_initialization_std, size=m.weight.size()), 2)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, CustomSAGEConv):
                # Initial weight of lin_l for SAGEConv
                m.lin.weight.data = torch.fmod(torch.normal(0, self.w_initialization_std, size=m.lin.weight.size()), 2)
        print("Successfully initialized weights for  GraphSAGE_Encoder_PyG_FINDER_v2 encoder!")

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, data):
        edge_index = data['edge_index']
        batch = data['batch']
        x = data['node_input']
        n2nsum_param = data.get('n2nsum_param', None)
        subgsum_param = data.get('subgsum_param', None)
        

        if n2nsum_param is not None and subgsum_param is not None:
            batch_size = subgsum_param['m']
            nodes_cnt = n2nsum_param['m']

        # since in pretrianing of the SSL model we may not have n2nsum_param and subgsum_param
        # we use the max batch value to determine the batch size
        else:
            batch_size = max(batch).item() + 1
            nodes_cnt = len(batch)


        # print("nodes_cnt: ", nodes_cnt)
        # print("batch_size: ", batch_size)
        # print('edge_index.shape: ', edge_index.shape)
        # print('x.shape: ', x.shape)
        # print('--------------------------------')

        if(x is None):
            #[node_cnt, num_node_features]
            x = torch.ones((nodes_cnt, self.num_node_features), dtype=torch.float)
            # sent to deivde of a paramter
            x = x.to(self.get_device())
        
        global_projection_input = torch.ones((batch_size, self.num_node_features), dtype=torch.float).to(self.get_device())

        #print("Time taken for data preparation: ", time.time() - start_time)
        x = self.act(self.lin(x))
        x = nn.functional.normalize(x, p=2, dim=1)
        #print("AVG x:", torch.mean(x))

        global_projection = self.act(self.lin(global_projection_input))
        global_projection = nn.functional.normalize(global_projection, p=2, dim=1)
        #print("AVG global_projection:", torch.mean(global_projection))

        lv = 0
        #print("Time taken for initial layer: ", time.time() - start_time)
        while lv < self.max_bp_iter:
            lv = lv + 1
            
            x_agg_lin = self.conv(x, edge_index)
            #print("lv: ", lv, "AVG node_linear: ", torch.mean(x_agg_lin))
            
            global_pool = custom_global_add_pool(x, batch, batch_size)
            #weird, but FINDER!
            global_lin = self.conv.lin(global_pool)
            #print("lv: ", lv, "AVG global_linear: ", torch.mean(global_lin))
            

            x_self_lin = self.mp_lin1(x)
            x_merged = torch.cat([x_agg_lin, x_self_lin], dim=1)
            x = self.act(self.mp_lin2(x_merged))

            global_projection_extra = self.mp_lin1(global_projection)
            globale_merged = torch.cat([global_lin, global_projection_extra], dim=1)
            global_projection = self.act(self.mp_lin2(globale_merged))

            x = torch.nn.functional.normalize(x, p=2, dim=1)
            global_projection = torch.nn.functional.normalize(global_projection, p=2, dim=1)
        
        #print("AVG final x:", torch.mean(x))
        #print("AVG final global_projection:", torch.mean(global_projection))
        return x, global_projection
    

class RandomEncoder(nn.Module):
    def __init__(self, embedding_size, mean=0, std=1, a=-2, b=2, seed=42):
        """
        Parameters:
        embedding_size (int): Dimension of the embedding
        mean (float): Mean of the truncated normal distribution
        std (float): Standard deviation of the truncated normal distribution
        a (float): Lower truncation bound (in terms of number of standard deviations away from the mean)
        b (float): Upper truncation bound (in terms of number of standard deviations away from the mean)
        """
        super(RandomEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.seed = seed
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b
        self.random_state = np.random.RandomState(seed)

    def forward(self, data):        
        n2nsum_param = data['n2nsum_param']
        subgsum_param = data['subgsum_param']
        batch = data['batch']
        
        batch_size = subgsum_param['m']
        nodes_cnt = n2nsum_param['m']
        # Generate values from a truncated normal distribution
        x = truncnorm.rvs(self.a, self.b, loc=self.mean, scale=self.std, 
                         size=(nodes_cnt, self.embedding_size),
                         random_state=self.random_state)


        # Convert values to a torch tensor
        x = torch.tensor(x, dtype=torch.float32)

        global_projection = custom_global_add_pool(x, batch, batch_size)
        
        return x, global_projection
    

class IdentityEncoder(nn.Module):
    def __init__(self, num_node_features=2):
        super(IdentityEncoder, self).__init__()
        self.num_node_features = num_node_features

    def forward(self, data):
        x = data['node_input']
        batch = data['batch']
        subgsum_param = data['subgsum_param']
        batch_size = subgsum_param['m']

        if x is None:
            nodes_cnt = data['n2nsum_param']['m']
            x = torch.ones((nodes_cnt, self.num_node_features), dtype=torch.float)
            x = x.to(self.get_device())

        global_projection = custom_global_add_pool(x, batch, batch_size)

        return x, global_projection

    def get_device(self):
        return next(self.parameters()).device