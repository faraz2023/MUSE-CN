import torch
import torch.nn as nn
from py_modules.utils import set_seed,get_activation_function
import torch_sparse
from py_modules.utils import set_seed
from copy import deepcopy



def get_normalization_function(normalization_str):
    if normalization_str == 'layer':
        return nn.LayerNorm
    elif normalization_str == 'batch':
        return nn.BatchNorm1d
    else:
        return nn.Identity

class MoE_DQN_simple(nn.Module):
    def __init__(self, experts, expert_params, decoder_args, seed=42):
        super(MoE_DQN_simple, self).__init__()
        
        self.seed = seed
        set_seed(seed)

        self.experts = nn.ModuleDict(experts)
        self.expert_params = expert_params
        self.decoder_args = decoder_args

        self.num_experts = len(experts)
        
        for expert_name, expert in self.experts.items():
            expert_load_mode = self.expert_params[expert_name]['load_mode']
            if expert_load_mode == 'freeze':
                for param in expert.parameters():
                    param.requires_grad = False
            elif expert_load_mode == 'finetune':
                for param in expert.parameters():
                    param.requires_grad = True
            else:
                raise ValueError(f"Invalid load mode for expert {expert_name}: {expert_load_mode}")

        self.decoder_type = decoder_args['decoder_type']

        if self.decoder_type == 'basic_mlp_weighted':
            raise ValueError("Basic MLP weighted decoder needs debugging")
            self.decoder_activation_str = decoder_args.get('activation', 'relu')
            self.decoder_dropout = decoder_args.get('dropout', 0.0)
            self.decoder_normalization = decoder_args.get('normalization', 'none')
            self.act = get_activation_function(self.decoder_activation_str)
            sum_embd_size = 0 
            for expert in self.experts.keys():
                sum_embd_size += self.expert_params[expert]['decoder_args']['embedding_size']

            num_layers = decoder_args.get('num_layers', 3)  # Default to 3 layers if not specified
            decoder_layer_sizes = [sum_embd_size]  # Input layer size
            
            # Calculate intermediate layer sizes with geometric progression
            for i in range(num_layers - 1):
                next_size = max(self.num_experts, decoder_layer_sizes[-1] // 2)  # Halve the size each time but don't go below num_experts
                decoder_layer_sizes.append(next_size)
            
            decoder_layer_sizes.append(self.num_experts)  # Output layer size = number of experts
            
            # Build the decoder layers
            layers = []
            for i in range(len(decoder_layer_sizes) - 1):
                layers.append(nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i + 1]))
                if i < len(decoder_layer_sizes) - 2:  # Don't add norm/activation/dropout after last layer
                    layers.append(self.act)
                    if self.decoder_normalization != 'none':
                        layers.append(get_normalization_function(self.decoder_normalization)(decoder_layer_sizes[i + 1]))
                    
                    if self.decoder_dropout > 0:
                        layers.append(nn.Dropout(self.decoder_dropout))

            self.decoder = nn.Sequential(*layers)

            # Initialize decoder weights
            for m in self.decoder.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = torch.fmod(torch.normal(0, decoder_args.get('w_initialization_std', 0.01), 
                                                          size=m.weight.size()), 2)
                    if m.bias is not None:
                        m.bias.data.zero_()

        elif self.decoder_type == 'mlp_with_dedicated_encoder':
            from py_modules.model_utils import get_encoder
            self.dedicated_encoder = get_encoder(decoder_args['encoder_type'], decoder_args['encoder_args'], seed,
                                        pretrained_encoder_path = None, pretrained_encoder_load_mode = None)
            
            self.decoder_activation_str = decoder_args.get('activation', 'relu')
            self.decoder_dropout = decoder_args.get('dropout', 0.0)
            self.decoder_normalization = decoder_args.get('normalization', 'none')
            self.decoder_embedding_size = decoder_args['encoder_args'].get('embedding_size')
            self.decoder_w_initialization_std = decoder_args.get('w_initialization_std', 0.01)
            self.decoder_reg_hidden = decoder_args.get('reg_hidden', 0)
            self.decoder_do_aux = decoder_args.get('auxilary_input', False)
            self.decoder_aux_dim = decoder_args.get('aux_dim', 0)
            self.decoder_rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size),2)
            self.act = get_activation_function(self.decoder_activation_str)

            if self.decoder_reg_hidden > 0:
                self.h1_weight = nn.parameter.Parameter(data=self.decoder_rand_generator(0, self.decoder_w_initialization_std, size=(self.decoder_embedding_size, self.decoder_reg_hidden)))
                self.h2_weight = nn.parameter.Parameter(data=self.decoder_rand_generator(0, self.decoder_w_initialization_std, size=(self.decoder_reg_hidden + self.decoder_aux_dim, self.num_experts)))
                self.last_w = self.h2_weight
            else:
                self.h1_weight = nn.parameter.Parameter(data=self.decoder_rand_generator(0, self.decoder_w_initialization_std, size=(2*self.decoder_embedding_size, self.decoder_reg_hidden)))
                self.last_w = self.h1_weight
                
            self.cross_product = nn.parameter.Parameter(data=self.decoder_rand_generator(0, self.decoder_w_initialization_std, size=(self.decoder_embedding_size, 1)))

                
        else:
            raise ValueError("Decoder type not supported")

    def train_forward(self, data, dtype=torch.float):
        if self.decoder_type == 'basic_mlp_weighted':
            expert_predictions = []
            expert_embeddings = []
            messages = []
            
            # Get predictions from each expert
            for expert in self.experts.keys():
                curr_data = deepcopy(data)
                expert_model = self.experts[expert]
                expert_params = self.expert_params[expert]

                if not expert_params['procedural_attrs']:
                    curr_data['node_input'] = None

                q_pred, cur_message_layer, action_embed = expert_model.train_forward(curr_data, return_action_embed=True, dtype=dtype)

                expert_predictions.append(q_pred)
                expert_embeddings.append(action_embed)
                messages.append(cur_message_layer)

            expert_embeddings = torch.stack(expert_embeddings)  # [num_experts, batch_size, emb_dim]
            expert_predictions = torch.stack(expert_predictions)  # [num_experts, batch_size, 1]
            messages = torch.stack(messages)
            
            # [num_experts, batch_size, emb_dim] -> [batch_size, num_experts * emb_dim]
            batch_size = expert_embeddings.size(1)
            emb_dim = expert_embeddings.size(2)
            expert_embeddings = expert_embeddings.transpose(0, 1)
            expert_embeddings = expert_embeddings.reshape(batch_size, -1)
            
            expert_weights = self.decoder(expert_embeddings)  # [batch_size, num_experts]
            expert_weights = torch.softmax(expert_weights, dim=1)  # [batch_size, num_experts]
            
            expert_weights = expert_weights.transpose(0, 1).unsqueeze(-1)  # [num_experts, batch_size, 1]
            
            weighted_pred = torch.sum(expert_predictions * expert_weights, dim=0)  # [batch_size, 1]

            # Return weighted prediction and average embedding
            return weighted_pred, messages.mean(dim=0)
        
        elif self.decoder_type == 'mlp_with_dedicated_encoder':
            expert_predictions = []

            for expert in self.experts.keys():
                curr_data = deepcopy(data)
                expert_model = self.experts[expert]
                expert_params = self.expert_params[expert]
                if not expert_params['procedural_attrs']:
                    curr_data['node_input'] = None
                q_pred, cur_message_layer, action_embed = expert_model.train_forward(curr_data, return_action_embed=True, dtype=dtype)
                expert_predictions.append(q_pred)

            expert_predictions = torch.stack(expert_predictions)
            expert_predictions = expert_predictions.squeeze(-1).transpose(0, 1) # [node_cnt, num_experts]


            action_select = data['action_select']
            aux_input = data['aux_input']

            cur_message_layer,  y_potential = self.dedicated_encoder(data)
            action_embed = torch_sparse.spmm(action_select['index'], action_select['value'],\
                    action_select['m'], action_select['n'], cur_message_layer)

            temp = torch.matmul(torch.unsqueeze(action_embed, dim=2),torch.unsqueeze(y_potential, dim=1))
            Shape = action_embed.size()
            embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product,[Shape[0],1]),\
                                                                    [Shape[0],Shape[1],1])),Shape)
            last_output = embed_s_a
            if self.decoder_reg_hidden > 0:
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = self.act(hidden)

            last_output = torch.concat([last_output, aux_input], 1)
            expert_weights = torch.matmul(last_output, self.last_w)
            expert_weights = torch.softmax(expert_weights, dim=1)
            

            weighted_pred = torch.sum(expert_predictions * expert_weights, dim=1)
            weighted_pred = weighted_pred.unsqueeze(-1)


            return weighted_pred, cur_message_layer
        else:
            raise ValueError("Decoder type not supported")
    
    def test_forward(self, data, return_embedding=False, dtype=torch.float):
        if self.decoder_type == 'basic_mlp_weighted':
            expert_predictions = []
            expert_embeddings = []
        
            # Get predictions from each expert
            for expert in self.experts.keys():
                curr_data = deepcopy(data)
                expert_model = self.experts[expert]
                expert_params = self.expert_params[expert]

                if not expert_params['procedural_attrs']:
                    curr_data['node_input'] = None


                pred, embedding = expert_model.test_forward(curr_data, return_embedding=True, dtype=dtype)
                expert_embeddings.append(embedding)
                expert_predictions.append(pred)

            expert_embeddings = torch.stack(expert_embeddings)
                
            # Stack predictions and apply softmax to weights
            expert_predictions = torch.stack(expert_predictions)  # [num_experts, node_cnt, 1]

            node_cnt = expert_embeddings.size(1)
            emb_dim = expert_embeddings.size(2)
            expert_embeddings = expert_embeddings.transpose(0, 1)
            expert_embeddings = expert_embeddings.reshape(node_cnt, -1)
            
            expert_weights = self.decoder(expert_embeddings)  
            expert_weights = torch.softmax(expert_weights, dim=1)  # [node_cnt, num_experts]
            
            # Transpose expert weights to match prediction dimensions and add extra dim for broadcasting
            expert_weights = expert_weights.transpose(0, 1).unsqueeze(-1)  # [num_experts, node_cnt, 1]
            
            # Calculate weighted average
            weighted_pred = torch.sum(expert_predictions * expert_weights, dim=0)  # [node_cnt, 1]

            if return_embedding:
                return weighted_pred, expert_embeddings
            return weighted_pred
        elif self.decoder_type == 'mlp_with_dedicated_encoder':

            expert_predictions = []
        
            # Get predictions from each expert
            for expert in self.experts.keys():
                curr_data = deepcopy(data)
                expert_model = self.experts[expert]
                expert_params = self.expert_params[expert]
                if not expert_params['procedural_attrs']:
                    curr_data['node_input'] = None
                pred, embedding = expert_model.test_forward(curr_data, return_embedding=True, dtype=dtype)
                expert_predictions.append(pred)

            expert_predictions = torch.stack(expert_predictions)  # [num_experts, node_cnt, 1]
            expert_predictions = expert_predictions.squeeze(-1).transpose(0, 1) # [node_cnt, num_experts]


            subgsum_param = data['subgsum_param']
            n2nsum_param = data['n2nsum_param']
            rep_global = data['rep_global']
            aux_input = data['aux_input']

            cur_message_layer,  y_potential = self.dedicated_encoder(data)

            rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                    rep_global['m'], rep_global['n'], y_potential)
            
            temp1 = torch.matmul(torch.unsqueeze(cur_message_layer, dim=2),torch.unsqueeze(rep_y, dim=1))
            Shape1 = cur_message_layer.size()
            embed_s_a_all = torch.reshape(torch.matmul(temp1, torch.reshape(torch.tile(self.cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)
            last_output = embed_s_a_all
            if self.decoder_reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = self.act(hidden)

            rep_aux = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                    rep_global['m'], rep_global['n'], aux_input)
            last_output = torch.concat([last_output, rep_aux], 1)
            expert_weights = torch.matmul(last_output, self.last_w)
            expert_weights = torch.softmax(expert_weights, dim=1)


            weighted_pred = torch.sum(expert_predictions * expert_weights, dim=1)
            weighted_pred = weighted_pred.unsqueeze(-1)



            if return_embedding:
                return weighted_pred, cur_message_layer
            return weighted_pred
        else:
            raise ValueError("Decoder type not supported")
        
    def forward(self, data, dtype=torch.float):
        if self.training:
            return self.train_forward(data, dtype=dtype)
        else:
            return self.test_forward(data, dtype=dtype)


class FINDER_DQN(nn.Module):
    def __init__(self, encoder, encoder_args, decoder_args, seed=42):
        super(FINDER_DQN, self).__init__()

        self.seed = seed
        set_seed(seed)
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size),2)

        self.embedding_size = decoder_args['embedding_size']
        self.w_initialization_std = decoder_args['w_initialization_std']
        self.reg_hidden = decoder_args['reg_hidden']
        self.aux_dim = decoder_args['aux_dim']


        self.activation_str = decoder_args['activation']
        self.act = get_activation_function(self.activation_str)
        
        # Use provided encoder or create based on the specified embedding method
        

        # Decoder part
        if self.reg_hidden > 0:
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, size=(self.embedding_size, self.reg_hidden)))
            self.h2_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, size=(self.reg_hidden + self.aux_dim, 1)))
            self.last_w = self.h2_weight
        else:
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, size=(2*self.embedding_size, self.reg_hidden)))
            self.last_w = self.h1_weight
            
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, size=(self.embedding_size, 1)))


    def train_forward(self, data, return_action_embed=False, dtype=torch.float):

        action_select = data['action_select']
        aux_input = data['aux_input']
        
        cur_message_layer,  y_potential = self._forward(data, dtype=dtype)

        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        #OLD action_embed = torch.matmul(action_select, cur_message_layer)
        # print(action_select['m'], action_select['n'])
        # print(cur_message_layer.shape)
        # print(action_select['index'].max(), action_select['index'].min())
        # print(action_select['value'].max(), action_select['value'].min())
        # print("currmessage: ", cur_message_layer.max(), cur_message_layer.min())
        # print("--------------------------------")
        action_embed = torch_sparse.spmm(action_select['index'], action_select['value'],\
                    action_select['m'], action_select['n'], cur_message_layer)

        # # [batch_size, embed_dim, embed_dim]
        temp = torch.matmul(torch.unsqueeze(action_embed, dim=2),torch.unsqueeze(y_potential, dim=1))
        # [batch_size, embed_dim]
        Shape = action_embed.size()
        # [batch_size, embed_dim], first transform
        embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product,[Shape[0],1]),\
                                                                    [Shape[0],Shape[1],1])),Shape)

        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = torch.matmul(embed_s_a, self.h1_weight)
            #[batch_size, reg_hidden]
            last_output = self.act(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = torch.concat([last_output, aux_input], 1)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = torch.matmul(last_output, self.last_w)
        
        if return_action_embed:
            return q_pred, cur_message_layer, action_embed
        else:
            return q_pred, cur_message_layer
    


    def train_forward_q_on_all(self, data, return_y=False):

        aux_input = data['aux_input']
        rep_global = data['rep_global']
        
        cur_message_layer,  y_potential = self._forward(data)

        rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                    rep_global['m'], rep_global['n'], y_potential)


        # # [batch_size, embed_dim, embed_dim]
        temp = torch.matmul(torch.unsqueeze(cur_message_layer, dim=2),torch.unsqueeze(rep_y, dim=1))
        # [batch_size, embed_dim]
        Shape = cur_message_layer.size()
        # [batch_size, embed_dim], first transform
        embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product,[Shape[0],1]),\
                                                                    [Shape[0],Shape[1],1])),Shape)

        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = torch.matmul(embed_s_a, self.h1_weight)
            #[batch_size, reg_hidden]
            last_output = self.act(hidden)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        
        aux_input_rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                    rep_global['m'], rep_global['n'], aux_input)
        

        last_output = torch.concat([last_output, aux_input_rep_y], 1)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = torch.matmul(last_output, self.last_w)
        
        if return_y:
            return q_pred, cur_message_layer, y_potential
        else:
            return q_pred, cur_message_layer
    
    def test_forward(self, data, return_embedding=False, dtype=torch.float):

        node_input = data['node_input']
        subgsum_param = data['subgsum_param']
        n2nsum_param = data['n2nsum_param']
        rep_global = data['rep_global']
        aux_input = data['aux_input']

        cur_message_layer,  y_potential = self._forward(data, dtype=dtype)

        #print("++++FROM DQN MODULE: shape of y_potential: ", y_potential.shape)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        #OLD rep_y = torch.matmul(rep_global, y_potential)
        if(y_potential.shape[0] != subgsum_param['m']):
            print("Red flag!!")
            print("Y potential shape: ", y_potential.shape)
            print("rep_global shape: ", rep_global['m'], rep_global['n'])
            print(subgsum_param)
            print(data['batch'])
            print(n2nsum_param)

        rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
                    rep_global['m'], rep_global['n'], y_potential)

        

        #[[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
        # # [node_cnt, embed_dim, embed_dim]
        cur_message_layer = cur_message_layer.to(dtype)
        rep_y = rep_y.to(dtype)
        temp1 = torch.matmul(torch.unsqueeze(cur_message_layer, dim=2),torch.unsqueeze(rep_y, dim=1))
        # [node_cnt embed_dim]
        Shape1 = cur_message_layer.size()
        # [batch_size, embed_dim], first transform

        embed_s_a_all = torch.reshape(torch.matmul(temp1, torch.reshape(torch.tile(self.cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = torch.matmul(embed_s_a_all, self.h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = self.act(hidden)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = torch_sparse.spmm(rep_global['index'], rep_global['value'],\
            rep_global['m'], rep_global['n'], aux_input)
        #rep_aux = torch.matmul(rep_global, aux_input)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = torch.concat([last_output,rep_aux],1)

        last_output = last_output.to(dtype)
        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = torch.matmul(last_output, self.last_w)

        if(return_embedding):
            return q_on_all, cur_message_layer
        return q_on_all



    def test_forward_chunked(self, data, return_embedding=False, dtype=torch.float,
                             node_chunk_size=20000):
        """Memory-scalable, *numerically identical* version of test_forward.

        ADDITIVE — does not replace test_forward. The only difference is that the
        per-node decoder computation (which otherwise materializes a
        [node_cnt, embed_dim, embed_dim] outer-product tensor — ~419 GB at 1.6M
        nodes, d=256) is evaluated in node chunks of `node_chunk_size`. The math
        per node is unchanged, so the output equals test_forward's output.

        Requires the encoder to expose `forward_chunked` (e.g. MEGA_Encoder) so the
        cross-attention distillation is also chunked; otherwise falls back to the
        encoder's normal forward (fine for single-encoder FINDER).
        """
        subgsum_param = data['subgsum_param']
        rep_global = data['rep_global']
        aux_input = data['aux_input']

        # --- encoder (chunked if supported) -> per-node embeddings [n, d] ---
        if hasattr(self.encoder, 'forward_chunked'):
            cur_message_layer, y_potential = self.encoder.forward_chunked(
                data, node_chunk_size=node_chunk_size, dtype=dtype)
        else:
            cur_message_layer, y_potential = self._forward(data, dtype=dtype)
        cur_message_layer = cur_message_layer.to(dtype)

        # broadcast the global ("virtual node") summary to every node: rep_y [n, d]
        rep_y = torch_sparse.spmm(rep_global['index'], rep_global['value'],
                                  rep_global['m'], rep_global['n'], y_potential).to(dtype)
        # per-node auxiliary features: rep_aux [n, aux_dim]
        rep_aux = torch_sparse.spmm(rep_global['index'], rep_global['value'],
                                    rep_global['m'], rep_global['n'], aux_input)

        n = cur_message_layer.shape[0]
        d = cur_message_layer.shape[1]
        cp = self.cross_product.reshape(1, d, 1)

        q_chunks = []
        for s in range(0, n, node_chunk_size):
            e = min(s + node_chunk_size, n)
            cur_c = cur_message_layer[s:e]                       # [c, d]
            rep_c = rep_y[s:e]                                   # [c, d]
            c = cur_c.shape[0]
            # outer product then contract with cross_product (identical to test_forward)
            temp1 = torch.matmul(torch.unsqueeze(cur_c, dim=2),
                                 torch.unsqueeze(rep_c, dim=1))  # [c, d, d]
            embed_s_a = torch.reshape(
                torch.matmul(temp1, cp.expand(c, d, 1)), (c, d))  # [c, d]
            last_output = embed_s_a
            if self.reg_hidden > 0:
                last_output = self.act(torch.matmul(embed_s_a, self.h1_weight))
            last_output = torch.concat([last_output, rep_aux[s:e]], 1).to(dtype)
            q_chunks.append(torch.matmul(last_output, self.last_w))
        q_on_all = torch.concat(q_chunks, dim=0)

        if return_embedding:
            return q_on_all, cur_message_layer
        return q_on_all

    def _forward(self, data, dtype=torch.float):
        cur_message_layer, y_cur_message_layer = self.encoder(data, dtype=dtype)
        return cur_message_layer, y_cur_message_layer
    
    def forward(self, data, dtype=torch.float):
        if self.training:
            return self.train_forward(data, dtype=dtype)
        else:
            return self.test_forward(data, dtype=dtype)
    