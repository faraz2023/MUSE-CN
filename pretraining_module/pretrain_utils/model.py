import dgl
import torch
import pretrain_utils.utils as utils
import torch.nn.functional as F
import math
import copy
from torch import Tensor 
import numpy as np
import torch.nn as nn
import math


def get_p_minsg_loss(h1, h2, temperature):
    f = lambda x: torch.exp(x / temperature)
    refl_sim = f(utils.sim_matrix(h1, h1))        # intra-view pairs
    between_sim = f(utils.sim_matrix(h1, h2))     # inter-view pairs
    x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
    loss = -torch.log(between_sim.diag() / x1)
    return loss

class TanhMultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(emb_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(emb_dim, self.all_head_size)
        self.key = nn.Linear(emb_dim, self.all_head_size)
        self.value = nn.Linear(emb_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        num_task, num_samples, emb_dim = query.size()


        query = self.query(query).view(-1, self.num_heads, self.attention_head_size)
        key = self.key(key).view(-1, self.num_heads, self.attention_head_size)
        value = self.value(value).view(-1, self.num_heads, self.attention_head_size)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = torch.tanh(attention_scores)
        
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)
        
        context_layer = context_layer.transpose(1, 2).contiguous().view(num_task, num_samples, self.all_head_size)
        return context_layer, attention_probs


class CrossAttentionTransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_dim, activation='softmax', dropout=0.1):
        super(CrossAttentionTransformerBlock, self).__init__()
        if(activation == 'softmax'):
            self.attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        elif(activation == 'tanh'):
            self.attention = TanhMultiheadAttention(emb_dim, num_heads, dropout=dropout)
        else:
            raise AssertionError("Unknown activation for attention in transformer block")

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_dim)
        )
        
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is of shape (num_tasks, batch_size, emb_dim)
        attention_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x





class SimpleTransformer_MT_Module(torch.nn.Module):
    def __init__(self, big_model, predictor_dim, tasks, num_heads=8):
        super(SimpleTransformer_MT_Module, self).__init__()
        self.big_model = big_model
        self.num_heads = num_heads
        self.predictor_dim = predictor_dim
        hid_dim = big_model.hid_dim

        self.tasks = tasks
        print(tasks)

        self._init_task_modules(tasks, predictor_dim, hid_dim)
        self.transformer_block = CrossAttentionTransformerBlock(hid_dim, num_heads, hid_dim, dropout=0.1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _forward_inital_embeddings(self, sample, opt):
        task_embeddings = {}
        #task_idx = {}
        idx = 0

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4]            
            h_plink = self.big_model(sg)
            task_embeddings['p_link'] = F.normalize(h_plink, dim=1)
            #task_idx['p_link'] = idx
            idx += 1

        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            h_ming = self.big_model(bg, feat)
            task_embeddings['p_ming'] = h_ming
            #task_idx['p_ming'] = idx
            idx += 1

        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            temperature = opt.temperature_minsg
            h_minsg = self.task_modules['p_minsg'](self.big_model(g1, feat1))
            task_embeddings['p_minsg'] = h_minsg
            #task_idx['p_minsg'] = idx
            idx += 1

        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            h_decor = self.big_model(g1, g1.ndata['feat'])
            task_embeddings['p_decor'] = h_decor
            #task_idx['p_decor'] = idx
            idx += 1

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            #x_target = g.ndata['feat'][mask_nodes].clone()
            feat = g.ndata['feat'].clone()
            feat[mask_nodes] = 0
            feat[mask_nodes] += self.task_modules['p_recon_mask'].parameter
            h_recon = self.big_model(g, feat)
            h_recon = self.task_modules['p_recon_recon_enc_dec'](h_recon)
            h_recon[mask_nodes] = 0
            task_embeddings['p_recon'] = h_recon
            #task_idx['p_recon'] = idx
            idx += 1

        return task_embeddings


    def _init_task_modules(self, tasks, predictor_dim, hid_dim):

        self.task_modules = nn.ModuleDict()

        if 'p_link' in tasks:
            self.task_modules['p_link_predictor_hid'] = torch.nn.Linear(hid_dim, predictor_dim)
            self.task_modules['p_link_predictor_class'] = torch.nn.Linear(predictor_dim, 1)

        if 'p_ming' in tasks:
            self.task_modules['p_ming_discriminator'] = Discriminator(hid_dim)

        if 'p_minsg' in tasks:
            self.task_modules['p_minsg'] = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        if 'p_decor' in tasks:
            self.task_modules['p_decor'] = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))
        
        if 'p_recon' in tasks:
            self.task_modules['p_recon_mask'] = ParameterModule(torch.nn.Parameter(torch.zeros(1, self.big_model.node_module.num_node_features)))
            self.task_modules['p_recon_recon_enc_dec'] = torch.nn.Linear(hid_dim, hid_dim, bias=False)
            self.task_modules['p_recon_decoder'] = dgl.nn.GraphConv(hid_dim, self.big_model.node_module.num_node_features, allow_zero_in_degree=True)

    def compute_representation(self, g, X):
        self.train(False)
        with torch.no_grad():
            h = self.big_model(g, X)
        self.train(True)
        return h.detach()
    
    def compute_post_representation(self, sample, opt, strategy='sum'):
        self.train(False)
        with torch.no_grad():
            init_embeddings_dict = self._forward_inital_embeddings(sample, opt)

        embeddings_sequence = torch.stack(list(init_embeddings_dict.values()), dim=0)

        transformed_sequence = self.transformer_block(embeddings_sequence)
        


        transformed_embeddings_dict = {task: embedding for task, embedding in zip(init_embeddings_dict.keys(), transformed_sequence)}

            

        if(strategy == 'sum'):
            sample_key = list(transformed_embeddings_dict.keys())[0]
            h = torch.zeros_like(transformed_embeddings_dict[sample_key])
            for task_name in transformed_embeddings_dict.keys():
                h += transformed_embeddings_dict[task_name]
            return h
        elif(strategy == 'max'):
            sample_key = list(transformed_embeddings_dict.keys())[0]
            h = torch.zeros_like(transformed_embeddings_dict[sample_key])
            for task_name in transformed_embeddings_dict.keys():
                h = torch.max(h, transformed_embeddings_dict[task_name])
            return h
        elif(strategy == 'raw'):
            return transformed_embeddings_dict
        else:
            raise AssertionError("Unknown strategy for computing post representation")
        
    def _forward_task_modules(self, sample, embeddings_dict, opt):
        res = {}

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4] 

            h_plink = embeddings_dict['p_link']
            h_plink = F.normalize(h_plink, dim=1)
            h_plink = F.relu(self.task_modules['p_link_predictor_hid'](h_plink))
            h_pos = h_plink[pos_u] * h_plink[pos_v]
            h_neg = h_plink[neg_u] * h_plink[neg_v]
            pos_logits = self.task_modules['p_link_predictor_class'](h_pos).squeeze()
            neg_logits = self.task_modules['p_link_predictor_class'](h_neg).squeeze()
            logits = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)])
            target = torch.cat([torch.ones_like(pos_logits),torch.zeros_like(neg_logits)])

            res['p_link'] = F.binary_cross_entropy(logits, target)

        
        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            h_ming = embeddings_dict['p_ming']
            positive = h_ming
            negative = self.big_model(bg, cor_feat)

            summary = torch.sigmoid(positive.mean(dim=0))
            positive = self.task_modules['p_ming_discriminator'](positive, summary)
            negative = self.task_modules['p_ming_discriminator'](negative, summary)
            l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
            l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
            res['p_ming'] = l1 + l2


        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            temperature = opt.temperature_minsg
            h_minsg = embeddings_dict['p_minsg']
            h1 = h_minsg
            h2 = self.task_modules['p_minsg'](self.big_model(g2, feat2))
            l1 = get_p_minsg_loss(h1, h2, temperature)
            l2 = get_p_minsg_loss(h2, h1, temperature)
            ret = (l1 + l2) * 0.5
            res['p_minsg'] = ret.mean()  # utils.constrastive_loss(h1, h2, temperature=temperature)

        
        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            N = g1.number_of_nodes()
            h1 = embeddings_dict['p_decor']
            h2 = self.big_model(g2, g2.ndata['feat'])

            z1 = (h1 - h1.mean(0)) / h1.std(0)
            z2 = (h2 - h2.mean(0)) / h2.std(0)

            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            c = (z1 - z2) / N
            c1 = c1 / N
            c2 = c2 / N

            loss_inv = torch.linalg.matrix_norm(c)
            iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
            loss_dec1 = torch.linalg.matrix_norm(iden - c1)
            loss_dec2 = torch.linalg.matrix_norm(iden - c2)

            res['p_decor'] = loss_inv + lambd * (loss_dec1 + loss_dec2)

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            
            h = embeddings_dict['p_recon']
            x_target = g.ndata['feat'][mask_nodes].clone()

            x_pred = self.task_modules['p_recon_decoder'](g, h)[mask_nodes]
            res['p_recon'] = sce_loss(x_pred, x_target)

        return res


        
    def forward(self, sample, opt):
        init_embeddings_dict = self._forward_inital_embeddings(sample, opt)

        embeddings_sequence = torch.stack(list(init_embeddings_dict.values()), dim=0)

        transformed_sequence = self.transformer_block(embeddings_sequence)
        


        transformed_embeddings_dict = {task: embedding for task, embedding in zip(init_embeddings_dict.keys(), transformed_sequence)}




        res = self._forward_task_modules(sample, transformed_embeddings_dict,  opt)

        return res





# Multi-task multi-head attention module
class XATT_MT_Module(torch.nn.Module):
    def __init__(self, big_model, predictor_dim, tasks, num_heads=8):
        super(XATT_MT_Module, self).__init__()
        print("Initializing MTMA Module")
        self.big_model = big_model
        self.num_heads = num_heads
        self.predictor_dim = predictor_dim
        hid_dim = big_model.hid_dim

        self.no_distil = False
        self.use_VxTAM = True
        self.use_alpha = True

        
        self.tasks = tasks
        print(tasks)
        

        self._init_distil_modules(tasks, hid_dim, self.use_alpha)
        self._init_task_modules(tasks, predictor_dim, hid_dim)
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _init_task_modules(self, tasks, predictor_dim, hid_dim):

        self.task_modules = nn.ModuleDict()

        if 'p_link' in tasks:
            self.task_modules['p_link_predictor_hid'] = torch.nn.Linear(hid_dim, predictor_dim)
            self.task_modules['p_link_predictor_class'] = torch.nn.Linear(predictor_dim, 1)

        if 'p_ming' in tasks:
            self.task_modules['p_ming_discriminator'] = Discriminator(hid_dim)

        if 'p_minsg' in tasks:
            self.task_modules['p_minsg'] = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        if 'p_decor' in tasks:
            self.task_modules['p_decor'] = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))
        
        if 'p_recon' in tasks:
            self.task_modules['p_recon_mask'] = ParameterModule(torch.nn.Parameter(torch.zeros(1, self.big_model.node_module.num_node_features)))
            self.task_modules['p_recon_recon_enc_dec'] = torch.nn.Linear(hid_dim, hid_dim, bias=False)
            self.task_modules['p_recon_decoder'] = dgl.nn.GraphConv(hid_dim, self.big_model.node_module.num_node_features, allow_zero_in_degree=True)

        

        
    def _init_distil_modules(self, tasks, in_dim, use_alpha):
        # Auxiliary function to get other tasks
        others = lambda t: {s for s in tasks if s != t}
            
        # Initialize VxTAM for each task pair
        self.xatt_block = nn.ModuleDict({
            t: nn.ModuleDict({
                **{f'VxTAM_{s}': VxTAM(in_dim, use_alpha) for s in others(t)},

                'post': torch.nn.Linear(in_dim*len(tasks), in_dim)
            })
            for t in tasks
        })

        # Provide easy access to parameters per-task. .parameters() won't return duplicates
        self.distil_params = nn.ModuleDict({
            t: nn.ModuleList([self.xatt_block[t]]) for t in tasks
        })
    
    def get(self, name, target, source=None):
        assert name in ["VxTAM"], f"Unknown distillation module {name}"
        if name == "VxTAM" and source is not None:
            k = f'{name}_{source}'
        else:
            #k = name
            raise AssertionError("Shouldn't be here!")
            

        module = self.xatt_block[target][k]
        return module



    def compute_representation(self, g, X):
        self.train(False)
        with torch.no_grad():
            h = self.big_model(g, X)
        self.train(True)
        return h.detach()
    
    def compute_post_representation(self, sample, opt, strategy='sum'):
        self.train(False)
        with torch.no_grad():
            init_embeddings_dict = self._forward_inital_embeddings(sample, opt)

            
            xatt_embeddings_dict = {}
            for task_name in init_embeddings_dict.keys():
                xatt_embeddings_dict[task_name] = self.apply_cross_attention(sample, init_embeddings_dict, task_name)

        if(strategy == 'sum'):
            sample_key = list(xatt_embeddings_dict.keys())[0]
            h = torch.zeros_like(xatt_embeddings_dict[sample_key])
            for task_name in xatt_embeddings_dict.keys():
                h += xatt_embeddings_dict[task_name]
            return h
        elif(strategy == 'max'):
            sample_key = list(xatt_embeddings_dict.keys())[0]
            h = torch.zeros_like(xatt_embeddings_dict[sample_key])
            for task_name in xatt_embeddings_dict.keys():
                h = torch.max(h, xatt_embeddings_dict[task_name])
            return h
        elif(strategy == 'raw'):
            return xatt_embeddings_dict
        else:
            raise AssertionError("Unknown strategy for computing post representation")
    
    def _forward_inital_embeddings(self, sample, opt):
        task_embeddings = {}
        #task_idx = {}
        idx = 0

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4]            
            h_plink = self.big_model(sg)
            task_embeddings['p_link'] = F.normalize(h_plink, dim=1)
            #task_idx['p_link'] = idx
            idx += 1

        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            h_ming = self.big_model(bg, feat)
            task_embeddings['p_ming'] = h_ming
            #task_idx['p_ming'] = idx
            idx += 1

        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            temperature = opt.temperature_minsg
            h_minsg = self.task_modules['p_minsg'](self.big_model(g1, feat1))
            task_embeddings['p_minsg'] = h_minsg
            #task_idx['p_minsg'] = idx
            idx += 1

        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            h_decor = self.big_model(g1, g1.ndata['feat'])
            task_embeddings['p_decor'] = h_decor
            #task_idx['p_decor'] = idx
            idx += 1

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            #x_target = g.ndata['feat'][mask_nodes].clone()
            feat = g.ndata['feat'].clone()
            feat[mask_nodes] = 0
            feat[mask_nodes] += self.task_modules['p_recon_mask'].parameter
            h_recon = self.big_model(g, feat)
            h_recon = self.task_modules['p_recon_recon_enc_dec'](h_recon)
            h_recon[mask_nodes] = 0
            task_embeddings['p_recon'] = h_recon
            #task_idx['p_recon'] = idx
            idx += 1

        return task_embeddings

    # Define a function to handle cross-attention for given embeddings and task name
    def apply_cross_attention(self, input_sample, embeddings_dict, task_name):
        task_embeddings = embeddings_dict[task_name]
        out_embs = [task_embeddings]
        other_tasks = [t for t in self.tasks if t != task_name]
        for other_task in other_tasks:
            if(other_task in input_sample.keys()):
                attention_module = self.xatt_block[task_name][f'VxTAM_{other_task}']
                embedding = attention_module(task_embeddings, embeddings_dict[other_task])
                out_embs.append(embedding)
            else:
                print("Warning: Task not found in input sample")

        embedding = torch.cat(out_embs, dim=-1)
        embedding = self.xatt_block[task_name]['post'](embedding)
        return embedding
    
    def _forward_task_modules(self, sample, embeddings_dict, opt):
        res = {}

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4] 

            h_plink = embeddings_dict['p_link']
            h_plink = F.normalize(h_plink, dim=1)
            h_plink = F.relu(self.task_modules['p_link_predictor_hid'](h_plink))
            h_pos = h_plink[pos_u] * h_plink[pos_v]
            h_neg = h_plink[neg_u] * h_plink[neg_v]
            pos_logits = self.task_modules['p_link_predictor_class'](h_pos).squeeze()
            neg_logits = self.task_modules['p_link_predictor_class'](h_neg).squeeze()
            logits = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)])
            target = torch.cat([torch.ones_like(pos_logits),torch.zeros_like(neg_logits)])

            res['p_link'] = F.binary_cross_entropy(logits, target)

        
        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            h_ming = embeddings_dict['p_ming']
            positive = h_ming
            negative = self.big_model(bg, cor_feat)

            summary = torch.sigmoid(positive.mean(dim=0))
            positive = self.task_modules['p_ming_discriminator'](positive, summary)
            negative = self.task_modules['p_ming_discriminator'](negative, summary)
            l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
            l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
            res['p_ming'] = l1 + l2


        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            temperature = opt.temperature_minsg
            h_minsg = embeddings_dict['p_minsg']
            h1 = h_minsg
            h2 = self.task_modules['p_minsg'](self.big_model(g2, feat2))
            l1 = get_p_minsg_loss(h1, h2, temperature)
            l2 = get_p_minsg_loss(h2, h1, temperature)
            ret = (l1 + l2) * 0.5
            res['p_minsg'] = ret.mean()  # utils.constrastive_loss(h1, h2, temperature=temperature)

        
        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            N = g1.number_of_nodes()
            h1 = embeddings_dict['p_decor']
            h2 = self.big_model(g2, g2.ndata['feat'])

            z1 = (h1 - h1.mean(0)) / h1.std(0)
            z2 = (h2 - h2.mean(0)) / h2.std(0)

            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            c = (z1 - z2) / N
            c1 = c1 / N
            c2 = c2 / N

            loss_inv = torch.linalg.matrix_norm(c)
            iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
            loss_dec1 = torch.linalg.matrix_norm(iden - c1)
            loss_dec2 = torch.linalg.matrix_norm(iden - c2)

            res['p_decor'] = loss_inv + lambd * (loss_dec1 + loss_dec2)

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            
            h = embeddings_dict['p_recon']
            x_target = g.ndata['feat'][mask_nodes].clone()

            x_pred = self.task_modules['p_recon_decoder'](g, h)[mask_nodes]
            res['p_recon'] = sce_loss(x_pred, x_target)

        return res


    def forward(self, sample, opt):
        init_embeddings_dict = self._forward_inital_embeddings(sample, opt)

        xatt_embeddings_dict = {}
        for task_name in init_embeddings_dict.keys():
            xatt_embeddings_dict[task_name] = self.apply_cross_attention(sample, init_embeddings_dict, task_name)


        res = self._forward_task_modules(sample, xatt_embeddings_dict,  opt)

        return res


    


class ParameterModule(torch.nn.Module):
    def __init__(self, parameter):
        super(ParameterModule, self).__init__()
        self.parameter = parameter

class VxTAM(nn.Module):
    """Vector cross-Task Attention Module"""
    def __init__(self, in_dim, use_alpha):
        super().__init__()

        ## Projection layers
        self.fc_b = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                  nn.BatchNorm1d(in_dim),
                                  nn.ReLU())

        self.fc_c = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                  nn.BatchNorm1d(in_dim),
                                  nn.ReLU())

        self.fc_d = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                  nn.BatchNorm1d(in_dim),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

        ## Channel-wise weights
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1, in_dim))

    def forward(self, x, y):
        B = self.fc_b(x)
        C = self.fc_c(y)
        D = self.fc_d(y)

        #print("B.shape", B.shape)
        #print("C.shape", C.shape)
        #print("D.shape", D.shape)

        # compute correlation matrix
        coeff = math.sqrt(B.size(1))
        corr = self.softmax(B.unsqueeze(2) @ C.unsqueeze(1) / coeff)

        out = (D.unsqueeze(1) @ corr).squeeze()

        if self.use_alpha:
            out *= self.alpha

        return out


class Link_Pred_v2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super(Link_Pred_v2, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, 1)
        

    def forward(self, h):
        h = self.linear1(h)
        h = self.relu(h)
        h = self.linear2(h)
        h = self.relu(h)
        return self.linear3(h).squeeze()



class Link_Pred(torch.nn.Module):
    def __init__(self, in_channels):
        super(Link_Pred, self).__init__()
        self.linear = torch.nn.Linear(in_channels, 1)
        self.linear_ =  torch.nn.Linear(in_channels, in_channels)

    def forward(self, h):
        return self.linear(h).squeeze()


class PretrainModule(torch.nn.Module):
    def __init__(self, big_model, predictor_dim):
        super(PretrainModule, self).__init__()
        hid_dim = big_model.hid_dim
        self.big_model = big_model
        
        # link prediction head
        self.link_predictor_hid = torch.nn.Linear(hid_dim, predictor_dim)
        self.link_predictor_class = torch.nn.Linear(predictor_dim, 1)
        
        # graph matching head
        self.graph_matcher = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(predictor_dim, hid_dim))
        # discriminator for ming
        self.discriminator = Discriminator(hid_dim)

        # head for metis partition cls
        self.metis_cls = torch.nn.Linear(hid_dim, 10)

        # head for metis partition clsss
        self.par_cls = torch.nn.Linear(hid_dim, 20)

        # head for minsg
        self.minsg = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        # head for decor
        self.decor = torch.nn.Sequential(torch.nn.Linear(hid_dim, predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(predictor_dim, hid_dim))

        # head for feature reconstruction
        self.recon_mask = torch.nn.Parameter(torch.zeros(1, big_model.node_module.num_node_features))
        self.recon_enc_dec = torch.nn.Linear(hid_dim, hid_dim, bias=False)
        self.decoder = dgl.nn.GraphConv(hid_dim, big_model.node_module.num_node_features, allow_zero_in_degree=True)
        for m in self.modules():
            self.weights_init(m)

    def get_device(self):
        return next(self.parameters()).device

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def compute_representation(self, g, X):
        self.train(False)
        with torch.no_grad():
            h = self.big_model(g, X)
        self.train(True)
        return h.detach()

    def forward(self, sample, opt):
        res = {}
        # print("forwarding plink")
        if 'p_link' in sample:
            data = sample['p_link']
            res['p_link'] = self.p_link(data[0], data[1], data[2], data[3], data[4])
        
        # print("forwarding pming")
        if 'p_ming' in sample:
            data = sample['p_ming']
            res['p_ming'] = self.p_ming(data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device))
        # print("finished pming")
        # print("forwarding pminsg")
        if 'p_minsg' in sample:
            data = sample['p_minsg']
            res['p_minsg'] = self.p_minsg(data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device), temperature=opt.temperature_minsg)
        # print("finished pminsg")
        # print("forwarding pdecor")
        if 'p_decor' in sample:
            data = sample['p_decor']
            res['p_decor'] = self.p_decor(data[0].to(opt.device), data[1].to(opt.device), lambd=opt.decor_lamb)

        # print("forwarding precon")
        if 'p_recon' in sample:
            data = sample['p_recon']
            res['p_recon'] = self.p_recon(data[0].to(opt.device), data[1])

        return res 


    def p_link(self, sg, pos_u, pos_v, neg_u, neg_v):
        h = self.big_model(sg)
        h = F.normalize(h, dim=1)
        h = F.relu(self.link_predictor_hid(h))
        h_pos = h[pos_u] * h[pos_v]
        h_neg = h[neg_u] * h[neg_v]
        pos_logits = self.link_predictor_class(h_pos).squeeze()
        neg_logits = self.link_predictor_class(h_neg).squeeze()

        logits = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)])
        target = torch.cat([torch.ones_like(pos_logits),torch.zeros_like(neg_logits)])
        return F.binary_cross_entropy(logits, target)
    
    def p_ming(self, bg, feat, cor_feat):
        src, dst = bg.edges()
        edge_index = torch.stack([src, dst], dim=0).to(self.get_device())
        positive = self.big_model({'edge_index': edge_index, 'node_input': feat, 'batch': torch.zeros(feat.shape[0], dtype=torch.int64).to(self.get_device())})
        negative = self.big_model({'edge_index': edge_index, 'node_input': cor_feat, 'batch': torch.zeros(cor_feat.shape[0], dtype=torch.int64).to(self.get_device())})
        summary = torch.sigmoid(positive.mean(dim=0)) 
        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        
        l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
        l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
        return l1 + l2

    def p_minsg(self, g1, feat1, g2, feat2, temperature):
        
        def get_loss(h1, h2, temperature):
            f = lambda x: torch.exp(x / temperature)
            refl_sim = f(utils.sim_matrix(h1, h1))        # intra-view pairs
            between_sim = f(utils.sim_matrix(h1, h2))     # inter-view pairs
            x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
            loss = -torch.log((between_sim.diag() + 1e-10) / (x1 + 1e-10))
            return loss



        g1_src, g1_dst = g1.edges()
        g1_edge_index = torch.stack([g1_src, g1_dst], dim=0).to(self.get_device())
        g2_src, g2_dst = g2.edges()
        g2_edge_index = torch.stack([g2_src, g2_dst], dim=0).to(self.get_device())

        h1 = self.minsg(self.big_model({'edge_index': g1_edge_index, 'node_input': feat1, 'batch': torch.zeros(feat1.shape[0], dtype=torch.int64).to(self.get_device())}))
        h2 = self.minsg(self.big_model({'edge_index': g2_edge_index, 'node_input': feat2, 'batch': torch.zeros(feat2.shape[0], dtype=torch.int64).to(self.get_device())}))

        # print("h1: ", h1)
        # print("h2: ", h2)
        # print("--------------------------------")

        l1 = get_loss(h1, h2, temperature)
        l2 = get_loss(h2, h1, temperature)
        ret = (l1 + l2) * 0.5
        return ret.mean()  # utils.constrastive_loss(h1, h2, temperature=temperature)

    def p_decor(self, g1, g2, lambd=1e-3):
        N = g1.number_of_nodes()

        g1_src, g1_dst = g1.edges()
        g1_edge_index = torch.stack([g1_src, g1_dst], dim=0).to(self.get_device())
        g2_src, g2_dst = g2.edges()
        g2_edge_index = torch.stack([g2_src, g2_dst], dim=0).to(self.get_device())

        h1 = self.big_model({'edge_index': g1_edge_index, 'node_input': g1.ndata['feat'], 'batch': torch.zeros(g1.number_of_nodes(), dtype=torch.int64).to(self.get_device())})
        h2 = self.big_model({'edge_index': g2_edge_index, 'node_input': g2.ndata['feat'], 'batch': torch.zeros(g2.number_of_nodes(), dtype=torch.int64).to(self.get_device())})

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = (z1 - z2) / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = torch.linalg.matrix_norm(c)
        iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
        loss_dec1 = torch.linalg.matrix_norm(iden - c1)
        loss_dec2 = torch.linalg.matrix_norm(iden - c2)

        return loss_inv + lambd * (loss_dec1 + loss_dec2)

    def p_recon(self, g, mask_nodes):
        x_target = g.ndata['feat'][mask_nodes].clone()
        feat = g.ndata['feat'].clone()
        feat[mask_nodes] = 0
        feat[mask_nodes] += self.recon_mask
        g_src, g_dst = g.edges()
        g_edge_index = torch.stack([g_src, g_dst], dim=0).to(self.get_device())
        h = self.big_model({'edge_index': g_edge_index, 'node_input': feat, 'batch': torch.zeros(feat.shape[0], dtype=torch.int64).to(self.get_device())})
        h = self.recon_enc_dec(h)
        h[mask_nodes] = 0
        x_pred = self.decoder(g, h)[mask_nodes]
        return sce_loss(x_pred, x_target)

class BigModel(torch.nn.Module):
    def __init__(self, node_module, graph_module, inter_dim):
        super(BigModel, self).__init__()

        self.node_module = node_module
        self.graph_module = graph_module
        self.hid_dim = node_module.embedding_size
        self.inter_mid = inter_dim
        print("inter Dim: ", inter_dim)
        # this is a universal projection head, agnostic of downstream task
        if inter_dim > 0:
            if graph_module != None:
                self.projection = torch.nn.Linear(self.hid_dim + graph_module.hid_dim , inter_dim)
            else:
                self.projection = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, inter_dim),
                                                    torch.nn.PReLU(),
                                                    torch.nn.Linear(inter_dim, self.hid_dim),
                                                    torch.nn.PReLU()
                )
            print("Self.projection: ", self.projection)
        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # G should be a PyG style data including edge_index, node_input, batch, edge_attr (optional)
    def forward(self, G):
        if type(G) is list:
            # for uniformity with rest of Qlearning code, we get the first element right here
            G = G[0]
            # node module return a tuple of node representation and graph representation
            node, _ = self.node_module(G)
            if self.graph_module == None:
                if self.inter_mid > 0:
                    return self.projection(node)
                else:
                    return node
            graph = self.graph_module(G)
            h = torch.cat([node, graph], dim= -1)
            if self.inter_mid > 0:
                return self.projection(node)
            else:
                return node
        else:
            # node module return a tuple of node representation and graph representation
            node, _ = self.node_module(G)
            if self.graph_module == None:
                if self.inter_mid > 0:
                    return self.projection(node)
                else:
                    return node
            graph = self.graph_module(G)
            h = torch.cat([node, graph], dim= -1)
            if self.inter_mid > 0:
                return self.projection(node)
            else:
                return node



class MultiEncoder_Transformer_MT_Module(torch.nn.Module):
    def __init__(self, tasks, predictor_dim, node_module_args, big_model_args, 
    attention_activation='softmax', num_heads=8, num_transformer_blocks=1):
        super(MultiEncoder_Transformer_MT_Module, self).__init__()
        
        self.tasks = tasks
        self.num_heads = num_heads
        self.predictor_dim = predictor_dim
        self.node_module_args = node_module_args
        self.big_model_args = big_model_args
        self.add_pre_xatt_loss = False
        self.attention_activation = attention_activation
        self.hid_dim = self.node_module_args['hidden_lst'][-1]#in a crazy way, the hid_dim in the big model is change to be the n_classes of the node_module. Furthermore, n_classes is the hid_lst[-1] in that class.

        print("BIG MODEL args: ", self.big_model_args)
        self._init_task_modules()


        # Create transformer blocks
        print("Creating transformer blocks with activation: ", self.attention_activation)
        self.transformer_block = nn.ModuleList([CrossAttentionTransformerBlock(self.hid_dim, num_heads, self.hid_dim,
             activation=self.attention_activation, dropout=0.1) for _ in range(num_transformer_blocks)])
        self.transformer_block = nn.Sequential(*self.transformer_block)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):

        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def _init_task_modules(self):

        self.task_modules = nn.ModuleDict()

        node_module = self.node_module_args['module']
        nm_args = self.node_module_args.copy()
        nm_args.pop('module')

        graph_module = None

        big_model_module = self.big_model_args['module']
        bm_args = self.big_model_args.copy()
        bm_args.pop('module')


        if 'p_link' in self.tasks:
            curr_node_module = node_module(**nm_args)
            curr_big_model = big_model_module(curr_node_module, graph_module, **bm_args)
            self.task_modules['p_link_encoder_module'] = curr_big_model
            self.task_modules['p_link_projection_module'] = nn.ModuleDict({
                'p_link_predictor_hid': torch.nn.Linear(self.hid_dim, self.predictor_dim),
                'p_link_predictor_class': torch.nn.Linear(self.predictor_dim, 1)
            })

        if 'p_ming' in self.tasks:
            curr_node_module = node_module(**nm_args)
            curr_big_model = big_model_module(curr_node_module, graph_module, **bm_args)
            self.task_modules['p_ming_encoder'] = curr_big_model
            self.task_modules['p_min_projection_module'] = nn.ModuleDict({
                'p_ming_discriminator': Discriminator(self.hid_dim)
            })
        
        if 'p_minsg' in self.tasks:
            curr_node_module = node_module(**nm_args)
            curr_big_model = big_model_module(curr_node_module, graph_module, **bm_args)
            self.task_modules['p_minsg_encoder'] = curr_big_model
            self.task_modules['p_minsg_projection_module'] = nn.ModuleDict({
                'p_minsg': torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.predictor_dim, self.hid_dim))
            })

            
        if 'p_decor' in self.tasks:
            curr_node_module = node_module(**nm_args)
            curr_big_model = big_model_module(curr_node_module, graph_module, **bm_args)
            self.task_modules['p_decor_encoder'] = curr_big_model
            self.task_modules['p_decor_projection_module'] = nn.ModuleDict({
                'p_decor': torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.predictor_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.predictor_dim, self.hid_dim))
            })
            
        if 'p_recon' in self.tasks:
            curr_node_module = node_module(**nm_args)
            curr_big_model = big_model_module(curr_node_module, graph_module, **bm_args)
            self.task_modules['p_recon_encoder'] = curr_big_model
            self.task_modules['p_recon_projection_module'] = nn.ModuleDict({
                'p_recon_mask': ParameterModule(torch.nn.Parameter(torch.zeros(1, curr_big_model.node_module.num_node_features))),
                'p_recon_recon_enc_dec': torch.nn.Linear(self.hid_dim, self.hid_dim, bias=False),
                'p_recon_decoder': dgl.nn.GraphConv(self.hid_dim, curr_big_model.node_module.num_node_features, allow_zero_in_degree=True)
            })

    def _forward_inital_embeddings(self, sample, opt):
        task_embeddings = {}

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4]     
            curr_big_model = self.task_modules['p_link_encoder_module']       
            h_plink = curr_big_model(sg)
            task_embeddings['p_link'] = F.normalize(h_plink, dim=1)
        
        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            curr_big_model = self.task_modules['p_ming_encoder']
            h_ming = curr_big_model(bg, feat)
            task_embeddings['p_ming'] = h_ming
        
        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            curr_big_model = self.task_modules['p_minsg_encoder']
            temperature = opt.temperature_minsg
            h_minsg = curr_big_model(g1, feat1)
            task_embeddings['p_minsg'] = h_minsg
        
        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            curr_big_model = self.task_modules['p_decor_encoder']
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            h_decor = curr_big_model(g1, g1.ndata['feat'])
            task_embeddings['p_decor'] = h_decor

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            curr_big_model = self.task_modules['p_recon_encoder']
            #x_target = g.ndata['feat'][mask_nodes].clone()
            feat = g.ndata['feat'].clone()
            feat[mask_nodes] = 0
            feat[mask_nodes] += self.task_modules['p_recon_projection_module']['p_recon_mask'].parameter
            h_recon = curr_big_model(g, feat)
            h_recon = self.task_modules['p_recon_projection_module']['p_recon_recon_enc_dec'](h_recon)
            h_recon[mask_nodes] = 0
            task_embeddings['p_recon'] = h_recon

        return task_embeddings
    
    def compute_representation(self, sample, opt, post_attn_block=False, strategy='sum'):
        self.train(False)
        with torch.no_grad():
            embedding_dict = self._forward_inital_embeddings(sample, opt)

        if(post_attn_block):
            embeddings_sequence = torch.stack(list(embedding_dict.values()), dim=0)

            transformed_sequence = self.transformer_block(embeddings_sequence)

            embedding_dict = {task: embedding for task, embedding in zip(embedding_dict.keys(), transformed_sequence)}

            
        if(strategy == 'sum'):
            sample_key = list(embedding_dict.keys())[0]
            h = torch.zeros_like(embedding_dict[sample_key])
            for task_name in embedding_dict.keys():
                h += embedding_dict[task_name]
            return h
        elif(strategy == 'max'):
            sample_key = list(embedding_dict.keys())[0]
            h = torch.zeros_like(embedding_dict[sample_key])
            for task_name in embedding_dict.keys():
                h = torch.max(h, embedding_dict[task_name])
            return h
        elif(strategy == 'concat'):
            #return concatenated embeddings created from the dictionary
            embedding_tensor = torch.stack(list(embedding_dict.values()), dim=0)
            _n = embedding_tensor.shape[1]
            embedding_tensor = embedding_tensor.transpose(0, 1)
            embedding_tensor = embedding_tensor.reshape(_n, -1)
            return embedding_tensor
        elif(strategy == 'raw'):
            # return the tensor with [num_task, sample_size, emb_dim]
            embedding_tensor = torch.stack(list(embedding_dict.values()), dim=0)
            return embedding_tensor
        else:
            raise AssertionError("Unknown strategy for computing post representation")
           
    def compute_post_representation(self, sample, opt, strategy='sum'):
        return self.compute_representation(sample, opt, post_attn_block=True, strategy=strategy)
    
    def _forward_task_modules(self, sample, embeddings_dict, opt):
        res = {}

        if 'p_link' in sample:
            data = sample['p_link']
            sg, pos_u, pos_v, neg_u, neg_v = data[0], data[1], data[2], data[3], data[4] 

            h_plink = embeddings_dict['p_link']
            h_plink = F.normalize(h_plink, dim=1)
            h_plink = F.relu(self.task_modules['p_link_projection_module']['p_link_predictor_hid'](h_plink))
            h_pos = h_plink[pos_u] * h_plink[pos_v]
            h_neg = h_plink[neg_u] * h_plink[neg_v]
            pos_logits = self.task_modules['p_link_projection_module']['p_link_predictor_class'](h_pos).squeeze()
            neg_logits = self.task_modules['p_link_projection_module']['p_link_predictor_class'](h_neg).squeeze()
            logits = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)])
            target = torch.cat([torch.ones_like(pos_logits),torch.zeros_like(neg_logits)])

            res['p_link'] = F.binary_cross_entropy(logits, target)

        if 'p_ming' in sample:
            data = sample['p_ming']
            bg, feat, cor_feat = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device)
            positive = embeddings_dict['p_ming']
            negative = self.task_modules['p_ming_encoder'](bg, cor_feat)

            summary = torch.sigmoid(positive.mean(dim=0))
            positive = self.task_modules['p_min_projection_module']['p_ming_discriminator'](positive, summary)
            negative = self.task_modules['p_min_projection_module']['p_ming_discriminator'](negative, summary)
            l1 = F.binary_cross_entropy(torch.sigmoid(positive), torch.ones_like(positive))
            l2 = F.binary_cross_entropy(torch.sigmoid(negative), torch.zeros_like(negative))
            res['p_ming'] = l1 + l2

        if 'p_minsg' in sample:
            data = sample['p_minsg']
            g1, feat1, g2, feat2 = data[0].to(opt.device), data[1].to(opt.device), data[2].to(opt.device), data[3].to(opt.device)
            temperature = opt.temperature_minsg
            h_minsg = embeddings_dict['p_minsg']
            h1 = h_minsg
            h2 = self.task_modules['p_minsg_projection_module']['p_minsg'](self.task_modules['p_minsg_encoder'](g2, feat2))
            l1 = get_p_minsg_loss(h1, h2, temperature)
            l2 = get_p_minsg_loss(h2, h1, temperature)
            ret = (l1 + l2) * 0.5
            res['p_minsg'] = ret.mean()  

        if 'p_decor' in sample:
            data = sample['p_decor']
            g1, g2 = data[0].to(opt.device), data[1].to(opt.device)
            lambd = 1e-3 if not opt.decor_lamb else opt.decor_lamb
            N = g1.number_of_nodes()
            h1 = embeddings_dict['p_decor']
            h2 = self.task_modules['p_decor_encoder'](g2, g2.ndata['feat'])

            z1 = (h1 - h1.mean(0)) / h1.std(0)
            z2 = (h2 - h2.mean(0)) / h2.std(0)

            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)

            c = (z1 - z2) / N
            c1 = c1 / N
            c2 = c2 / N

            loss_inv = torch.linalg.matrix_norm(c)
            iden = torch.tensor(np.eye(c1.shape[0])).to(h1.device)
            loss_dec1 = torch.linalg.matrix_norm(iden - c1)
            loss_dec2 = torch.linalg.matrix_norm(iden - c2)

            res['p_decor'] = loss_inv + lambd * (loss_dec1 + loss_dec2)

        if 'p_recon' in sample:
            data = sample['p_recon']
            g, mask_nodes = data[0].to(opt.device), data[1]
            
            h = embeddings_dict['p_recon']
            x_target = g.ndata['feat'][mask_nodes].clone()

            x_pred = self.task_modules['p_recon_projection_module']['p_recon_decoder'](g, h)[mask_nodes]
            res['p_recon'] = sce_loss(x_pred, x_target)

        return res



    def set_add_pre_xatt_loss(self, add_pre_xatt_loss):
        self.add_pre_xatt_loss = add_pre_xatt_loss

    def forward(self, sample, opt, pretrain=False):
        init_embeddings_dict = self._forward_inital_embeddings(sample, opt)

        embeddings_sequence = torch.stack(list(init_embeddings_dict.values()), dim=0)

        if(pretrain):
            return self._forward_task_modules(sample, init_embeddings_dict,  opt)
        # it is possible to calculate the losses based on only the inital embeddings as well
        # a potential loss function could be the sum of losses from the initial embeddings
        # plus the losses after the cross attention module. This way, and I may be wrong, there is
        # an additionally provided incentive for the task-specific encoders to performs well for their respective tasks

        if(self.add_pre_xatt_loss):
            res_pre = self._forward_task_modules(sample, init_embeddings_dict,  opt)

        transformed_sequence = self.transformer_block(embeddings_sequence)

        transformed_embeddings_dict = {task: embedding for task, embedding in zip(init_embeddings_dict.keys(), transformed_sequence)}

        res = self._forward_task_modules(sample, transformed_embeddings_dict,  opt)

        if(self.add_pre_xatt_loss):
            for task_name in res.keys():
                res[task_name] += res_pre[task_name]

        return res

    # Experiments:
        # 1. Try with single and double projection heads 
            # (first with shared projection heades before and after the cross-task attention block)
            #  (then with seperate projection heads for each task before and after the cross-task attention block)
        # 2. Either  freeze sperate encoders after individual pretraining or not (end to end training after pretraining)
        # 3. Do with and without pretraining of the seperate encoders

    ## How should the pretraining work? Should we use the same sample for each of the tasks
    ## at each iteration? (saving time for dataloading) or should we use different samples (effectivly training each task fully seperately)
    ## How about the schedule and optimizer? Should they be shared across the different encoders and with the final overall model?
    


    ### Warning: reset optimizer and scheduler after pretraining
    #def pre_train_encoders(self, dataloader, opt, num_iters=10*1000):
    #    #pass
    #    for sample in dataloader:
    #        # one forward backward for enc1
    #        # one forward backward for enc2
    #        # one forward backward for enc3
    #        # one forward backward for enc4
    #        pass
    #    return None




class GCN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 hidden_lst,
                 dropout,
                 norm,
                 prelu):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.num_node_features = num_node_features
        hidden_lst = [num_node_features] + hidden_lst
        for in_, out_ in zip(hidden_lst[:-1], hidden_lst[1:]):
            self.layers.append(dgl.nn.GraphConv(in_, out_, allow_zero_in_degree=True))
            self.norms.append(torch.nn.BatchNorm1d(out_, momentum=0.99) if norm == 'batch' else \
                              torch.nn.LayerNorm(out_))
            self.activations.append(torch.nn.PReLU() if prelu else torch.nn.ReLU())

        self.dropout = torch.nn.Dropout(p=dropout)
        self.n_classes = hidden_lst[-1]

    def forward(self, g, features=None):
        if type(g) is list:
            h = g[0].ndata['feat']['_N'].to(self.layers[-1].weight.device)
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g[i].to(self.layers[-1].weight.device), h)
                h = self.activations[i](self.norms[i](h))
        else:
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(g, h)
                h = self.activations[i](self.norms[i](h))
        return h



class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features

class BatchNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)

    def reset_parameters(self):
        self.module.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.module(x)


    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss