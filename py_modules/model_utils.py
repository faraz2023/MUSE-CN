from py_modules.encoders import  GraphSAGE_Encoder_PyG_FINDER_v2, IdentityEncoder, MEGA_Encoder
from py_modules.models import FINDER_DQN, MoE_DQN_simple
import torch.optim as optim
import torch
import json
import os

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


def get_best_CNDP_DQN_model(agent_class, exp_path):
    best_model_path = find_best_model_path(exp_path)
            
    # Load agent and best model
    with open(os.path.join(exp_path, 'params.json'), 'r') as f:
        params = json.load(f)

    agent = agent_class(params)
    agent.LoadModel(best_model_path)
    print("!!! Successfully loaded best model from:\n", best_model_path)
    model = agent.DQN


    return model, params


def get_encoder(encoder_type, encoder_args={}, seed=42, pretrained_encoder_path = None, pretrained_encoder_load_mode = 'freeze'):
    if encoder_type == 'FINDER_encoder_PyG':
        encoder = GraphSAGE_Encoder_PyG_FINDER_v2(**encoder_args, seed=seed)
    elif encoder_type.lower() == 'identity':
        encoder = IdentityEncoder(**encoder_args)
    elif encoder_type.lower() == 'mega':
        encoder = MEGA_Encoder(**encoder_args)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    if pretrained_encoder_path is not None:
        checkpoint = torch.load(pretrained_encoder_path)
        encoder.load_state_dict(checkpoint)
        print("Successfully loaded pretrained encoder from: \n\n", pretrained_encoder_path)

        if pretrained_encoder_load_mode == 'reset':
            encoder._initialize_weights()
            for param in encoder.parameters():
                param.requires_grad = True

        if pretrained_encoder_load_mode == 'freeze':
            for param in encoder.parameters():
                param.requires_grad = False
        elif pretrained_encoder_load_mode == 'finetune':
            for param in encoder.parameters():
                param.requires_grad = True
    
        else:
            raise ValueError(f"Unsupported pretrained encoder load mode: {pretrained_encoder_load_mode}")
        
        print("set encoder to {} mode".format(pretrained_encoder_load_mode))
    
    return encoder
def get_model(model_type, encoder_type=None, encoder_args={}, decoder_args={},seed=42,
              pretrained_encoder_path = None, pretrained_encoder_load_mode = 'freeze',
              experts_args=None):
    
    if 'MoE' in model_type:
        from Q_CNDP import Q_CNDP_Agent

        experts = experts_args['experts']
        expert_models = {}
        expert_params = {}
        for expert_name, expert_args in experts.items():
            expert_model, curr_expert_params = get_best_CNDP_DQN_model(Q_CNDP_Agent, expert_args['model_dir'])

            curr_expert_params['load_mode'] = expert_args['load_mode']
            expert_models[expert_name] = expert_model
            expert_params[expert_name] = curr_expert_params

        model = MoE_DQN_simple(experts=expert_models, expert_params=expert_params, decoder_args=decoder_args, seed=seed)

        return model


    else: 
        # encoder = None
        # if encoder_type == 'FINDER_encoder_PyG':
        #     encoder = GraphSAGE_Encoder_PyG_FINDER_v2(**encoder_args, seed=seed)
        #     decoder_args['embedding_size'] = encoder_args['embedding_size']
        # elif encoder_type.lower() == 'identity':
        #     encoder = IdentityEncoder(**encoder_args)
        #     decoder_args['embedding_size'] = encoder_args['num_node_features']
        # elif encoder_type.lower() == 'mega':
        #     encoder = MEGA_Encoder(**encoder_args)
        #     decoder_args['embedding_size'] = encoder_args['embedding_size']
        # else:
        #     raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # if pretrained_encoder_path is not None:
        #     checkpoint = torch.load(pretrained_encoder_path)
        #     encoder.load_state_dict(checkpoint)
        #     print("Successfully loaded pretrained encoder from: \n\n", pretrained_encoder_path)

        #     if pretrained_encoder_load_mode == 'freeze':
        #         for param in encoder.parameters():
        #             param.requires_grad = False
        #     elif pretrained_encoder_load_mode == 'finetune':
        #         for param in encoder.parameters():
        #             param.requires_grad = True
        #     else:
        #         raise ValueError(f"Unsupported pretrained encoder load mode: {pretrained_encoder_load_mode}")
            
        #     print("set encoder to {} mode".format(pretrained_encoder_load_mode))

        encoder = get_encoder(encoder_type, encoder_args, seed, pretrained_encoder_path, pretrained_encoder_load_mode)
        decoder_args['embedding_size'] = encoder_args['embedding_size']
        if model_type.lower() == 'finder_dqn':
            # return FINDER_DQN(encoder, seed=seed, decoder_args=decoder_args, encoder_args=encoder_args)
            return FINDER_DQN(encoder, seed=seed, decoder_args=decoder_args, encoder_args=encoder_args)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    

def get_optimizer(model, optimizer, optimizer_args):
    # return optimizer class
    if optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), **optimizer_args)
    
    else:
        raise NotImplementedError("Optimizer not implemented not supported!")
