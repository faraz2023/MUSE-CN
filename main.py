from Q_CNDP import Q_CNDP_Agent
from py_modules.curriclum import curriculum
from py_modules.utils import get_device, set_seed, signal_handler
from datetime import datetime
import os, json
import signal
import wandb
import copy
import cProfile
import pstats
from memory_profiler import profile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
'''
EXP_CONFIG_PATH = os.path.join("exp_configs", 'SSL_exps')
# EXPS = ['SW_FINDER_proc_imitation.json', 'SW_FINDER_proc.json']
# EXPS = ['BA_MC_proc_imitation.json', 'BA_MC_proc.json', 'SW_MC_proc_imitation.json', 'SW_MC_proc.json']
# EXPS = ['BA_FINDER_proc_SingleLevel.json', 'SW_FINDER_proc_SingleLevel.json']

EXPS = [
    # "ORIGINAL_DiverseD_FINDER.json",
    # "MTSSL_MEGA_finetune_DiverseD_FINDER_proc.json",
    # 'MTSSL_MEGA_finetune_BA_FINDER_proc.json',
    # 'MTSSL_MEGA_freeze_BA_FINDER_proc.json',
    # 'BA_FINDER_proc.json', 
    'ORIGINAL_BA_FINDER.json',
    # 'MTSSL_freeze_BA_FINDER_proc.json',
    # 'MTSSL_finetune_BA_FINDER_proc.json',
    # 'MTSSL_MEGA_BA_FINDER_proc.json'
    ]
'''

# EXP_CONFIG_PATH = os.path.join("exp_configs", 'MoE_exps')
# EXPS = [
#     'precon_freeze_BA_FINDER_proc.json',
#     'precon_finetune_BA_FINDER_proc.json',
#     'plink_freeze_BA_FINDER_proc.json',
#     'plink_finetune_BA_FINDER_proc.json',
#     'pming_freeze_BA_FINDER_proc.json',
#     'pming_finetune_BA_FINDER_proc.json',
#     'pminsg_freeze_BA_FINDER_proc.json',
#     'pminsg_finetune_BA_FINDER_proc.json',
#     'pdecor_freeze_BA_FINDER_proc.json',
#     'pdecor_finetune_BA_FINDER_proc.json',
#     'ORIGINAL_BA_FINDER.json',

# ]

# EXP_CONFIG_PATH = os.path.join("exp_configs")
# EXPS = [
#     'RealWorld_FINDER_proc.json',

# ]


EXP_CONFIG_PATH = os.path.join("exp_configs", 'SSL_exps')
# EXP_CONFIG_PATH = os.path.join("exp_configs", 'SSL_exps')
EXPS = [
    # "MTSSL_MEGA_CrossAttention_freeze_BA_FINDER_proc.json",
    # "MTSSL_MEGA_CrossAttention_finetune_BA_FINDER_proc.json",
    # "MTSSL_MEGA_CrossAttention_freeze_BA_FINDER_proc.json", 
    # 'MoE_multitask_finetune_finetune_BA.json',

    # 'MTSSL_MEGA_finetune_BA_FINDER_noProc.json',
    # 'MTSSL_MEGA_freeze_BA_FINDER_noProc.json',
    # 'MTSSL_MEGA_finetune_BA_FINDER_proc.json',
    # 'MTSSL_MEGA_freeze_BA_FINDER_proc.json',
    
    # 'MTSSL_MEGA_reset_BA_FINDER_noProc.json'
    # 'MoE_multitask_finetune_freeze_dedicated_encoder_BA copy'


    # "MTSSL_MEGA_CrossAttention_freeze_BA_FINDER_noProc.json",
    # "MTSSL_MEGA_CrossAttention_finetune_BA_FINDER_noProc.json",
    # "PlayGround.json"

    # "MTSSL_MEGA_freeze_BA_FINDER_noProc_d128.json",
    # "ORIGINAL_BA_FINDER_d128.json",
    # "MTSSL_MEGA_finetune_BA_FINDER_noProc_d128.json",
    # "MTSSL_MEGA_reset_BA_FINDER_noProc_d128.json",

    # "MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc_d128.json",
    "MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc_d128.json",
    "MTSSL_MEGA_CrossAttention_6l_freeze_BA_FINDER_noProc_d128.json",


]

EXPS_PATH = [os.path.join(EXP_CONFIG_PATH, exp) for exp in EXPS]

# check all paths exist
for exp_path in EXPS_PATH:
    if not os.path.exists(exp_path):
        print(f"Experiment path {exp_path} does not exist")
        exit(1)


@profile
def main(exp_config):
    print("Starting experiment with seed: ", exp_config["seed"])
    set_seed(exp_config["seed"])

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Using device: {exp_config['device']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = exp_config["export_path"]

    # if export_path exists, exit with error 
    # if os.path.exists(exp_config["export_path"]):
    #     print("Experiment already exists")
    #     exit(1)

    # if experiment directory exists, re-read exp_config
    if os.path.exists(experiment_dir):
        if exp_config["overwrite"]:
            print("Overwriting existing experiment directory: ", experiment_dir)
            os.system(f"rm -rf {experiment_dir}")
        else:
            with open(os.path.join(experiment_dir, "params.json"), "r") as f:
                exp_config = json.load(f)

    os.makedirs(experiment_dir, exist_ok=True)

    exp_config["export_path"] = experiment_dir

    

    # Initialize wandb if enabled
    # if exp_config["use_wandb"]:
    #     wandb.init(project="MC-CNDP", config=exp_config)

    print("Calculating the acutal number of node features...")
    automatic_numb_node_featuees = len(exp_config.get("procedural_attrs", [])) + len(exp_config.get("contextual_attrs", []))
    if automatic_numb_node_featuees > 0 and exp_config.get('pretrained_encoder', None) is None\
        and 'MoE' not in exp_config.get('model_type', ''):
        exp_config["encoder_args"]["num_node_features"] = automatic_numb_node_featuees
        if 'prone' in exp_config['procedural_attrs']:
            exp_config["encoder_args"]["num_node_features"] += exp_config["procedural_attrs_args"].get('prone', {}).get('emb_size', 32) - 1
        if 'ones' in exp_config['procedural_attrs']:
            exp_config["encoder_args"]["num_node_features"] += exp_config["procedural_attrs_args"].get('ones', {}).get('num_features', 1) - 1
        automatic_numb_node_featuees = exp_config["encoder_args"]["num_node_features"]
        print("Warning: automatically setting automatic_numb_node_featuees to ", automatic_numb_node_featuees)
        print(f"Automatic in_channels value: {automatic_numb_node_featuees}")
        print(f"Automatic in_channels value overwrites in_channels to {automatic_numb_node_featuees}")

    else:
        if 'MoE' in exp_config.get('model_type', ''):
            print(f"Using MoE model, the number of node features to be determined at a later step.")
        elif exp_config.get('pretrained_encoder', None) is None:
                print(f"Using manually set in_channels value: {exp_config['encoder_args']['num_node_features']} | each node gets a feature vector of length {exp_config['encoder_args']['num_node_features']}")
        else:
            print(f"Using pretrained mode, the number of node features to be determined at a later step.")


    # exp_config['curriculum'][0]['num_training_iters'] = 1001
    
    agent = Q_CNDP_Agent(exp_config)

    with open(os.path.join(experiment_dir, "params.json"), "w") as f:
        json.dump({k: str(v) if callable(v) else v for k, v in exp_config.items()}, f, indent=4)

    agent.train()

    if exp_config["use_wandb"]:
        wandb.finish()

def train_moe_actor_critic(model, env, optimizer, ppo_epochs=4, clip_param=0.2, value_loss_coef=0.5, 
                          entropy_coef=0.01, max_grad_norm=0.5, num_episodes=1000):
    """
    Train the MoE model using PPO
    """
    for episode in range(num_episodes):
        # Storage for episode data
        states = []
        actions = []
        rewards = []
        values = []
        action_log_probs = []
        masks = []
        
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                q_pred, policy, value = model(state)
                action = policy.multinomial(1)
                action_log_prob = policy.log().gather(1, action)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action.item())
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            action_log_probs.append(action_log_prob)
            masks.append(1 - done)
            
            state = next_state
            episode_reward += reward
        
        # Compute returns and advantages
        returns = compute_returns(rewards, masks, values[-1])
        advantages = compute_gae(rewards, masks, values)
        
        # PPO update
        for _ in range(ppo_epochs):
            for state_batch, action_batch, old_log_prob_batch, return_batch, advantage_batch in \
                ppo_iter(states, actions, action_log_probs, returns, advantages):
                
                # Evaluate actions
                new_log_probs, new_values, entropy = model.evaluate_actions(state_batch, action_batch)
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, return_batch)
                
                # Total loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

def compute_returns(rewards, masks, last_value, gamma=0.99):
    returns = []
    R = last_value
    for r, mask in zip(reversed(rewards), reversed(masks)):
        R = r + gamma * R * mask
        returns.insert(0, R)
    return returns

def compute_gae(rewards, masks, values, gamma=0.99, tau=0.95):
    gae = 0
    advantages = []
    for r, mask, v, next_v in zip(reversed(rewards), reversed(masks),
                                 reversed(values[:-1]), reversed(values[1:])):
        delta = r + gamma * next_v * mask - v
        gae = delta + gamma * tau * mask * gae
        advantages.insert(0, gae)
    return advantages

def ppo_iter(states, actions, log_probs, returns, advantages, batch_size=64):
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        
        yield (states[batch_indices], actions[batch_indices],
               log_probs[batch_indices], returns[batch_indices],
               advantages[batch_indices])

if __name__ == "__main__":
    
    for exp_path in EXPS_PATH:
        print("Number of experiments: ", len(EXPS_PATH))
        with open(exp_path, 'r') as f:
            exp_config = json.load(f)
        cProfile.run('main(exp_config)', 'profile_stats')

        # Print the profiling results
        p = pstats.Stats('profile_stats')
        p.strip_dirs().sort_stats('cumulative').print_stats(10)
        # main(exp)