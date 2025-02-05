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


EXP_CONFIG_PATH = os.path.join("exp_configs", 'SSL_exps')
EXPS = [
    "ORIGINAL_BA_FINDER.json",
    "MTSSL_MEGA_CrossAttention_freeze_BA_FINDER_noProc.json",

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

    
    agent = Q_CNDP_Agent(exp_config)

    with open(os.path.join(experiment_dir, "params.json"), "w") as f:
        json.dump({k: str(v) if callable(v) else v for k, v in exp_config.items()}, f, indent=4)

    agent.train()

    if exp_config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    
    for exp_path in EXPS_PATH:
        with open(exp_path, 'r') as f:
            exp_config = json.load(f)
        cProfile.run('main(exp_config)', 'profile_stats')

        # Print the profiling results
        p = pstats.Stats('profile_stats')
        p.strip_dirs().sort_stats('cumulative').print_stats(10)
        # main(exp)