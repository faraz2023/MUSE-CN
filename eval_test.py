import os
import json
import networkx as nx
from py_modules.baselines import get_baseline_sol
from py_modules.utils import calc_ANC_score, calc_pairwise_connectivity
from Q_CNDP import Q_CNDP_Agent
import pandas as pd
from py_modules.morPOP_utils import prep_G_for_tensor

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

def evaluate_experiments(exps, Gs, baselines, export_path, overwrite=False):
    results = []


    df_export_path = os.path.join(export_path, 'results.csv')

    if(not overwrite and os.path.exists(df_export_path)):
        df = pd.read_csv(df_export_path)
        results = df.to_dict(orient='records')
        print(f"Loaded {len(results)} results from {df_export_path}")


    else:
        print(f"No results found at {df_export_path}, creating new results file")


    for G_dict in Gs:
        G = G_dict['G']
        G_label = G_dict['G_label']
        budget = G_dict['budget']
        G_directory_path = os.path.join(export_path, G_label)
        os.makedirs(G_directory_path, exist_ok=True)

        nx.write_edgelist(G, os.path.join(G_directory_path, f"G.el"))
        
        G_pairwise_connectivity = calc_pairwise_connectivity(G, normalize=True, total_nodes=G.number_of_nodes())
        
        
        for exp in exps:
            row = {'G_label': G_label}
            exp_path = exp['exp_path']
            exp_label = exp['exp_label']
            
            # Find best model path
            best_model_path = find_best_model_path(exp_path)
            
            # Load agent and best model
            with open(os.path.join(exp_path, 'params.json'), 'r') as f:
                params = json.load(f)

            step_size = 1
            if (G.number_of_nodes() > 20000):
                step_size = int(budget * 0.01)
                # manually set device to cpu
                params['device'] = 'cpu'

            agent = Q_CNDP_Agent(params)

            print(f"Loading model from {best_model_path}")
            agent.LoadModel(best_model_path)
            print(f"Model loaded")
            
            # Calculate model solution
            removed_nodes = agent.calc_solution_nx(G, budget, step_size=step_size, print_progress=True)
            
            # Calculate metrics
            ANC_score, _ = calc_ANC_score(G, removed_nodes, return_scores=True, normalize=True, step_size=step_size)
            G_removed = G.copy()
            G_removed.remove_nodes_from(removed_nodes)
            pairwise_connectivity = calc_pairwise_connectivity(G_removed, normalize=True, total_nodes=G.number_of_nodes())
            
            row[f"ANC"] = ANC_score
            row[f"start_pairwise_connectivity"] = G_pairwise_connectivity
            row[f"end_pairwise_connectivity"] = pairwise_connectivity
            # row[f"removed_nodes"] = removed_nodes
            row[f"baseline"] = False
            row[f"method_name"] = exp_label


            # write removed nodes
            sol_path = os.path.join(G_directory_path, f"{exp_label}_sol.txt")
            with open(sol_path, 'w') as f:
                for node in removed_nodes:
                    f.write(f"{node}\n")

            results.append(row)

            
        
        # Calculate baseline solutions
        for baseline in baselines:
            row = {'G_label': G_label}
            print(f"Calculating {baseline} baseline")
            removed_nodes = get_baseline_sol(G, budget, baseline)
            step_size = 1
            if (G.number_of_nodes() > 20000):
                step_size = int(budget * 0.01)
            ANC_score, _ = calc_ANC_score(G, removed_nodes, return_scores=True, normalize=True, step_size=step_size)
            G_removed = G.copy()
            G_removed.remove_nodes_from(removed_nodes)
            pairwise_connectivity = calc_pairwise_connectivity(G_removed, normalize=True, total_nodes=G.number_of_nodes())
            
            row[f"ANC"] = ANC_score
            row[f"start_pairwise_connectivity"] = G_pairwise_connectivity
            row[f"end_pairwise_connectivity"] = pairwise_connectivity
            # row[f"removed_nodes"] = removed_nodes
            row[f"baseline"] = True
            row[f"method_name"] = baseline

            sol_path = os.path.join(G_directory_path, f"{baseline}_sol.txt")
            with open(sol_path, 'w') as f:
                for node in removed_nodes:
                    f.write(f"{node}\n")

            results.append(row)
        # export as df 
        df = pd.DataFrame(results)
        df.to_csv(df_export_path, index=False)
    
    return results

# Example usage
exps = [
    # {'exp_path': os.path.join("experiments", "MEGA_integration", "ORIGINAL_DiverseD_FINDER"), 'exp_label': "ORIGINAL_DiverseD_FINDER"},
    # {'exp_path': os.path.join("experiments", "MEGA_integration", "MTSSL_MEGA_finetune_DiverseD_FINDER_proc"), 'exp_label': "MTSSL_MEGA_finetune_DiverseD_FINDER_proc"},


    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "SW_FINDER_proc_imitation"), 'exp_label': "DQN SW_FINDER_proc_imitation"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "SW_FINDER_proc"), 'exp_label': "DQN SW_FINDER_proc"},
    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "SW_FINDER_proc_imitation"), 'exp_label': "MC SW_FINDER_proc_imitation"},
    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "SW_FINDER_proc"), 'exp_label': "MC SW_FINDER_proc"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "BA_FINDER_proc_imitation"), 'exp_label': "DQN BA_FINDER_proc_imitation"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "BA_FINDER_proc"), 'exp_label': "DQN BA_FINDER_proc"},
    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "BA_FINDER_proc_imitation"), 'exp_label': "MC BA_FINDER_proc_imitation"},
    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "BA_FINDER_proc"), 'exp_label': "MC BA_FINDER_proc"},

    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "SW_FINDER"), 'exp_label': "DQN SW_FINDER"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "BA_FINDER"), 'exp_label': "DQN BA_FINDER"},

    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "ORIGINAL_BA_FINDER"), 'exp_label': "DQN ORIGINAL_BA_FINDER"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "ORIGINAL_SW_FINDER"), 'exp_label': "DQN ORIGINAL_SW_FINDER"},

    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "BA_FINDER_proc_SingleLevel"), 'exp_label': "MC BA_FINDER_proc_SingleLevel"},
    # {'exp_path': os.path.join("experiments", "MC", "FINDER", "SW_FINDER_proc_SingleLevel"), 'exp_label': "MC SW_FINDER_proc_SingleLevel"},

    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "BA_FINDER_proc_ones_SingleLevel"), 'exp_label': "DQN BA_FINDER_proc_ones_SingleLevel"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "SW_FINDER_proc_ones_SingleLevel"), 'exp_label': "DQN SW_FINDER_proc_ones_SingleLevel"},

    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "BA_FINDER_proc_noNorm_ones_SingleLevel"), 'exp_label': "DQN BA_FINDER_proc_noNorm_ones_SingleLevel"},
    # {'exp_path': os.path.join("experiments", "DQN", "FINDER", "SW_FINDER_proc_noNorm_ones_SingleLevel"), 'exp_label': "DQN SW_FINDER_proc_noNorm_ones_SingleLevel"},

    


]

exp_names = [
    # "MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_proc",
    # "MTSSL_MEGA_CrossAttention_8l_finetune_BA_FINDER_proc",
    # "MTSSL_MEGA_CrossAttention_1l_finetune_BA_FINDER_proc"
    # "MTSSL_MEGA_CrossAttention_freeze_BA_FINDER_proc"
    # "ORIGINAL_BA_FINDER",
    # "BA_FINDER_proc",
    # "MTSSL_MEGA_finetune_BA_FINDER_proc",
    # "MTSSL_MEGA_finetune_DiverseD_FINDER_proc",
    # "MTSSL_MEGA_freeze_BA_FINDER_proc",
    # "ORIGINAL_DiverseD_FINDER"
    # "MoE_multitask_finetune_freeze_dedicated_encoder_BA",


    # 'MTSSL_MEGA_finetune_BA_FINDER_noProc',
    # 'MTSSL_MEGA_freeze_BA_FINDER_noProc',
    # 'MTSSL_MEGA_finetune_BA_FINDER_proc',
    # 'MTSSL_MEGA_freeze_BA_FINDER_proc',
    # 'ORIGINAL_BA_FINDER',
    # 'MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc',
    # 'MTSSL_MEGA_CrossAttention_4l_finetune_BA_FINDER_noProc',
    # 'MTSSL_MEGA_reset_BA_FINDER_noProc',

    # "MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc",
    # "MTSSL_MEGA_CrossAttention_1l_finetune_BA_FINDER_noProc",


    # "MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc",

    'MTSSL_MEGA_freeze_BA_FINDER_noProc',
    'MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc',
    'MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc',
    'MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc',

    'MTSSL_MEGA_finetune_BA_FINDER_noProc',
    'MTSSL_MEGA_reset_BA_FINDER_noProc'



]
# for exp_name in os.listdir(os.path.join("experiments", "MEGA_integration")):
for exp_name in exp_names:
    # if exp_name == 'ORIGINAL_BA_FINDER':
    #     continue
    # else:
    exps.append({'exp_path': os.path.join("experiments", "MEGA_integration", exp_name), 'exp_label': exp_name})

baselines =  ['random', 'degree', 'HDA', 'CI']




# NL_G = nx.read_edgelist('Data/NL_Day2/day2_0.el', create_using=nx.Graph(), nodetype=int)
# print(f"NL_G has {NL_G.number_of_nodes()} nodes and {NL_G.number_of_edges()} edges")
# # remove all nodes with degree 0
# NL_G.remove_nodes_from([node for node in NL_G.nodes() if NL_G.degree(node) == 0])
# print(f"After removing nodes with degree 0, NL_G has {NL_G.number_of_nodes()} nodes and {NL_G.number_of_edges()} edges")
# # relable nodes such that they are 0, n-1 with no gaps
# NL_G = nx.convert_node_labels_to_integers(NL_G)
# print(f"After relabeling, NL_G has {NL_G.number_of_nodes()} nodes and {NL_G.number_of_edges()} edges")

# NL_G = prep_G_for_tensor(f'Data/NL_Day2/day2_0.el')


Gs = [
    # {'G': nx.barabasi_albert_graph(100, 4, seed=42), 'G_label': 'BA_100_4', 'budget': 70},
    # {'G': nx.barabasi_albert_graph(200, 4, seed=42), 'G_label': 'BA_200_4', 'budget': 140},
    # {'G': nx.barabasi_albert_graph(500, 4, seed=42), 'G_label': 'BA_500_4', 'budget': 350},
    # {'G': nx.barabasi_albert_graph(1000, 4, seed=42), 'G_label': 'BA_1000_4', 'budget': 700},

    # {'G': nx.barabasi_albert_graph(200, 5, seed=42), 'G_label': 'BA_200_5', 'budget': 140},
    # {'G': nx.barabasi_albert_graph(200, 6, seed=42), 'G_label': 'BA_200_6', 'budget': 140},
    # {'G': nx.barabasi_albert_graph(200, 7, seed=42), 'G_label': 'BA_200_7', 'budget': 140},

    # {'G': nx.watts_strogatz_graph(100, 8, 0.1, seed=42), 'G_label': 'WS_100_8', 'budget': 70},
    # {'G': nx.watts_strogatz_graph(200, 8, 0.1, seed=42), 'G_label': 'WS_200_8', 'budget': 140},
    # {'G': nx.watts_strogatz_graph(500, 8, 0.1, seed=42), 'G_label': 'WS_500_8', 'budget': 350},
    # {'G': nx.watts_strogatz_graph(1000, 8, 0.1, seed=42), 'G_label': 'WS_1000_8', 'budget': 700},

    # {'G': nx.watts_strogatz_graph(200, 10, 0.1, seed=42), 'G_label': 'WS_200_10', 'budget': 140},
    # {'G': nx.watts_strogatz_graph(200, 12, 0.1, seed=42), 'G_label': 'WS_200_12', 'budget': 140},
    # {'G': nx.watts_strogatz_graph(200, 14, 0.1, seed=42), 'G_label': 'WS_200_14', 'budget': 140},

    # {'G': NL_G, 'G_label': 'NL Day 2', 'budget': NL_G.number_of_nodes() * 0.1},

]

### PPI networks
# for i in range(20):
for i in [13]:
    G = prep_G_for_tensor(f'Data/PPI_networks/graph_{i}.txt')
    Gs.append({'G': G, 'G_label': f'PPI_{i}', 'budget': int(G.number_of_nodes() * 0.1)})

    print(f"PPI_{i} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


# Benchmark
# for file in os.listdir('Data/Data_benchmark'):
#     if not file.endswith('.txt'):
#         print(f"Skipping {file} because it is not a txt file")
#         continue


#     G = prep_G_for_tensor(f'Data/Data_benchmark/{file}')
#     # if G.number_of_nodes() > 2000:
#     #     continue
#     Gs.append({'G': G, 'G_label': file.split('.')[0], 'budget': int(G.number_of_nodes() * 0.1)})

#     print(f"{file} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


export_path = os.path.join('Evals', 'MUSE_CN_PPI_ablations')
# export_path = os.path.join('Evals', 'PPI')
# export_path = os.path.join('Evals', 'MEGA_integration_PPI_2')
# export_path = os.path.join('Evals', 'MEGA_integration_benchmark_4_p10')    
# export_path = os.path.join('Evals', 'MEGA_integration_synthetic_2')
os.makedirs(export_path, exist_ok=True)
results = evaluate_experiments(exps, Gs,  baselines, export_path, overwrite=False)





