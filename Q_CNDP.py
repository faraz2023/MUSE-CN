import torch
from torch import nn
import torch_sparse
import numpy as np
import networkx as nx
import random
import time
import gc
from tqdm import tqdm
import traceback

import cy_modules.PrepareBatchGraph as PrepareBatchGraph
import cy_modules.graph as graph
import cy_modules.nstep_replay_mem as nstep_replay_mem
import cy_modules.nstep_replay_mem_prioritized as nstep_replay_mem_prioritized
import cy_modules.mvc_env as mvc_env
import cy_modules.utils as utils
from py_modules.model_utils import get_model, get_optimizer
from py_modules.utils import load_real_world_graph, bfs_subgraph
import logging
from datetime import datetime
import os
from copy import deepcopy
import json
from torch.autograd import Variable
import wandb
import logging
from datetime import datetime



def setup_logger(export_path): # issue with the logger getting closed in MoE experiments
    """Set up logging to both file and stdout with append mode"""
    logger_name = 'Q_CNDP'
    logger = logging.getLogger(logger_name)
    
    # Close any existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler in append mode ('a' instead of 'w')
    log_path = os.path.join(export_path, 'logs.log')
    os.makedirs(export_path, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add a divider line when starting new session
    logger.info("\n" + "="*80)
    logger.info("New session started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("="*80 + "\n")
    
    return logger


class Q_CNDP_Agent:
    def __init__(self, config):
        """Initialize Q-CNDP Agent with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing all experiment parameters
        """
        self.logger = setup_logger(config['export_path'])

        self.logger.info("Initializing Q-CNDP Agent...")

        # Core training parameters
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = config.get('dtype', torch.float32) # Default to float32
        print("SELF.DTYPE: ", self.dtype)
        self.logger.info(f"Using dtype: {self.dtype}")
        self.logger.info(f"Device: {self.device}")
        self.seed = config['seed']
        self.use_wandb = config['use_wandb']


        # For the MC algorithm we may need to include the remove ratio meassure
        self.RL_algorithm = config['RL_algorithm']
        
        # Model configuration
        self.model_type = config['model_type']
        self.is_MoE = False


        if 'MoE' in self.model_type:
            self.is_MoE = True
            self.experts_args = config['experts_args']
            self.encoder_args = {'num_node_features': 2}

        else:
            config['pretrained_encoder'] = config.get('pretrained_encoder', None)

            if(config['pretrained_encoder'] is None):
                self.pretrained_encoder = False
                self.encoder_type = config['encoder_type']
                self.encoder_args = config['encoder_args']
            else:
                self.pretrained_encoder = True
                self.pretrained_encoder_vars = config['pretrained_encoder']
                with open(config['pretrained_encoder']['config_file_path'], 'r') as f:
                    config_vars = json.load(f)
                self.encoder_type = config_vars['encoder_type']
                self.encoder_args = config_vars['encoder_args']

        self.decoder_args = config['decoder_args']
        
        # Feature configuration
        self.procedural_attrs = config.get('procedural_attrs', None)
        self.procedural_attrs_args = config.get('procedural_attrs_args', None)
        self.contextual_attrs = config.get('contextual_attrs', [])


        if self.contextual_attrs not in [None, []]:
            raise("Not implemented yet")

        if self.procedural_attrs not in [None, []]:
            self.logger.info(f"Procedural attrs detected, automatically setting encoder num_node_features to breakdown:")

            self.encoder_args['num_node_features'] = len(self.procedural_attrs)

            prone_dim, ones_dim = 0, 0
            if 'prone' in self.procedural_attrs:
                prone_dim = self.procedural_attrs_args.get('prone', {}).get('emb_size', 32)
                self.encoder_args['num_node_features'] += prone_dim - 1
                self.logger.info(f"  - prone: {prone_dim}")
            if 'ones' in self.procedural_attrs:
                ones_dim = self.procedural_attrs_args.get('ones', {}).get('num_features', 1)
                self.encoder_args['num_node_features'] += ones_dim - 1
                self.logger.info(f"  - ones: {ones_dim}")
            
            self.logger.info(f"  - total: {self.encoder_args['num_node_features']}")
        
        self.logger.info(f"Encoder num_node_features: {self.encoder_args['num_node_features']}, breakdown: ")


        
        # RL specific parameters
        self.rl_config = config['RL_algorithm_args']
        self.gamma = self.rl_config['gamma']
        self.aggregatorID = config['aggregatorID']

        self.SSL_losses = config['SSL_losses']
        self.SSL_loss_args = config['SSL_loss_args']
        
        # Buffer configuration
        self.buffer_capacity = config['buffer_capacity']
        
        # Optimizer configuration
        self.optimizer_type = config['optimizer']
        self.optimizer_args = config['optimizer_args']
        
        # Reward configuration
        self.reward_function = config['reward_function']
        
        # Curriculum configuration
        self.curriculum = config['curriculum']

        # Export configuration
        self.export_path = config['export_path']
        self.exp_name = config['exp_name']
        
        # Store full config for logging purposes
        self.config = config


        self.IsHuberloss = False
        if(self.IsHuberloss):
            self.loss = nn.HuberLoss(delta=1.0)
        else:
            self.loss = nn.MSELoss()

        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)


        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.utils = utils.py_Utils()
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            # following paramters should be passed by RL algorithm args
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(self.epsilon,self.alpha,self.beta,self.beta_increment_per_sampling,self.TD_err_upper,self.buffer_capacity)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(self.buffer_capacity)

        self.setup_envs()
        torch.set_num_threads(16)

        if self.is_MoE:
            self.DQN = get_model(self.model_type, encoder_type=None, encoder_args={}, decoder_args=self.decoder_args, seed=self.seed,\
                        pretrained_encoder_path=None, pretrained_encoder_load_mode=None,
                        experts_args=self.experts_args)
            
            print("MoE model created, number of experts: ", len(self.experts_args['experts']))
            print("Number of parameters in MoE model: ", sum(p.numel() for p in self.DQN.parameters()))
        
        else:
            if self.pretrained_encoder:
                self.DQN = get_model(self.model_type, self.encoder_type, self.encoder_args, self.decoder_args, self.seed,\
                    self.pretrained_encoder_vars['model_file_path'], self.pretrained_encoder_vars.get('load_mode', 'freeze'))
            else:
                self.DQN = get_model(self.model_type, self.encoder_type, self.encoder_args, self.decoder_args, self.seed)
        self.DQN.to(self.device)
        self.DQN = self.DQN.to(dtype=self.dtype)
        self.logger.info(f"Model {self.model_type} initialized and moved to {self.device} with dtype {self.dtype}")

        if self.pretrained_encoder and self.pretrained_encoder_vars.get('model_path', None) and 'MoE' not in self.model_type:
            self.logger.info(f"Attempting to load pretrained model weights for the main DQN from {self.pretrained_encoder_vars.get('model_path')}")
            self.DQN.load_state_dict(torch.load(self.pretrained_encoder_vars.get('model_path'), map_location=self.device))

        if(self.RL_algorithm == 'DQN'):
            self.logger.info("Creating target model...")

            if self.is_MoE:
                self.DQN_T = deepcopy(self.DQN)


            else: 
                if self.pretrained_encoder:
                    # shouldn't matter because we take snapshot, but just keeping it in
                    self.DQN_T = get_model(self.model_type, self.encoder_type, self.encoder_args, self.decoder_args, self.seed,\
                        self.pretrained_encoder_vars['model_file_path'], self.pretrained_encoder_vars.get('load_mode', 'freeze')) # shouldn't matter freeze or finetune but keeping it in for now. 
                else:
                    self.DQN_T = get_model(self.model_type, self.encoder_type, self.encoder_args, self.decoder_args, self.seed)
            self.DQN_T.to(self.device)
            self.DQN_T = self.DQN_T.to(dtype=self.dtype)
            self.logger.info(f"Target model {self.model_type} initialized and moved to {self.device} with dtype {self.dtype}")
            self.TakeSnapShot()

        self.optimizer = get_optimizer(self.DQN, self.optimizer_type, self.optimizer_args)
        self.logger.info(f"Created optimizer: {self.optimizer_type}")


        pytorch_total_params = sum(p.numel() for p in self.DQN.parameters())
        self.logger.info(f"EXP: {self.exp_name} \t Agent created with total number of parameters: {pytorch_total_params}")

        self.inf = float('inf')


    def TakeSnapShot(self):
        if (self.RL_algorithm == 'DQN'):
            self.DQN_T.load_state_dict(self.DQN.state_dict())
        else:
            self.logger.error("Snapshot only applicable to DQN training")
            raise ValueError("Snapshot only applicable to DQN training")

    def set_export_path(self, path):
        self.export_path = path
        
    def setup_envs(self):
        self.num_env = self.rl_config['num_env']
        self.max_episode_length = self.rl_config['max_episode_length']
        for i in range(self.num_env):
            # if reward_function is cndp
            if self.reward_function == 'cndp':
                self.logger.info("Setting the reward function to cndp...")
                self.env_list.append(mvc_env.py_MvcEnv(self.max_episode_length))
            elif 'dcndp' in self.reward_function:
                # format is k-dcndp, export k as the integer 
                k = int(self.reward_function.split('-')[0])
                self.logger.info("Setting the reward function to k-dcndp with k: ", k)
                if (k < 1): raise ValueError("k must be greater than 0")
                self.env_list.append(mvc_env.py_MvcEnv(self.max_episode_length, "k-dcndp", k))
            else:
                self.logger.error("Unknown reward function")
                raise ValueError("Unknown reward function")
            self.g_list.append(graph.py_Graph())
            
        if self.reward_function == 'cndp':
            self.test_env = mvc_env.py_MvcEnv(self.max_episode_length)
        elif 'dcndp' in self.reward_function:
            # format is k-dcndp, export k as the integer 
            k = int(self.reward_function.split('-')[0])
            if (k < 1): raise ValueError("k must be greater than 0")
            self.test_env = mvc_env.py_MvcEnv(self.max_episode_length, "k-dcndp", k)


    def InsertGraph(self,g,is_test=False):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, g)
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, g)

    def GenCyNetwork(self, g, node_attributes=None):    #networkx2four
        edges = g.edges()
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
        else:
            A = np.array([0])
            B = np.array([0])
        if node_attributes is not None:
            return graph.py_Graph(len(g.nodes()), len(edges), A, B, node_attributes)
        else:
            g = graph.py_Graph(len(g.nodes()), len(edges), A, B)
            if(self.procedural_attrs):
                g.set_procedural_attributes(self.procedural_attrs, self.procedural_attrs_args)

            return g

    def get_random_sample(self, large_G, large_G_attrs, sample_size, return_nx=False) -> graph.py_Graph:
        # select a random node and get a random subgrpah with bfs_subgraph()
        random_root_node = random.choice(range(large_G.number_of_nodes()))
        if(self.contextual_attrs):
            new_g_nx, new_g_attributes = bfs_subgraph(large_G, random_root_node, sample_size, node_attributes=large_G_attrs)
            new_g = self.GenCyNetwork(new_g_nx, node_attributes=new_g_attributes)
        else: 
            new_g_nx = bfs_subgraph(large_G, random_root_node, sample_size)
            new_g = self.GenCyNetwork(new_g_nx)
                

        if(return_nx==True):
                return new_g, new_g_nx
        return new_g
    

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', ''
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(self.GenCyNetwork(g), solutions)
        return Robustness, sol

    def randomize_realworld_train_graphs(self, level, level_G, node_attrs_df=None):
        self.logger.info('Randomizing new training graphs from real world network...')

        self.ClearTrainGraphs()
        level_min_n = level['min_n']
        level_max_n = level['max_n']
        g_instance_number = level['num_graphs']
        for i in tqdm(range(g_instance_number)):
            sample_size = random.randint(level_min_n, level_max_n)
            g = self.get_random_sample(level_G, node_attrs_df, sample_size)
            #print("Shape of g attrs in randomise_train_graphs:", g.node_attributes.shape)
            self.InsertGraph(g, is_test=False)

    # def PrepareValidData(self, level):
    #     self.logger.info(f"Preparing validation data for level: \n{level}...")
    #     result_degree = 0.0
    #     level_type = level['type']

    #     if level_type == 'real-world':
    #         G, node_attrs_df = load_real_world_graph(level['edge_list_path'], level.get('node_attributes_path'))
    #     for i in tqdm(range(level['num_valid'])):
    #         if(level_type == 'real-world'):
    #             sample_size = level['sample_size']
    #             g_cy, g_nx = self.get_random_sample(G, node_attrs_df, sample_size, return_nx=True)
    #         else:
    #             generator = level['generator']
    #             generator = getattr(nx, level['generator'])
    #             min_n, max_n = level['min_n'], level['max_n']
    #             curr_n = random.randint(min_n, max_n)
    #             g_nx = generator(curr_n, **level['args'])
    #             g_cy = self.GenCyNetwork(g_nx)
    #         g_degree = g_nx.copy()
    #         val_degree, sol = self.HXA(g_degree, 'HDA')
    #         result_degree += val_degree
    #         self.InsertGraph(g_cy, is_test=True)
    #     self.logger.info('Validation of HDA: %.6f'%(result_degree / level['num_valid']))

    def PrepareValidData(self, level):
        self.logger.info(f"Preparing validation data for level:\n{level}...")
        result_degree = 0.0
        level_type = level['type']

        # If real-world, load the entire big graph first
        if level_type == 'real-world':
            # node attributes are not used in current version but are the code is kept for extendability
            G, node_attrs_df = load_real_world_graph(level['edge_list_path'], level.get('node_attributes_path'))
            def new_validation_graph():
                min_n, max_n = level['min_n'], level['max_n']
                sample_size = random.randint(min_n, max_n)
                # returns (g_cy, g_nx)
                return self.get_random_sample(G, node_attrs_df, sample_size, return_nx=True)

        elif level_type == 'synthetic':
            # Single generator approach
            def new_validation_graph():
                generator = getattr(nx, level['generator'])
                min_n, max_n = level['min_n'], level['max_n']
                curr_n = random.randint(min_n, max_n)
                g_nx = generator(curr_n, **level['args'])
                return (self.GenCyNetwork(g_nx), g_nx)

        elif level_type == 'synthetic_diverse':
            # Emulate the "diverse" approach as in gen_synthetic_graphs
            generators_args = level['generators_args']
            # Convert to list of (generator_name, config_dict) for random selection
            gen_config_items = list(generators_args.items())
            gen_weights = [cfg['weight'] for _, cfg in gen_config_items]

            def new_validation_graph():
                # Randomly pick one of the graph families
                chosen_idx = random.choices(range(len(gen_config_items)), weights=gen_weights, k=1)[0]
                gen_name, cfg = gen_config_items[chosen_idx]
                generator = getattr(nx, gen_name)
                curr_n = random.randint(cfg['min_n'], cfg['max_n'])
                g_nx = generator(curr_n, **cfg['args'])
                return (self.GenCyNetwork(g_nx), g_nx)

        else:
            raise ValueError(f"Unsupported level type: {level_type}")

        # Actually generate and evaluate the num_valid graphs
        for _ in tqdm(range(level['num_valid'])):
            g_cy, g_nx = new_validation_graph()

            # Compute HDA solution  (or any other baseline as needed)
            g_degree = g_nx.copy()
            val_degree, _ = self.HXA(g_degree, 'HDA')
            result_degree += val_degree

            # Insert the generated graph into TestSet
            self.InsertGraph(g_cy, is_test=True)

        self.logger.info("Validation of HDA: %.6f", (result_degree / level['num_valid']))

    def simulate_game_round(self, n_traj, epsilon, imitation_algorithm=None, imitation_deterministic=False):

        num_env = len(self.env_list)
        n = 0
        TrainSet = self.TrainSet
        if(self.RL_algorithm == 'DQN'):
            n_step = self.rl_config.get('n_steps', 1)
        elif self.RL_algorithm == 'MC':
            n_step = 1
        else:
            self.logger.error(f"Unsupported RL algorithm: {self.RL_algorithm}")
            raise ValueError(f"Unsupported RL algorithm: {self.RL_algorithm}")
            
        sample = TrainSet.Sample()

        while n < n_traj:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                    g_sample= TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= n_traj:
                break

            explore = False
            if random.uniform(0,1) >= epsilon:

                pred = self.predict_with_current_qnet(self.g_list, [env.action_list for env in self.env_list])
            else:
                explore = True

            for i in range(num_env):
                if (explore):
                    if imitation_algorithm:
                        a_t = self.env_list[i].imitateAction(imitation_algorithm, imitation_deterministic)
                    else:
                        a_t = self.env_list[i].randomAction()
                else:
                    a_t = np.argmax(pred[i])


                self.env_list[i].step(a_t)
                # get covered
                covered = self.env_list[i].covered_set
                # set new procedural node attributes if needed
                if(self.procedural_attrs):
                    self.g_list[i].set_procedural_attributes(self.procedural_attrs, self.procedural_attrs_args, covered=covered)


    def train(self):

        training_log = {}
        self.logger.info("Training the agent...")
        if(self.use_wandb):
            wandb.init(project="Q-CNDP", name=self.exp_name)
            wandb.watch(self.DQN, log='all')

        levels_export_path = os.path.join(self.export_path, "levels")
        os.makedirs(levels_export_path, exist_ok=True)


        for i, level in enumerate(self.curriculum):


            self.logger.info(f"Training on level {i}...")
            level['level_export_path'] = os.path.join(levels_export_path, f"level_{i}")

            # if log level exists, load it
            if os.path.exists(os.path.join(level['level_export_path'], 'level_log.json')):
                with open(os.path.join(level['level_export_path'], 'level_log.json'), 'r') as f:
                    level_log = json.load(f)

                if level_log.get('done'):
                    self.logger.info(f"Level {i} already completed, skipping...")
                    continue
            

            level_log = self.train_level(level)

            training_log[f"level_{i}"] = level_log


            with open(os.path.join(self.export_path, 'training_log.json'), 'w') as f:
                json.dump(training_log, f, indent=4)




    def train_level(self, level):

        curr_level_export_path = level['level_export_path']
        level_models_export_path = os.path.join(curr_level_export_path, 'models')
        os.makedirs(level_models_export_path, exist_ok=True)
        eps_start = level['epsilon_start']
        eps_end = level['epsilon_end']
        eps_step = level['epsilon_step']


        level_imitation_algorithm = level.get('imitation_algorithm', None)
        level_imitation_deterministic = level.get('imitation_deterministic', False)

        self.logger.info(f"Level's imitation algorithm: {level_imitation_algorithm}, deterministic: {level_imitation_deterministic}")

        # if log level exists, load it

        level_log = {
                "training_iterations": [],
                'multibatch_durations': [],
                "validation_scores": [],
                'checkpoint_vcs': []
        }

        iter_offset = 0
        if os.path.exists(os.path.join(curr_level_export_path, 'level_log.json')):
            with open(os.path.join(curr_level_export_path, 'level_log.json'), 'r') as f:
                level_log = json.load(f)

            if level_log.get('checkpoint_vcs'):
                last_checkpoint_vc = level_log['checkpoint_vcs'][-1]['vc']
                self.logger.info(f"Resuming training from current level with iteration {level_log['checkpoint_vcs'][-1]['iteration']} last checkpoint VC: {last_checkpoint_vc}")
                # load the last checkpoint
                self.LoadModel(os.path.join(level_models_export_path, f"model_iter_{level_log['checkpoint_vcs'][-1]['iteration']}.ckpt"))
                iter_offset = level_log['checkpoint_vcs'][-1]['iteration']
            else:
                self.logger.info(f"Starting training from level {iter_offset}")


        self.logger.info(f"Preparing validation data for level {level}...")
        self.PrepareValidData(level)

        if(level['type'] == 'real-world'):
            #node attributes are not used in current version but are the code is kept for extendability
            level_G, _ = load_real_world_graph(level['edge_list_path'], level.get('node_attributes_path'))
            self.randomize_realworld_train_graphs(level, level_G)
            self.logger.info("Generated randomized real world instances")
        else:
            self.gen_synthetic_graphs(level)
            self.logger.info("Generated synthetic instances")


        n_cndp_rounds = level.get('num_initial_game_rounds', 10)
        self.logger.info(f"Playing {n_cndp_rounds} rounds of the game...")
        # playing some rounds of the game
        for _ in range(n_cndp_rounds):
            # self.simulate_game_round(100, eps_start)
            self.simulate_game_round(10, eps_start, level_imitation_algorithm, level_imitation_deterministic)

        if (self.RL_algorithm == 'DQN'):
            self.TakeSnapShot()

        
        

        multibatch_size = 100
        
        # checkpoint interval % validation interval must be 0
        validation_interval = 500
        checkpoint_interval = 1000
        update_train_graphs_interval = 5000
        simulate_game_round_interval = 100

        num_training_iters = level['num_training_iters']


        start_time = time.time()
        multibatch_interval_time = time.time()
        for iter in range(iter_offset, num_training_iters):
            
            losses_dict = self.Fit(level['batch_size'])
            level_log['training_iterations'].append({"iteration": iter, **losses_dict})


            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)



            if(iter % update_train_graphs_interval == 0):
                self.logger.info("Generating new training graphs...")
                if(level['type'] == 'real-world'):
                    self.randomize_realworld_train_graphs(level, level_G)
                    self.logger.info("Generated randomized real world instances")
                else:
                    self.gen_synthetic_graphs(level)
                    self.logger.info("Generated synthetic instances")

            if iter % simulate_game_round_interval == 0:
                self.simulate_game_round(10, eps, level_imitation_algorithm, level_imitation_deterministic)


            if (self.use_wandb):
                wandb.log({"epsilon": eps}, step=iter)

            if (iter % multibatch_size == 0 and iter != 0):
                # get average losses for the past multibatch_size iterations
                avg_losses_dict = {key: sum(level_log['training_iterations'][-multibatch_size:][i][key] for i in range(multibatch_size)) / multibatch_size for key in losses_dict.keys()}
                multibatch_duration = time.time() - multibatch_interval_time
                multibatch_interval_time = time.time()

                self.logger.info(f'iteration: {iter} | avg loss: {round(avg_losses_dict["loss"], 4)}, ' +
                               f'avg loss_rl: {round(avg_losses_dict["loss_rl"], 4)} | ' +
                               f'avg loss_reconstruction: {round(avg_losses_dict["loss_reconstruction"], 4)} | ' +
                               f'{multibatch_size} iterations total time: {multibatch_duration:.2f}s\n')
                level_log['multibatch_durations'].append({'iteration': iter, 'duration': multibatch_duration})

            if (iter % validation_interval == 0):
                gc.collect()
                frac = 0.0
                test_start = time.time()
                n_valid = level['num_valid']
                for idx in range(n_valid):
                    frac += self.test_on_gid(idx)
                test_end = time.time()
                test_time = test_end - test_start

                level_log['validation_scores'].append({'iteration': iter, 'vc': frac/n_valid,
                                                        'test_duration': test_end - test_start, 'epsilon': eps})
                self.logger.info('Validation: iteration %d, eps %.4f, average size of vc:%.6f'%(iter, eps, frac/n_valid))
                self.logger.info('\tvalidation on %s graphs time: %.2fs'%(n_valid, test_time))
                if(self.use_wandb):
                    validation_vc = frac/n_valid
                    wandb.log({'validation_vc': validation_vc}, step=iter)
                    wandb.log({'test_time': test_time}, step=iter)

                if (iter % checkpoint_interval == 0 and iter != 0):
                    self.SaveModel(os.path.join(level_models_export_path, f"model_iter_{iter}.ckpt"))
                    # save the log
                    with open(os.path.join(curr_level_export_path, 'level_log.json'), 'w') as f:
                        json.dump(level_log, f, indent=4)
                    self.logger.info(f"Iter: {iter} | checkpoint saved")
                    level_log['checkpoint_vcs'].append({'iteration': iter, 'vc': frac/n_valid})

            if (self.RL_algorithm == 'DQN' and iter % level['target_update_freq'] == 0 and iter != 0):
                self.TakeSnapShot()
                self.logger.info(f"Iter: {iter} | target model updated")



            if(self.use_wandb):
                wandb.log(losses_dict, step=iter)

        
        level_log['done'] = True
        # save the log
        with open(os.path.join(curr_level_export_path, 'level_log.json'), 'w') as f:
            json.dump(level_log, f, indent=4)
        self.logger.info("Training for level completed")


        return level_log
    
    def Fit(self, batch_size):
        sample = self.nStepReplayMem.Sampling(batch_size)
        # print("sample: ", sample)
        # print("sample reward: ", sample.list_rt)
        # print("sample list_normalized_rt: ", sample.list_normalized_rt)
        sample_is_not_all_terminal = False
        
        for i in range(batch_size):
            if (not sample.list_term[i]):
                sample_is_not_all_terminal = True
                break
        if sample_is_not_all_terminal:
            if (self.RL_algorithm == 'DQN'):
                if self.IsDoubleDQN:
                    double_list_pred = self.predict_with_current_qnet(sample.g_list, sample.list_s_primes, batch_size=batch_size)
                    double_list_predT = self.predict_with_snapshot(sample.g_list, sample.list_s_primes, batch_size=batch_size)
                    list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
                else:
                    list_pred = self.predict_with_snapshot(sample.g_list, sample.list_s_primes, batch_size=batch_size)
            elif(self.RL_algorithm == 'MC'):
                list_pred = self.predict_with_current_qnet(sample.g_list, sample.list_s_primes, batch_size=batch_size)
            else:
                self.logger.error(f"Unsupported RL algorithm: {self.RL_algorithm}")
                raise ValueError(f"Unsupported RL algorithm: {self.RL_algorithm}")


        # here, only samples for the actual actions are passed to the models
        # not the whole graphs and individual actions
        list_target = np.zeros([batch_size, 1])

        for i in range(batch_size):
            if self.RL_algorithm == 'DQN':
                # DQN target calculation
                q_rhs = 0

                ## this is weird since for the actual n_step, it should be
                # q_hrs = (self.gamma ** n_step) * list_pred[i]
                if (not sample.list_term[i]):
                    if self.IsDoubleDQN:
                        q_rhs = self.gamma * list_pred[i]
                    else:
                        q_rhs = self.gamma * max(list_pred[i])
                q_rhs += sample.list_rt[i]
                list_target[i] = q_rhs
            elif self.RL_algorithm == 'MC':
                # Monte Carlo target calculation - use actual returns
                list_target[i] = sample.list_normalized_rt[i]
            else:
                self.logger.error(f"Unsupported RL algorithm: {self.RL_algorithm}")
                raise ValueError(f"Unsupported RL algorithm: {self.RL_algorithm}")
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target, batch_size)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target, batch_size)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target, batch_size):
        '''
        double loss = 0.0
        n_graphs = len(g_list)
        i, j, bsize = 0, 0, 0
        for i in range(0,n_graphs,self.BATCH_SIZE):
            bsize = self.BATCH_SIZE
            if (i + self.BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)
        '''
        pass

    def fit(self,g_list,covered,actions,list_target, batch_size):
        loss_values = 0.0
        losses_dict = {
            'loss': 0,
            'loss_rl': 0,
            'loss_reconstruction': 0
        }
        n_graphs = len(g_list)
        i, j, bsize = 0, 0, 0
        for i in range(0,n_graphs,batch_size):
            self.optimizer.zero_grad()

            bsize = batch_size
            if (i + batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            data = self.SetupPyGAll(batch_idxes, g_list, covered, actions=actions, target=list_target, train=True)


            q_pred, cur_message_layer = self.DQN.train_forward(data)

            l_d = self.calc_loss(data, q_pred, cur_message_layer)
            loss = l_d['loss']
            loss.backward()
            self.optimizer.step()


            for key in losses_dict.keys():
                losses_dict[key] += l_d[key].item()*bsize

        for key in losses_dict.keys():
            losses_dict[key] /= len(g_list)

        return losses_dict

    def calc_loss(self, data, q_pred, cur_message_layer) :
        ## first order reconstruction loss
        #OLD loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer,0,1),\
        #    torch.matmul(self.inputs['laplacian_param'], cur_message_layer)))


        if self.IsPrioritizedSampling:
            self.TD_errors = torch.sum(torch.abs(data['target'] - q_pred), dim=1)    # for updating Sumtree
            if self.IsHuberloss:
                pass
                #loss_rl = self.loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                pass
                #loss_rl = torch.sum(self.ISWeights * self.loss(self.target, q_pred))
        else:
            if self.IsHuberloss:
                pass
                #loss_rl = self.loss(self.inputs['target'], q_pred)
            else:
                loss_rl = self.loss(data['target'], q_pred)

        loss = loss_rl

        loss_dict = {
            'loss': loss,
            'loss_rl': loss_rl,
        }

        if 'recon_loss' in self.SSL_losses:
            recon_loss_args = self.SSL_loss_args['recon_loss']
            alpha = recon_loss_args['alpha']
            loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer,0,1),\
                torch_sparse.spmm(data['laplacian_param']['index'], data['laplacian_param']['value'],\
                data['laplacian_param']['m'], data['laplacian_param']['n'],\
             cur_message_layer)))


            edge_num = torch.sum(data['n2nsum_param']['value'])



            loss_recons = torch.divide(loss_recons, edge_num)

            loss = torch.add(loss_rl, loss_recons, alpha = alpha)

            loss_dict['loss_reconstruction'] = loss_recons

        

        return loss_dict



    
    def predict(self,g_list,covered, batch_size=128, isSnapSnot=False, return_embedding=False):
        #print("Predict called!")
        n_graphs = len(g_list)
        i, j, k, bsize = 0, 0, 0, 0
        for i in range(0, n_graphs, batch_size):
            bsize = batch_size
            if (i + batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            data, idx_map_list = self.SetupPyGAll(batch_idxes, g_list, covered, train=False)
            #Node input is NONE for not costed scnario
            if isSnapSnot:
                result = self.DQN_T.test_forward(data, return_embedding, dtype=self.dtype)
            else:
                result = self.DQN.test_forward(data, return_embedding, dtype=self.dtype)
            # TOFIX: line below used to be raw_output = result[0]. This is weird because results is supposed to be 
            # [node_cnt, 1] (Q-values per node). And indeed it resulted in an error! I have fixed it by the line below
            # look inito it later.
            if return_embedding:
                result, embeddings = result
                embeddings = embeddings.cpu().detach().numpy()
            raw_output = result[:,0]
            pos = 0
            pred = []
            if(return_embedding):
                final_embeddings = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                if return_embedding:
                    cur_embedding = np.zeros((len(idx_map), embeddings.shape[1]))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -self.inf
                        if return_embedding:
                            cur_embedding[k] = np.zeros(embeddings.shape[1])
                    else:
                        cur_pred[k] = raw_output[pos]
                        if return_embedding:
                            cur_embedding[k] = embeddings[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -self.inf
                pred.append(cur_pred)
                if return_embedding:
                    final_embeddings.append(cur_embedding)
            if (pos != len(raw_output)):
                self.logger.warning(f"Warning: pos != len(raw_output)\n\t pos: {pos} \n\t len(raw_output): {len(raw_output)}")
            #assert (pos == len(raw_output))
        if(return_embedding):
            embeddings = np.array(final_embeddings)
            #print('pred[0]: ', pred[0].shape, 'embeddings: ', embeddings[0].shape)
            return pred, embeddings
        return pred


    def predict_with_snapshot(self,g_list,covered, batch_size=1, return_embedding=False):
        result = self.predict(g_list, covered, batch_size=batch_size, isSnapSnot=True, return_embedding=return_embedding)
        return result

    def predict_with_current_qnet(self,g_list,covered, batch_size=1, return_embedding=False):
        result = self.predict(g_list, covered, batch_size=batch_size, isSnapSnot=False, return_embedding=return_embedding)
        return result

    def predict_chunked(self, g_list, covered, node_chunk_size=20000, return_embedding=False):
        """Memory-scalable inference for very large graphs (ADDITIVE).

        Mirrors `predict` (batch_size=1) but calls the current Q-net's
        `test_forward_chunked`, which evaluates the cross-attention distillation
        and the decoder outer-product in node chunks. Numerically identical to
        `predict_with_current_qnet` — only the peak memory differs.
        """
        n_graphs = len(g_list)
        pred = []
        final_embeddings = [] if return_embedding else None
        for i in range(n_graphs):
            batch_idxes = np.int32([i])
            data, idx_map_list = self.SetupPyGAll(batch_idxes, g_list, covered, train=False)
            result = self.DQN.test_forward_chunked(
                data, return_embedding=return_embedding, dtype=self.dtype,
                node_chunk_size=node_chunk_size)
            if return_embedding:
                result, embeddings = result
                embeddings = embeddings.cpu().detach().numpy()
            raw_output = result[:, 0].cpu().detach().numpy()
            idx_map = idx_map_list[0]
            cur_pred = np.zeros(len(idx_map))
            if return_embedding:
                cur_embedding = np.zeros((len(idx_map), embeddings.shape[1]))
            pos = 0
            for k in range(len(idx_map)):
                if idx_map[k] < 0:
                    cur_pred[k] = -self.inf
                else:
                    cur_pred[k] = raw_output[pos]
                    if return_embedding:
                        cur_embedding[k] = embeddings[pos]
                    pos += 1
            for k in covered[i]:
                cur_pred[k] = -self.inf
            pred.append(cur_pred)
            if return_embedding:
                final_embeddings.append(cur_embedding)
            del data, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if return_embedding:
            return pred, np.array(final_embeddings)
        return pred

    def test_on_gid(self, gid: int):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cost = 0.0
        i = 0
        sol = []
        while (not self.test_env.isTerminal()):
            #print("not terminal! :/")
            # cost += 1
            list_pred = self.predict_with_current_qnet(g_list, [self.test_env.action_list], batch_size=1)
            # new_action = self.argMax(list_pred[0])
            list_pred = list_pred[0]
            new_action = np.argmax(list_pred)
            #print(list_pred)
            #print(new_action)
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
            if(self.procedural_attrs):
                self.test_env.graph.set_procedural_attributes(self.procedural_attrs, self.procedural_attrs_args, covered=sol)
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness


    def SaveModel(self,model_path):
        torch.save(self.DQN.state_dict(), model_path)
        self.logger.info(f'model has been saved successfully at {model_path}')

    def LoadModel(self,model_path):
        try:
            self.DQN.load_state_dict(torch.load(model_path))
        except:
            self.DQN.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.logger.info('restore model from file successfully')

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def gen_synthetic_graphs(self, level):
        self.logger.info('\ngenerating new random training graphs...')
        self.ClearTrainGraphs()
        if level['type'] == 'synthetic':
            generator = level['generator']
            min_n, max_n = level['min_n'], level['max_n']
            generators_args = {generator: { 'weight': 1, 'min_n': min_n, 'max_n': max_n, 'args': level['args']}}
        
        elif level['type'] == 'synthetic_diverse':
            generators_args = level['generators_args']
        else:
            raise ValueError(f"Unsupported level type: {level['type']}")
    
        gen_w_tuples = [(generator, args['weight']) for generator, args in generators_args.items()]
        generators_strs, generators_weights = zip(*gen_w_tuples)
        for i in tqdm(range(level['num_graphs'])):
            # select a random generator based on the weights
            generator_str = random.choices(generators_strs, weights=generators_weights)[0]
            generator = getattr(nx, generator_str)
            min_n, max_n = generators_args[generator_str]['min_n'], generators_args[generator_str]['max_n']
            curr_n = random.randint(min_n, max_n)
            g_nx = generator(curr_n, **generators_args[generator_str]['args'])
            g_cy = self.GenCyNetwork(g_nx)
            self.InsertGraph(g_cy, is_test=False)

            
    def SetupSparseT(self, sparse_dict):        
        if(sparse_dict is None):
            return None
        sparse_dict['index'] = Variable(sparse_dict['index']).to(self.device).type(torch.int64) 
        sparse_dict['value'] = Variable(sparse_dict['value']).to(self.device)

        return sparse_dict

    def SetupPyGAll(self, idxes, g_list, covered, actions=None, target=None, train=False):
        if(train):
            # assert (actions and target)
            assert (actions is not None and target is not None)
            data, prepareBatchGraph = self.SetupTrain(idxes, g_list, covered, actions, target, return_prepareBatchGraph=True)
        else:
            data, idx_map_list, prepareBatchGraph = self.SetupPredAll(idxes, g_list, covered, return_prepareBatchGraph=True)


        data['node_input'] = prepareBatchGraph.node_attributes.to(self.device)

        #update dtype
        data['node_input'] = data['node_input'].type(self.dtype)
        data['aux_input'] = data['aux_input'].type(self.dtype)
        if data['node_input'].shape == torch.Size([0]):
            assert not self.contextual_attrs 
            assert not self.procedural_attrs
            data['node_input'] = None
        else:
            # cannot support both yet
            assert (self.contextual_attrs or self.procedural_attrs)
            
        data['edge_index'] = prepareBatchGraph.edge_index.to(self.device)
        data['batch'] = prepareBatchGraph.batch_tensor.to(self.device)

        if(train):
            return data
        else:
            return data, idx_map_list


    def SetupPredAll(self, idxes, g_list, covered, return_prepareBatchGraph=False):
        
        #for g in g_list:
        #    print("Shape of g_attrs: ", g.node_attributes.shape)
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(self.aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)

        data = {}

        data['rep_global'] = self.SetupSparseT(prepareBatchGraph.rep_global)
        data['n2nsum_param'] = self.SetupSparseT(prepareBatchGraph.n2nsum_param)
        data['subgsum_param'] = self.SetupSparseT(prepareBatchGraph.subgsum_param)
        data['node_input'] = None
        data['aux_input'] = Variable(torch.tensor(prepareBatchGraph.aux_feat).type(torch.FloatTensor)).to(self.device)
        if return_prepareBatchGraph:
            return data, prepareBatchGraph.idx_map_list, prepareBatchGraph

        return data, prepareBatchGraph.idx_map_list

    def SetupTrain(self, idxes, g_list, covered, actions, target, return_prepareBatchGraph=False):
        m_y = target

        data = {}
        data['target'] = Variable(torch.tensor(m_y).type(torch.FloatTensor)).to(self.device)
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(self.aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)

        data['action_select'] = self.SetupSparseT(prepareBatchGraph.act_select)
        data['rep_global'] = self.SetupSparseT(prepareBatchGraph.rep_global)
        data['n2nsum_param'] = self.SetupSparseT(prepareBatchGraph.n2nsum_param)
        data['laplacian_param'] = self.SetupSparseT(prepareBatchGraph.laplacian_param)
        data['subgsum_param'] = self.SetupSparseT(prepareBatchGraph.subgsum_param)

        data['node_input'] = None
        data['aux_input'] = Variable(torch.tensor(prepareBatchGraph.aux_feat).type(torch.FloatTensor)).to(self.device)
        
        if return_prepareBatchGraph:
            return data, prepareBatchGraph
        return data

    def get_node_scores_nx(self, g_nx, node_attributes=None, covered=[], return_embedding=False):
        g = self.GenCyNetwork(g_nx, node_attributes=node_attributes)
        self.InsertGraph(g, is_test=True)
        g_list = []
        self.test_env.s0(self.TestSet.Get(0))
        infer_start = time.time()


        #print("Covered in function: ", covered)
        for action in covered:
            #print("Adding action... ", action)
            self.test_env.stepWithoutReward(action)
        #print("!!! Action list: ", self.test_env.action_list)
        if(self.test_env.isTerminal()):
            scores = np.zeros(g.num_nodes)
            print("Warning: Graph is already fully covered")

        else:
            if(self.procedural_attrs):
                self.test_env.graph.set_procedural_attributes(self.procedural_attrs, self.procedural_attrs_args, covered=covered)

            g_list.append(self.test_env.graph)
            if(return_embedding):
                #print('g_list[0]')
                #print(g_list[0])
                # print covered
                #print("\t ----- Action list: ", self.test_env.action_list)
                scores, embeddings = self.predict_with_current_qnet(g_list, [self.test_env.action_list], return_embedding=True)
                scores = scores[0]
                embeddings = embeddings[0]
            else:
                scores = self.predict_with_current_qnet(g_list, [self.test_env.action_list], return_embedding=False)
                scores = scores[0]


        infer_time = time.time() - infer_start
        self.ClearTestGraphs()

        if(return_embedding):
            #embeddings = embeddings.cpu().detach().numpy()
            return scores, embeddings
        # turn scores into dict
        else:
            scores = {i: scores[i] for i in range(len(scores))}
            return scores

    def calc_solution_nx_chunked(self, g, budget, step_size=1, node_chunk_size=20000,
                                 node_attributes=None, print_progress=False):
        """Memory-scalable rollout for very large graphs (ADDITIVE).

        Identical autoregressive node-removal loop as `calc_solution_nx`, but scores
        nodes with `predict_chunked` (chunked cross-attention + decoder) so graphs
        with millions of nodes fit in memory. Output is the same solution
        `calc_solution_nx` would produce for the same step_size.
        """
        import psutil
        process = psutil.Process()
        budget = int(budget)
        print(f"[chunked] node_chunk_size={node_chunk_size} step_size={step_size}")
        print(f"Initial memory usage: {process.memory_info().rss/1024/1024:.2f} MB")

        g_cy = self.GenCyNetwork(g, node_attributes=node_attributes)
        self.InsertGraph(g_cy, is_test=True)
        g_list = []
        self.test_env.s0(self.TestSet.Get(0))

        if print_progress:
            pbar = tqdm(total=budget, desc="Removing nodes (chunked)")

        removed_nodes = []
        self.DQN.eval()
        # Enable memory-scalable sparse-matmul aggregation in every SAGE conv so
        # the encoder does not materialize a [num_edges, dim] message tensor
        # (which OOMs on million-edge graphs). Numerically identical sum-agg.
        n_spmm = 0
        for m in self.DQN.modules():
            if type(m).__name__ == 'CustomSAGEConv':
                m.use_spmm_aggr = True
                n_spmm += 1
        print(f"[chunked] enabled spmm aggregation on {n_spmm} SAGE conv layers")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Graph size: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        print(f"Target budget: {budget} nodes")

        try:
            with torch.no_grad():
                processed_count = 0
                while len(removed_nodes) < budget:
                    if self.test_env.isTerminal():
                        break
                    if self.procedural_attrs:
                        self.test_env.graph.set_procedural_attributes(
                            self.procedural_attrs, self.procedural_attrs_args, covered=removed_nodes)
                    g_list.append(self.test_env.graph)
                    scores = self.predict_chunked(
                        g_list, [self.test_env.action_list],
                        node_chunk_size=node_chunk_size, return_embedding=False)
                    scores = scores[0]
                    top_nodes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:step_size]
                    for node in top_nodes:
                        if node not in removed_nodes:
                            removed_nodes.append(node)
                            self.test_env.stepWithoutReward(node)
                            if print_progress:
                                pbar.update(1)
                    g_list.clear()
                    processed_count += 1
                    if processed_count % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error during chunked node removal: {str(e)}")
            if print_progress:
                pbar.close()
            print(f"Completed {len(removed_nodes)}/{budget} nodes before error")
            traceback.print_exc()
            self.ClearTestGraphs()
            # Propagate CUDA OOM so the caller can fall back to CPU; only swallow
            # other errors (returning partial progress).
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                raise
            return removed_nodes

        self.DQN.train()
        if print_progress:
            pbar.close()
        print(f"Final memory usage: {process.memory_info().rss/1024/1024:.2f} MB")
        self.ClearTestGraphs()
        return removed_nodes

    def calc_solution_nx(self, g, budget, step_size=1, node_attributes=None, print_progress=False):
        # Add memory monitoring
        import psutil
        process = psutil.Process()

        budget = int(budget)
        
        # Log initial memory usage
        initial_mem = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_mem:.2f} MB")
        
        g_cy = self.GenCyNetwork(g, node_attributes=node_attributes)
        self.InsertGraph(g_cy, is_test=True)
        g_list = []
        self.test_env.s0(self.TestSet.Get(0))
        
        if print_progress:
            pbar = tqdm(total=budget, desc="Removing nodes")

        removed_nodes = []
        # set model to eval mode
        self.DQN.eval()
        
        # Force garbage collection before starting
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU
        
        # Log graph size information
        print(f"Graph size: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        print(f"Target budget: {budget} nodes")
        
        try:
            with torch.no_grad():
                processed_count = 0
                while len(removed_nodes) < budget:
                    if self.test_env.isTerminal():
                        break
                    
                    # if processed_count % 10 == 0:  # Log memory every 10 iterations
                    #     current_mem = process.memory_info().rss / 1024 / 1024
                    #     print(f"Memory after {processed_count} nodes: {current_mem:.2f} MB")
                    
                    if self.procedural_attrs:
                        self.test_env.graph.set_procedural_attributes(self.procedural_attrs, self.procedural_attrs_args, covered=removed_nodes)
                    
                    g_list.append(self.test_env.graph)
                    
                    # Try to catch out-of-memory issues during prediction
                    try:
                        scores = self.predict_with_current_qnet(g_list, [self.test_env.action_list], return_embedding=False)
                        scores = scores[0]
                    except RuntimeError as e:
                        print("Runtime error: ", e)
                        traceback.print_exc()
                        if "out of memory" in str(e):
                            self.logger.error(f"OOM during prediction after {len(removed_nodes)} nodes")
                            if torch.cuda.is_available():
                                print(f"GPU memory: {torch.cuda.memory_allocated()/1024/1024:.2f} MB allocated, {torch.cuda.memory_reserved()/1024/1024:.2f} MB reserved")
                        raise e
                    
                    # Get the top step_size scored nodes
                    top_nodes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:step_size]
                    
                    # Remove the top nodes from the graph and add them to removed_nodes
                    for node in top_nodes:
                        if node not in removed_nodes:
                            removed_nodes.append(node)
                            self.test_env.stepWithoutReward(node)
                            if print_progress:
                                pbar.update(1)
                    
                    # Clear the g_list for the next iteration
                    g_list.clear()
                    
                    # Force garbage collection periodically
                    processed_count += 1
                    if processed_count % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error during node removal: {str(e)}")
            # Log final state before exiting
            if print_progress:
                pbar.close()
            print(f"Completed {len(removed_nodes)}/{budget} nodes before error")
            # print stack
            traceback.print_exc()
            self.ClearTestGraphs()
            return removed_nodes
                    
        # set model to train mode
        self.DQN.train()

        if print_progress:
            pbar.close()
        
        final_mem = process.memory_info().rss / 1024 / 1024
        print(f"Final memory usage: {final_mem:.2f} MB (change: {final_mem-initial_mem:.2f} MB)")
        
        self.ClearTestGraphs()
        return removed_nodes