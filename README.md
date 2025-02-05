# MUSE-CN
Repo for MUSE-CN: MUlti-encoder Self-supervised Expert for learning to identify Critical Nodes in large graphs

## Instructions

1. First recreate the conda environment: 
```
conda env create -f environment.yml
```

2. The dependencies for MUSE-CN are: cython, pytorch, torch-geometric, numpy, pandas, scipy, networkx. 

3. To perform  MT-SSL pretraining, run the following command:

```

cd pretraining_module
bash general_pretraining.sh
```

(optional) you can use `P_01_SSL_data_generator.ipynb` to generate custome synthetic datasets for pretraining.

4. Make all the cython extensions needed for MUSE-CN:
```
python setup.py build_ext --inplace
```

5. The experiment configurations are stored in `exp_configs/`. You can run the experiments by selecting (or creating custome experiment configs) in `main.py` and running the following command to train the models:

```
python main.py
```

6. You can evalauted the trained models with the following command:

```
python eval_test.py
```





## Training models using exp_configs

The project uses JSON configuration files (located in `exp_configs/`) to define training experiments. This section details the available configuration options and their usage.

### Configuration File Structure

Each experiment configuration file contains several key sections:

1. **Basic Configuration**
   - `exp_name`: Name of the experiment
   - `export_path`: Directory to save experiment results
   - `seed`: Random seed for reproducibility
   - `device`: Training device ('cuda' or 'cpu')
   - `use_wandb`: Enable/disable Weights & Biases logging

2. **Model Configuration**
   - `model_type`: Type of model to use
     - `FINDER_DQN`: Standard DQN model
   - `encoder_type`: Type of graph encoder
     - `FINDER_encoder_PyG`: GraphSAGE encoder
     - `identity`: Identity encoder
     - `MEGA`: MEGA encoder
   - `encoder_args`: Encoder-specific arguments
     - `num_node_features`: Number of input node features
     - `embedding_size`: Size of node embeddings
   - `decoder_args`: Decoder-specific arguments

3. **Training Algorithm Configuration**
   - `RL_algorithm`: Choice of RL algorithm
     - `DQN`: Deep Q-Network
     - `MC`: Monte Carlo
   - `RL_algorithm_args`: Algorithm-specific parameters
     - `gamma`: Discount factor
     - `n_steps`: Number of steps for n-step returns (DQN)
     - `num_env`: Number of parallel environments
     - `max_episode_length`: Maximum episode length

4. **Feature Configuration**
   - `procedural_attrs`: List of procedural node attributes
     - `prone`: ProNE embeddings
     - `ones`: One-hot features
   - `procedural_attrs_args`: Arguments for procedural attributes
   - `contextual_attrs`: List of contextual node attributes

5. **Buffer & Optimization Configuration**
   - `buffer_capacity`: Replay buffer size
   - `optimizer`: Optimizer type (e.g., 'adam')
   - `optimizer_args`: Optimizer parameters

6. **Curriculum Learning Configuration**
   - `curriculum`: List of training levels
   Each level contains:
   - `type`: Type of graphs ('synthetic', 'synthetic_diverse', 'real-world')
   - `num_training_iters`: Number of training iterations
   - `batch_size`: Training batch size
   - `epsilon_start`, `epsilon_end`, `epsilon_step`: Exploration parameters
   - `target_update_freq`: Target network update frequency (DQN)
   - Graph generation parameters:
     - For synthetic: `generator`, `min_n`, `max_n`, `args`
     - For real-world: `edge_list_path`, `node_attributes_path`

