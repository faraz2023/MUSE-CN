
curriculum = []

num_training_iters = 10001
curriculum.append(
    {
    'type': 'synthetic',
    'generator': 'barabasi_albert_graph',
    'min_n': 40,
    'max_n': 60,
    'args': {'m': 4},
    'num_graphs': 200,
    'num_episodes': 10,
    'epsilon_start': 1,
    'epsilon_end': 0.7,
    'epsilon_step': 1000,
    'target_update_freq': 500,
    'num_training_iters': num_training_iters,
    'batch_size': 128,
    'num_valid': 2, # 2 is too low, just for testing for now
    # 'loss_function': 'regression', # only regression is supported for now
    'imitation_algorithm': 'pagerank',
    'imitation_deterministic': True,
    }
)

curriculum.append(
    {
    'type': 'synthetic',
    'generator': 'barabasi_albert_graph',
    'min_n': 40,
    'max_n': 60,
    'args': {'m': 4},
    'num_graphs': 200,
    'num_episodes': 10,
    'epsilon_start': 1,
    'epsilon_end': 0.7,
    'epsilon_step': 1000,
    'target_update_freq': 500,
    'num_training_iters': num_training_iters,
    'batch_size': 128,
    'num_valid': 2, # 2 is too low, just for testing for now
    # 'loss_function': 'regression', # only regression is supported for now
    'imitation_algorithm': 'degree',
    'imitation_deterministic': True,
    }
)


curriculum.append(
    {
    'type': 'synthetic',
    'generator': 'barabasi_albert_graph',
    'min_n': 40,
    'max_n': 60,
    'args': {'m': 4},
    'num_graphs': 200,
    'num_episodes': 10,
    'epsilon_start': 1,
    'epsilon_end': 0.05,
    'epsilon_step': 10000,
    'target_update_freq': 1000,
    'num_training_iters': num_training_iters * 10,
    'batch_size': 128,
    'num_valid': 2, # 2 is too low, just for testing for now
    # 'loss_function': 'regression', # only regression is supported for now
    }
)
