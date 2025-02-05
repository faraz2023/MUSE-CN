import argparse
import os
from pathlib import Path

from numpy import require


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_MultiheadAttention_options(self):
        self.parser.add_argument('--attention_activation', type=str, default='softmax', choices=['softmax', 'tanh'])
        self.parser.add_argument('--num_transformer_blocks', type=int, default=1)
        self.parser.add_argument('--num_heads', type=int, default=8)
        self.parser.add_argument('--accumulate_gradients', action='store_true',
                    help='Accumulate gradients across tasks before updating model parameters')
        self.parser.add_argument('--use_post_representation', action='store_true',
                                 help='Use post representation for the final prediction (instead of the smaller shared embedding module)')
        self.parser.add_argument('--post_agg_strategy', type=str, default='sum', choices=['max', 'sum'])
        self.parser.add_argument('--weight_scaling', type=str, default='none', choices=['none', 'init_loss'])
        self.parser.add_argument('--add_pre_att_loss', action='store_true')
        
    def multi_encoder_options(self):
        self.parser.add_argument('--add_pre_xatt_loss', action='store_true')
        self.parser.add_argument('--do_kstep_sequential_training', action='store_true')
        self.parser.add_argument('--k_step', type=int, default=10)
        self.parser.add_argument('--do_pretrain', action='store_true')
        self.parser.add_argument('--pretrain_steps', type=int, default=25*1000)
        self.parser.add_argument('--only_pretrain', action='store_true')
        self.parser.add_argument('--inference_agg_strategy', type=str, default='sum', choices=['max', 'sum', 'raw'])

    def add_DQN_options(self):
        self.parser.add_argument('--no_weight_scaling', action='store_true', help='Do not scale weight (minsg)')

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=100000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_general_model_options(self):
        self.parser.add_argument('--not_use_pareto', action='store_true')

        self.parser.add_argument('--model_name', type=str, default='mega', choices=['mega', 'paretognn', 'xatt_mt', 'simple_transformer', 'dqn', 'multienc_transformer'])
        
        self.parser.add_argument('--hid_dim', type=int, default=256)
        self.parser.add_argument('--predictor_dim', type=int, default=512)
        self.parser.add_argument('--inter_dim', type=int, default=0)
        self.parser.add_argument('--tasks', type=str, nargs='+', default=['p_gm', 'p_link', 'p_ming', 'p_minsg'])
        self.parser.add_argument('--n_layer', type=int, default=100)
        self.parser.add_argument('--n_group', type=int, default=2)
        self.parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
        self.parser.add_argument('--temperature_gm', type=float, default=0.2, help='gm: tau for infoNCE')
        self.parser.add_argument('--temperature_minsg', type=float, default=0.1, help='minsg: tau for infoNCE')
        self.parser.add_argument('--sub_size', type=int, default=100, help='size for subgraphs used in gm')
        self.parser.add_argument('--decor_size', type=int, default=100, help='size for subgraphs used in decor')
        self.parser.add_argument('--decor_lamb', type=float, default=1e-3, help='lambda for decor')
        self.parser.add_argument('--decor_der', type=float, default=0.1, help='der for decor')
        self.parser.add_argument('--decor_dfr', type=float, default=0.0, help='dfr for decor, set to 0.0 to disable, especially if not too many node features')
        self.parser.add_argument('--minsg_der', type=float, default=0.1, help='der for minsg')
        self.parser.add_argument('--minsg_dfr', type=float, default=0.0, help='dfr for minsg, set to 0.0 to disable, especially if not too many node features')
        self.parser.add_argument('--use_prelu', action='store_true')
        self.parser.add_argument('--grad_norm', type=str, default='l2', choices=['l2', 'loss', 'loss+', 'none'])
        self.parser.add_argument('--split', type=str, default='random', choices=['public', 'random'], help='dataset')
        self.parser.add_argument('--mask_edge', action='store_true', help='mask edge for link prediction')
        self.parser.add_argument('--tvt_addr', type=str, default='none', help='tvt file for the dataset')
        self.parser.add_argument('--hetero_graph_path', type=str, default='none', help='tvt file for the dataset')
        self.parser.add_argument('--dual_encoder', action='store_true')
        self.parser.add_argument('--use_saint', action='store_true')
        self.parser.add_argument('--lp_neg_ratio', type=int, default=1, help='negative ratio for link prediction pretrain')
        self.parser.add_argument('--no_self_loop', action='store_true')
        self.parser.add_argument('--norm', type=str, default='batch')

        
    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--pretrain_label_dir', required=True, type=str, default='path to pretrain labels')
        self.parser.add_argument('--dataset', type=str, required=True, help='dataset')

        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--is_distributed', action='store_true')
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--batch_size_multiplier_minsg", default=10, type=int, 
                        help="Batch size multiplier for minsg.")
        self.parser.add_argument("--batch_size_multiplier_ming", default=10, type=int, 
                        help="Batch size multiplier for ming.")
        self.parser.add_argument('--khop_ming', type=int, default=3, help='order for ming sampling')
        self.parser.add_argument('--khop_minsg', type=int, default=3, help='order for minsg sampling')
        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--world_size", type=int, default=1,
                        help="For distributed training: world_size")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--save_freq', type=int, default=1000, help="Save freq | not working in favor of best model saving") #not used for now for now as switching to best model saving
        self.parser.add_argument("--worker", type=int, default=3,
                        help="number of workers for dataloader")
        self.parser.add_argument('--wandb', action='store_true')
        self.parser.add_argument('--debug', action='store_true')

    def parse(self):
        opt = self.parser.parse_args()
        return opt