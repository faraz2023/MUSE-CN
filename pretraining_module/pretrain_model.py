import sys
import os
parent_dir = os.path.abspath(os.path.join(".."))
sys.path.append(parent_dir)

from pretrain_utils.options import Options
import torch
import dgl
from pathlib import Path
import numpy as np
import pretrain_utils.utils
from py_modules.encoders import GraphSAGE_Encoder_PyG_FINDER_v2
from pretrain_utils.model import PretrainModule, BigModel
import json
import pretrain_utils.data
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
# import pretrain_utils.min_norm_solvers
from pretrain_utils.data_train import Graph_Dataset, Universal_Collator
import pretrain_utils
import gc


mp.set_start_method('spawn', force=True)


WANDB_PROJECT_NAME = "QCNDP_SSL_v1"





def get_model_str_repr(opt):
    # fp = opt.dataset+f'_lp' if opt.mask_edge else opt.dataset+f'_ssnc'
    # fp += f'_s{opt.seed}'
    # fp += str(opt.tasks)
    # fp += '_saint' if opt.use_saint else '_k-order'
    # fp += '_prelu_' if opt.use_prelu else '_relu'
    # fp += 'hid_dim_{}_'.format(str(opt.hid_dim))
    # fp += 'optim_{}_'.format(str(opt.optim))
    # fp += 'inter_dim_{}_'.format(str(opt.inter_dim))
    # fp += 'pred_dim_{}_'.format(str(opt.predictor_dim))
    # fp += 'lr_{}_'.format(str(opt.lr))
    # fp += 'decay_{}_'.format(str(opt.weight_decay))
    # fp += 'no_self_loop_' if opt.no_self_loop else ''
    # fp += opt.grad_norm + '_'
    # fp += 'no_pareto' if opt.not_use_pareto else ''

    fp = f'seed_{opt.seed}_tasks'
    for task in opt.tasks:
        fp += f'_{task}'

    
    return fp

def main(opt):

    def train(model, optimizer, scheduler, step, opt, checkpoint_path):
        torch.manual_seed(opt.local_rank + opt.seed)
        tasks = opt.tasks
        collator = Universal_Collator(g, opt.use_saint, opt.per_gpu_batch_size, opt.device, tasks, \
                                            opt.minsg_der, opt.minsg_dfr, opt.batch_size_multiplier_minsg, opt.khop_minsg, \
                                            opt.khop_ming, opt.batch_size_multiplier_ming, \
                                            opt.sub_size, 0.15, 0.15, 0.15, \
                                            opt.lp_neg_ratio, \
                                            opt.decor_size, opt.decor_der, opt.decor_dfr, \
                                            20000, 0.5,
                                            num_layers=opt.ModelARGS['node_module']['encoder_args']['max_bp_iter'])
        dataset = Graph_Dataset(g)
        curr_losses = {}
        model.train()
        model.zero_grad()
        inner_step = 0
        if opt.not_use_pareto:
            logger.info('Not using Pareto MTL.')
        else:
            logger.info('Using Pareto MTL.')

        best_loss, best_step = 1e9, 0
        while step < opt.total_steps:
            dataloader = DataLoader(dataset=dataset, shuffle=False, prefetch_factor=1, persistent_workers=False, \
                                    collate_fn=collator, num_workers=opt.worker, pin_memory=False)

            for sample in dataloader:
                loss_data = {}
                grads = {}
                # -------------- Begin of Pareto Multi-Tasking Learning --------------
                if opt.not_use_pareto:
                    #sol = {t:1. for t in tasks}
                    len_tasks = len(tasks)
                    sol = {t:1./len_tasks for t in tasks}
                else:
                    if 'p_link' in tasks:
                        sg, pos_u, pos_v, neg_u, neg_v = sample['p_link']
                        loss = model.p_link(sg, pos_u, pos_v, neg_u, neg_v)
                        grads['p_link'] = []
                        loss_data['p_link'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_link'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_ming' in tasks:
                        ming_graph, ming_feat, ming_cor_feat = sample['p_ming']
                        loss = model.p_ming(ming_graph.to(opt.device), ming_feat.to(opt.device), ming_cor_feat.to(opt.device))
                        grads['p_ming'] = []
                        loss_data['p_ming'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_ming'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_minsg' in tasks:
                        minsg_g1, minsg_f1, minsg_g2, minsg_f2 = sample['p_minsg']
                        loss = model.p_minsg(minsg_g1.to(opt.device), minsg_f1.to(opt.device),
                                            minsg_g2.to(opt.device), minsg_f2.to(opt.device), opt.temperature_minsg)
                        grads['p_minsg'] = []
                        loss_data['p_minsg'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_minsg'].append(param.grad.data.detach().cpu())
                        model.zero_grad() 
                    
                    if 'p_decor' in tasks:
                        decor_g1, decor_g2 = sample['p_decor']
                        loss = model.p_decor(decor_g1.to(opt.device), decor_g2.to(opt.device), opt.decor_lamb)
                        grads['p_decor'] = []
                        loss_data['p_decor'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_decor'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_recon' in tasks:
                        p_recon_g, p_recon_mask = sample['p_recon']
                        loss = model.p_recon(p_recon_g.to(opt.device), p_recon_mask)
                        grads['p_recon'] = []
                        loss_data['p_recon'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_recon'].append(param.grad.data.detach().cpu())
                        model.zero_grad() 

                    if len(tasks) > 1:
                        gn = src.min_norm_solvers.gradient_normalizers(grads, loss_data, opt.grad_norm)
                        for t in loss_data:
                            for gr_i in range(len(grads[t])):
                                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
                        sol, _ = src.min_norm_solvers.MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
                        # sum of values in sol should be 1
                        sol = {k:sol[i] for i, k in enumerate(tasks)}
                    else:
                        sol = {tasks[0]:1.}
                # if wandb enabled, log each of the task's weight
                if opt.wandb and opt.is_main:
                    wandb.log({k+'_weight': v for k, v in sol.items()})
                # -------------- End of Pareto Multi-Tasking Learning --------------

                model.zero_grad()
                train_loss = 0
                actual_loss = 0
                scaled_loss = 0
                unscaled_loss = 0
                loss_dict = model(sample, opt)  

                for i, l in loss_dict.items():
                    train_loss += float(sol[i]) * l
                    actual_loss += l

                    unscaled_loss += l
                    scaled_loss += float(sol[i]) * l
                
                train_loss.backward()

                loss_dict['train_loss'] = actual_loss.detach()
                loss_dict['unscaled_loss'] = unscaled_loss.detach()
                loss_dict['scaled_loss'] = scaled_loss.detach()

                for k, v in sol.items():
                    loss_dict[k+'_weight'] = torch.tensor(float(v))
                    if k not in curr_losses:
                        curr_losses[k] = loss_dict[k].item()
                    else:
                        curr_losses[k] += loss_dict[k].item()
                if 'train_loss' not in curr_losses:
                    curr_losses['train_loss'] = loss_dict['train_loss']
                else:
                    curr_losses['train_loss'] += loss_dict['train_loss']
                if opt.wandb and opt.is_main:
                    wandb.log({k: v.item() for k, v in loss_dict.items()})

                inner_step += 1
    
                if inner_step == opt.accumulation_steps:
                    inner_step = 0
                    step += 1
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    if opt.is_main and step % 100 == 0:
                        logger.info(f"Train loss at step {step}: {loss_dict['train_loss']}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        print("", flush=True)
                    if opt.is_main and (loss_dict['train_loss'] < best_loss or step == int(opt.total_steps * 1/2)):
                        best_loss = loss_dict['train_loss']
                        best_step = step
                        logger.info(f"New best loss: {best_loss} at step {best_step}")
                        if not opt.debug:
                            ckpt_name = f"best_model" 
                            pretrain_utils.utils.save(model, optimizer, scheduler, step,
                                            opt, checkpoint_path, ckpt_name)
                            ## Bypassing the emebedding saving for now as we don't need it for Q-CNDP
                            # model.eval()
                            # with torch.no_grad():
                            #     if opt.no_self_loop:
                            #         use_g = dgl.add_self_loop(g)
                            #     else:
                            #         use_g = g
                            #     # check number of nodes
                            #     if use_g.num_nodes() > 2: #20000
                            #         inference_device = 'cpu'
                            #         do_save = False
                            #         if step >= int(opt.total_steps * 1/2):
                            #             do_save = True
                            #     else:
                            #         inference_device = opt.device
                            #         do_save = True
                            #     # move model to cpu to save memory
                            #     if(do_save):
                            #         model.to(inference_device)
                            #         if opt.is_distributed:
                            #             X = model.module.compute_representation(use_g.to(inference_device) , g.ndata['feat'].to(inference_device))
                            #         else:
                            #             X = model.compute_representation(use_g.to(inference_device), g.ndata['feat'].to(inference_device))
                                    

                            #         fp = os.path.join(checkpoint_path, 'X.pt')
                            #         torch.save(X, fp)
                            #         del X
                            #         logger.info(f"Saved embeddings to {fp}")
                            #         # move model back to cuda
                            #         model.to(opt.device)
                            # model.train()
                        # evaluate(model, g, opt) # skipping the node evaluation for now 
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train loss: {curr_losses['train_loss']/(opt.save_freq*opt.accumulation_steps):.3f} |"
                        for t in sample:
                            log += f"{t} loss: {curr_losses[t]/(opt.save_freq*opt.accumulation_steps):.3f} |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        logger.info(log)
                        for k in curr_losses:
                            curr_losses[k] = 0

                del loss_dict
                del train_loss
                del actual_loss
                del scaled_loss
                del unscaled_loss
                del sol
                del grads
                del sample

                if step >= opt.total_steps:
                    step += 1
                    break

        opt.best_loss = best_loss.item()
        opt.best_step = best_step

        opt.checkpoint_path = str(opt.checkpoint_path)
        with open('{}/opt.json'.format(checkpoint_path), 'w') as fp: json.dump(vars(opt), fp, sort_keys=True, indent=4)
        

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.is_main = True
    opt.local_rank = 0
    opt.world_size = 1

    if opt.wandb and opt.is_main:
        import wandb
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(opt.dataset, str(opt.tasks), str(opt.hid_dim), str(opt.n_layer), \
             str(opt.total_steps), 'saint' if opt.use_saint else 'k-order', \
             str(opt.lr), str(opt.weight_decay))
        # add idenfier for opt.not_use_pareto
        if opt.not_use_pareto:
            name += '_{}'.format('no_pareto')
        else: 
            name += '_{}'.format('pareto')
        if opt.mask_edge:
            name += '_{}'.format('mask_edge')
        wandb.init(project=WANDB_PROJECT_NAME, config=opt, name=name, resume='allow')
        wandb.config = opt
        
    np.random.seed(opt.seed+opt.local_rank)
    # dgl.seed(opt.seed+opt.local_rank)
    torch.manual_seed(opt.seed+opt.local_rank)
    checkpoint_path = os.path.join(opt.checkpoint_dir, opt.name)
    checkpoint_path = os.path.join(checkpoint_path, get_model_str_repr(opt))
    os.makedirs(checkpoint_path, exist_ok=True)
    logger = pretrain_utils.utils.init_logger(
        opt.is_main,
        opt.is_distributed, # is_distributed=
        os.path.join(checkpoint_path, 'run.log')
    )
    opt.checkpoint_path = checkpoint_path

    logger.info(f"Initializing Data..")

    data_path = os.path.join(opt.root_data_path, opt.dataset)
    g = pretrain_utils.data.load_data(data_path)

    logger.info(f"Initializing Model..")

    print("g.ndata['feat'].shape[1]: ", g.ndata['feat'].shape[1])

    ENCODER_TYPE = "FINDER_encoder_PyG"
    ENCODER_ARGS = {
        "embedding_size": opt.hid_dim,
        "activation": "prelu",
        "weight_init": True,
        "num_node_features": g.ndata['feat'].shape[1],
        "max_bp_iter": 3,
        "inner_conv_normalize": False
        }
    opt.ModelARGS = {
        'node_module':{
            "encoder_type": ENCODER_TYPE,
            "encoder_args": ENCODER_ARGS,
        },
        'BigModel':{
            'inter_dim': opt.inter_dim,
        },
        'PretrainModule':{
            'predictor_dim': opt.predictor_dim,
        },
    }
    print("Model Args: ", opt.ModelARGS)
    node_module = GraphSAGE_Encoder_PyG_FINDER_v2(**ENCODER_ARGS).to(opt.device)

    bigM = BigModel(node_module, None, opt.inter_dim)
    MEGA = PretrainModule(bigM, opt.predictor_dim).to(opt.device)

    # number of mode parameters
    num_params = sum(p.numel() for p in MEGA.parameters() if p.requires_grad)
    print("Num params: ", num_params)
    logger.info(f"Number of model parameters: {num_params}")
    
    # MEGA_config = {'input_dim':g.ndata['feat'].shape[1], 'hid_dim':opt.hid_dim, 
    #             'n_layer':len(opt.hid_dim), 'inter_dim':opt.inter_dim, 'dropout':opt.dropout, 'num_params':num_params,}
    # opt.MEGA_config = MEGA_config
    # logger.info("MEGA CONFIG: "+json.dumps(MEGA_config, indent=2))
    logger.info("MEGA CONFIG:  "+ json.dumps(ENCODER_ARGS, indent=2))
    model = MEGA.to(opt.device)

    #### Counting number of parameters for each componetn ####
    # number of parameters in model.big_model
    num_params = sum(p.numel() for p in model.big_model.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters in big_model: {num_params}")

    # number of parameters dedicated to p_link (and link_predictor_hid, link_predictor_class)
    li_pred_class = sum(p.numel() for p in model.link_predictor_class.parameters() if p.requires_grad)
    li_pred_hid = sum(p.numel() for p in model.link_predictor_hid.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters for link prediction: {li_pred_class + li_pred_hid}")

    # number of parameters dedicated to p_ming ( discriminator)
    ming_disc = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters for discriminator: {ming_disc}")

    # number of parameters dedicated to p_minsg ( minsg)
    minsg_disc = sum(p.numel() for p in model.minsg.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters for minsg: {minsg_disc}")

    # number of parameters dedicated to p_decor ( decor)
    logger.info(f"Num params in decor: 0")

    # number of parameters dedicated to p_recon ( recon_enc_dec, decoder)
    recon_enc_dec = sum(p.numel() for p in model.recon_enc_dec.parameters() if p.requires_grad)
    decoder = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    logger.info(f"Number of model parameters for reconstruction: {recon_enc_dec + decoder}")




    optimizer, scheduler = pretrain_utils.utils.set_optim(opt, model)
    step = 0

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, static_graph=True)

    logger.info("Start training")


    train(
        model,
        optimizer,
        scheduler,
        step,
        opt,
        checkpoint_path)
    
    logger.info("Successfully finished training")


if __name__ == '__main__':
    options = Options()
    options.add_general_model_options()
    options.add_optim_options()
    opt = options.parse()
    opt.wandb = False
    opt.root_data_path = os.path.join("..", "Data", "SSL_data")
    world_size = opt.world_size
    if opt.is_distributed:
        # mp.spawn(
        #     main,
        #     args=(world_size, opt),
        #     nprocs=world_size,
        #     start_method='spawn',
        #     join=True
        # )
        pass
    else:
        main(opt)