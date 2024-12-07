import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right

import evaluator
import utils
from flags import parser, DATA_FOLDER
from dataset import CompositionDataset
from oadis_base import OADIS
from model import convert_weights

ever_best_hm_1, ever_best_hm_2, ever_best_hm_3 = 0., 0., 0.
ever_best_seen_1, ever_best_seen_2, ever_best_seen_3 = 0., 0., 0.
ever_best_unseen_comp_1, ever_best_unseen_comp_2, ever_best_unseen_comp_3 = 0., 0., 0. 
ever_best_unseen_pair_1, ever_best_unseen_pair_2, ever_best_unseen_pair_3 = 0., 0., 0.

ever_best_hm_1_stage, ever_best_hm_2_stage, ever_best_hm_3_stage = 0, 0, 0
ever_best_seen_1_stage, ever_best_seen_2_stage, ever_best_seen_3_stage = 0, 0, 0
ever_best_unseen_comp_1_stage, ever_best_unseen_comp_2_stage, ever_best_unseen_comp_3_stage = 0, 0, 0 
ever_best_unseen_pair_1_stage, ever_best_unseen_pair_2_stage, ever_best_unseen_pair_3_stage = 0, 0, 0

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def decay_learning_rate(optimizer, cfg):
    """Decays learning rate using the decay factor in cfg.
    """
    print('# of param groups in optimizer: %d' % len(optimizer.param_groups))
    param_groups = optimizer.param_groups
    for i, p in enumerate(param_groups):
        current_lr = p['lr']
        new_lr = current_lr * cfg.decay_factor
        print('Group %d: current lr = %.8f, decay to lr = %.8f' %(i, current_lr, new_lr))
        p['lr'] = new_lr


def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    """Decays learning rate following milestones in cfg.
    """
    milestones = cfg.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.decay_factor ** it
    
    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print("Group %i, lr = %.8f" %(i, lr * gamma_group))


def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    state_dict = model_or_optim.state_dict()
    path = os.path.join(cfg.dataset_name, name+'.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, trainloader, device, cfg):
    model.train()
    for idx, batch in enumerate(trainloader):
        for k in batch:
            if isinstance(batch[k], list): 
                continue
            batch[k] = batch[k].to(device, non_blocking=True)
        
        out = model(batch)
        loss = out['loss_total']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_ge(epoch, model, testloader, evaluator, device, cfg):
    model.eval()
    topk = cfg.topk
    dset = testloader.dataset
    all_attr_gt, all_obj_gt, all_pair_gt, all_pred_1, all_pred_2, all_pred_3 = [], [], [], [], [], []
    hm_seen_1, hm_seen_2, hm_seen_3 = 0., 0., 0.
    hm_unseen_comp_1, hm_unseen_comp_2, hm_unseen_comp_3 = 0., 0., 0.
    hm_unseen_pair_1, hm_unseen_pair_2, hm_unseen_pair_3 = 0., 0., 0.
    best_hm_1, best_hm_2, best_hm_3 = 0., 0., 0.
    best_seen_1, best_seen_2, best_seen_3 = 0., 0., 0.
    best_unseen_comp_1, best_unseen_comp_2, best_unseen_comp_3 = 0., 0., 0. 
    best_unseen_pair_1, best_unseen_pair_2, best_unseen_pair_3 = 0., 0., 0.
    bias_1, bias_2 = 0., 0.
    
    for _, data in enumerate(testloader):
        for k in data:
            if isinstance(data[k], list): 
                continue
            data[k] = data[k].to(device, non_blocking=True)

        out = model(data)
        predictions_1 = out['scores_1']
        predictions_2 = out['scores_2']
        predictions_3 = out['scores_3']
        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']
        
        all_pred_1.append(predictions_1)
        all_pred_2.append(predictions_2)
        all_pred_3.append(predictions_3)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)
    
    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to('cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict_1,  all_pred_dict_2, all_pred_dict_3 = {}, {}, {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred_1[0].keys():
        all_pred_dict_1[k] = torch.cat([all_pred_1[i][k].to('cpu') for i in range(len(all_pred_1))])
    for k in all_pred_2[0].keys():
        all_pred_dict_2[k] = torch.cat([all_pred_2[i][k].to('cpu') for i in range(len(all_pred_2))])
    for k in all_pred_3[0].keys():
        all_pred_dict_3[k] = torch.cat([all_pred_3[i][k].to('cpu') for i in range(len(all_pred_3))])
    
    # Calculate best unseen accuracy
    bias_1, bias_2, hm_seen_1, hm_unseen_comp_1, hm_unseen_pair_1, best_hm_1, best_seen_1, best_unseen_comp_1, best_unseen_pair_1 = evaluator.evaluate(all_pred_dict_1, all_pair_gt, all_attr_gt, all_obj_gt, topk=topk)
    print('Test Epoch: %d Domain %s' %(epoch, cfg.train_domain_1))
    print('Best HM: %.4f Best Seen Acc: %.4f Best Unseen Comp Acc: %.4f Best Unseen Pair Acc: %.4f'%(best_hm_1, best_seen_1, best_unseen_comp_1, best_unseen_pair_1))
    print('Bias 1: %.4f Bias 2: %.4f HM Seen Acc: %.4f HM Unseen Comp Acc: %.4f HM Unseen Pair Acc: %.4f'%(bias_1, bias_2, hm_seen_1, hm_unseen_comp_1, hm_unseen_pair_1))
    
    bias_1, bias_2, hm_seen_2, hm_unseen_comp_2, hm_unseen_pair_2, best_hm_2, best_seen_2, best_unseen_comp_2, best_unseen_pair_2 = evaluator.evaluate(all_pred_dict_2, all_pair_gt, all_attr_gt, all_obj_gt, topk=topk)
    print('Test Epoch: %d Domain %s' %(epoch, cfg.train_domain_2))
    print('Best HM: %.4f Best Seen Acc: %.4f Best Unseen Comp Acc: %.4f Best Unseen Pair Acc: %.4f'%(best_hm_2, best_seen_2, best_unseen_comp_2, best_unseen_pair_2))
    print('Bias 1: %.4f Bias 2: %.4f HM Seen Acc: %.4f HM Unseen Comp Acc: %.4f HM Unseen Pair Acc: %.4f'%(bias_1, bias_2, hm_seen_2, hm_unseen_comp_2, hm_unseen_pair_2))
    
    bias_1, bias_2, hm_seen_3, hm_unseen_comp_3, hm_unseen_pair_3, best_hm_3, best_seen_3, best_unseen_comp_3, best_unseen_pair_3 = evaluator.evaluate(all_pred_dict_3, all_pair_gt, all_attr_gt, all_obj_gt, topk=topk)
    print('Test Epoch: %d Domain %s' %(epoch, cfg.test_domain))
    print('Best HM: %.4f Best Seen Acc: %.4f Best Unseen Comp Acc: %.4f Best Unseen Pair Acc: %.4f'%(best_hm_3, best_seen_3, best_unseen_comp_3, best_unseen_pair_3))
    print('Bias 1: %.4f Bias 2: %.4f HM Seen Acc: %.4f HM Unseen Comp Acc: %.4f HM Unseen Pair Acc: %.4f'%(bias_1, bias_2, hm_seen_3, hm_unseen_comp_3, hm_unseen_pair_3))
    
    renew(best_hm_1, best_seen_1, best_unseen_comp_1, best_unseen_pair_1, 1, epoch) 
    renew(best_hm_2, best_seen_2, best_unseen_comp_2, best_unseen_pair_2, 2, epoch) 
    renew(best_hm_3, best_seen_3, best_unseen_comp_3, best_unseen_pair_3, 3, epoch) 
    
    return


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def renew(best_hm, best_seen, best_unseen_comp, best_unseen_pair, domain, epoch):
    
    global ever_best_hm_1, ever_best_hm_2, ever_best_hm_3
    global ever_best_seen_1, ever_best_seen_2, ever_best_seen_3
    global ever_best_unseen_comp_1, ever_best_unseen_comp_2, ever_best_unseen_comp_3
    global ever_best_unseen_pair_1, ever_best_unseen_pair_2, ever_best_unseen_pair_3
    global ever_best_hm_1_stage, ever_best_hm_2_stage, ever_best_hm_3_stage
    global ever_best_seen_1_stage, ever_best_seen_2_stage, ever_best_seen_3_stage
    global ever_best_unseen_comp_1_stage, ever_best_unseen_comp_2_stage, ever_best_unseen_comp_3_stage
    global ever_best_unseen_pair_1_stage, ever_best_unseen_pair_2_stage, ever_best_unseen_pair_3_stage
    
    if domain == 1:
        if best_hm > ever_best_hm_1:
            ever_best_hm_1 = best_hm
            ever_best_hm_1_stage = epoch
        if best_seen > ever_best_seen_1:
            ever_best_seen_1 = best_seen
            ever_best_seen_1_stage = epoch
        if best_unseen_comp > ever_best_unseen_comp_1:
            ever_best_unseen_comp_1 = best_unseen_comp
            ever_best_unseen_comp_1_stage = epoch
        if best_unseen_pair > ever_best_unseen_pair_1:
            ever_best_unseen_pair_1 = best_unseen_pair
            ever_best_unseen_pair_1_stage = epoch
    
    if domain == 2:
        if best_hm > ever_best_hm_2:
            ever_best_hm_2 = best_hm
            ever_best_hm_2_stage = epoch
        if best_seen > ever_best_seen_2:
            ever_best_seen_2 = best_seen
            ever_best_seen_2_stage = epoch
        if best_unseen_comp > ever_best_unseen_comp_2:
            ever_best_unseen_comp_2 = best_unseen_comp
            ever_best_unseen_comp_2_stage = epoch
        if best_unseen_pair > ever_best_unseen_pair_2:
            ever_best_unseen_pair_2 = best_unseen_pair
            ever_best_unseen_pair_2_stage = epoch
    
    if domain == 3:
        if best_hm > ever_best_hm_3:
            ever_best_hm_3 = best_hm
            ever_best_hm_3_stage = epoch
        if best_seen > ever_best_seen_3:
            ever_best_seen_3 = best_seen
            ever_best_seen_3_stage = epoch
        if best_unseen_comp > ever_best_unseen_comp_3:
            ever_best_unseen_comp_3 = best_unseen_comp
            ever_best_unseen_comp_3_stage = epoch
        if best_unseen_pair > ever_best_unseen_pair_3:
            ever_best_unseen_pair_3 = best_unseen_pair
            ever_best_unseen_pair_3_stage = epoch
    

def main_worker(gpu, cfg):
    
    print('Use GPU %d for training' %gpu)
    torch.cuda.set_device(gpu)
    device = 'cuda:%d'%gpu
    ckpt_dir = cfg.checkpoint_dir

    print('Batch size on each gpu: %d' %cfg.batch_size)
    print('Prepare dataset')
    trainset = CompositionDataset(phase='train', cfg=cfg)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False, worker_init_fn=seed_worker)
    
    testset = CompositionDataset(phase='test', cfg=cfg)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = OADIS(trainset, cfg)
    model.to(device)
    if cfg.use_fp16 == True:
        convert_weights(model)
    #print(model)

    if not cfg.finetune_backbone :
        freeze(model.feat_extractor)
    #total_params = utils.count_parameters(model)

    evaluator_test_ge = evaluator.Evaluator(testset, model, cfg)
    
    torch.backends.cudnn.benchmark = True

    params_word_embedding = []
    params_encoder = []
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if 'attr_embedder' in name or 'obj_embedder' in name:
            if cfg.lr_word_embedding > 0:
                params_word_embedding.append(p)
                print('params_word_embedding: %s' % name)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
            #print('params_encoder: %s' % name)
        else:
            params.append(p)
            #print('params_main: %s' % name)

    if cfg.lr_word_embedding > 0:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.lr_encoder},
            {'params': params_word_embedding, 'lr': cfg.lr_word_embedding},
            {'params': params, 'lr': cfg.lr},
        ], lr=cfg.lr, weight_decay=cfg.wd)  
        group_lrs = [cfg.lr_encoder, cfg.lr_word_embedding, cfg.lr]
    else:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.lr_encoder},
            {'params': params, 'lr': cfg.lr},
        ], lr=cfg.lr, weight_decay=cfg.wd)
        group_lrs = [cfg.lr_encoder, cfg.lr]
    
    if not cfg.finetune_backbone and not cfg.use_precomputed_features:
        freeze(model.feat_extractor)
    
    epoch = 1
    while epoch <= cfg.max_epoch:
        train(epoch, model, optimizer, trainloader, device, cfg)
        max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0
        print('Max GPU usage in MB till now: %.4f' %max_gpu_usage_mb)

        if cfg.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.start_epoch_validate:
            epoch += 1
            continue
        
        validate_ge(epoch, model, testloader, evaluator_test_ge, device, cfg)
        epoch += 1
    
    print('Domain %s Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(cfg.train_domain_1, ever_best_hm_1, ever_best_hm_1_stage, ever_best_seen_1, ever_best_seen_1_stage, ever_best_unseen_comp_1, ever_best_unseen_comp_1_stage, ever_best_unseen_pair_1, ever_best_unseen_pair_1_stage))
    print('Domain %s Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(cfg.train_domain_2, ever_best_hm_2, ever_best_hm_2_stage, ever_best_seen_2, ever_best_seen_2_stage, ever_best_unseen_comp_2, ever_best_unseen_comp_2_stage, ever_best_unseen_pair_2, ever_best_unseen_pair_2_stage))
    print('Domain %s Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(cfg.test_domain, ever_best_hm_3, ever_best_hm_3_stage, ever_best_seen_3, ever_best_seen_3_stage, ever_best_unseen_comp_3, ever_best_unseen_comp_3_stage, ever_best_unseen_pair_3, ever_best_unseen_pair_3_stage))
                

def main():
    cfg = parser.parse_args()
    domain_num = 0
    for domain in ['photo', 'art', 'cartoon']:
        if cfg.test_domain == domain:
            pass
        elif domain_num == 0:
            cfg.train_domain_1 = domain
            domain_num = 1
        else:
            cfg.train_domain_2 = domain
    
    print('Train Domain 1: %s Train Domain 2: %s Test Domain: %s' %(cfg.train_domain_1, cfg.train_domain_2, cfg.test_domain))
    utils.fix_seeds(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    main_worker(0, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(ever_best_hm_1, ever_best_hm_1_stage, ever_best_seen_1, ever_best_seen_1_stage, ever_best_unseen_comp_1, ever_best_unseen_comp_1_stage, ever_best_unseen_pair_1, ever_best_unseen_pair_1_stage))
        print('Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(ever_best_hm_2, ever_best_hm_2_stage, ever_best_seen_2, ever_best_seen_2_stage, ever_best_unseen_comp_2, ever_best_unseen_comp_2_stage, ever_best_unseen_pair_2, ever_best_unseen_pair_2_stage))
        print('Best HM ACC %.4f Achieved %d Seen %.4f Achieved %d Unseen Comp %.4f Achieved %d Unseen Pair %.4f Achieved %d' %(ever_best_hm_3, ever_best_hm_3_stage, ever_best_seen_3, ever_best_seen_3_stage, ever_best_unseen_comp_3, ever_best_unseen_comp_3_stage, ever_best_unseen_pair_3, ever_best_unseen_pair_3_stage))