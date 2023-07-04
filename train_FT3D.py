"""
References:
PointPWC-Net: https://github.com/DylanWusee/PointPWC
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
from model import SPFlowNet
# from model_v2 import SPFlowNet
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *

from losses.unsupervised_losses import UnSupervisedL1Loss
from evaluation_utils import evaluate_2d, evaluate_3d


losses_dict = {
    'unsup_l1': UnSupervisedL1Loss
               }

def sequence_loss(pos1, pos2, flows_pred, flow_gt, hparams, loss_func):
    if 'loss_iters_w' in hparams:
        assert (len(hparams['loss_iters_w']) == len(flows_pred))
        loss = torch.zeros(1).cuda()
        for i, w in enumerate(hparams['loss_iters_w']):
            loss += w * loss_func(pos1, pos2, flows_pred[i], flow_gt, i)
    else:
        loss = loss_func(pos1, pos2, flows_pred[-1], flow_gt)
    return loss

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.exp_params['gpu'] if args.exp_params['multi_gpu'] is None else '0,1'

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-Flyingthings3d-'%args.exp_params['model_name'] + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    print(file_dir)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    saved_dir = file_dir.joinpath('models/')
    saved_dir.mkdir(exist_ok=True)

    # os.system('cp %s %s' % ('model.py', saved_dir))
    os.system('cp %s %s' % ('model_v2.py', saved_dir))
    os.system('cp %s %s' % ('train_FT3D.py', saved_dir))
    os.system('cp %s %s' % ('./configs_without_occlusions/config_evaluate.yaml', saved_dir))
    os.system('cp %s %s' % ('./configs_without_occlusions/config_train_FT3D.yaml', saved_dir))
    os.system('cp %s %s' % ('./utils/pointconv_util.py', saved_dir))
    os.system('cp %s %s' % ('./utils/modules.py', saved_dir))
    os.system('cp %s %s' % ('./losses/unsupervised_losses.py', saved_dir))
    os.system('cp %s %s' % ('./losses/common_losses.py', saved_dir))

    '''LOG'''
    logger = logging.getLogger(args.exp_params['model_name'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.exp_params['model_name'])
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    # blue = lambda x: '\033[94m' + x + '\033[0m'
    model = SPFlowNet(args)

    loss_func = losses_dict[args.exp_params['loss']['loss_type']](**args.exp_params['loss'])

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            num_points=args.num_points),
        num_points=args.num_points,
        data_root = args.data_root,
        full=args.full
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.exp_params['batch_size'],
        shuffle=True,
        num_workers=args.exp_params['workers'],
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.exp_params['batch_size'],
        shuffle=False,
        num_workers=args.exp_params['workers'],
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    '''GPU selection and multi-GPU'''
    if args.exp_params['multi_gpu'] is not None:
        device_ids = [int(x) for x in args.exp_params['multi_gpu'].split(',')]
        torch.backends.cudnn.benchmark = True 
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()

    if args.exp_params['pretrain'] is not None:
        model.load_state_dict(torch.load(args.exp_params['pretrain']))
        print('load model %s'%args.exp_params['pretrain'])
        logger.info('load model %s'%args.exp_params['pretrain'])
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    pretrain = args.exp_params['pretrain'] 
    init_epoch = int(pretrain[-14:-11]) if args.exp_params['pretrain'] is not None else 0 
    print(init_epoch)

    if args.exp_params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.exp_params['learning_rate'], momentum=0.9)
    elif args.exp_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.exp_params['learning_rate'])
                
    optimizer.param_groups[0]['initial_lr'] = args.exp_params['learning_rate']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.exp_params['scheduler']['milestones'], gamma=args.exp_params['scheduler']['gamma'])
    LEARNING_RATE_CLIP = 1e-5 

    history = defaultdict(lambda: list())
    best_epe = 1000.0
    best_acc_3d = -1
    for epoch in range(args.exp_params['epochs']):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, norm1, norm2, flow, _ = data  
            #move to cuda 
            pos1 = pos1.cuda()
            pos2 = pos2.cuda() 
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda() 

            model = model.train() 
            with torch.autograd.detect_anomaly():
                pred_flows, loss_consistency = model(pos1, pos2, norm1, norm2)

            loss = sequence_loss(pos1, pos2, pred_flows, flow, args.exp_params['loss'], loss_func)
            loss += loss + 0.01*loss_consistency.sum()

            history['loss'].append(loss.cpu().data.numpy())
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.exp_params['batch_size']
            total_seen += args.exp_params['batch_size']

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, 'train', train_loss)
        print(str_out)
        logger.info(str_out)

        logger.info('test of ft3d')
        eval_epe3d, eval_loss, eval_as = eval_sceneflow(model.eval(), val_loader, logger, loss_func, args)
        str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, 'eval', eval_epe3d, eval_loss)
        print(str_out)
        logger.info(str_out)

        if (eval_epe3d < best_epe) or (eval_as > best_acc_3d):
            # best_epe = eval_epe3d
            best_epe = eval_epe3d if eval_epe3d < best_epe else best_epe
            best_acc_3d = eval_as if eval_as > best_acc_3d else best_acc_3d
            torch.save(optimizer.state_dict(), '%s/optimizer.pth'%(checkpoints_dir))
            if args.exp_params['multi_gpu'] is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.exp_params['model_name'], epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.exp_params['model_name'], epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader, logger, loss_func, args):

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data  
        
        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        with torch.no_grad():
            pred_flows, loss = model(pos1, pos2, norm1, norm2)

            eval_loss = sequence_loss(pos1, pos2, pred_flows, flow, args.exp_params['loss'], loss_func)
            eval_loss += eval_loss + 0.01*loss.sum()
            
            epe3d = torch.norm(pred_flows[-1] - flow, dim = 2).mean()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_flows[-1].cpu().numpy(), flow.cpu().numpy())
        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)


        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())
        metrics['acc_3d'].append(acc3d_strict)

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])
    mean_as = np.mean(metrics['acc_3d'])

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       ))

    print(res_str)
    logger.info(res_str)


    return mean_epe3d, mean_eval, mean_as

if __name__ == '__main__':
    main()




