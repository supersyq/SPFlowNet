"""
References:
PointPWC-Net: https://github.com/DylanWusee/PointPWC
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
FlowStep3D: https://github.com/yairkit/flowstep3d
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
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

import transforms
import datasets
import cmd_args 
from main_utils import *

from losses.unsupervised_losses import UnSupervisedL1Loss

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
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%snon_occlusions-'%args.exp_params['model_name'] + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    print(log_dir)
    os.system('cp %s %s' % ('model.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('./configs_without_occlusions/config_evaluate.yaml', log_dir))
    os.system('cp %s %s' % ('./utils/pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('./utils/modules.py', log_dir))
    os.system('cp %s %s' % ('./losses/unsupervised_losses.py', log_dir))
    os.system('cp %s %s' % ('./losses/common_losses.py', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.exp_params['model_name'])
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.exp_params['model_name'])
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    # blue = lambda x: '\033[94m' + x + '\033[0m'
    model = SPFlowNet(args)

    loss_func = losses_dict[args.exp_params['loss']['loss_type']](**args.exp_params['loss'])

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.exp_params['workers'],
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, path = data

        #move to cuda 
        pos1 = pos1.cuda()
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda() 

        model = model.eval()
        with torch.no_grad(): 
            pred_flows,_ = model(pos1, pos2, norm1, norm2)
            loss = sequence_loss(pos1, pos2, pred_flows, flow, args.exp_params, loss_func)
            full_flow = pred_flows[-1]
            epe3d = torch.norm(full_flow - flow, dim = 2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = pos1.cpu().numpy()
        pc2_np = pos2.cpu().numpy() 
        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        np.set_printoptions(suppress=True)
        
        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_flows[-1].cpu().numpy(), sf_np)
        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
                                                        pc1_np+sf_np,
                                                        pc1_np+pred_sf,
                                                        path)
        EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%('Evaluate', mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds
                       ))
    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




