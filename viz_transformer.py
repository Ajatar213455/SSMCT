import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from utils.skeleton import Skeleton
from utils.interpolate import interpolate_local
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from modules.warmup import GradualWarmupScheduler
import torch.optim as optim
import numpy as np
import yaml
import time
import random
from model import Encoder
# from visdom import Visdom
import utils.benchmarks as bench
import utils.utils_func as uf
from tqdm import tqdm

from mirrored_supModel.BABELdata.rotation2xyz import Rotation2xyz
from mirrored_supModel.LaFanBABEL import BABEL
from mirrored_supModel.ACTORanim import plot_3d_motion_dico

BABELparents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

def RecordBatch(losses, mode, epoch):
    for key, val in losses.items():
        losses[key] = np.mean(val)
    for key, val in losses.items():
        # writer.add_scalar(f'{mode}_{key}', val, global_step = epoch)
        losses[key] = format(val, '.3f')
    print(f"epoch = {epoch}, mode = {mode}, losses = {losses}")

def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))

def interpolation(X, Q, n_past=10, n_future=10, n_trans = 40):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape     numpy(Batchsize, Timesteps, Joints, 3)
    :param Q: Local quaternions array of shape     numpy(B, T, J, 4)
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :param n_trans:
    :return:  B, curr_window, xxx
    """
    batchsize = X.shape[0]  #  B

    # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
    curr_window = n_trans + n_past + n_future
    curr_x = X[:, :curr_window, ...]    # B, curr_window, J, 3
    curr_q = Q[:, :curr_window, ...]    # B, curr_window, J, 4
    gt_pose = np.concatenate([X.reshape((batchsize, X.shape[1], -1)), Q.reshape((batchsize, Q.shape[1], -1))], axis=2)  # [B, curr_window, J*3+J*4]

    # Interpolation pos/quats
    x, q = curr_x, curr_q    # x: B,curr_window,J,3       q: B, curr_window, J, 4
    inter_pos, inter_local_quats = interpolate_local(x.numpy(), q.numpy(), n_past, n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4

    trans_inter_pos = inter_pos[:, 1:-1, :, :]    #  B, n_trans, J, 3  把头尾2帧去掉
    inter_local_quats = inter_local_quats[:, 1:-1, :, :]  #  B, n_trans, J, 4
    total_interp_positions = np.concatenate([X[:, 0: n_past, ...], trans_inter_pos, X[:, n_past+n_trans: , ...]], axis = 1)    # B, curr_window, J, 3  重新拼接
    total_interp_rotations = np.concatenate([Q[:, 0: n_past, ...], inter_local_quats, Q[:, n_past+n_trans: , ...]], axis = 1)  # B, curr_window, J, 4
    interp_pose = np.concatenate([total_interp_positions.reshape((batchsize, X.shape[1], -1)), total_interp_rotations.reshape((batchsize, Q.shape[1], -1))], axis=2)  # [B, curr_window, xxx]
    return gt_pose, interp_pose

from tensorboardX import SummaryWriter

def do_batch(batch_i, batch_data, mode):
    if mode == "train":
        model.train()
        grad_env = torch.enable_grad
    elif mode == "valid" or mode == "viz":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    with grad_env():
        positions = batch_data['local_x'] # B, F, J, 3
        rotations = batch_data['local_q']
        global_p_gt = batch_data['X'].cuda()
        # 插值 求ground truth 和 插值结果
        # gt_pose numpy [B, F, dim] interp_pose numpy[B, F, dim]
        gt_pose, interp_pose = interpolation(positions,
                                                rotations,
                                                n_past = opt['model']['n_past'],
                                                n_future = opt['model']['n_future'],
                                                n_trans = opt['model']['n_trans'])

        # 数据放到GPU to_device
        gt_pose = gt_pose.astype(np.float32)
        interp_pose = interp_pose.astype(np.float32)
        input = torch.from_numpy(interp_pose).to(device)
        target_output = torch.from_numpy(gt_pose).to(device)

        # Training
        output = model(input)
        if mode == 'train':
            optimizer.zero_grad()

        # Results output
        local_q_pred = output[:, :, opt['model']['num_joints']*3:]       # B, F, J*4            局部四元数
        local_p_pred = output[:, :, 0:opt['model']['num_joints']*3]       # B, F, J*3            局部位置坐标

        #------------------------global or local data-----------------------------------
        local_q_pred = local_q_pred.view(local_q_pred.shape[0], local_q_pred.shape[1], -1, 4)
        local_q_pred_ = local_q_pred / torch.norm(local_q_pred, dim=-1, keepdim=True) #BTJC
        local_p_pred_ = local_p_pred.view(local_p_pred.shape[0], local_p_pred.shape[1], -1, 3)

        global_q_pred, global_p_pred = uf.quat_fk_cuda(local_q_pred_, local_p_pred_, parents)

        if args.get_3dpos_method == 'smpl':
            param2xyz = {"pose_rep": "rotquat", "glob_rot": None, "glob": True, "jointstype": 'smpl', "translation": False, "vertstrans": False}
            rotation2xyz = Rotation2xyz(device=torch.device("cuda"))
            global_p_pred = rotation2xyz(local_q_pred_.permute(0,2,3,1), None, **param2xyz)
            global_p_pred = global_p_pred.permute(0,3,1,2)


        if mode == "viz":
            fps = 10
            global_p_gt = global_p_gt.permute(0,2,3,1).detach().cpu().numpy()
            global_p_pred = global_p_pred.permute(0,2,3,1).detach().cpu().numpy()
            assert opt['model']['n_past'] == opt['model']['n_future']
            seedLength = opt['model']['n_past']
            for i in range(global_p_gt.shape[0]):
                from copy import deepcopy
                lastGtPose = global_p_gt[i, :, :, -seedLength]
                lastGtPose = deepcopy(lastGtPose)
                # lastGtPose[:, [2,0,1]] = lastGtPose[:, [0,1,2]]
                savePath = './gifResults_{}'.format('transformer')
                if not os.path.exists(savePath): os.makedirs(savePath)
                # print("pred_list[i].shape = ", pred_list[i].shape)
                plot_3d_motion_dico((global_p_pred[i], 60, savePath + '/{}_th_predGT.gif'.format(i), {'pose_rep':'xyz'},
                    {"title": "gen", "interval": 1000/fps, "labelSeq":None, "seedLength":seedLength, "lastGtPose":lastGtPose, "GT":global_p_gt[i]}))

        loss_dic = {}
        return loss_dic

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train-BABEL_large_newmodel.yaml')
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--dataset_mode', type=str, default='rand', choices=['same', 'diff', 'rand'])
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--mode', type=str, default="viz", choices=['viz', 'valid'])
    parser.add_argument('--get_3dpos_method', type=str, default="fk", choices=['fk', 'smpl'])
    # parser.add_argument('--name_suffix', type=str, default='default')
    # parser.add_argument('--model_type', type=str, default='uni_dir', choices=['uni_dir', 'bi_dir'])

    args = parser.parse_args()

    # ===========================================读取配置信息===============================================
    opt = yaml.load(open('./config/test_config_BABEL.yaml', 'r').read(), Loader=yaml.FullLoader)      # 用mocap_bfa, mocap_xia数据集训练
    # opt = yaml.load(open('./config/train_config_lafan.yaml', 'r').read())     # 用lafan数据集训练
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    print(stamp)
    # assert 0
    
    log_dir = opt['train']['log_dir']
    writer = SummaryWriter(log_dir)

    output_dir = opt['train']['output_dir']     # 模型输出路径
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    random.seed(opt['train']['seed'])
    torch.manual_seed(opt['train']['seed'])
    if opt['train']['cuda']:
        torch.cuda.manual_seed(opt['train']['seed'])

    # ===================================使用GPU==================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)


    #==========================初始化Skel和数据========================================
    parents = BABELparents

    lafan_data_train = BABEL(opt['data']['data_dir'], seq_len = opt['model']['seq_length'])
    x_mean_n = lafan_data_train.x_mean.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n = lafan_data_train.x_std.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n[0, 0, 0] = torch.tensor([1.,1.,1.], dtype=torch.float32, device=device)

    lafan_loader_train = DataLoader(lafan_data_train, \
                                    batch_size=opt['train']['batch_size'], \
                                    shuffle=True, num_workers=opt['data']['num_workers'])

    lafan_data_valid = BABEL(opt['data']['vald_dir'], seq_len = opt['model']['seq_length'])
    x_mean_n_vld = lafan_data_train.x_mean.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n_vld = lafan_data_train.x_std.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n_vld[0, 0, 0] = torch.tensor([1.,1.,1.], dtype=torch.float32, device=device)

    lafan_loader_valid = DataLoader(lafan_data_valid, \
                                    batch_size=opt['train']['batch_size'], \
                                    shuffle=True, num_workers=opt['data']['num_workers'])

    #===============================初始化模型=======================================
    ## initialize model ##
    model = Encoder(device = device,
                    seq_len=opt['model']['seq_length'],
                    input_dim=opt['model']['input_dim'],
                    n_layers=opt['model']['n_layers'],
                    n_head=opt['model']['n_head'],
                    d_k=opt['model']['d_k'],
                    d_v=opt['model']['d_v'],
                    d_model=opt['model']['d_model'],
                    d_inner=opt['model']['d_inner'],
                    dropout=opt['train']['dropout'],
                    n_past=opt['model']['n_past'],
                    n_future=opt['model']['n_future'],
                    n_trans=opt['model']['n_trans'])
    # print(model)
    model.to(device)
    print('Encoder params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    epoch_i = 1
    if opt['test']['model_dir']:
        checkpoint = torch.load(opt['test']['model_dir'])
        model.load_state_dict(checkpoint['model'])
        epoch_i = checkpoint['epoch']

    optimizer = optim.Adam(filter(lambda x: x.requires_grad,  model.parameters()),
                           lr=opt['train']['lr'],)
    scheduler_steplr = StepLR(optimizer, step_size=200, gamma=opt['train']['weight_decay'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler_steplr)

    #============================================= train ===================================================

    curr_window = opt['model']['n_past'] + opt['model']['n_trans'] + opt['model']['n_future']
    print(f"curr_window: {curr_window}")
    for epoch_i in range(1, opt['train']['num_epoch']+1):  # 每个epoch轮完一遍所有的训练数据
        #validate:
        lossTerms = ["loss_total","loss_pos","loss_quat","loss_fk","l2p_error"]
        losses = {key:[] for key in lossTerms}
        for i_batch, sampled_batch in tqdm(enumerate(lafan_loader_valid)):
            loss_dic = do_batchloss_dic = do_batch(i_batch, sampled_batch, args.mode)
            for key, val in loss_dic.items():
                losses[key].append(val)
            if args.mode == 'viz': break
        
        if args.mode == 'viz': break
        RecordBatch(losses, 'valid', epoch_i)


