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
from model import Encoder, Encoder_BiRNN, CVAE_Transformer
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

def CalcDiffNorm_Rot(jointRot, jointPos, selectedJoint, fps, params):
    '''
        jointRot.shape = JT4
        jointPos.shape = JT3
        selectedJoint.shape = (J-1)
        fps = int
    '''
    w_rot = np.zeros((jointRot.shape[1]-1, selectedJoint.shape[0], 3), dtype=np.float32)
    for k in range(jointRot.shape[1]-1):
        q_nxt = jointRot[selectedJoint, k+1]
        q_now = jointRot[selectedJoint, k]
        q_now_inv = geometry.quaternion_invert(q_now)

        delta_q = geometry.quaternion_multiply(q_nxt, q_now_inv).cpu().numpy()
        delta_q_len = np.linalg.norm(delta_q[:, 1:], axis=-1)
        delta_q_angle = 2*np.arctan2(delta_q_len, delta_q[:, 0])
        w_rot[k] = delta_q[:, 1:] * delta_q_angle.reshape(-1,1) * fps

    diffNorm_rot = np.linalg.norm(w_rot[1:]-w_rot[:-1], ord=1, axis=-1).mean()
    return diffNorm_rot

# def calcAccLoss(q_pred, q_gt):


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

        if opt['data']['mask'] == 'rand':
            n_trans = np.random.randint(2, opt['model']['seq_length']-2)
            n_past = (opt['model']['seq_length'] - n_trans) // 2
            n_future = opt['model']['seq_length'] - n_past - n_trans
        elif opt['data']['mask'] == 'static':
            n_past, n_future, n_trans = opt['model']['n_past'], opt['model']['n_future'], opt['model']['n_trans']

        gt_pose, interp_pose = interpolation(positions, rotations, n_past = n_past, n_future = n_future, n_trans = n_trans)

        # 数据放到GPU to_device
        gt_pose = gt_pose.astype(np.float32)
        interp_pose = interp_pose.astype(np.float32)
        input = torch.from_numpy(interp_pose).to(device)
        target_output = torch.from_numpy(gt_pose).to(device)

        # Training
        loss_KL = torch.tensor(0.0)
        if model.name == 'CVAE':
            # output, loss_KL = model({'gt':target_output, 'interpolated': input}, n_past=n_past, n_future=n_future, n_trans=n_trans)
            outputList = []
            for i in range(args.nsample):
                tmp = model.generate({'gt':target_output, 'interpolated': input}, n_past=n_past, n_future=n_future, n_trans=n_trans)
                outputList.append(tmp[:, None, :, :])
            output = torch.cat(outputList, dim=1)
            output = output.reshape(-1, output.shape[2], output.shape[3]) # (B*S, T, C)
        else: output = model(input, n_past=n_past, n_future=n_future, n_trans=n_trans)

        # Results output
        local_q_pred = output[:, :, opt['model']['num_joints']*3:]       # B, F, J*4            局部四元数
        local_q_gt = target_output[:, :, opt['model']['num_joints']*3:]  # B, F, J*4
        local_p_pred = output[:, :, 0:opt['model']['num_joints']*3]       # B, F, J*3            局部位置坐标
        local_p_gt = target_output[:, :, 0:opt['model']['num_joints']*3]  # B, F, J*3

        #------------------------global or local data-----------------------------------
        local_q_pred = local_q_pred.view(local_q_pred.shape[0], local_q_pred.shape[1], -1, 4)
        local_q_pred_ = local_q_pred / torch.norm(local_q_pred, dim=-1, keepdim=True) #BTJC
        local_p_pred_ = local_p_pred.view(local_p_pred.shape[0], local_p_pred.shape[1], -1, 3)
        
        local_q_gt = local_q_gt.view(local_q_gt.size(0), local_q_gt.size(1), -1, 4)  # ground truth rotation and position data
        local_p_gt_ = local_p_gt.view(local_p_gt.size(0), local_p_gt.size(1), -1, 3)  # B, F, J, 3

        root_pred = local_p_pred[:, :, 0:3]         # B, F, 3   根节点预测值
        root_gt = local_p_gt[:, :, 0:3]           # B, F,  3

        global_q_pred, global_p_pred = uf.quat_fk_cuda(local_q_pred_, local_p_pred_, parents)

        if args.get_3dpos_method == 'smpl':
            param2xyz = {"pose_rep": "rotquat", "glob_rot": None, "glob": True, "jointstype": 'smpl', "translation": False, "vertstrans": False}
            rotation2xyz = Rotation2xyz(device=torch.device("cuda"))
            global_p_pred = rotation2xyz(local_q_pred_.permute(0,2,3,1), None, **param2xyz)
            global_p_pred = global_p_pred.permute(0,3,1,2)

        loss_dic = {}
        if mode == 'valid':
            # loss --------------------------------------
            # loss_ik += torch.mean(torch.abs(glbl_p_pred_ - glbl_p_gt_) / x_std_n)   # ik运动学损失                                                            # Lik反运动学损失
            loss_quat = torch.mean(torch.abs(local_q_pred - local_q_gt))       # 旋转四元数损失
            loss_position = torch.mean(torch.abs(root_pred - root_gt))     # 位移损失
            loss_fk = torch.mean(torch.abs(global_p_pred - global_p_gt) / x_std_n)

            # 计算损失函数
            loss_total = opt['train']['loss_quat_weight'] * loss_quat + \
                            opt['train']['loss_fk_weight'] * loss_fk + \
                            opt['train']['loss_position_weight'] * loss_position + \
                            opt['train']['loss_KL_weight'] * loss_KL
                            # opt['train']['loss_fk_weight'] * loss_fk

            # update parameters
            loss_fk = opt['train']['loss_fk_weight'] * loss_fk
            loss_quat = opt['train']['loss_quat_weight'] * loss_quat
            loss_pos = opt['train']['loss_position_weight'] * loss_position
            loss_KL = opt['train']['loss_KL_weight'] * loss_KL
            # local to global for metrics----------------------------------------
            mean = x_mean_n if mode == 'train' else x_mean_n_vld
            std = x_std_n if mode == 'train' else x_std_n_vld
            trans_global_p_pred = (global_p_pred[:, n_past:n_past+n_trans, ...] - mean).detach().cpu().numpy() / std.detach().cpu().numpy()  # Normalization
            trans_global_p_gt = (global_p_gt[:, n_past:n_past+n_trans, ...] - mean).detach().cpu().numpy() / std.detach().cpu().numpy()
            # B*T*J*3
            l2p_error = np.mean(np.sqrt(np.sum((trans_global_p_pred - trans_global_p_gt) ** 2.0, axis=(2, 3))))

            loss_dic = {
                "loss_total": loss_total.item(),
                "loss_pos": loss_pos.item(),
                "loss_quat": loss_quat.item(),
                "loss_fk": loss_fk.item(),
                "loss_KL": loss_KL.item(),
                # "loss_Acc": calcAccLoss(local_q_pred, local_q_gt),
                "l2p_error": l2p_error.item()
            }

        if mode == "viz":
            fps = 10
            global_p_gt = global_p_gt.permute(0,2,3,1).detach().cpu().numpy()
            global_p_pred = global_p_pred.permute(0,2,3,1).detach().cpu().numpy()
            assert opt['model']['n_past'] == opt['model']['n_future']
            seedLength = opt['model']['n_past']

            B, J, C, T = global_p_gt.shape
            global_p_pred = global_p_pred.reshape(B, -1, J, C, T)
            savePath = './gifResults_{}'.format(opt['train']['method'])
            if not os.path.exists(savePath): os.makedirs(savePath)

            for i in range(B):
                from copy import deepcopy
                lastGtPose = deepcopy(global_p_gt[i, :, :, -seedLength])
                allMotions = np.concatenate((global_p_pred[i], global_p_gt[i][None, :, :, :]), axis=0)
                plot_3d_motion_dico((allMotions, 60, savePath + '/{}_th_predGT.gif'.format(i), {'pose_rep':'xyz'},
                    {"title": "gen", "interval": 1000/fps, "labelSeq":None, "seedLength":seedLength, "lastGtPose":lastGtPose}))

        return loss_dic

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/test_config_BABEL_rm_cvae.yaml')
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--dataset_mode', type=str, default='rand', choices=['same', 'diff', 'rand', 'select'])
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--mode', type=str, default="viz", choices=['viz', 'valid'])
    parser.add_argument('--get_3dpos_method', type=str, default="fk", choices=['fk', 'smpl'])
    parser.add_argument('--model', type=str, default='cvae', choices=['transformer', 'birnn', 'cvae'])
    parser.add_argument('--nsample', type=int, default=6)

    # parser.add_argument('--name_suffix', type=str, default='default')
    # parser.add_argument('--model_type', type=str, default='uni_dir', choices=['uni_dir', 'bi_dir'])

    args = parser.parse_args()

    # ===========================================读取配置信息===============================================
    opt = yaml.load(open(args.config, 'r').read(), Loader=yaml.FullLoader)      # 用mocap_bfa, mocap_xia数据集训练
    # opt = yaml.load(open('./config/train_config_lafan.yaml', 'r').read())     # 用lafan数据集训练
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    print(stamp)
    # assert 0
    
    # log_dir = opt['train']['log_dir']
    # writer = SummaryWriter(log_dir)

    # output_dir = opt['train']['output_dir']     # 模型输出路径
    # if not os.path.exists(output_dir): os.mkdir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if opt['train']['cuda']:
        torch.cuda.manual_seed(args.seed)

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
                                    shuffle=args.shuffle, num_workers=opt['data']['num_workers'])

    lafan_data_valid = BABEL(opt['data']['vald_dir'], seq_len = opt['model']['seq_length'], motion_type = args.dataset_mode)
    x_mean_n_vld = lafan_data_train.x_mean.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n_vld = lafan_data_train.x_std.cuda().view(1, 1, opt['model']['num_joints'], 3)
    x_std_n_vld[0, 0, 0] = torch.tensor([1.,1.,1.], dtype=torch.float32, device=device)

    lafan_loader_valid = DataLoader(lafan_data_valid, \
                                    batch_size=opt['train']['batch_size'], \
                                    shuffle=args.shuffle, num_workers=opt['data']['num_workers'])

    #===============================初始化模型=======================================
    ## initialize model ##
    kargs = {
        'device': device,
        'seq_len': opt['model']['seq_length'],
        'input_dim': opt['model']['input_dim'],
        'n_layers': opt['model']['n_layers'],
        'n_head': opt['model']['n_head'],
        'd_k': opt['model']['d_k'],
        'd_v': opt['model']['d_v'],
        'd_model': opt['model']['d_model'],
        'd_inner': opt['model']['d_inner'],
        'dropout': opt['train']['dropout'],
        'n_past': opt['model']['n_past'],
        'n_future': opt['model']['n_future'],
        'n_trans': opt['model']['n_trans']
    }
    modelDict = {
        'transformer': Encoder,
        'birnn': Encoder_BiRNN,
        'cvae': CVAE_Transformer
    }
    model = modelDict[args.model](**kargs)
    # print(model)
    model.to(device)
    print('Encoder params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    epoch_i = 1
    if opt['test']['model_dir']:
        checkpoint = torch.load(opt['test']['model_dir'])
        model.load_state_dict(checkpoint['model'])
        epoch_i = checkpoint['epoch']

    #============================================= train ===================================================

    curr_window = opt['model']['n_past'] + opt['model']['n_trans'] + opt['model']['n_future']
    print(f"curr_window: {curr_window}")
    for epoch_i in range(1, opt['train']['num_epoch']+1):  # 每个epoch轮完一遍所有的训练数据
        #validate:
        lossTerms = ["loss_total","loss_pos","loss_quat","loss_fk","loss_KL","l2p_error"]
        losses = {key:[] for key in lossTerms}
        for i_batch, sampled_batch in tqdm(enumerate(lafan_loader_valid)):
            loss_dic = do_batch(i_batch, sampled_batch, args.mode)
            for key, val in loss_dic.items():
                losses[key].append(val)
            if args.mode == 'viz': break
        
        if args.mode == 'viz': break
        RecordBatch(losses, 'valid', epoch_i)

#python viz_transformer.py --model birnn --dataset_mode select --config config/test_config_BABEL_rm_birnn.yaml

