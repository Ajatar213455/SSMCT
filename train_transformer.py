import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from utils.skeleton import Skeleton

try: from utils.interpolate import interpolate_local
except: from SSMCT.utils.interpolate import interpolate_local

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

BABELparents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

def RecordBatch(losses, mode, epoch):
    for key, val in losses.items():
        losses[key] = np.mean(val)
    for key, val in losses.items():
        writer.add_scalar(f'{mode}_{key}', val, global_step = epoch)
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
    # inter_pos, inter_local_quats = interpolate_local(x.numpy(), q.numpy(), n_past, n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4
    inter_pos, inter_local_quats = interpolate_local(x, q, n_past, n_future)

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
    elif mode == "valid":
        model.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    with grad_env():
        loss_fk = 0
        loss_quat = 0
        loss_position = 0
        loss_root = 0
        positions = batch_data['local_x'] # B, F, J, 3
        rotations = batch_data['local_q']
        global_p_gt = batch_data['X'].cuda()
        global_q_gt = batch_data['Q']
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
        local_q_gt = target_output[:, :, opt['model']['num_joints']*3:]  # B, F, J*4
        local_p_pred = output[:, :, 0:opt['model']['num_joints']*3]       # B, F, J*3            局部位置坐标
        local_p_gt = target_output[:, :, 0:opt['model']['num_joints']*3]  # B, F, J*3

        #------------------------global or local data-----------------------------------
        local_q_pred = local_q_pred.view(local_q_pred.shape[0], local_q_pred.shape[1], -1, 4)
        local_q_pred_ = local_q_pred / torch.norm(local_q_pred, dim=-1, keepdim=True)
        local_p_pred_ = local_p_pred.view(local_p_pred.shape[0], local_p_pred.shape[1], -1, 3)


        local_q_gt = local_q_gt.view(local_q_gt.size(0), local_q_gt.size(1), -1, 4)  # ground truth rotation and position data
        local_p_gt_ = local_p_gt.view(local_p_gt.size(0), local_p_gt.size(1), -1, 3)  # B, F, J, 3


        root_pred = local_p_pred[:, :, 0:3]         # B, F, 3   根节点预测值
        root_gt = local_p_gt[:, :, 0:3]           # B, F,  3

        #----------------------------local data-------------------------------------

        # global_p_gt = skeleton_mocap.forward_kinematics(local_q_gt.detach().cpu(), root_gt.detach().cpu())
        # global_q_pred, global_p_pred = uf.quat_fk(local_q_pred.detach().cpu().numpy(), local_p_pred_.detach().cpu().numpy(), parents)
        # global_p_pred = torch.from_numpy(global_p_pred) #error: detach before calc loss
        # global_p_pred = global_p_pred.to(device)

        global_q_pred, global_p_pred = uf.quat_fk_cuda(local_q_pred, local_p_pred_, parents)
        

        # loss --------------------------------------
        # loss_ik += torch.mean(torch.abs(glbl_p_pred_ - glbl_p_gt_) / x_std_n)   # ik运动学损失                                                            # Lik反运动学损失
        loss_quat += torch.mean(torch.abs(local_q_pred - local_q_gt))       # 旋转四元数损失
        loss_position += torch.mean(torch.abs(root_pred - root_gt))     # 位移损失
        loss_fk += torch.mean(torch.abs(global_p_pred - global_p_gt) / x_std_n)


        # 计算损失函数
        loss_total = opt['train']['loss_quat_weight'] * loss_quat + \
                        opt['train']['loss_fk_weight'] * loss_fk + \
                        opt['train']['loss_position_weight'] * loss_position
                        # opt['train']['loss_fk_weight'] * loss_fk

        # update parameters
        if mode == 'train':
            loss_total.backward()
            optimizer.step()

        # loss_fk = opt['train']['loss_fk_weight'] * loss_fk
        loss_fk = opt['train']['loss_fk_weight'] * loss_fk
        loss_quat = opt['train']['loss_quat_weight'] * loss_quat
        loss_pos = opt['train']['loss_position_weight'] * loss_position
        # local to global for metrics----------------------------------------

        mean = x_mean_n if mode == 'train' else x_mean_n_vld
        std = x_std_n if mode == 'train' else x_std_n_vld
        trans_global_p_pred = (global_p_pred[:,opt['model']['n_past']: opt['model']['n_past'] + opt['model']['n_trans'],...] - mean).detach().cpu().numpy() / std.detach().cpu().numpy()  # Normalization
        trans_global_p_gt = (global_p_gt[:,opt['model']['n_past']: opt['model']['n_past'] + opt['model']['n_trans'],...] - mean).detach().cpu().numpy() / std.detach().cpu().numpy()
        l2p_error = np.mean(np.sqrt(np.sum((trans_global_p_pred - trans_global_p_gt) ** 2.0, axis=(2, 3))))


        loss_dic = {
            "loss_total": loss_total.item(),
            "loss_pos": loss_pos.item(),
            "loss_quat": loss_quat.item(),
            "loss_fk": loss_fk.item(),
            "l2p_error": l2p_error.item()
        }
        return loss_dic

if __name__ == '__main__':
    # ===========================================读取配置信息===============================================
    opt = yaml.load(open('./config/train_config_BABEL.yaml', 'r').read(), Loader=yaml.FullLoader)      # 用mocap_bfa, mocap_xia数据集训练
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
    
    optimizer = optim.Adam(filter(lambda x: x.requires_grad,  model.parameters()),
                           lr=opt['train']['lr'],)
    scheduler_steplr = StepLR(optimizer, step_size=200, gamma=opt['train']['weight_decay'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler_steplr)

    #============================================= train ===================================================

    curr_window = opt['model']['n_past'] + opt['model']['n_trans'] + opt['model']['n_future']
    print(f"curr_window: {curr_window}")
    for epoch_i in range(1, opt['train']['num_epoch']+1):  # 每个epoch轮完一遍所有的训练数据
        model.train()
        scheduler_warmup.step(epoch_i)
        print("epoch: ",epoch_i, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))

        # 每个batch训练一批数据
        lossTerms = ["loss_total","loss_pos","loss_quat","loss_fk","l2p_error"]
        losses = {key:[] for key in lossTerms}
        for batch_i, batch_data in tqdm(enumerate(lafan_loader_train)):  # mini_batch
            loss_dic = do_batch(batch_i, batch_data, 'train')
            for key, val in loss_dic.items():
                losses[key].append(val)
        
        RecordBatch(losses, 'train', epoch_i)

        if epoch_i % opt['train']['save_per_epochs'] == 0 or epoch_i == 1:
            #validate:
            losses = {key:[] for key in lossTerms}
            for i_batch, sampled_batch in tqdm(enumerate(lafan_loader_valid)):
                loss_dic = do_batchloss_dic = do_batch(i_batch, sampled_batch, 'valid')
                for key, val in loss_dic.items():
                    losses[key].append(val)
            RecordBatch(losses, 'valid', epoch_i)
            
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch_i
            }
            filename = os.path.join(opt['train']['output_dir'], f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)


