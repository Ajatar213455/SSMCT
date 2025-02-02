import pickle as pkl
import numpy as np
import zipfile
import os

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import time
from utils.skeleton import Skeleton
import torch
import utils.utils_func as uf
import utils.quaternion as qua
np.set_printoptions(precision=3)

def fast_npss(gt_seq, pred_seq):
    """
    Computes Normalized Power Spectrum Similarity (NPSS).

    This is the metric proposed by Gropalakrishnan et al (2019).
    This implementation uses numpy parallelism for improved performance.

    :param gt_seq: ground-truth array of shape : (Batchsize, Timesteps, Dimension)
    :param pred_seq: shape : (Batchsize, Timesteps, Dimension)
    :return: The average npss metric for the batch
    """
    # Fourier coefficients along the time dimension
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seq, axis=1))
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seq, axis=1))

    # Square of the Fourier coefficients
    gt_power = np.square(gt_fourier_coeffs)
    pred_power = np.square(pred_fourier_coeffs)

    # Sum of powers over time dimension
    gt_total_power = np.sum(gt_power, axis=1)
    pred_total_power = np.sum(pred_power, axis=1)

    # Normalize powers with totals
    gt_norm_power = gt_power / gt_total_power[:, np.newaxis, :]
    pred_norm_power = pred_power / pred_total_power[:, np.newaxis, :]

    # Cumulative sum over time
    cdf_gt_power = np.cumsum(gt_norm_power, axis=1)
    cdf_pred_power = np.cumsum(pred_norm_power, axis=1)

    # Earth mover distance
    emd = np.linalg.norm((cdf_pred_power - cdf_gt_power), ord=1, axis=1)

    # Weighted EMD
    power_weighted_emd = np.average(emd, weights=gt_total_power)

    return power_weighted_emd


def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))


def benchmark_interpolation(X, Q, x_mean, x_std, offsets, parents, out_path=None, n_past=10, n_future=10):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    对于给定的数据，分析过渡生成任务的基线
    :param X: Local positions array of shape 局部位置 numpy数组 (Batchsize, Timesteps, Joints, 3)
    :param Q: Local quaternions array of shape 局部四元数 numpy数组 (B, T, J, 4)
    :param x_mean : Mean vector of local positions of shape (1, J*3, 1)
    :param x_std: Standard deviation vector of local positions (1, J*3, 1)
    :param offsets: Local bone offsets tensor of shape (1, 1, J, 3) 子关节的偏移量 常量
    :param parents: List of bone parents indices defining the hierarchy 每个关节的父关节的下标构成的列表
    :param out_path: optional path for saving the results
    :param n_past: Number of frames used as past context 用于已知起始帧的数量
    :param n_future: Number of frames used as future context (only the first frame is used as the target) 用于已知未来帧的数量，注意只有第1帧用于目标
    :return: Results dictionary
    """

    trans_lengths = [5, 15, 30, 45]
    n_joints = 22
    res = {}

    for n_trans in trans_lengths:
        print('Computing errors for transition length = {}...'.format(n_trans))

        # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
        curr_window = n_trans + n_past + n_future   # 当前考虑的总帧数
        curr_x = X[:, :curr_window, ...]    # 当前考虑的局部位置 B, curr_window, J, 3
        curr_q = Q[:, :curr_window, ...]    # 当前考虑的局部旋转四元数 B, curr_window, J, 4
        batchsize = curr_x.shape[0] #  数据集长度B

        # Ground-truth positions/quats/eulers
        gt_glbl_quats = curr_q     # 全局旋转四元数真实值 B, curr_window, J, 4
        gt_glbl_poses = curr_x     # 所有关节的全局位置真实值 B, curr_window, J, 3

        trans_gt_glbl_poses = gt_glbl_poses[:, n_past: -n_future, ...]  # 过渡区间的全局位置真实值  B, n_trans, J, 3
        trans_gt_glbl_quats = torch.from_numpy(gt_glbl_quats[:, n_past: -n_future, ...])   # 过渡区间的全局旋转真实值  B, n_trans, J, 4
        # # Local to global with Forward Kinematics (FK) 局部位置和旋转转换为全局位置和旋转
        # trans_gt_global_quats, trans_gt_global_poses = uf.quat_fk(trans_gt_local_quats, trans_gt_local_poses, parents)

        # Normalize 全局位置 正则化处理
        trans_gt_glbl_poses = (torch.from_numpy(trans_gt_glbl_poses) - x_mean_n) / x_std_n

        # Zero-velocity pos/quats
        trans_gt_glbl_poses_ = trans_gt_glbl_poses.numpy()

        zerov_trans_glbl_quats, zerov_trans_glbl_poses = np.zeros_like(trans_gt_glbl_quats), np.zeros_like(trans_gt_glbl_poses_)
        zerov_trans_glbl_quats[:, :, :, :] = gt_glbl_quats[:, n_past - 1:n_past, :, :]
        zerov_trans_glbl_quats = torch.from_numpy(zerov_trans_glbl_quats)
        zerov_trans_glbl_poses[:, :, :, :] = gt_glbl_poses[:, n_past - 1:n_past, :, :]
        # Normalize
        trans_zerov_glbl_poses = (torch.from_numpy(zerov_trans_glbl_poses) - x_mean_n) / x_std_n

        # Interpolation pos/quats
        r, q = curr_x[:, :, :, :], curr_q    # r: L,curr_window,1,3       q:当前考虑的局部旋转四元数 B, curr_window, J, 4
        inter_pos, inter_glbl_quats = uf.interpolate_local(r, q, n_past, n_future)  # inter_root: B, n_trans + 2, 1, 3   inter_local_quats: B, n_trans + 2, J, 4
        # trans_inter_root = torch.from_numpy(inter_root[:, 1:-1, :, :].reshape(inter_root.shape[0], -1, 3)).to(torch.float32)     # 根节点位移 插值的中间帧 B, n_trans, 1, 3
        # trans_inter_glbl_quats = torch.from_numpy(inter_glbl_quats[:, 1:-1, :, :]).to(torch.float32)  #旋转四元数 插值的中间帧 B, n_trans, J, 4
        # trans_inter_offsets = np.tile(offsets, [batchsize, n_trans, 1, 1]) # 子关节的偏移量真实值 B, n_trans, J, 3
        trans_inter_glbl_quats = inter_glbl_quats[:, 1:-1, :, :]
        trans_inter_glbl_poses = inter_pos[:, 1:-1, :, :]
        # trans_inter_glbl_poses = skeleton_mocap.forward_kinematics(trans_inter_glbl_quats.to(device), trans_inter_root.to(device))  # 所有关节的局部位置插值值 B, n_trans, J, 3

        # To global
        # trans_inter_glbl_poses = trans_inter_glbl_poses.reshape((trans_inter_glbl_poses.shape[0], -1, n_joints * 3)).transpose([0, 2, 1])
        # # # Normalize
        trans_inter_glbl_poses = (torch.from_numpy(trans_inter_glbl_poses) - x_mean_n) / x_std_n

        # print(f" trans_gt_glbl_quats :{type(trans_gt_glbl_quats)}")
        # print(f"zerov_trans_glbl_quats :{type(zerov_trans_glbl_quats)}")
        # print(f"zerov_inter_glbl_quats :{type(trans_inter_glbl_quats)}")
        # print(f"trans_zerov_glbl_poses :{type(trans_zerov_glbl_poses)}")
        # print(f"trans_inter_glbl_poses :{type(trans_inter_glbl_poses)}")
        # print(f"trans_zerov_glbl_poses :{type(trans_zerov_glbl_poses)}")
        # print(f"trans_inter_glbl_quats :{type(trans_inter_glbl_quats)}")

        # 与baseline的比较实验 评估指标： L2Q、L2P和NPSS
        # Local quaternion loss L2Q
        res[('zerov_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((zerov_trans_glbl_quats.numpy() - trans_gt_glbl_quats.numpy()) ** 2.0, axis=(2, 3))))
        res[('interp_quat_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_inter_glbl_quats - trans_gt_glbl_quats.numpy()) ** 2.0, axis=(2, 3))))  # 插值的全局旋转四元数 - 真实的全局旋转四元数 二范数 最后求均值

        # Global positions loss L2P
        res[('zerov_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_zerov_glbl_poses.numpy() - trans_gt_glbl_poses.numpy())**2.0, axis=(2, 3))))
        res[('interp_pos_loss', n_trans)] = np.mean(np.sqrt(np.sum((trans_inter_glbl_poses.numpy() - trans_gt_glbl_poses.numpy())**2.0, axis=(2, 3))))  # 插值的全局位置 - 真实的全局位置 二范数

        # NPSS loss on global quaternions
        res[('zerov_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_glbl_quats), flatjoints(zerov_trans_glbl_quats))      # 保持tensor的前两个维度不动，序列化剩余的维度
        res[('interp_npss_loss', n_trans)] = fast_npss(flatjoints(trans_gt_glbl_quats), flatjoints(trans_inter_glbl_quats))

    print()
    avg_zerov_quat_losses  = [res[('zerov_quat_loss', n)] for n in trans_lengths]
    avg_interp_quat_losses = [res[('interp_quat_loss', n)] for n in trans_lengths]
    print("=== Global quat losses ===")
    print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d}".format("Lengths", 5, 15, 30, 45))
    print("{0: <16} | {1:6.2f} | {2:6.2f} | {3:6.2f} | {4:6.2f}".format("Zero-V", *avg_zerov_quat_losses))
    print("{0: <16} | {1:6.2f} | {2:6.2f} | {3:6.2f} | {4:6.2f}".format("Interp.", *avg_interp_quat_losses))
    print()

    avg_zerov_pos_losses = [res[('zerov_pos_loss', n)] for n in trans_lengths]
    avg_interp_pos_losses = [res[('interp_pos_loss', n)] for n in trans_lengths]
    print("=== Global pos losses ===")
    print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d}".format("Lengths", 5, 15, 30, 45))
    print("{0: <16} | {1:6.3f} | {2:6.3f} | {3:6.3f} | {4:6.3f}".format("Zero-V", *avg_zerov_pos_losses))
    print("{0: <16} | {1:6.3f} | {2:6.3f} | {3:6.3f} | {4:6.3f}".format("Interp.", *avg_interp_pos_losses))
    print()

    avg_zerov_npss_losses = [res[('zerov_npss_loss', n)] for n in trans_lengths]
    avg_interp_npss_losses = [res[('interp_npss_loss', n)] for n in trans_lengths]
    print("=== NPSS on global quats ===")
    print("{0: <16} | {1:5d}  | {2:5d}  | {3:5d}  | {4:5d}".format("Lengths", 5, 15, 30, 45))
    print("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}".format("Zero-V", *avg_zerov_npss_losses))
    print("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}".format("Interp.", *avg_interp_npss_losses))
    print()

    # Write to file is desired 输出到文件中
    if out_path is not None:
        res_txt_file = open(os.path.join(out_path, 'h36m_transitions_benchmark.txt'), "a")
        res_txt_file.write("\n=== Global quat losses ===\n")
        res_txt_file.write("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d}\n".format("Lengths", 5, 15, 30, 45))
        res_txt_file.write("{0: <16} | {1:6.2f} | {2:6.2f} | {3:6.2f} | {4:6.2f}\n".format("Zero-V", *avg_zerov_quat_losses))
        res_txt_file.write("{0: <16} | {1:6.2f} | {2:6.2f} | {3:6.2f} | {4:6.2f}\n".format("Interp.", *avg_interp_quat_losses))
        res_txt_file.write("\n\n")
        res_txt_file.write("=== Global pos losses ===\n")
        res_txt_file.write("{0: <16} | {1:5d}  | {2:5d}  | {3:5d}  | {4:5d}\n".format("Lengths", 5, 15, 30, 45))
        res_txt_file.write("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}\n".format("Zero-V", *avg_zerov_pos_losses))
        res_txt_file.write("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}\n".format("Interp.", *avg_interp_pos_losses))
        res_txt_file.write("\n\n")
        res_txt_file.write("=== NPSS on global quats ===\n")
        res_txt_file.write("{0: <16} | {1:5d}  | {2:5d}  | {3:5d}  | {4:5d}\n".format("Lengths", 5, 15, 30, 45))
        res_txt_file.write("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}\n".format("Zero-V", *avg_zerov_npss_losses))
        res_txt_file.write("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f} | {4:5.4f}\n".format("Interp.", *avg_interp_npss_losses))
        res_txt_file.write("\n\n\n\n")
        res_txt_file.close()

    return res



if __name__ == '__main__':
    opt = yaml.load(open('../config/train_config_lafan.yaml', 'r').read(), Loader=yaml.FullLoader)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    stamp = stamp + '-' + opt['train']['method']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt['train']['debug']:
        stamp = 'debug'

    lafan_data_train = LaFan1(opt['data']['data_dir'],
                              opt['data']['data_set'],
                              seq_len = 65,
                              offset = opt['data']['offset'],
                              train = False,
                              debug= False)
    # print("train_positions.shape", lafan_data_train.data['X'].shape)
    # print("train_rotations.shape", lafan_data_train.data['Q'].shape)
    lafan_loader_train = DataLoader(lafan_data_train,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=True,
                                    num_workers=opt['data']['num_workers'])

    x_mean = lafan_data_train.x_mean
    x_std = lafan_data_train.x_std
    x_mean_n = lafan_data_train.x_mean.view(1, 1, opt['model']['num_joints'], 3)
    x_std_n = lafan_data_train.x_std.view(1, 1, opt['model']['num_joints'], 3)
    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    skeleton_mocap.to(device)

    if opt['data']['data_set'] == "lafan":
        skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])

    offsets = skeleton_mocap.offsets().detach().cpu().numpy()
    parents = skeleton_mocap.parents()
    # for batch_i, batch_data in tqdm(enumerate(lafan_loader_train)):
    #         positions = batch_data['X']  # B, F, J, 3
    #         rotations = batch_data['local_q']
    benchmark_interpolation(X=lafan_data_train.data['X'], Q= lafan_data_train.data['Q'], x_mean=x_mean, x_std = x_std,
                                        offsets=offsets,parents=opt['data']['parents'],
                                        n_future=opt['model']['n_future'], n_past=opt['model']['n_past'])


