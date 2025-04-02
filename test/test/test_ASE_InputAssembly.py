import os
import numpy as np
from ase import Atoms
from ase.io import read, write

working_loc = 'Lingjiang'

dataset_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE/ISO17'}

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE/ISO17/structures'}

if __name__ == '__main__':
    datafile_MDTraj = f'{dataset_dir_dict[working_loc]}/datasets/MDTraj_nstep-3999.npy'
    datafile_ModelOutput = f'{dataset_dir_dict[working_loc]}/Nstep-3999/ODE-VAE_ISO17_Epoch-899.npy'

    MDTraj = np.load(datafile_MDTraj)  # 读取数据
    model_output = np.load(datafile_ModelOutput)  # 读取数据

    permutation = np.load(f'{dataset_dir_dict[working_loc]}/Nstep-3999/Permutation.npy')  # 读取采样的时间序列
    # permutation = s
    anchor_l = np.random.choice(permutation, size=1)[0]  # 随机选择一个采样点作为左锚点
    idx_anchor_l = list(permutation).index(anchor_l)  # 找到左锚点在原始数据中的索引
    anchor_r = permutation[idx_anchor_l+1]  # 找到右锚点

    # frame_interpolated = np.random.randint(anchor_l, anchor_r)
    frame_interpolated = 2000
    frame_extrapolated = 3500

    print(f'frame_interpolated: {frame_interpolated}')

    config_name = 'OC3OC4H10'
    mol_inter_MD = Atoms(config_name,positions=MDTraj[frame_interpolated])
    mol_extrap_MD = Atoms(config_name,positions=MDTraj[frame_extrapolated])
    mol_inter_model = Atoms(config_name,positions=model_output[frame_interpolated])
    mol_extrap_model = Atoms(config_name,positions=model_output[frame_extrapolated])

    mol_list = [mol_inter_MD, mol_extrap_MD, mol_inter_model, mol_extrap_model]
    tag_list = ['MD_inter', 'MD_extrap', 'model_inter', 'model_extrap']

    for i in range(4):
        molecule = mol_list[i]
        tag = tag_list[i]

        for fmt in ['png', 'cif', 'xyz', 'gaussian-in']:
            os.makedirs(f'{saving_dir_dict[working_loc]}/{fmt}', exist_ok=True)  # 创建子目录用于存放数据

            if fmt == 'gaussian-in':
                write(f'{saving_dir_dict[working_loc]}/{fmt}/{config_name}_{tag}.gjf', molecule, format=fmt, method='b3lyp', basis='def2TZVP')
            else:
                write(f'{saving_dir_dict[working_loc]}/{fmt}/{config_name}_{tag}.{fmt}', molecule, format=fmt)
