import numpy as np
import matplotlib.pyplot as plt

working_loc = 'Lingjiang'

data_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE/ISO17'}

saving_dir_dict = {'Lingjiang': 'E:/PhD_research/AI4Science/Working_dir/NeuralODE/ISO17/demo'}

if __name__ == '__main__':
    time, MSE_trace = np.loadtxt(f'{data_dir_dict[working_loc]}/Nstep-3999/ODE-VAE_ISO17_Epoch-1499_MSE.txt')
    l_data = len(MSE_trace)
    time_int = np.arange(0, l_data, 1)

    plt.plot(time_int, MSE_trace, label='MSE')
    plt.xlim(0, l_data)

    # for fmt in ['png', 'eps', 'pdf']:
        # plt.savefig(f'{saving_dir_dict[working_loc]}/MSE_trace.{fmt}', dpi=300)

    plt.show(block=True)
    plt.close()

    mean_loss, median_loss = np.load(f'{data_dir_dict[working_loc]}/Nstep-3999/ODE-VAE_ISO17_loss.npy')
    num_epochs = len(mean_loss)

    epoch_idx = np.arange(0, num_epochs, 1)

    plt.plot(epoch_idx, mean_loss, label='Loss')
    # plt.plot(epoch_idx, median_loss, label='Median Loss')
    plt.xlim(0, 500)

    for fmt in ['png', 'eps', 'pdf']:
        plt.savefig(f'{saving_dir_dict[working_loc]}/Loss.{fmt}', dpi=300)

    plt.show(block=True)

