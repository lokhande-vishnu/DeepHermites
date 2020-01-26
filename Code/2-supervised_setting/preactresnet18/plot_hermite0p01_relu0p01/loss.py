import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import csv
import numpy as np

image_folder = '.'
curr_dir = '../data/'
hermite_dir = 'hermite_lr0p01'
relu_dir = 'relu_lr0p01'

label1 = 'h4'
label2 = 'relu'

# loss, train_acc, test_acc
plot_type = 'loss'

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the

def read_csv(filename, idx):
    data_sim = [0]
    with open(filename) as f:
        reader = csv.DictReader(f)
        next(reader) # skip header
        for r in reader:
            data_sim.append(float(r[idx]))
    data = np.asarray(data_sim)
    return data

def plot_fid_shade(axs, plot_type, label, idx):
    if label == 'h4':
        cl = 'b'
        label_name = '4 Hermites'
        dir_name = os.path.join(hermite_dir, 'hermite4_exp{}/stat/preactresnet_18_hermite_exp{}.csv')
    elif label == 'relu':
        cl = 'r'
        label_name = 'relu'
        dir_name = os.path.join(relu_dir, 'relu_lr0p01_exp{}/stat/preactresnet_18_relu.csv')

    matrix = []
    for i in range(4):
        filename = os.path.join(curr_dir, dir_name.format(i+1, i+1))
        data = read_csv(filename, idx)
        matrix.append(data[1:])
    matrix = np.asarray(matrix)
    x = [i for i in range(matrix.shape[1])]

    means = np.mean(matrix, axis=0)
    errs = np.std(matrix, axis=0)
    axs.plot(x, means, c=cl, linewidth=1, label=label_name)
    axs.fill_between(x, means - errs, means + errs, alpha=0.35,
                     edgecolor='#1B2ACC', facecolor=cl, linewidth=0.1, antialiased=True, interpolate=True)

    return x

if __name__ == '__main__':
    
    if plot_type == 'loss':
        y_label = 'Loss Value'
        idx = 'train_loss'
    elif plot_type == 'train_acc':
        y_label = 'Train Accuracy'
        idx = 'train_acc'
    else:
        y_label = 'Test Accuracy'
        idx = 'test_acc'

    fig, axs = plt.subplots(1, 1, figsize=(8, 7), sharey=False)
    # axs.grid(which='both')
    x = plot_fid_shade(axs, plot_type, label1, idx)
    x = plot_fid_shade(axs, plot_type, label2, idx)
    axs.set_xlabel('Epochs')
    axs.set_ylabel(y_label)
    axs.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('{}/preactresnet18_hermitelr0p01_relulr0p01_{}.png'.format(image_folder, plot_type), bbox_inches='tight')
