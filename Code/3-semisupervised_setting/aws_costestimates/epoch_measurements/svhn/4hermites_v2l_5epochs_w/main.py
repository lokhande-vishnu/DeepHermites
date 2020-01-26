from lib.hermite_resnet_v2l import ResNet18
from lib.svhn import get_loaders, shuffle_loader
from lib.utils import unsup_nll, scalar2onehot, calc_entropy
from torch.autograd import Variable
import copy
import csv
import cvxpy
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

# will turn on the cudnn autotuner that selects efficient algorithms.
cudnn.benchmark = True

if os.name == "nt":
    nll_loss = nn.NLLLoss(reduction="elementwise_mean")
else:
    nll_loss = nn.NLLLoss(reduction="mean")
logsoft = nn.LogSoftmax()
if torch.__version__[2] != "1":
    logsoft = nn.LogSoftmax(dim=1)

DATASET = "svhn"
NET_NAME = "hermite_resnet_v2l"
PLOT_FOLDER = 'plots/'  # folder to save the plot of y_u accuracies in time
Y_U_FOLDER = 'estimated_labels/'  # folder to save the y_u estimates
Y_FILE = ''
AUGMENT_TYPE = "affine"

# hyperparameters:
LAMBDA_ENT = 1.0  # weight of entropy in the overall loss
ETA_W = 0.01  # learning rate for weights in the inner loop
ETA_Y = 1.0  # learning rate for estimated labels in the outer loop
EPSILON = 0.05  # label threshold
BATCH_SIZE = 100  # batch size for both labeled and unlabeled data
""" Percentage of the unlabeled data to be used. If 1.0, entire training data
    is used. """
UNLAB_RAT = 1.0
# determines the deviation of the noise added to weights
LANGEVIN_COEF = 10**-5
MOMENTUM = 0.9
WEIGHT_DECAY = 0

# setup directories and set dataset/net_name dependent parameters:
NB_LABELLED = 1000
NB_OUTER_ITER = 75
NB_INNER_ITER = 5
nb_outer_start = 0
net_w_orig = ResNet18().cuda()
file_name = '%s_%s' % (DATASET, NET_NAME)
unsup_nll_loss = unsup_nll(BATCH_SIZE)

for path in [PLOT_FOLDER, Y_U_FOLDER]:
    if not os.path.exists(path):
        os.makedirs(path)

# load dataloaders:
loaders = get_loaders(NB_LABELLED, BATCH_SIZE, UNLAB_RAT, AUGMENT_TYPE)
lab_inds = loaders["lab_inds"]
test_set = loaders["test_set"]
testloader = loaders["testloader"]
trainloader_l = loaders["trainloader_l"]
trainloader_u = loaders["trainloader_u"]
trainset_l = loaders["trainset_l"]
trainset_u_org = loaders["trainset_u"]
train_data_u_np = trainset_u_org.data
train_labels_u_org = trainset_u_org.labels

# prepare unlabeled data for pytorch setting:
train_labels_u_org = np.array(train_labels_u_org)
train_data_u = torch.from_numpy(np.float32(train_data_u_np))

# load or randomly initalize y_u estimates:
if Y_FILE == '':  # start y_u estimates randomly
    estimated_y = np.random.randint(0, 10, len(train_labels_u_org))
    estimated_y = scalar2onehot(estimated_y)
else:  # load y_u estimates from given .npy file
    npy_obj = np.load(Y_FILE)
    npy_obj = np.atleast_1d(npy_obj)[0]
    estimated_y = npy_obj["estimated_y"]
    nb_outer_start = npy_obj["epoch"] + 1

    # Reload dataset with stored labels
    lab_inds = npy_obj["lab_inds"]
    loaders = get_loaders(NB_LABELLED, BATCH_SIZE, UNLAB_RAT, AUGMENT_TYPE,
                          lab_inds)
    test_set = loaders["test_set"]
    testloader = loaders["testloader"]
    trainloader_l = loaders["trainloader_l"]
    trainloader_u = loaders["trainloader_u"]
    trainset_l = loaders["trainset_l"]
    trainset_u_org = loaders["trainset_u"]
    train_data_u_np = trainset_u_org.data
    train_labels_u_org = trainset_u_org.labels

    # prepare unlabeled data for pytorch setting:
    train_labels_u_org = np.array(train_labels_u_org)
    train_data_u = torch.from_numpy(np.float32(train_data_u_np))

estimated_y = Variable(
    torch.FloatTensor(estimated_y).cuda(), requires_grad=True)

st_time = time.time()
for epoch in range(nb_outer_start, NB_OUTER_ITER):
    """ permute entire trainloader for unlabeled data and shuffling indexes to
        be used with estimated_y """
    trainloader_u, ind_shuff_all = shuffle_loader(trainset_u_org, BATCH_SIZE)

    # start with random weights at the beggining of each outer epoch:
    net_w = copy.deepcopy(net_w_orig)
    net_w.train()
    var_list = [{'params': net_w.parameters()}]
    optimizer_w = optim.SGD(
        var_list, ETA_W, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # argmax the current posterior to plot y_u accurucies:
    est_y_arg = np.array(np.argmax(estimated_y.data.cpu().numpy(), axis=1))
    y_acc = ((est_y_arg == np.array(train_labels_u_org)).sum()
             ) * 1.0 / len(train_labels_u_org)
    print(file_name)
    print("epoch=%d,time=%f,y_acc=%f" % (epoch, time.time() - st_time, y_acc))

    with open("log.csv", mode='a', newline='') as logfile:
        writer = csv.writer(
            logfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([epoch, time.time() - st_time, y_acc])

    st_time = time.time()
    print(len(trainloader_u))
    for epoch_w in range(NB_INNER_ITER):
        loss_l_cum, loss_u_cum, loss_ent_cum = 0.0, 0.0, 0.0
        timeset = []
        start = time.time()
        for batch_idx, (inps_u, targs_u) in enumerate(trainloader_u):
            # load labeled batches:
            inps_l, targs_l = iter(trainloader_l).next()
            inps_l, targs_l = Variable(inps_l.cuda()), Variable(targs_l.cuda())
            outs_l = logsoft(net_w(inps_l))
            ind_shuff = copy.deepcopy(
                ind_shuff_all[batch_idx * BATCH_SIZE:(batch_idx + 1) *
                              BATCH_SIZE])
            loss_l = nll_loss(outs_l, targs_l.long())

            # extract the current label estimates for the batch:
            ind_shuff = torch.tensor(ind_shuff, dtype=torch.long)
            est_y = estimated_y[ind_shuff.long()]

            # calculate the losses for unlabeled data:
            inps_u = Variable(inps_u.cuda())
            outs_u = net_w(inps_u)
            loss_ent = LAMBDA_ENT * calc_entropy(outs_u)
            outs_u = logsoft(outs_u)
            loss_u = unsup_nll_loss(est_y, 0, outs_u)

            # sum up loss for logging
            loss_l_cum += loss_l.item()
            loss_u_cum += loss_u.item()
            loss_ent_cum += loss_ent.item()

            # sum all the losses:
            loss = loss_l + loss_u + loss_ent

            # Add Gaussian noise to weights:
            for param in net_w.parameters():
                noise = torch.cuda.FloatTensor(param.size()).normal_()
                param.data += ((ETA_W * LANGEVIN_COEF)**0.5) * noise

            # update weights:
            loss.backward()
            optimizer_w.step()
            net_w.zero_grad()
            timeset.append(time.time() - start)
            start = time.time()
        print(np.mean(timeset))
        raise RuntimeError

        with open("losslog.csv", mode='a', newline='') as logfile:
            writer = csv.writer(
                logfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [epoch, epoch_w, loss_l_cum, loss_u_cum, loss_ent_cum])

    # update unlabel estimates with cumulative loss:
    target_var_list = [{'params': estimated_y, 'lr': ETA_Y}]
    optimizer_y = optim.SGD(target_var_list, momentum=0, weight_decay=0)
    optimizer_y.step()
    optimizer_y.zero_grad()
    estimated_y.grad.data.zero_()

    # Project onto probability polytope
    est_y_num = estimated_y.data.cpu().numpy()
    U = cvxpy.Variable((est_y_num.shape[0], est_y_num.shape[1]))
    objective = cvxpy.Minimize(cvxpy.sum(cvxpy.square(U - est_y_num)))
    constraints = [U >= EPSILON, cvxpy.sum(U, axis=1) == 1]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()
    estimated_y.data = torch.from_numpy(np.float32(U.value)).cuda()

    # save estimated labels and labeled indexes on unlabeled data:
    state = {
        "lab_inds": lab_inds,
        "estimated_y": estimated_y.data.cpu().numpy(),
        "epoch": epoch,
        "y_acc": y_acc,
    }
    np.save(Y_U_FOLDER + file_name + '.npy', state)