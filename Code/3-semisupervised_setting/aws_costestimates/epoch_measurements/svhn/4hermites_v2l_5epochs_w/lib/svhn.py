from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import copy
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
sys.path.append("lib/torchsample/")
sys.path.append("lib/torchsample/torchsample/")
sys.path.append("lib/torchsample/torchsample/transforms/")
from affine_transforms import RandomRotate, RandomChoiceShear  # noqa: E402
from affine_transforms import RandomChoiceZoom  # noqa: E402


def all_transforms(x):
    tens = transforms.ToTensor()
    norm = transforms.Normalize((0., 0., 0.), (1., 1., 1.))
    obj0 = RandomChoiceZoom([0.8, 0.9, 1.1, 1.2])
    obj1 = RandomRotate(10)
    obj2 = RandomChoiceShear([-10, -5, 0, 5, 10])
    obj3 = transforms.Lambda(augmented_crop_svhn)

    case = np.random.randint(3, size=1)[0]
    x = obj3(x)
    if case == 0:
        return obj0(norm(tens(x)))
    elif case == 1:
        return obj1(norm(tens(x)))
    else:
        return obj2(norm(tens(x)))


def augment_affine_svhn():

    transform_train = transforms.Compose([
        transforms.Lambda(all_transforms),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])
    return transform_train, transform_test


class crop_middle:
    def __call__(self, old_image):
        crop_size = np.random.randint(5, 10, size=2)
        crop_x = np.random.randint(0, 32 - crop_size[0])
        crop_y = np.random.randint(0, 32 - crop_size[1])
        pixels = old_image.load()
        for i in range(crop_x, crop_x + crop_size[0]):
            for j in range(crop_y, crop_y + crop_size[1]):
                pixels[i, j] = (0, 0, 0)
        return old_image


def augmented_crop_svhn(x):
    obj = transforms.RandomCrop(32, padding=np.random.randint(5, size=1)[0])
    obj2 = crop_middle()
    return obj2(obj(x))


def augment_mean_svhn():
    transform_train = transforms.Compose([
        transforms.Lambda(augmented_crop_svhn),
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])
    return transform_train, transform_test


def noaug_SVHN():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0., 0., 0.), (1., 1., 1.)),
    ])
    return transform_train, transform_test


def get_loaders(nb_labelled,
                batch_size,
                unlab_rat,
                augment_type,
                lab_inds=[],
                is_balanced=True):

    if augment_type == "affine":
        transform_train, transform_test = augment_affine_svhn()
    elif augment_type == "mean":
        transform_train, transform_test = augment_mean_svhn()
    elif augment_type == "no":
        transform_train, transform_test = noaug_SVHN()

    trainset_l = SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    test_set = SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    print(trainset_l.data.shape, len(trainset_l.labels))
    if len(lab_inds) == 0:
        if is_balanced:
            lab_inds = []
            for i in range(10):
                labels = np.array(trainset_l.labels)
                inds_i = np.where(labels == i)[0]
                inds_i = np.random.permutation(inds_i)
                lab_inds.extend(inds_i[0:int(nb_labelled / 10)].tolist())
            lab_inds = np.array(lab_inds)
        else:
            lab_inds = np.arange(0, nb_labelled)

    all_inds = np.arange(len(trainset_l.labels))
    unlab_inds = np.setdiff1d(all_inds, lab_inds)

    trainset_u = copy.deepcopy(trainset_l)
    unlab_inds = unlab_inds[0:int(unlab_rat * len(unlab_inds))]
    trainset_u.data = np.array(trainset_u.data)[unlab_inds]
    trainset_u.labels = np.array(trainset_u.labels)[unlab_inds]
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False, num_workers=1)
    print(trainset_u.data.shape, len(trainset_u.labels))

    trainset_l.data = np.array(trainset_l.data)[lab_inds]
    trainset_l.labels = np.array(trainset_l.labels)[lab_inds]

    print(trainset_l.data.shape, len(trainset_l.labels))
    trainloader_l = DataLoader(
        trainset_l, batch_size=batch_size, shuffle=True, num_workers=1)

    testloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    loaders = {
        "trainloader_l": trainloader_l,
        "testloader": testloader,
        "trainloader_u": trainloader_u,
        "trainset_l": trainset_l,
        "test_set": test_set,
        "trainset_u": trainset_u,
        "lab_inds": lab_inds
    }
    return loaders


def shuffle_loader(trainset_u_org, batch_size=100):
    nb_unlabelled = len(trainset_u_org.labels)
    inds_all = np.arange(nb_unlabelled)
    inds_shuff = np.random.permutation(inds_all)
    trainset_u = copy.deepcopy(trainset_u_org)
    trainset_u.data = trainset_u_org.data[inds_shuff]
    trainset_u.labels = np.array(trainset_u_org.labels)[inds_shuff]
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False, num_workers=1)
    return trainloader_u, torch.from_numpy(inds_shuff).cuda()
