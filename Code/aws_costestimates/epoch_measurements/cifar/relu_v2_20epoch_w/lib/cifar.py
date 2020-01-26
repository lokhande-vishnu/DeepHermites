from PIL import Image
from torch.utils.data import DataLoader
import copy
import numpy as np
import sys
import torch
import torchvision
import torchvision.transforms as transforms
sys.path.append("lib/torchsample/")
sys.path.append("lib/torchsample/torchsample/")
sys.path.append("lib/torchsample/torchsample/transforms/")
from affine_transforms import RandomRotate, RandomChoiceShear, RandomChoiceZoom


class CIFAR10(torchvision.datasets.CIFAR10):
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def all_transforms(x):
    tens = transforms.ToTensor()
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
    obj0 = RandomChoiceShear([-10, -5, 0, 5, 10])
    obj1 = RandomChoiceZoom([0.8, 0.9, 1.1, 1.2])
    obj2 = RandomRotate(10)
    obj3 = transforms.Lambda(augmented_crop_cifar)
    obj4 = transforms.RandomHorizontalFlip()
    obj5 = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    x = obj5(obj4(obj3(x)))
    case = np.random.randint(3, size=1)[0]
    if case == 0:
        return obj0(norm(tens(x)))
    elif case == 1:
        return obj1(norm(tens(x)))
    else:
        return obj2(norm(tens(x)))


def augment_affine_cifar10():
    transform_train = transforms.Compose([
        transforms.Lambda(all_transforms),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def augmented_crop_cifar(x):
    case = np.random.randint(2, size=1)[0]
    if case == 0:
        obj = RandomTranslateWithReflect(4)
    else:
        obj = transforms.RandomCrop(
            32, padding=np.random.randint(5, size=1)[0])
    return obj(x)


def augment_mean_cifar10():
    transform_train = transforms.Compose([
        transforms.Lambda(augmented_crop_cifar),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def noaug_cifar10():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def get_loaders(nb_labelled, batch_size, unlab_rat, augment_type, lab_inds=[]):

    if augment_type == "affine":
        transform_train, transform_test = augment_affine_cifar10()
    elif augment_type == "mean":
        transform_train, transform_test = augment_mean_cifar10()
    elif augment_type == "no":
        transform_train, transform_test = noaug_cifar10()

    trainset_l = CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    nb_class = 10

    print(trainset_l.train_data.shape, len(trainset_l.train_labels))
    if len(lab_inds) == 0:
        lab_inds = []
        for i in range(nb_class):
            labels = np.array(trainset_l.train_labels)
            inds_i = np.where(labels == i)[0]
            inds_i = np.random.permutation(inds_i)
            lab_inds.extend(inds_i[0:int(nb_labelled / nb_class)].tolist())
        lab_inds = np.array(lab_inds)

    all_inds = np.arange(len(trainset_l.train_labels))
    unlab_inds = np.setdiff1d(all_inds, lab_inds)

    trainset_u = copy.deepcopy(trainset_l)

    trainset_u.train_data = np.array(trainset_u.train_data)[unlab_inds]
    trainset_u.train_labels = np.array(trainset_u.train_labels)[unlab_inds]
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False, num_workers=1)
    print(trainset_u.train_data.shape, len(trainset_u.train_labels))

    trainset_l.train_data = np.array(trainset_l.train_data)[lab_inds]
    trainset_l.train_labels = np.array(trainset_l.train_labels)[lab_inds]
    print(trainset_l.train_data.shape, len(trainset_l.train_labels))
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


class RandomTranslateWithReflect:
    """Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(
            -self.max_translation, self.max_translation + 1, size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image,
                        (xpad, ypad))  # 2-tuple giving the upper left corner

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop(
            (xpad - xtranslation, ypad - ytranslation,
             xpad + xsize - xtranslation, ypad + ysize - ytranslation))
        return new_image


def shuffle_loader(trainset_u_org, batch_size=100):
    nb_unlabelled = len(trainset_u_org.train_labels)
    inds_all = np.arange(nb_unlabelled)
    inds_shuff = np.random.permutation(inds_all)
    trainset_u = copy.deepcopy(trainset_u_org)
    trainset_u.train_data = trainset_u_org.train_data[inds_shuff]
    trainset_u.train_labels = np.array(trainset_u_org.train_labels)[inds_shuff]
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False, num_workers=1)
    return trainloader_u, torch.from_numpy(inds_shuff).cuda()
