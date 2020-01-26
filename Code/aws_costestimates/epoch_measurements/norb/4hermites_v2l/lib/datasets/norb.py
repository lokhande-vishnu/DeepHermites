from PIL import Image
from requests import get  # to make GET request
from torch.utils.data import DataLoader
import codecs
import copy
import errno
import gzip
import hashlib
import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.io


class NORB(data.Dataset):
    BASE_URL = "https://cs.nyu.edu/~ylclab/data/norb-v1.0/"
    TRAINING_FILE = 'training.pt'
    TEST_FILE = 'test.pt'

    def __init__(self, root="./data/", transform=None):
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, "raw/")
        self.processed_folder = os.path.join(self.root, "processed/")

        dirs = [
            self.root,
            self.raw_folder,
            self.processed_folder,
        ]
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)

        if not self._check_exists():
            self.download()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        pass

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(
            os.path.join(self.processed_folder, self.TRAINING_FILE))
                and os.path.exists(
                    os.path.join(self.processed_folder, self.TEST_FILE)))

    def download(self):
        # Get training images
        for i in range(1, 3):
            num = "{0:0=2d}".format(i)
            url = (self.BASE_URL +
                   "norb-5x46789x9x18x6x2x108x108-training-{}-dat.mat.gz".
                   format(num))
            filename = os.path.basename(url)
            fpath = os.path.join(self.raw_folder, filename)
            if not os.path.exists(fpath.replace(".gz", "")):
                download_url(url, fpath)
                self.extract_gzip(gzip_path=fpath, remove_finished=True)

        # Get training labels
        for i in range(1, 3):
            num = "{0:0=2d}".format(i)
            url = (self.BASE_URL +
                   "norb-5x46789x9x18x6x2x108x108-training-{}-cat.mat.gz".
                   format(num))
            filename = os.path.basename(url)
            fpath = os.path.join(self.raw_folder, filename)
            if not os.path.exists(fpath.replace(".gz", "")):
                download_url(url, fpath)
                self.extract_gzip(gzip_path=fpath, remove_finished=True)

        # Get testing images
        for i in range(1, 3):
            num = "{0:0=2d}".format(i)
            url = (self.BASE_URL +
                   "norb-5x46789x9x18x6x2x108x108-testing-{}-dat.mat.gz".
                   format(num))
            filename = os.path.basename(url)
            fpath = os.path.join(self.raw_folder, filename)
            if not os.path.exists(fpath.replace(".gz", "")):
                download_url(url, fpath)
                self.extract_gzip(gzip_path=fpath, remove_finished=True)

        # Get testing labels
        for i in range(1, 3):
            num = "{0:0=2d}".format(i)
            url = (self.BASE_URL +
                   "norb-5x46789x9x18x6x2x108x108-testing-{}-cat.mat.gz".
                   format(num))
            filename = os.path.basename(url)
            fpath = os.path.join(self.raw_folder, filename)
            if not os.path.exists(fpath.replace(".gz", "")):
                download_url(url, fpath)
                self.extract_gzip(gzip_path=fpath, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_files(
                os.path.join(
                    self.raw_folder,
                    "norb-5x46789x9x18x6x2x108x108-training-{}-dat.mat")),
            read_label_files(
                os.path.join(
                    self.raw_folder,
                    "norb-5x46789x9x18x6x2x108x108-training-{}-cat.mat")))
        test_set = (
            read_image_files(
                os.path.join(
                    self.raw_folder,
                    "norb-5x46789x9x18x6x2x108x108-testing-{}-dat.mat")),
            read_label_files(
                os.path.join(
                    self.raw_folder,
                    "norb-5x46789x9x18x6x2x108x108-testing-{}-cat.mat")))
        with open(
                os.path.join(self.processed_folder, self.training_file),
                'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file),
                  'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)


def read_label_files(path_template):
    pass


def read_image_files(path_template):
    # Get training images
    for i in range(1, 3):
        fpath = path_template.format(i)
        mat = scipy.io.loadmat(fpath)
        print(mat)


def download_url(url, file_name):
    print("Downloading " + url + " to " + file_name)
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


