import numpy as np
import tensorflow as tf
import pickle
import gzip
import os
import struct

from utils import DataSpec, load_data, DATA_PATH


try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

DOWNLOAD = False


def load_F_MNIST(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
               data_test_monitor: DataSpec, data_run: DataSpec, adversarial_data_suffix=None):
    fmnist_dataset_folder_path = DATA_PATH + "Fashion_MNIST"

    # URLs for the train image and label data
    url_train_image = 'train-images-idx3-ubyte.gz'
    url_train_labels = 'train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    # print("Downloading train data")
    # train = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = 't10k-images-idx3-ubyte.gz'
    url_test_labels = 't10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    # print("Downloading test data")
    # test = try_download(url_test_image, url_test_labels, num_test_samples)

    """
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    """

    x_train = loadData(url_train_image, num_train_samples)
    y_train = loadLabels(url_train_labels, num_train_samples)

    x_test = loadData(url_test_image, num_test_samples)
    y_test = loadLabels(url_test_labels, num_test_samples)

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    if adversarial_data_suffix is not None:
        with open(fmnist_dataset_folder_path + "/adversarial{}".format(adversarial_data_suffix), mode='rb') as file:
            adversarial = pickle.load(file, encoding='latin1')
        x_run = np.array(adversarial['data'])
        y_run = np.array(adversarial['labels'])
    else:
        x_run = x_test
        y_run = y_test

    data_train_model.set_data(inputs=x_train, labels=y_train, assertion=False)
    data_train_monitor.set_data(inputs=x_train, labels=y_train, assertion=False)
    data_test_model.set_data(inputs=x_test, labels=y_test, assertion=False)
    data_test_monitor.set_data(inputs=x_test, labels=y_test, assertion=False)
    data_run.set_data(inputs=x_run, labels=y_run, assertion=False)
    pixel_depth = 255.0
    class_label_map, all_labels = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth,
        is_adversarial_data=(adversarial_data_suffix is not None))

    return class_label_map, all_labels


# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by
#   the CNTK text reader


def loadData(src, cimg):
    gzfname = DATA_PATH + "Fashion_MNIST/{}".format(src)

    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype=np.uint8)
    finally:
        if DOWNLOAD:
            os.remove(gzfname)
    return res.reshape((cimg, crow, ccol))


def loadLabels(src, cimg):
    gzfname = DATA_PATH + "Fashion_MNIST/{}".format(src)
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype=np.uint8)
    finally:
        if DOWNLOAD:
            os.remove(gzfname)
    return res.reshape((cimg,))


def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

