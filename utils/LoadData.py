from pickle import load
from random import sample
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

from utils import *


def load_data(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
              data_test_monitor: DataSpec, data_run: DataSpec, pixel_depth=None, is_adversarial_data=False):
    data_specs = [data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run]
    data_specs_normalize = [data_train_model, data_test_model, data_train_monitor, data_test_monitor]\
        if is_adversarial_data else data_specs

    # load data from files and pickle it
    for ds in data_specs:
        if not ds.has_data():
            if ds.file is None:
                raise(ValueError("Got a DataSpec with neither data nor a file name specified!"))

            with open(ds.file, mode='rb') as f:
                data = load(f)
                ds.set_data(data=data, assertion=False)
            assert ds.has_data(), "Was not able to find data!"

    # filter data such that only the specified classes occur
    for ds in data_specs:
        if ds.classes is not None:
            x_new = []
            y_new = []
            for x, y in zip(ds.inputs(), ds.ground_truths()):
                if y in ds.classes:
                    x_new.append(x)
                    y_new.append(y)
            ds.set_data(inputs=np.array(x_new), labels=np.array(y_new), assertion=True)

    # account for the correct number of data points
    for ds in data_specs:
        length = len(ds.inputs())
        if ds.n_max is not None and ds.n_max < length:
            if ds.randomize:
                # sample uniformly from the whole data
                indices = sample(range(length), ds.n_max)
                x = ds.inputs()[indices]
                y = ds.ground_truths()[indices]
            else:
                # choose the first n data points
                x = ds.inputs()[:ds.n_max]
                y = ds.ground_truths()[:ds.n_max]
            ds.set_data(inputs=x, labels=y, assertion=True)

    # normalize data by pixel depth
    if pixel_depth is not None:
        for ds in data_specs_normalize:
            x = (ds.inputs().astype(np.float32) - (pixel_depth * 0.5)) / (pixel_depth * 0.5)
            ds.set_inputs(x)

    # normalize labels to "categorical vector"
    data_specs_network = [data_train_model]
    all_labels_network = get_labels(data_specs_network)
    class_label_map = class_label_map_from_labels(all_labels_network)
    data_specs_all = [data_train_model, data_train_monitor, data_run]
    all_labels = get_labels(data_specs_all)

    return class_label_map, all_labels


def get_labels(data_specs):
    all_labels_total = set()

    for ds in data_specs:
        for cd in ds.ground_truths():
            all_labels_total.add(cd)

    return sorted(all_labels_total)
