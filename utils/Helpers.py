from math import sqrt
from time import time
from datetime import time as time_type

import numpy
import numpy as np
import colorsys
import random
from copy import copy
import csv
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.data import Dataset
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn import mixture
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, accuracy_score
from itertools import groupby
from operator import itemgetter

from .CoreStatistics import CoreStatistics
from .Options import *

import seaborn as sns


def to_classes(list_of_bit_vectors):
    return [to_class(b) for b in list_of_bit_vectors]


def to_class(bit_vector):
    return np.where(bit_vector == 1)[0][0]


def to_dataset(x_train, y_train, batch_size):
    return Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)


def ratio(part, total):
    try:
        return part / total * 100
    except ZeroDivisionError:
        return 100.0
    except TypeError:
        return "?"


def extend(strings):
    n = max(len(s) for s in strings)
    return (s.rjust(n) for s in strings)


def get_rgb_colors(n):
    fixed_colors = [
        (1.0, 0.0, 0.0),  # red
        (0.33, 0.42, 0.2),  # green
        (0.0, 0.0, 1.0),  # blue
        (0.8, 0.0, 1.0)  # purple
    ]
    # if n == 2:
    #     # special options for binary case
    #     return [(0.33, 0.42, 0.2), (0.96, 0.52, 0.26)]  # green & orange
    if n <= 4:
        return fixed_colors[0:n]

    # based on https://stackoverflow.com/a/876872
    hsv_tuples = [(x * 1.0 / n, 0.8, 1) for x in range(n)]
    rgb_colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    rgb_colors_shuffled = []
    step = int(sqrt(n))
    i = 0
    while i < step:
        for j in range(step):
            rgb_colors_shuffled.append(rgb_colors[i + j * step])
        i += 1
    rgb_colors_shuffled.extend(rgb_colors[i+(step-1)*step:])
    return rgb_colors_shuffled


def color_blind(n):
    return sns.color_palette(palette='colorblind', n_colors=n)


def get_markers(n_classes):
    all_markers = ["o", "s", "^", "p", "X", "D", "v", "P", "<", ">", "H"]
    if n_classes > len(all_markers):
        markers = copy(all_markers)
        while n_classes > len(markers):
            markers.extend(markers)
    else:
        markers = all_markers
    return markers[:n_classes]


def set_random_seed(n):
    print("Setting random seed to", n)
    random.seed(n)
    np.random.seed(n)
    tf.random.set_seed(n)


def get_image_shape(images):
    return images.shape[1:4]


def number_of_neurons(model, layer):
    return model.layers[layer].output_shape[1]


def categoricals2numbers(categorical_vectors):
    """convert categorical vectors to numbers"""
    return [categorical2number(categorical_vector) for categorical_vector in categorical_vectors]


def categorical2number(categorical_vector):
    """convert categorical vector to number"""
    return np.where(categorical_vector == 1)[0][0]


def number2categoricals(numbers, sorted_set_of_numbers=None):
    """convert numbers to categorical vectors"""
    # map labels to consecutive numbers
    numbers_mapped, sorted_set_of_numbers = labels2classes(numbers, sorted_set_of_numbers)
    # use numpy conversion function
    return to_categorical(numbers_mapped, num_classes=len(sorted_set_of_numbers), dtype='float32')


def labels2classes(labels, sorted_set_of_numbers=None):
    """convert list of labels to class form, e.g., from [1, 4] to [0, 1]"""
    if sorted_set_of_numbers is None:
        sorted_set_of_numbers = sorted(set(labels))
    old2new = dict()
    for i, ni in enumerate(sorted_set_of_numbers):
        old2new[ni] = i
    numbers_mapped = np.array(list(map(old2new.__getitem__, labels)))
    return numbers_mapped, sorted_set_of_numbers


def number_of_model_classes(model):
    # to account for one-class classifiers
    if model.layers[-1].output_shape[1] == 1:
        return 2
    else:
        return model.layers[-1].output_shape[1]


def rate_fraction(num, den):
    if den == 0:
        return 0  # convention: return 0
    return num/den


def obtain_predictions(model, data, class_label_map, layers=None, ignore_misclassifications: bool = False,
                       transfer: bool = False):
    delta_t = time()
    data_filtered = data
    if layers is None:
        predictions = model.predict(data.inputs())
        result = to_classifications(predictions, class_label_map)
    else:
        if ignore_misclassifications:
            # compare classes to ground truths
            classes, _, _ = obtain_predictions(model=model, data=data, class_label_map=class_label_map)
            filter = []
            for i, (p, gt) in enumerate(zip(classes, data.ground_truths())):
                if p == gt:
                    filter.append(i)
            if len(filter) < 100:
                print("WARN: The network classified fewer than 100 samples correctly!")
            data_filtered = data.filter(filter, copy=True)
        layer2values = dict()
        for layer_index in layers:
            try:
                manual_model = model.is_manual_model()
            except:
                manual_model = False
            if manual_model:
                result = model.predict(data_filtered.inputs(), layer_index)
            else:
                # construct pruned model following
                # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
                # NOTE: we have to use .get_output_at(1) instead of .output because for the
                # transferred model the graph has to connect through the right outputs
                if transfer and len(model.layers[layer_index]._inbound_nodes) > 1:
                    layer_output = model.layers[layer_index].get_output_at(1)
                else:
                    layer_output = model.layers[layer_index].output
                model_until_layer = Model(inputs=model.input, outputs=layer_output)
                result = model_until_layer.predict(data_filtered.inputs())
            layer2values[layer_index] = result
        result = layer2values
    timer = time() - delta_t

    return result, data_filtered, timer


def to_classifications(list_of_predictions, class_label_map):
    return [to_classification(p, class_label_map) for p in list_of_predictions]


def to_classification(prediction, class_label_map):
    class_idx = np.argmax(prediction)
    if class_label_map is not None:
        class_idx = class_label_map.get_label(class_idx)
    return class_idx


def normalize_layer(model, raw_layer):
    layer_index = None
    if isinstance(raw_layer, str):
        # search for layer in the model
        for idx, layer in enumerate(model.layers):
            if layer.name == raw_layer:
                layer_index = idx
                break
    elif isinstance(raw_layer, int):
        if raw_layer < 0:
            layer_index = len(model.layers) + raw_layer
            assert layer_index >= 0, "Negative layer indices should be such that their absolute value is smaller " + \
                                     "than the number of layers."
        else:
            layer_index = raw_layer
            assert layer_index < len(model.layers), "Layer index exceeds the number of layers."
    else:
        raise (ValueError("A layer needs to be a string or an integer, but got ", raw_layer))

    if layer_index is None:
        raise (ValueError("Could not find layer", raw_layer))

    return layer_index


def float_printer(timer):
    if isinstance(timer, time_type):
        f = timer.second + timer.microsecond / 1000000
    else:
        assert isinstance(timer, int) or isinstance(timer, float)
        f = timer
    if f < 1e-2:
        if f == 0:
            return "0.00"
        return "< 0.01"
    return "{:.2f}".format(f)


def print_data_information(data_train_monitor, data_test_monitor, data_run):
    print("Loaded the following data:")
    print("- classes {} with {:d} inputs (monitor training),".format(
        classes2string(data_train_monitor.classes), data_train_monitor.n()))
    print("- classes {} with {:d} inputs (monitor test),".format(
        classes2string(data_test_monitor.classes), data_test_monitor.n()))
    print("- classes {} with {:d} inputs (monitor run)".format(
        classes2string(data_run.classes), data_run.n()))


def uniform_bins(n: int, max=1.0):
    step = max / float(n)
    return [i * step for i in range(n + 1)]


def determine_zero_filters(values: dict, data, n_neurons, layer=None):
    class2nonzeros = dict()
    for class_id in data.classes:
        class2nonzeros[class_id] = [0 for _ in range(n_neurons)]
    for vj, gt in zip(values, data.ground_truths()):
        for i, vi in enumerate(vj):
            if vi > 0:
                class2nonzeros[gt][i] += 1
    # create mask of all dimensions with entry 'True' whenever there is at least one non-zero entry
    class2nonzero_mask = dict()
    for class_id, nonzeros in class2nonzeros.items():
        nonzero_mask = []
        n_zeros = 0
        for i, nzi in enumerate(nonzeros):
            if nzi > 0:
                nonzero_mask.append(True)
            else:
                nonzero_mask.append(False)
                n_zeros += 1
        class2nonzero_mask[class_id] = nonzero_mask
        if layer is not None:
            print("filtering zeros removes {:d}/{:d} dimensions from layer {:d} for class {:d}".format(
                n_zeros, n_neurons, layer, class_id))
    return class2nonzero_mask


def classes2string(classes):
    comma = ""
    string = ""
    for k, g in groupby(enumerate(classes), lambda ix: ix[0] - ix[1]):
        consecutive_classes = list(map(itemgetter(1), g))
        if len(consecutive_classes) > 1:
            string += comma + "{:d}-{:d}".format(min(consecutive_classes), max(consecutive_classes))
        else:
            string += comma + "{:d}".format(consecutive_classes[0])
        comma = ","
    #if classes == [k for k in range(len(classes))]:
        # short version for consecutive classes
    #    return "0-{:d}".format(len(classes) - 1)
    #else:
        # long version with enumeration of all classes
    #    comma = ""
    #    string = ""
    #    for c in classes:
    #        string += comma + str(c)
    #        comma = ","
    return string


def store_core_statistics(storages, name, filename_prefix="results"):
    if isinstance(name, str):
        filename = "{}-{}.csv".format(filename_prefix, name)
        _store_csv_helper(filename, storages, CoreStatistics.row_header())
    else:
        assert isinstance(name, list)
        for storages_alpha, alpha in zip(storages, name):
            filename = "{}-at{}.csv".format(filename_prefix, int(alpha * 100))
            _store_csv_helper(filename, storages_alpha, CoreStatistics.row_header())


def _store_csv_helper(filename, storages, row_header):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_header)
        for storage in storages:
            storage.write(writer)


def store_csv_storage(filename, storage):
    _store_csv_helper(filename, [storage], storage.row_header)


def load_core_statistics(name, filename_prefix="results"):
    if isinstance(name, str):
        filename = "{}-{}.csv".format(filename_prefix, name)
        storages = _load_core_statistics_helper(filename)
        return storages
    else:
        assert isinstance(name, list)
        storages_all = []
        for alpha in name:
            filename = "{}-at{}.csv".format(filename_prefix, int(alpha * 100))
            storages = _load_core_statistics_helper(filename)
            storages_all.append(storages)
        return storages_all


def _load_core_statistics_helper(filename):
    storages = []
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            cs = CoreStatistics.parse(row)
            storages.append(cs)
    return storages


def number_of_hidden_layers(model):
    return len(model.layers) - 2


def number_of_hidden_neurons(model):
    n = 0
    for layer_idx in range(1, len(model.layers) - 1):
        layer = model.layers[layer_idx]
        prod = 1
        for j in range(1, len(layer.output_shape)):
            prod *= layer.output_shape[j]
        n += prod
    return n


def reduce_dimension(layer2data, layers, layer2n_components, layer2components=None, method_name="PCA",
                     keep_dimension=False):
    layer2components_new = {}
    layer2reduced = {}
    for layer in layers:
        data = layer2data[layer]
        n_components = layer2n_components[layer]
        if n_components > 0:
            print("applying reduction to layer {:d}".format(layer))
            if layer2components is None:
                # compute new reduction mapping
                if method_name == "PCA":
                    # choose the variance to capture
                    layer2components_new[layer] = PCA(n_components=n_components)
                elif method_name == "KernelPCA":
                    # choose the variance to capture
                    layer2components_new[layer] = KernelPCA(n_components=n_components, kernel='rbf')
                elif method_name == "ICA":
                    # choose the number of independent components to consider
                    layer2components_new[layer] = FastICA(n_components=n_components)
                else:
                    raise(ValueError("unknown method name {}".format(method_name)))
                # store components for the layer
                layer2components_new[layer].fit(data)
                if PLOT_ADDITIONAL_FEEDBACK:
                    plt.plot(np.cumsum(layer2components_new[layer].explained_variance_ratio_))
                    plt.xlabel('number of components')
                    plt.ylabel('cumulative explained variance')
                    plt.show()
            else:
                # use previous reduction mapping
                layer2components_new[layer] = layer2components[layer]

            # transform to component space
            layer2reduced[layer] = layer2components_new[layer].transform(data)

            # project the data back from components to original dimension
            if keep_dimension:
                layer2reduced[layer] = layer2components_new[layer].inverse_transform(layer2reduced[layer])
        else:
            print("no reduction for layer {:d}".format(layer))
            layer2reduced[layer] = data

    return layer2reduced, layer2components_new


def inside_the_box(layer2data, layers, method_name=DISTRIBUTION_METHOD):
    layer2distribution = {}
    for layer in layers:
        layer2distribution[layer] = inside_the_box_layer(layer2data[layer], method_name=method_name)
    return layer2distribution


def inside_the_box_layer(data, method_name=DISTRIBUTION_METHOD):
    # taken from https://github.com/scikit-learn/scikit-learn/blob/master/examples/mixture/plot_gmm_pdf.py
    # fit a Gaussian Mixture Model with n_components components
    if method_name == "GMM":
        # find the optimal number of components using AIC
        n_components_list = np.arange(2, 100, 10)
        n_components = n_components_list[0]
        model = mixture.GaussianMixture(n_components, covariance_type="full", random_state=0)
        aic_prev = model.fit(data).aic(data)
        for n_components in n_components_list[1:]:
            model = mixture.GaussianMixture(n_components, covariance_type="full", random_state=0)
            aic = model.fit(data).aic(data)
            # the optimal number of components gives minimal AIC
            if aic < aic_prev:
                aic_prev = aic
            else:
                break
    else:
        raise(ValueError("unknown method name: {}".format(method_name)))
    # no need for model.fit(data) again because we did that above already

    return model


def isNaN(num):
    return num != num


def rejected_inputs(history, confidence_threshold=INCREDIBLE_CONFIDENCE, n_min_acceptance=None):
    monitor2indices = dict()
    for monitor_index in history.monitor2results.keys():
        monitor_results = history.monitor2results[monitor_index]
        indices = [i for i in range(len(history.predictions)) if
                   not monitor_results[i].accepts(confidence_threshold=confidence_threshold,
                                                  n_min_acceptance=n_min_acceptance)]
        monitor2indices[monitor_index] = indices
    return monitor2indices


def normalize_layer_map(layer2something, model):
    layer2something_new = dict()
    for layer, something in layer2something.items():
        layer_normalized = normalize_layer(model, layer)
        if layer_normalized in layer2something_new:
            raise(ValueError(
                "Duplicate layer index {:d} found. Please use unique indexing.".format(layer_normalized)))
        layer2something_new[layer_normalized] = something
    return layer2something_new


def compute_alpha_thresholding(monitor_results: list, model, data):
    confidences_at = model.predict(data.inputs())  # Note: model.predict_proba() was deprecated
    for monitor_result, confidence_at_vec in zip(monitor_results, confidences_at):
        confidence = 1 - np.max(confidence_at_vec)  # values closer to 0 mean higher confidence
        monitor_result.add_confidence(confidence)


def get_random_classes(k, n_total):
    # return a sorted list of k non-repeated indices from [0, n_total]
    classes = random.sample(range(n_total), k)
    classes.sort()
    return classes
