from copy import copy as make_copy
from numpy.random import permutation

from . import *
import numpy as np


class DataSpec(object):
    def __init__(self, file=None, randomize=False, classes=None, inputs=None, labels=None, n_max=None, uniform=False):
        self.file = file
        self.randomize = randomize
        assert classes is None or classes == sorted(classes), "List of classes must be sorted for filtering!"
        self.classes = classes
        self._inputs = inputs
        self._labels = labels
        if labels is not None:
            if classes is None:
                self.update_classes()
            else:
                assert self._check_classes_consistency(), "Invalid classes used!"
        self.n_max = n_max  # only used to load a bounded amount of data
        if uniform:
            # have a uniform amount of (shuffled) data points per class
            self._filter_uniform()

    def inputs(self):
        return self._inputs

    def ground_truths(self):
        return self._labels

    def ground_truths_as_classes(self):
        result, _ = labels2classes(self._labels, sorted_set_of_numbers=self.classes)
        return result

    def categoricals(self):
        return number2categoricals(self._labels, sorted_set_of_numbers=self.classes)

    def n_classes(self):
        return len(self.classes)

    def n(self):
        return len(self._inputs)

    def set_data(self, data=None, inputs=None, labels=None, assertion=None):
        if data is not None:
            # extract data from dict
            if "features" in data:
                new_inputs = data["features"]
            else:
                new_inputs = data["data"]
            self.set_inputs(new_inputs)
            self.set_labels(data["labels"], assertion=assertion)
        else:
            self.set_inputs(inputs)
            self.set_labels(labels, assertion=assertion)

    def set_inputs(self, x):
        self._inputs = x

    def set_labels(self, labels, assertion):
        self._labels = labels
        assert not assertion or self._check_classes_consistency(), "Invalid classes used!"

    def has_data(self):
        return self._inputs is not None and self._labels is not None

    def merge(self, other):
        data = make_copy(self)
        data.merge_inplace(other=other)

    def merge_inplace(self, other):
        old_length = self.n()
        self._inputs = np.append(self._inputs, other.inputs(), axis=0)
        self._labels = np.append(self._labels, other.ground_truths(), axis=0)
        self.classes = sorted(set(self.classes + other.classes))
        return old_length

    def filter(self, filter, copy, classes=None):
        data = make_copy(self) if copy else self
        data._inputs = data._inputs[filter]
        data._labels = data._labels[filter]
        if classes is not None:
            assert classes == sorted(classes), "List of classes must be sorted for filtering!"
            data.classes = classes
        else:
            data.update_classes()
        if isinstance(filter, list):
            len_filter = len(filter)
        elif isinstance(filter, slice):
            len_filter = filter.stop - filter.start
        else:
            raise(ValueError("Unknown filter type received: {}".format(type(filter))))
        assert len_filter == len(data._inputs) == len(data._labels)
        return data

    def filter_by_classes(self, classes, copy):
        data = make_copy(self) if copy else self
        indices = []
        for i, label_i in enumerate(data.ground_truths()):
            if label_i in classes:
                indices.append(i)
        data.filter(filter=indices, copy=False)
        return data

    def filter_range(self, lo, hi, copy=False):
        data = make_copy(self) if copy else self
        data._inputs = data._inputs[lo:hi]
        data._labels = data._labels[lo:hi]
        data.update_classes()
        return data

    def update_classes(self):
        self.classes = sorted(set(self.ground_truths()))

    def shuffle(self):
        # sample uniformly from the whole data
        indices = permutation([i for i in range(len(self.inputs()))])
        x = self.inputs()[indices]
        y = self.ground_truths()[indices]
        self.set_data(inputs=x, labels=y, assertion=True)

    def _check_classes_consistency(self):
        labels = sorted(set(self._labels))
        return labels == self.classes

    def n_data_per_class(self):
        class2n = dict()
        for class_id in self.classes:
            class2n[class_id] = 0
        for label in self._labels:
            class2n[label] += 1
        return class2n

    def _filter_uniform(self):
        # shuffle the data and then randomly remove data points until there is the same (maximal) amount of data points
        # for each class

        # get number of data points and data points per class
        n_data = self.n()
        class2n = self.n_data_per_class()
        n_min = min(class2n.values())
        class2n = {c: n_min for c in self._labels}

        # permute data
        self.shuffle()

        # create copy of first n_min data points for each class
        x_new = []
        y_new = []
        for x, y in zip(self._inputs, self._labels):
            if class2n[y] > 0:
                x_new.append(x)
                y_new.append(y)
                class2n[y] -= 1
        self.set_data(inputs=np.array(x_new), labels=np.array(y_new), assertion=False)
