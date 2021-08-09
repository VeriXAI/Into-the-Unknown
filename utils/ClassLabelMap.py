import numpy as np
from copy import copy


class ClassLabelMap(object):
    # map between data labels and network classes
    # A data label is the index assigned in the labeled data set.
    # A network class is the index used in the output layer of the network.
    def __init__(self):
        self.c2l = dict()
        self.l2c = dict()

    def __len__(self):
        return len(self.c2l)

    def known_labels(self):
        return sorted(self.l2c.keys())

    def add_label(self, label):
        if label not in self.l2c:
            #raise(ValueError("label {} is already known".format(label)))
            class_idx = len(self.l2c)  # next class index in consecutive order
            self.c2l[class_idx] = label
            self.l2c[label] = class_idx

    def add_labels(self, labels):
        for label in labels:
            self.add_label(label)

    def __contains__(self, item):
        return item in self.l2c

    def get_label(self, class_idx):
        label = self.c2l.get(class_idx, None)
        if label is None:
            raise(ValueError("class index {} is not known".format(class_idx)))
        return label

    def get_labels(self, class_indices, as_np_array):
        labels = []
        for class_idx in class_indices:
            label = self.c2l.get(class_idx, None)
            if label is None:
                raise(ValueError("class index {} is not known".format(class_idx)))
            labels.append(label)
        if as_np_array:
            labels = np.array(labels)
        return labels

    def get_class(self, label):
        class_idx = self.l2c.get(label, None)
        if class_idx is None:
            raise(ValueError("label {} is not known".format(label)))
        return class_idx

    def get_classes(self, labels):
        class_indices = []
        for label in labels:
            class_idx = self.l2c.get(label, None)
            if class_idx is None:
                raise(ValueError("label {} is not known".format(label)))
            class_indices.append(class_idx)
        return np.array(class_indices)

    def copy(self):
        result = ClassLabelMap()
        result.c2l = copy(self.c2l)
        result.l2c = copy(self.l2c)
        return result


def class_label_map_from_labels(labels):
    class_label_map = ClassLabelMap()
    for label in labels:
        class_label_map.add_label(label=label)
    return class_label_map
