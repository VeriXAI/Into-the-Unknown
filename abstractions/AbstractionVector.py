from copy import deepcopy
from utils import *

from .BoxAbstraction import BoxAbstraction


class AbstractionVector(object):
    def __init__(self, abstraction, n_classes_total):
        self._abstractions = [deepcopy(abstraction) for _ in range(n_classes_total)]
        self._distributions = dict()

    def __str__(self):
        return str(self._abstractions[0])

    def abstractions(self):
        return self._abstractions

    def nonempty_labels(self):
        return [i for i in range(len(self._abstractions)) if not self._abstractions[i].isempty()]

    def long_str(self):
        string = str(self)
        for i, abstraction in enumerate(self._abstractions):
            string += "\n class", i, "-> " + abstraction.long_str()
        return string

    def short_str(self):
        return self._abstractions[0].short_str()

    def initialize(self, n_watched_neurons):
        for abstraction in self._abstractions:
            abstraction.initialize(n_watched_neurons)

    def add(self, class_id, vector):
        self._abstractions[class_id].add(vector)

    def finalize(self):
        for abstraction in self._abstractions:
            if not abstraction.isempty():
                abstraction.finalize()

    def isknown(self, class_id, vector, skip_confidence=False, novelty_mode=False):
        # posterior of the data point in the layer
        if self._distributions[class_id] is not None:
            distribution = self._distributions[class_id].predict_proba(vector.reshape(1, -1))[0]
        else:
            distribution = None
        return self._abstractions[class_id].isknown(vector=vector, distribution=distribution,
                                                    skip_confidence=skip_confidence or novelty_mode,
                                                    novelty_mode=novelty_mode)

    def clear(self, class_indices=None):
        if class_indices is None:
            class_indices = [i for i in range(len(self._abstractions))]
        for class_index in class_indices:
            abstraction = self._abstractions[class_index]
            abstraction.clear()

    def add_finalized(self, class_id, vector):
        self._abstractions[class_id].add_finalized(vector)

    def default_options(self):
        return self._abstractions[0].default_options()

    def coarsen_options(self, options):
        return self._abstractions[0].coarsen_options(options)

    def refine_options(self, options):
        return self._abstractions[0].refine_options(options)

    def propose(self, vector):
        # proposal is only based on mean
        class_proposed = -1
        min_distance = float("inf")
        for class_index, abstraction in enumerate(self._abstractions):
            if abstraction.isempty():
                continue
            distance = abstraction.euclidean_mean_distance_absolute(vector)
            if distance < min_distance:
                min_distance = distance
                class_proposed = class_index
        assert class_proposed >= 0, "Did not find any nonempty abstraction."
        return class_proposed

    def update_clustering(self, class_index, clusters):
        self._abstractions[class_index].update_clustering(clusters)

    def add_clustered(self, class_index, values, clusters, distribution_method=DISTRIBUTION_METHOD):
        # fit a distribution to the data of each class
        if distribution_method is not None:
            self._distributions[class_index] = inside_the_box_layer(data=np.array(values),
                                                                    method_name=distribution_method)
        else:
            self._distributions[class_index] = None
        # cluster the data of each class, excluding distribution outliers
        self._abstractions[class_index].add_clustered(values, clusters, self._distributions[class_index])

    def n_data(self, class_index):
        return self._abstractions[class_index].n_data()

    def distance_to_boxes(self):
        for abstraction in self._abstractions:
            assert isinstance(abstraction, BoxAbstraction), "box_distance is only available for box abstractions!"
        m = len(self._abstractions)
        distances = [[[] for _ in range(m)] for _ in range(m)]
        for i, abstraction1 in enumerate(self._abstractions):
            for j, abstraction2 in enumerate(self._abstractions):
                if i == j:
                    d = []
                else:
                    d = abstraction1.distance_to_boxes(abstraction2)
                distances[i][j] = d
        return distances

    def box_distance(self, class_id, point):
        return self._abstractions[class_id].box_distance(point)

    def euclidean_distance(self, class_id, point):
        return self._abstractions[class_id].euclidean_distance(point)
