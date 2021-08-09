from monitoring import *
from monitoring.online import DistanceMonitorResult
from utils import *


def box_distance_parameter(abstraction, label, point):
    return abstraction.box_distance(class_id=label, point=point)


def euclidean_distance_parameter(abstraction, label, point):
    return abstraction.euclidean_distance(class_id=label, point=point)


class DistanceMonitor(Monitor):
    def __init__(self, monitor: Monitor, distance_fun):
        super().__init__(layer2abstraction=monitor._layer2abstraction,
                         score_fun=monitor._score_fun,
                         layer2dimensions=monitor._layer2dimensions,
                         learn_from_test_data=monitor._learn_from_test_data,
                         is_novelty_training_active=monitor._is_novelty_training_active,
                         class_label_map=monitor._class_label_map,
                         id=monitor._id)
        self._distance_fun = distance_fun

    # some arguments are just there for interface reasons
    def run(self, layer2values: dict, predictions: list, history: History, zero_filter=None, skip_confidence=None):
        results = []
        # TODO only works for a single layer
        assert(len(self._layer2abstraction) == 1)
        for layer, abstraction in self._layer2abstraction.items():
            for j, vj in enumerate(layer2values[layer]):
                distances = dict()
                for label in abstraction.nonempty_labels():
                    # for each layer compute distance from the layer output
                    #       corresponding to the monitored input
                    #       to the abstraction of this layer
                    distances[label] = self._distance_fun(abstraction, label, vj)

                label_predicted = predictions[j]
                results.append(DistanceMonitorResult(prediction_network=label_predicted, distances=distances))
        history.set_monitor_results(m_id=self.id(), results=results)
        return results

    def box_distance(self, layer2values_fun, label):
        """compute the box distance to a given class for multiple layers"""
        dist = float('-inf')
        for layer in self.layers():
            point = layer2values_fun(layer)  # layer2values_fun is a function, not a dictionary
            dist_layer = super().box_distance(layer=layer, class_id=label, point=point)
            dist = max(dist, dist_layer)
        return dist

    def euclidean_distance(self, layer2values_fun, label):
        """compute the Euclidean distance to a given class for multiple layers"""
        dist = float('-inf')
        for layer in self.layers():
            point = layer2values_fun(layer)  # layer2values_fun is a function, not a dictionary
            dist_layer = super().euclidean_distance(layer=layer, class_id=label, point=point)
            dist = max(dist, dist_layer)
        return dist
