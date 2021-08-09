class DistanceMonitorResult(object):
    def __init__(self, prediction_network, distances):
        self._prediction_network = prediction_network
        self._distances = distances

    def __str__(self):
        return "p: {:d}, d: {:f}, s: {:d}".format(self._prediction_network, self._distances, self.suggestion())

    def prediction(self):
        return self._prediction_network

    def suggestion(self):
        return min(self._distances, key=self._distances.get)

    def distances(self):
        return self._distances

    def distance(self, label):
        return self._distances[label]

    def accepts(self, confidence_threshold, n_min_acceptance=None):
        # confidence_threshold is the distance threshold but called like this for interface reasons
        # n_min_acceptance is just there for interface reasons
        return self._distances[self._prediction_network] <= confidence_threshold

    def is_zero_filtered(self):  # just there for interface reasons
        return False
