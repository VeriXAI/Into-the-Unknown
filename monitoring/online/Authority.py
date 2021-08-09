from .AuthorityDataThreshold import AuthorityDataThreshold
from utils import *


class Authority(object):
    def __init__(self, threshold_percentage=AUTHORITY_THRESHOLD_PERCENTAGE):
        self.threshold_percentage = threshold_percentage

    # TODO: make labelling interactive
    def label(self, data: DataSpec, monitor_wrapper):
        # report outsider and ask for a label
        authority_labels = data.ground_truths()
        # increment number of collected samples
        for label in authority_labels:
            if label not in monitor_wrapper.n_samples.keys():
                monitor_wrapper.n_samples[label] = 1
            else:
                monitor_wrapper.n_samples[label] += 1
        return authority_labels

    def threshold(self, n_initial_samples, n_initial_classes):
        # TODO: make thresholding interactive

        # define sample thresholds as a percentage of the average initial class dataset size
        average_size = n_initial_samples / n_initial_classes
        threshold = self.threshold_percentage * average_size
        return AuthorityDataThreshold(lower=threshold, upper=threshold)
