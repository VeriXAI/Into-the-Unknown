from .AuthorityDataThreshold import AuthorityDataThreshold
from utils import *


class Evaluator(object):
    def evaluate(self, monitor_wrapper, new_labels, authority, n_initial_samples, n_initial_classes, status):
        # ask authority how much data is enough
        data_threshold = authority.threshold(n_initial_samples=n_initial_samples, n_initial_classes=n_initial_classes)  # type: AuthorityDataThreshold

        other_labels = []
        # check which class has accumulated enough data for training
        for other_label, n_samples in monitor_wrapper.n_samples.items():
            if other_label in new_labels and data_threshold.can_train(n_samples):
                other_labels.append(other_label)
                if data_threshold.must_train(n_samples):
                    status = STATUS_RETRAIN_NETWORK
        return status, other_labels
