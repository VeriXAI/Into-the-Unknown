from monitoring import *
from utils import *


class RandomMonitor(Monitor):
    def __init__(self, monitor: Monitor, acceptance_probability):
        super().__init__(layer2abstraction=monitor._layer2abstraction,
                         score_fun=monitor._score_fun,
                         layer2dimensions=monitor._layer2dimensions,
                         learn_from_test_data=monitor._learn_from_test_data,
                         is_novelty_training_active=monitor._is_novelty_training_active,
                         class_label_map=monitor._class_label_map,
                         id=monitor._id)
        self._acceptance_probability = acceptance_probability

    def acceptance_probability(self):
        return self._acceptance_probability

    def requires_layer_data(self):
        return False

    # some arguments are just there for interface reasons
    def run(self, layer2values: dict, predictions: list, history: History, zero_filter=None, skip_confidence=None):
        results = []
        for i in range(len(predictions)):
            monitor_result = MonitorResult()
            confidence = random.random()  # random number in [0, 1]
            monitor_result.add_confidence(confidence)
            results.append(monitor_result)
        history.set_monitor_results(m_id=self.id(), results=results)
        return results
