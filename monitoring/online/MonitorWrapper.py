from utils import *
from monitoring import MonitorManager
from monitoring.online import DistanceMonitor, RandomMonitor, OnlineStatistics


class MonitorWrapper(object):
    def __init__(self, monitor_manager: MonitorManager, alpha_threshold=0,
                 n_false_warnings_threshold=DEFAULT_N_FALSE_WARNINGS_THRESHOLD,
                 initial_distance_threshold=INITIAL_DISTANCE_THRESHOLD,
                 adapt_score_thresholds=True):
        self.monitor_manager = monitor_manager
        self.monitor_statistics = 0.0  # cumulative statistics about fp,fn,tp,tn
        self.network_statistics = 0.0  # cumulative statistics about fp,fn,tp,tn
        self.score_thresholds = {}  # thresholds for distance scores
        # TODO: as baseline for statistics use
        #  the ones computed for the network and monitor trained for all combinations of classes from scratch
        self.monitor_statistics_threshold = 0.0  # threshold for monitor statistics
        self.network_statistics_threshold = 0.0  # threshold for network statistics
        self.n_samples = {}  # number of samples per class
        self.n_false_warnings = 0  # number of false monitor warnings since the last monitor training
        # allowed number of false monitor warnings before monitor training
        self.n_false_warnings_threshold = n_false_warnings_threshold
        self.label2avgdist = {}  # average distances for each label (TODO CS: currently not used - potentially remove)
        self.alpha_threshold = alpha_threshold
        self.initial_distance_threshold = initial_distance_threshold
        self.adapt_score_thresholds = adapt_score_thresholds

    def process(self, data: DataSpec, monitor: DistanceMonitor, history_run: History, class_label_map: ClassLabelMap,
                skip_image_plotting=True):
        outside_the_box = []

        for i, (prediction_i, ground_truth_i) in enumerate(zip(history_run.predictions, history_run.ground_truths)):
            is_outlier = self.update_history_result(i=i, monitor=monitor, prediction_i=prediction_i,
                                                    ground_truth_i=ground_truth_i, history=history_run)

            if is_outlier:
                outside_the_box.append(i)

            # TODO CS: only plot actual warnings?
            # report it now
            # plot warning and novelty images
            if not skip_image_plotting:
                labels = ['label' + str(i) for i in range(max(data.classes) + 1)]
                warnings_full = history_run.warnings(monitor=monitor, data=data)
                plot_images(images=warnings_full, labels=labels, classes=data.classes,
                            iswarning=True,
                            monitor_id=monitor.id())

        # TODO CS: commented this out because the caller only needs to know whether there is at least one outsider
        #   this needs to be changed once we go to batches
        # outsiders = DataSpec(randomize=False)
        # data_x = np.array(outside_the_box)
        # outsiders.set_x(data_x)
        # outsiders.n = len(data_x)
        # outsiders.classes = sorted(set(outsiders.ground_truths()))
        # TODO: not to forget to keep these outsiders to show to the authority for labelling

        return len(outside_the_box) > 0

    def update_history_results(self, monitor, history: History):
        history.reset_statistics()
        for i, (prediction_i, ground_truth_i) in enumerate(zip(history.predictions, history.ground_truths)):
            self.update_history_result(i=i, monitor=monitor, prediction_i=prediction_i, ground_truth_i=ground_truth_i,
                                       history=history)

    def update_history_result(self, i, monitor, prediction_i, ground_truth_i, history):
        monitor_results = history.monitor2results[monitor.id()][i]
        if isinstance(monitor, RandomMonitor):
            confidence_threshold = monitor.acceptance_probability()
        elif self.alpha_threshold > 0:
            confidence_threshold = self.alpha_threshold
        elif isinstance(monitor, DistanceMonitor):
            confidence_threshold = self.score_thresholds[monitor_results.prediction()]
        else:
            confidence_threshold = ACCEPTANCE_CONFIDENCE
        accepts = monitor_results.accepts(confidence_threshold=confidence_threshold)
        is_outlier = not accepts

        # meta statistics
        is_correct = prediction_i == ground_truth_i
        history.update_statistics_one(accepts=accepts, is_correct=is_correct)

        return is_outlier

    def adapt(self, data_new: DataSpec, data_train_old, history, sample_threshold, statistics):
        # Note: data_new is already labeled with the authority labels
        predictions = history.predictions[len(history.predictions) - data_new.n():]
        results = history.monitor2results[1][len(history.predictions) - data_new.n():]
        # TODO CS: with data batches we first need to filter the relevant indices; note that history already only
        #  contains the relevant entries
        for label_i, prediction_i, result_i in zip(data_new.ground_truths(), predictions, results):
            if prediction_i == label_i:
                # network was correct
                print("false positive")
                # TODO CS: find relevant set and increase distance (by how much?)?
                self.n_false_warnings += 1
            else:
                print("true positive")
                # network was incorrect
                if isinstance(self.monitor_manager.monitor(), DistanceMonitor):
                    # additional adaptation of distance monitor
                    distance = result_i.distance(label_i)
                    if label_i not in self.label2avgdist:
                        self.label2avgdist[label_i] = AverageDistanceWrapper()
                    self.label2avgdist[label_i].add(distance)
                    if self.adapt_score_thresholds and distance > self.score_thresholds[label_i]:
                        # increase the class score threshold
                        # we discount the change by the number of samples in the respective class
                        monitor = self.monitor_manager.monitor()
                        old_threshold = self.score_thresholds[label_i]
                        delta = distance - old_threshold
                        n_samples = monitor.n_data(label_i)
                        scaled_delta = delta / n_samples * sample_threshold._lower
                        self.score_thresholds[label_i] += scaled_delta
                        self.print_threshold(label=label_i, old_threshold=old_threshold)

        # policy: always re-cluster
        # if self.n_false_warnings >= self.n_false_warnings_threshold:
        recluster = True

        if recluster and isinstance(self.monitor_manager.monitor(), DistanceMonitor):
            print("additional training of monitors with authority labels")
            self.retrain_monitor_incrementally(data_train_old=data_train_old,
                                               updated_classes=data_new.classes,
                                               history=history,
                                               statistics=statistics)

    def retrain_monitor_incrementally(self, data_train_old, updated_classes, history, statistics):
        if self.skip_computations():
            self._retrain_monitor_shared()
            return  # skip retraining in alpha-threshold mode

        statistics.timer_online_adapting_monitors.start()
        monitor = self.monitor_manager.monitor()
        data_train_combined = data_train_old.filter_by_classes(classes=updated_classes, copy=True)
        # filter history
        layer2values = dict()
        ground_truths = []
        for i, gt_i in enumerate(history.ground_truths):
            if gt_i in updated_classes:
                ground_truths.append(gt_i)
                for layer, values in history.layer2values.items():
                    if layer not in layer2values:
                        layer2values[layer] = []
                    layer2values[layer].append(values[i])
        for layer, values in layer2values.items():
            layer2values[layer] = np.array(values)
        ground_truths = np.array(ground_truths)

        timer_monitor = time()
        # re-clustering with initialization from training data
        layer2class2clusterer = self.monitor_manager._clustering(data=data_train_combined,
                                                                 layer2values=layer2values,
                                                                 statistics=statistics,
                                                                 initialized=self.monitor_manager.layer2class2clusterer)
        for layer, class2clusterer in layer2class2clusterer.items():
            class2clusterer_old = self.monitor_manager.layer2class2clusterer[layer]
            for c, clusterer in class2clusterer.items():
                class2clusterer_old[c] = clusterer
        # monitor.train_with_novelties(authority_labels, history_run.layer2values)
        # clear current abstraction first to update it
        for layer in monitor.layers():
            monitor.abstraction(layer).clear(class_indices=updated_classes)
        # update abstraction with re-clustered data
        monitor.add_clustered(layer2values=layer2values,
                              ground_truths=ground_truths,
                              layer2class2clusterer=self.monitor_manager.layer2class2clusterer,
                              layer2distribution=self.monitor_manager.layer2distribution,
                              distribution_method=self.monitor_manager.fit_distribution_method)
        duration = time() - timer_monitor
        if monitor.id() in statistics.time_tweaking_each_monitor.keys():
            statistics.time_tweaking_each_monitor[monitor.id()] += duration
        else:
            statistics.time_tweaking_each_monitor[monitor.id()] = duration
        self._retrain_monitor_shared()
        statistics.timer_online_adapting_monitors.stop()

    def retrain_monitor_from_scratch(self, data_train, data_test, network, known_labels, statistics: OnlineStatistics):
        statistics.timer_online_retraining_monitors.start()
        # clear current abstraction first to update it
        monitor = self.monitor_manager.monitor()
        for layer in monitor.layers():
            monitor.abstraction(layer).clear()
        #data_train_new_classes = data_train.filter_by_classes(classes=known_labels, copy=True)
        self.monitor_manager.train(model=network,
                                   data_train=data_train,
                                   data_test=data_test,
                                   statistics=statistics,
                                   initialized=self.monitor_manager.layer2class2clusterer)
        # initialize score thresholds for new labels
        for label in known_labels:
            if label not in self.score_thresholds:
                self.score_thresholds[label] = self.initial_distance_threshold
        # reset average-distance information
        self.label2avgdist = {}
        # shared code when retraining a monitor
        self._retrain_monitor_shared()
        statistics.timer_online_retraining_monitors.stop()

    def _retrain_monitor_shared(self):
        self.n_false_warnings = 0

    def skip_computations(self):
        return self.monitor_manager._monitoring_mode != MONITOR_MANAGER_MODE_NORMAL

    def update_results(self, monitor2results):
        pass

    def print_threshold(self, label, old_threshold=None):
        new_threshold = self.score_thresholds[label]
        print("changed threshold for class {:d} from {} to {}".format(label, old_threshold, new_threshold))

    def print_thresholds(self):
        for label, threshold in self.score_thresholds.items():
            print("threshold for class {:d}: {}".format(label, threshold))


class AverageDistanceWrapper(object):
    def __init__(self):
        self._total_distances = 0.0
        self._n = 0

    def add(self, distance):
        self._total_distances += distance
        self._n += 1

    def average_distance(self):
        if self._n == 0:
            return 0
        return self._total_distances / self._n

    def reset(self):
        self._total_distances = 0.0
        self._n = 0
