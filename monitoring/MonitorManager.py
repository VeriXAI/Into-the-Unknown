from . import *
from utils import *


class MonitorManager(object):
    # --- public --- #

    def __init__(self, monitors: list, clustering_algorithm="KMeans", clustering_threshold=0.1, n_clusters=None,
                 filter_zeros=FILTER_ZERO_DIMENSIONS, alpha_thresholding=False,
                 skip_confidence=SKIP_CONFIDENCE_COMPUTATION,
                 layer2n_components=None, reduction_method=REDUCTION_METHOD,
                 fit_distribution_method=DISTRIBUTION_METHOD, transfer=USE_TRANSFER_LEARNING,
                 class_label_map=None):
        self._monitors = monitors
        self._alpha_thresholding = alpha_thresholding
        self._layers = []
        self.clustering_algorithm = clustering_algorithm
        self.clustering_threshold = clustering_threshold
        self.n_clusters = n_clusters
        self.layer2class2clusterer = dict()
        self.layer2class2nonzero_mask = dict() if filter_zeros else None
        self.skip_confidence = skip_confidence
        self.learn_from_test_data = any(m.is_test_training_active() for m in self._monitors)
        self.learn_from_novelty_data = any(m.is_novelty_training_active() for m in self._monitors)
        self.layer2n_components = layer2n_components  # mapping: layer -> number of principal components (or 'None')
        self.layer2components = dict()  # mapping: layer -> PCA transformation
        self.layer2distribution = None  # mapping: layer -> distribution (or 'None')
        self.fit_distribution_method = fit_distribution_method
        self.reduction_method = reduction_method
        self.transfer = transfer
        self.class_label_map = class_label_map  # see ClassLabelMap
        self._monitoring_mode = MONITOR_MANAGER_MODE_NORMAL
        if alpha_thresholding > 0:
            # special mode where monitor only checks alpha threshold
            self._monitoring_mode = MONITOR_MANAGER_MODE_ALPHA
        elif all(not monitor.requires_layer_data() for monitor in monitors):
            # special mode where monitor is random and does not require data processing
            self._monitoring_mode = MONITOR_MANAGER_MODE_PASSIVE

    def layers(self):
        return self._layers

    def monitor(self):
        assert len(self._monitors) == 1, "Monitor is not unique!"
        return self._monitors[0]

    def monitors(self):
        return self._monitors

    def normalize_and_initialize(self, model, class_label_map, n_classes_total):
        self.class_label_map = class_label_map
        layers = set()
        for monitor in self._monitors:  # type: Monitor
            monitor.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=n_classes_total)
            layers.update(monitor.layers())
        self._layers = list(layers)
        do_reduce_dimension = self.layer2n_components is not None
        for layer in self._layers:
            if do_reduce_dimension:
                self.layer2n_components = normalize_layer_map(self.layer2n_components, model)
                n_neurons = number_of_neurons(model, layer)
                if layer in self.layer2n_components and self.layer2n_components[layer] > 0:
                    if self.layer2n_components[layer] > n_neurons:
                        raise ValueError("The number of components {:d} in layer {:d} is larger than the number of "
                                         "neurons {:d}".format(self.layer2n_components[layer], layer, n_neurons))
                else:
                    # if unspecified, use the default number of components for this layer
                    self.layer2n_components[layer] = min(n_neurons, Options.MAX_N_COMPONENTS)
        print("Watching the following layers:")
        for layer in self._layers:
            print("- layer {:d} with {:d} neurons{}".format(
                layer, number_of_neurons(model, layer),
                " (reducing to {:d} components)".format(self.layer2n_components[layer]) if do_reduce_dimension else ""))

    def train(self, model, data_train: DataSpec, data_test: DataSpec, statistics: Statistics,
              ignore_misclassifications=ONLY_LEARN_FROM_CORRECT_CLASSIFICATIONS, initialized=None):
        if self._monitoring_mode == MONITOR_MANAGER_MODE_NORMAL:
            pass  # continue below
        elif self._monitoring_mode in [MONITOR_MANAGER_MODE_ALPHA, MONITOR_MANAGER_MODE_PASSIVE]:
            data_train_filtered = data_train
            return data_train_filtered
        else:
            raise(ValueError("Unknown monitor-manager mode {}".format(self._monitoring_mode)))

        print("\n--- monitor training ---\n")

        # extract values for watched layers
        print("extracting data for watched layers")
        layer2values, data_train_filtered, timer =\
            obtain_predictions(model=model, data=data_train, layers=self.layers(), class_label_map=self.class_label_map,
                               ignore_misclassifications=ignore_misclassifications, transfer=self.transfer)
        # data_train_filtered is the data that is actually used for training (ignoring misclassifications)

        # project layer values using dimensionality reduction
        if self.layer2n_components is not None:
            print("reducing dimension")
            layer2values, self.layer2components = \
                reduce_dimension(layer2data=layer2values, layers=self.layers(),
                                 layer2n_components=self.layer2n_components, method_name=self.reduction_method)

        # fit a distribution to the layer data
        if self.fit_distribution_method is not None:
            self.layer2distribution = inside_the_box(layer2data=layer2values, layers=self.layers(),
                                                     method_name=self.fit_distribution_method)

        timer_sum = timer
        assert not self.learn_from_novelty_data and not self.learn_from_novelty_data, \
            "Learning from novelty or test data is not implemented for reduced layer data"
        if self.learn_from_test_data or self.learn_from_novelty_data:
            # project layer values using dimensionality reduction
            layer2values_novel, _, timer = obtain_predictions(model=model, data=data_test,
                                                              class_label_map=self.class_label_map,
                                                              layers=self.layers(), transfer=self.transfer)
            if self.layer2n_components is not None:
                layer2values_novel, layer2components_novel = \
                    reduce_dimension(layer2data=layer2values_novel, layers=self.layers(),
                                     layer2n_components=self.layer2n_components, method_name=self.reduction_method)

            if self.fit_distribution_method is not None:
                # fit a distribution to the reduced layer data
                # TODO: the data has to be combined first and then fit to
                layer2distribution_novel = inside_the_box(layer2data=layer2values, layers=self.layers(),
                                                          method_name=self.fit_distribution_method)
            else:
                layer2distribution_novel = None
            timer_sum += timer
            predictions_novel, _, timer = obtain_predictions(model=model, data=data_test,
                                                             class_label_map=self.class_label_map,
                                                             transfer=self.transfer)
            timer_sum += timer
        statistics.time_training_monitor_value_extraction = timer_sum

        # filter out zero dimensions
        if self.layer2class2nonzero_mask is not None:
            assert not self.learn_from_test_data
            self._determine_zero_filters(layer2values, model, data_train_filtered)
            layer2values = self._remove_zero_dimensions(layer2values, data_train_filtered.ground_truths(),
                                                        compute_violation_indices=False)
            for monitor in self._monitors:
                monitor.initialize_abstractions(self.layer2class2nonzero_mask)

        # clustering
        print("determining optimal clusters for each layer and class ({}, {})".format(
            self.clustering_threshold, self.n_clusters))
        self.layer2class2clusterer = self._clustering(data=data_train_filtered, layer2values=layer2values,
                                                      statistics=statistics, initialized=initialized)
        if self.learn_from_test_data:
            data_train_combined = DataSpec(inputs=np.append(data_train_filtered.inputs(), data_test.inputs(), axis=0),
                                           labels=np.append(data_train_filtered.ground_truths(), data_test.ground_truths(),
                                                            axis=0))
            layer2values_combined = self._combine_layer2values(layer2values, layer2values_novel)
            layer2class2clusterer_combined = self._clustering(data=data_train_combined,
                                                              layer2values=layer2values_combined, statistics=statistics,
                                                              includes_test_data=True)

        # monitor training
        print("training monitors on the obtained data")
        self._train_monitors(data=data_train_filtered, layer2values=layer2values,
                             layer2distribution=self.layer2distribution,
                             layer2class2clusterer=self.layer2class2clusterer,
                             predictions=None, statistics=statistics, includes_test_data=False)
        if self.learn_from_test_data:
            self._train_monitors(data=data_train_combined, layer2values=layer2values_combined,
                                 layer2distribution=self.layer2distribution,
                                 layer2class2clusterer=layer2class2clusterer_combined, predictions=None,
                                 statistics=statistics, includes_test_data=True)

        if self.learn_from_novelty_data:
            # novelty training
            print("additional training of monitors on novelty data")
            self._train_monitors(data=data_test, layer2values=layer2values_novel,
                                 layer2distribution=layer2distribution_novel, layer2class2clusterer=None,
                                 predictions=predictions_novel, statistics=statistics, includes_test_data=False)

        return data_train_filtered

    def run(self, model, data: DataSpec, statistics: Statistics):
        print("\n--- running monitored session ---\n")

        history = History()
        history.set_ground_truths(data.ground_truths())

        if self._monitoring_mode == MONITOR_MANAGER_MODE_NORMAL:
            pass  # continue below
        elif self._monitoring_mode in [MONITOR_MANAGER_MODE_ALPHA, MONITOR_MANAGER_MODE_PASSIVE]:
            timer = time()
            predictions, _, _ = obtain_predictions(model=model, data=data, class_label_map=self.class_label_map,
                                                   transfer=self.transfer)
            statistics.time_running_monitor_value_extraction = time() - timer
            history.set_predictions(predictions)

            if self._monitoring_mode == MONITOR_MANAGER_MODE_ALPHA:
                monitor_results = [MonitorResult() for _ in predictions]
                compute_alpha_thresholding(monitor_results, model, data)
            elif self._monitoring_mode == MONITOR_MANAGER_MODE_PASSIVE:
                monitor_results = self.monitor().run(layer2values=dict(), predictions=predictions, history=history)
            else:
                raise(ValueError("Unknown monitor-manager mode: {}".format(self._monitoring_mode)))
            history.set_monitor_results(self.monitor().id(), monitor_results)

            return history
        else:
            raise(ValueError("Unknown monitor-manager mode {}".format(self._monitoring_mode)))

        # extract values for watched layers and predictions of model
        layer2values, _, timer = obtain_predictions(model=model, data=data, class_label_map=self.class_label_map,
                                                    layers=self.layers(), transfer=self.transfer)
        # project layer values using dimensionality reduction obtained during training
        if self.layer2n_components is not None:
            layer2values, _ = reduce_dimension(layer2data=layer2values, layers=self.layers(),
                                               layer2n_components=self.layer2n_components,
                                               layer2components=self.layer2components,
                                               method_name=self.reduction_method)

        timer_sum = timer
        predictions, _, timer = obtain_predictions(model=model, data=data, class_label_map=self.class_label_map,
                                                   transfer=self.transfer)
        timer_sum += timer
        statistics.time_running_monitor_value_extraction = timer_sum
        history.set_layer2values(layer2values)
        history.set_predictions(predictions)

        # filter out zero dimensions
        if self.layer2class2nonzero_mask is not None:
            layer2values, zero_filter = self._remove_zero_dimensions(layer2values, predictions,
                                                                     compute_violation_indices=True)
            print("Found {:d} inputs that do not match the 'zero pattern'.".format(len(zero_filter)))
        else:
            zero_filter = []

        # run monitors
        timer = time()
        monitor2results = dict()
        for monitor in self._monitors:  # type: Monitor
            m_id = monitor.id()
            print("running monitor {:d} on the inputs".format(m_id))
            timer_monitor = time()
            monitor_results = monitor.run(layer2values=layer2values,
                                          predictions=predictions, history=history,
                                          zero_filter=zero_filter, skip_confidence=self.skip_confidence)
            statistics.time_running_each_monitor[m_id] = time() - timer_monitor
            monitor2results[m_id] = monitor_results
            if self._alpha_thresholding:
                compute_alpha_thresholding(monitor_results=monitor_results, model=model, data=data)
                monitor2results[-m_id] = monitor_results
                history.set_monitor_results(-m_id, monitor_results)
        statistics.time_running_monitor_classification = time() - timer

        return history

    # --- private --- #

    def _clustering(self, data, layer2values, statistics, initialized=None, includes_test_data=False):
        layers = self.layers()

        # cluster classes in each layer
        timer = time()
        layer2class2clusterer = dict()
        for layer in layers:
            class2values = dict()  # mapping: class_index -> values from watched layer
            values = layer2values[layer]
            print("data size before clustering: {:}".format(len(values)))
            assert len(values) == data.n(), "inconsistent data sizes {:d} and {:d}".format(len(values), data.n())
            for j, cj in enumerate(data.ground_truths()):
                vj = values[j]
                if cj in class2values:
                    class2values[cj].append(vj)
                else:
                    class2values[cj] = [vj]

            # find number of clusters
            print("Layer {:d}:".format(layer))
            initialized_layer = initialized[layer] if initialized is not None else None
            class2clusters = cluster_refinement(class2values, algorithm=self.clustering_algorithm,
                                                threshold=self.clustering_threshold,
                                                n_clusters=self.n_clusters,
                                                initialized=initialized_layer)
            layer2class2clusterer[layer] = class2clusters

            # update abstraction with number of clusters
            for monitor in self._monitors:  # type: Monitor
                if includes_test_data != monitor.is_test_training_active():
                    continue
                monitor.update_clustering(layer, class2clusters)

        statistics.time_training_monitor_clustering = time() - timer
        return layer2class2clusterer

    def _train_monitors(self, data, layer2values, layer2distribution, layer2class2clusterer, predictions, statistics,
                        includes_test_data):
        timer = time()
        ground_truths = data.ground_truths()
        novelty_training_mode = layer2class2clusterer is None
        for monitor in self._monitors:  # type: Monitor
            if (novelty_training_mode and not monitor.is_novelty_training_active()) or \
                    (includes_test_data != monitor.is_test_training_active()):
                continue
            print("training monitor {:d}{}".format(monitor.id(),
                                                   " with novelties" if layer2class2clusterer is None else ""))
            timer_monitor = time()
            if novelty_training_mode:
                monitor.train_with_novelties(predictions, layer2values)
            else:
                monitor.add_clustered(layer2values, ground_truths, layer2class2clusterer, layer2distribution,
                                      distribution_method=self.fit_distribution_method)
            duration = time() - timer_monitor
            if monitor.id() in statistics.time_tweaking_each_monitor.keys():
                statistics.time_tweaking_each_monitor[monitor.id()] += duration
            else:
                statistics.time_tweaking_each_monitor[monitor.id()] = duration
        duration = time() - timer
        if statistics.time_training_monitor_tweaking == -1:
            statistics.time_training_monitor_tweaking = duration
        else:
            statistics.time_training_monitor_tweaking += duration

    def _determine_zero_filters(self, layer2values: dict, model: Model, data: DataSpec):
        for layer, values in layer2values.items():
            n_neurons = number_of_neurons(model, layer)
            self.layer2class2nonzero_mask[layer] = determine_zero_filters(values, data, n_neurons, layer)

    def _remove_zero_dimensions(self, layer2values, classes, compute_violation_indices: bool):
        layer2values_new = dict()
        zero_indices = set()
        for layer, values in layer2values.items():
            class2nonzero_indices = self.layer2class2nonzero_mask[layer]
            filtered_values = []
            layer2values_new[layer] = filtered_values
            for j, (class_id, vj) in enumerate(zip(classes, values)):
                vj_filtered = []
                for vi, filter_i in zip(vj, class2nonzero_indices[class_id]):
                    if filter_i:
                        vj_filtered.append(vi)
                    elif vi != 0:
                        # found a violation
                        zero_indices.add(j)
                filtered_values.append(vj_filtered)
        if compute_violation_indices:
            return layer2values_new, sorted(zero_indices)
        else:
            return layer2values_new

    def _combine_layer2values(self, layer2values_1: dict, layer2values_2: dict):
        res = dict()
        if layer2values_1:
            for k, v in layer2values_1.items():
                res[k] = np.append(v, layer2values_2[k], axis=0)
        else:
            return layer2values_2
        return res
