from monitoring.online import *
from run.experiment_helper import *
import time
import threading

STATS_FILE = "precision_with_retraining_"
META_STATS_FILE = "extrinsic_classifier_accuracy_"
META_TESTSET_FILE = "testset_classifier_accuracy_"
INTERACTION_STATS_FILE = "interaction_with_authority_"


class TimerClass(threading.Thread):
    def __init__(self, count, worker_func):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.count = count
        self.worker_func = worker_func

    def run(self):
        while self.count > 0 and not self.event.is_set():
            self.count -= 1
            self.worker_func()
            self.event.wait(1)

    def stop(self):
        self.event.set()


def do_every(interval, worker_func, iterations=0):
    if iterations != 1:
        threading.Timer(
            interval,
            do_every, [interval, worker_func, 0 if iterations == 0 else iterations - 1]
        ).start()

    worker_func()


def collect_stats(data_name, history_test, data_test_monitor, history_meta_test,
                  history_meta_test_retrained, history_meta_test_with_monitor,
                  history_meta_from_scratch,
                  test_history_meta_test,
                  all_test_history_meta_test,
                  test_history_meta_test_with_monitor,
                  all_test_history_meta_test_with_monitor,
                  test_history_meta_from_scratch,
                  class_label_map, authority_requests, processed_inputs, options, test=False):
    # concurrent updates can result in inconsistencies here; hence we make a copy and compare length first
    history_test = copy(history_test)
    history_meta_test = copy(history_meta_test)
    history_meta_test_retrained = copy(history_meta_test_retrained)
    history_meta_test_with_monitor = copy(history_meta_test_with_monitor)
    history_meta_from_scratch = copy(history_meta_from_scratch)
    data_test_monitor = copy(data_test_monitor)
    authority_requests = copy(authority_requests)
    class_label_map = copy(class_label_map)
    test_history_meta_test = copy(test_history_meta_test)
    all_test_history_meta_test = copy(all_test_history_meta_test)
    test_history_meta_test_with_monitor = copy(test_history_meta_test_with_monitor)
    all_test_history_meta_test_with_monitor = copy(all_test_history_meta_test_with_monitor)
    test_history_meta_from_scratch = copy(test_history_meta_from_scratch)
    if len(history_test.predictions) > 1 \
            and len(data_test_monitor.ground_truths()) > 1 \
            and len(history_meta_test_with_monitor.predictions) > 1:
        if len(history_test.predictions) == len(data_test_monitor.ground_truths()):
            # evaluate network performance
            indices = []
            for i, label_i in enumerate(data_test_monitor.ground_truths()):
                if label_i in class_label_map.known_labels():
                    indices.append(i)
            data_test_network = data_test_monitor.filter(filter=indices,
                                                         copy=True)
            # Precision-score for the network based on how well it can predict known classes
            network_score_known = precision_score(
                y_pred=np.array(history_test.predictions)[indices],
                y_true=data_test_network.ground_truths(),
                average='weighted')

            # performance on all classes
            network_score_all = precision_score(
                y_pred=history_test.predictions,
                y_true=data_test_monitor.ground_truths(),
                average='weighted')
            # evaluate monitor performance on all classes
            monitor_precision = history_test.precision()
            with open(_csv_name(STATS_FILE, data_name, options), "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([time.ctime(),
                                 network_score_known,
                                 network_score_all,
                                 history_test._fn,
                                 history_test._fp,
                                 history_test._tn,
                                 history_test._tp,
                                 monitor_precision,
                                 len(class_label_map.known_labels()),
                                 authority_requests,
                                 processed_inputs])

            # evaluate overall accuracy of the classifier without monitor
            network_accuracy_no_monitor = accuracy_score(y_pred=history_meta_test.predictions,
                                                         y_true=history_meta_test.ground_truths)
            # evaluate overall accuracy of the classifier when monitor is active
            network_accuracy_with_monitor = accuracy_score(y_pred=history_meta_test_with_monitor.predictions,
                                                           y_true=history_meta_test_with_monitor.ground_truths)
            # evaluate overall accuracy of the classifier when monitor is active but no authority
            network_accuracy_no_authority = accuracy_score(y_pred=history_meta_test_retrained.predictions,
                                                           y_true=history_meta_test_retrained.ground_truths)
            # evaluate overall accuracy of the classifier trained from scratch
            network_accuracy_from_scratch = accuracy_score(y_pred=history_meta_from_scratch.predictions,
                                                           y_true=history_meta_from_scratch.ground_truths)
            monitor_accuracy = history_meta_test_with_monitor.accuracy()

            # ----------- printing to files ----------------
            with open(_csv_name(META_STATS_FILE, data_name, options), "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [time.ctime(),
                     network_accuracy_no_monitor,
                     network_accuracy_no_authority,
                     network_accuracy_with_monitor,
                     network_accuracy_from_scratch,
                     monitor_accuracy, len(class_label_map.known_labels())])# / options.n_classes_total])

            with open(_csv_name(INTERACTION_STATS_FILE, data_name, options), "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([time.ctime(),
                                 authority_requests,
                                 processed_inputs,
                                 len(class_label_map.known_labels())])# / options.n_classes_total])
    if test and len(test_history_meta_test_with_monitor.predictions) > 1:
        # ---------- on testset only ------------------
        # evaluate overall accuracy of the classifier without monitor
        network_test_accuracy_no_monitor = accuracy_score(y_pred=test_history_meta_test.predictions,
                                                          y_true=test_history_meta_test.ground_truths)
        # evaluate overall accuracy on ALL classes of the classifier without monitor
        all_network_test_accuracy_no_monitor = accuracy_score(y_pred=all_test_history_meta_test.predictions,
                                                              y_true=all_test_history_meta_test.ground_truths)
        # evaluate overall accuracy on known classes of the classifier when monitor is active
        network_test_accuracy_with_monitor = accuracy_score(
            y_pred=test_history_meta_test_with_monitor.predictions,
            y_true=test_history_meta_test_with_monitor.ground_truths)
        # evaluate overall accuracy on ALL classes of the classifier when monitor is active
        all_network_test_accuracy_with_monitor = accuracy_score(
            y_pred=all_test_history_meta_test_with_monitor.predictions,
            y_true=all_test_history_meta_test_with_monitor.ground_truths)
        # evaluate overall accuracy of the classifier trained from scratch
        network_test_accuracy_from_scratch = accuracy_score(y_pred=test_history_meta_from_scratch.predictions,
                                                            y_true=test_history_meta_from_scratch.ground_truths)
        monitor_test_accuracy = test_history_meta_test_with_monitor.accuracy()
        with open(_csv_name(META_TESTSET_FILE, data_name, options), "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [time.ctime(),
                 network_test_accuracy_no_monitor,
                 all_network_test_accuracy_no_monitor,
                 network_test_accuracy_with_monitor,
                 all_network_test_accuracy_with_monitor,
                 network_test_accuracy_from_scratch,
                 monitor_test_accuracy,
                 test_history_meta_test_with_monitor._fn,
                 test_history_meta_test_with_monitor._fp,
                 test_history_meta_test_with_monitor._tn,
                 test_history_meta_test_with_monitor._tp,
                 len(class_label_map.known_labels()),  # / options.n_classes_total,
                 authority_requests,
                 processed_inputs]
            )


def online_loop(original_network, network_builder: NetworkBuilder, monitor_wrapper: MonitorWrapper,
                data_train: DataSpec, data_test: DataSpec,
                data_train_monitor: DataSpec, data_test_monitor: DataSpec, data_name,
                from_scratch_network, class_label_map_all,
                evaluator: Evaluator, authority: Authority, data_stream: DataStream, options: OnlineFrameworkOptions,
                statistics: OnlineStatistics, class_label_map: ClassLabelMap):
    """
    Main loop for online monitoring (incl. learning of monitor and re-training neural network)
    :param class_label_map_all: all labels in the training data used in the network trained from scratch
    :param from_scratch_network: model trained on all classes from scratch
    :param data_name: the name of the original dataset
    :param data_test_monitor: data monitors were tested on (same as network)
    :param data_train_monitor: data monitors were trained on (same as network)
    :param class_label_map: see ClassLabelMap
    :param statistics: all monitor training statistics
    :param original_network: pre-trained neural network
    :param network_builder: builds a neural network from a labeled data set
    :param monitor_wrapper: builds a monitor from a labeled data set and a neural network
    :param data_train: data for training the initial network/monitor (modified in this function)
    :param data_test: data for testing network models (NOT modified in this function)
    :param evaluator: evaluates the monitor performance and can be tweaked
    :param authority: authority that labels a given data set
    :param data_stream: stream of input data
    :param options: options
    """
    statistics.timer_online_total.start()
    statistics.timer_online_until_all_classes_known.start()
    """extract monitor and network information"""
    monitor_manager = monitor_wrapper.monitor_manager
    monitor = monitor_manager.monitor()
    n_initial_samples = data_train_monitor.n()
    n_initial_classes = len(data_train_monitor.classes)
    if len(data_train_monitor.inputs().shape) > 2:
        # images
        input_shape = get_image_shape(data_train_monitor.inputs())
    else:
        # sequential data
        input_shape = int(data_train_monitor.inputs().shape[1])
    # clone original model
    cloned_network = network_builder._model_constructor(weights=None, classes=len(class_label_map),
                                                        input_shape=input_shape)
    # load pre-trained weights.
    cloned_network.load_weights(network_builder.model_path)

    timer_online_until_all_classes_known_runs = True
    adapt_iter = 0
    monitor_iter = 0
    history_test = History()
    history_meta_test = History()
    history_meta_test_retrained = History()
    history_meta_test_with_monitor = History()
    history_meta_from_scratch = History()
    test_history_meta_test = History()
    test_history_meta_from_scratch = History()
    all_test_history_meta_test = History()
    all_test_history_meta_test_with_monitor = History()
    # define a lambda function
    worker_func = lambda: collect_stats(data_name,
                                        history_test,
                                        data_test_monitor,
                                        history_meta_test,
                                        history_meta_test_retrained,
                                        history_meta_test_with_monitor,
                                        history_meta_from_scratch,
                                        test_history_meta_test,
                                        all_test_history_meta_test,
                                        test_history_meta_test_with_monitor,
                                        all_test_history_meta_test_with_monitor,
                                        test_history_meta_from_scratch,
                                        class_label_map,
                                        authority_requests,
                                        processed_inputs,
                                        options=options)
    tmr = TimerClass(count=36000, worker_func=worker_func)
    network_score_all = 0
    authority_requests = 0
    processed_inputs = 0
    open(_csv_name(STATS_FILE, data_name, options), 'w', newline='')
    open(_csv_name(META_STATS_FILE, data_name, options), 'w', newline='')
    open(_csv_name(META_TESTSET_FILE, data_name, options), 'w', newline='')
    open(_csv_name(INTERACTION_STATS_FILE, data_name, options), 'w', newline='')
    """Start online loop"""
    while True:
        # print_general_statistics(statistics, data_train_monitor=data_train, data_run=None, show_running=False)
        history_plot, history_known, history_test, test_history_meta_test_with_monitor, \
        all_labels = _update_training_history(
            cloned_network,
            data_test,
            data_train_monitor,
            data_test_monitor,
            monitor_wrapper,
            class_label_map,
            history_test, adapt_iter)
        # ------------- testset only -------------------
        data_test_model = data_test.filter_by_classes(classes=[ind for ind in data_test.classes
                                                               if ind in class_label_map.known_labels()], copy=True)
        # testset predictions for the original network without transfer learning
        test_predictions_meta_test, _, _ = obtain_predictions(model=original_network, data=data_test_model,
                                                              class_label_map=class_label_map)
        test_history_meta_test.set_ground_truths(data_test_model.ground_truths())
        test_history_meta_test.set_predictions(test_predictions_meta_test)

        # testset predictions for the active network
        test_predictions_meta_test_with_monitor, _, _ = obtain_predictions(model=cloned_network, data=data_test_model,
                                                                           class_label_map=class_label_map)
        test_history_meta_test_with_monitor.set_ground_truths(data_test_model.ground_truths())
        test_history_meta_test_with_monitor.set_predictions(test_predictions_meta_test_with_monitor)

        # testset predictions for the original network on ALL classes without transfer learning
        all_test_predictions_meta_test, _, _ = obtain_predictions(model=original_network, data=data_test,
                                                                  class_label_map=class_label_map)
        all_test_history_meta_test.set_ground_truths(data_test.ground_truths())
        all_test_history_meta_test.set_predictions(all_test_predictions_meta_test)

        # testset predictions for the network trained from scratch
        test_predictions_baseline, _, _ = obtain_predictions(model=from_scratch_network, data=data_test_model,
                                                             class_label_map=class_label_map_all)
        test_history_meta_from_scratch.set_ground_truths(data_test_model.ground_truths())
        test_history_meta_from_scratch.set_predictions(test_predictions_baseline)

        # testset predictions for the active network on ALL classes
        all_test_predictions_with_monitor, _, _ = obtain_predictions(model=cloned_network, data=data_test,
                                                                     class_label_map=class_label_map)
        all_test_history_meta_test_with_monitor.set_ground_truths(data_test.ground_truths())
        all_test_history_meta_test_with_monitor.set_predictions(all_test_predictions_with_monitor)

        # set baselines for statistics:
        # for monitor, the threshold can be set after first novelties accumulate
        # for network, the threshold is set based on the data of known classes
        monitor_wrapper.network_statistics_threshold = .95 * precision_score(y_pred=history_test.predictions,
                                                                             y_true=data_test_monitor.ground_truths(),
                                                                             average='weighted')
        if not adapt_iter:
            data_test_monitor = DataSpec()
            history_test = History()
            tmr.start()
        while True:
            break_loop = False
            """Monitoring mode"""
            collect_stats(data_name,
                          history_test,
                          data_test_monitor,
                          history_meta_test,
                          history_meta_test_retrained,
                          history_meta_test_with_monitor,
                          history_meta_from_scratch,
                          test_history_meta_test,
                          all_test_history_meta_test,
                          test_history_meta_test_with_monitor,
                          all_test_history_meta_test_with_monitor,
                          test_history_meta_from_scratch,
                          class_label_map,
                          authority_requests,
                          processed_inputs,
                          options=options, test=True)
            # obtain new data (one by one)
            # TODO: set up batch processing
            timer_online_batch = StopWatch()
            statistics.timers_online_batch.append(timer_online_batch)
            timer_online_batch.start()
            data_all = data_stream.get(n=options.batch_size)
            if data_all is None:
                # data stream has ended
                tmr.stop()
                statistics.timer_online_total.stop()
                timer_online_batch.stop()
                monitor_wrapper.print_thresholds()
                return  # TODO CS: return some results here?
            n_batch = data_all.n()
            processed_inputs += n_batch

            # run monitoring on given data
            history_run_all = monitor_manager.run(model=cloned_network, data=data_all, statistics=statistics)
            # print_general_statistics(statistics, data_train_monitor=None, data_run=data_all, show_training=False)

            # --- collect meta statistics ---
            # pause timers
            statistics.timer_online_total.stop()
            timer_online_batch.stop()
            if timer_online_until_all_classes_known_runs:
                statistics.timer_online_until_all_classes_known.stop()

            # combine predictions for the original network without transfer learning
            predictions_meta_test, _, _ = obtain_predictions(model=original_network, data=data_all,
                                                             class_label_map=class_label_map)
            predictions_meta_test_combined = history_meta_test.predictions + predictions_meta_test
            # merge history for meta-analysis of the original network without transfer learning
            history_meta_test.set_ground_truths(np.append(history_meta_test.ground_truths,
                                                          data_all.ground_truths()))
            history_meta_test.set_predictions(predictions_meta_test_combined)

            # combine predictions for the network trained from scratch
            predictions_baseline, _, _ = obtain_predictions(model=from_scratch_network, data=data_all,
                                                            class_label_map=class_label_map_all)
            predictions_baseline_combined = history_meta_from_scratch.predictions + predictions_baseline
            # merge history for meta-analysis of the network trained from scratch
            history_meta_from_scratch.set_ground_truths(np.append(history_meta_from_scratch.ground_truths,
                                                                  data_all.ground_truths()))
            history_meta_from_scratch.set_predictions(predictions_baseline_combined)

            # merge layer values for meta-analysis of retrained network without authority labels
            layer2values_retrained_combined = monitor_manager._combine_layer2values(
                history_meta_test_retrained.layer2values,
                history_run_all.layer2values)
            # combine predictions and results for the monitor with previous ones
            predictions_retrained_combined = history_meta_test_retrained.predictions + history_run_all.predictions
            if not history_meta_test_retrained.monitor2results:
                results_retrained_combined = history_run_all.monitor2results[monitor.id()]
            else:
                results_retrained_combined = history_meta_test_retrained.monitor2results[monitor.id()] + \
                                             history_run_all.monitor2results[
                                                 monitor.id()]
            # merge history for meta-analysis of retrained network without authority labels
            history_meta_test_retrained.update_history(layer2values_retrained_combined,
                                                       np.append(history_meta_test_retrained.ground_truths,
                                                                 history_run_all.ground_truths),
                                                       predictions_retrained_combined,
                                                       monitor.id(),
                                                       results_retrained_combined)
            history_meta_test_retrained.add_statistics(new_history=history_run_all)

            # restart timers again
            statistics.timer_online_total.start()
            timer_online_batch.start()
            if timer_online_until_all_classes_known_runs:
                statistics.timer_online_until_all_classes_known.start()
            # --- end of collect meta statistics ---

            # temporary workaround: split batch into minibatches of size 1
            for data_i in range(n_batch):
                # split history_run_all into history_run of size 1
                history_run = History()
                history_run.set_predictions(history_run_all.predictions[data_i:data_i + 1])
                history_run.set_ground_truths(history_run_all.ground_truths[data_i:data_i + 1])
                results = history_run_all.monitor2results[monitor.id()][data_i:data_i + 1]
                history_run.set_monitor_results(m_id=monitor.id(), results=results)
                data = data_all.filter(slice(data_i, data_i + 1), copy=True)
                if history_run_all.layer2values:  # only update if it has a value (e.g., not in alpha-threshold mode)
                    layer2values = dict()
                    for layer in monitor.layers():
                        layer2values[layer] = history_run_all.layer2values[layer][data_i:data_i + 1]
                    history_run.set_layer2values(layer2values)

                has_outsiders = monitor_wrapper.process(data=data,
                                                        monitor=monitor,
                                                        history_run=history_run,
                                                        class_label_map=class_label_map,
                                                        skip_image_plotting=True)

                status = STATUS_ONLINE
                other_labels = []
                all_labels_in_the_stream = []
                if has_outsiders:
                    # monitor would request authority
                    authority_requests += 1

                if has_outsiders and (authority_requests <= options.interaction_limit):
                    # if there are outsider reports and the authority is still available...

                    # Note: since our data is labeled already, we can later forget about these labels
                    authority_labels = authority.label(data=data, monitor_wrapper=monitor_wrapper)
                    all_labels_in_the_stream += list(authority_labels)

                    # merge new data into training data (for later)
                    # TODO CS: add a mechanism to have a bounded amount of data for each class (ideally representative)
                    #   currently we also add data for known classes
                    data_train.merge_inplace(other=data)

                    # add data from the stream to the test data
                    if not data_test_monitor.classes:
                        data_test_monitor = data
                    else:
                        data_test_monitor.merge_inplace(other=data)
                    # merge layer values for testing
                    if monitor_wrapper.skip_computations():
                        layer2values_test_combined = None
                    else:
                        layer2values_test_combined = monitor_manager._combine_layer2values(history_test.layer2values,
                                                                                           history_run.layer2values)
                    # combine predictions and results for the monitor with test ones
                    predictions_test_combined = history_test.predictions + history_run.predictions
                    if not history_test.monitor2results:
                        results_test_combined = history_run.monitor2results[monitor.id()]
                    else:
                        results_test_combined = history_test.monitor2results[monitor.id()] + \
                                                history_run.monitor2results[
                                                    monitor.id()]
                    # merge history for testing
                    history_test.update_history(layer2values_test_combined,
                                                np.append(history_test.ground_truths, authority_labels),
                                                predictions_test_combined,
                                                monitor.id(),
                                                results_test_combined)
                    history_test.add_statistics(new_history=history_run)

                    # identify new labels
                    new_labels = set(authority_labels) - set(class_label_map.known_labels())

                    # new labels have been assigned
                    # evaluate results and decide on next step
                    all_new_labels = set(data_train.classes) - set(class_label_map.known_labels())
                    status, other_labels = evaluator.evaluate(monitor_wrapper=monitor_wrapper,
                                                              new_labels=all_new_labels,
                                                              authority=authority,
                                                              n_initial_samples=n_initial_samples,
                                                              n_initial_classes=n_initial_classes,
                                                              status=status)
                    if status == STATUS_RETRAIN_NETWORK:
                        # evaluate network performance
                        data_test_network = data_test_monitor.filter_by_classes(class_label_map.known_labels(),
                                                                                copy=True)
                        indices = []
                        for i, label_i in enumerate(data_test_monitor.ground_truths()):
                            if label_i in class_label_map.known_labels():
                                indices.append(i)
                        # Precision-score for the network based on how well it can predict known classes
                        monitor_wrapper.network_statistics = precision_score(
                            y_pred=np.array(history_test.predictions)[indices],
                            y_true=data_test_network.ground_truths(),
                            average='weighted')
                        # print("Network Precision Score is {}".format(monitor_wrapper.network_statistics))
                        # performance on all classes
                        network_score_all = precision_score(
                            y_pred=history_test.predictions,
                            y_true=data_test_monitor.ground_truths(),
                            average='weighted')
                    if not new_labels:
                        # no new labels have been assigned but monitor disagrees with the authority -> adapt monitor
                        if monitor_wrapper.skip_computations():
                            layer2values_known_combined = None
                        else:
                            layer2values_known_combined = monitor_manager._combine_layer2values(
                                history_known.layer2values,
                                history_run.layer2values)
                        # combine predictions and results for the monitor with previous ones
                        predictions_known_combined = history_known.predictions + history_run.predictions
                        if not history_known.monitor2results:
                            results_known_combined = history_run.monitor2results[monitor.id()]
                        else:
                            results_known_combined = history_known.monitor2results[monitor.id()] + \
                                                     history_run.monitor2results[
                                                         monitor.id()]
                        # history for adapting abstraction
                        history_known.update_history(layer2values_known_combined,
                                                     np.append(history_known.ground_truths, authority_labels),
                                                     predictions_known_combined,
                                                     monitor.id(),
                                                     results_known_combined)
                        history_known.add_statistics(new_history=history_run)

                        # Precision-score for the monitor based on how well it can predict unknown classes
                        monitor_wrapper.monitor_statistics = history_test.precision()
                        # print("Monitor Precision Score is {}".format(monitor_wrapper.monitor_statistics))
                        if not monitor_wrapper.monitor_statistics_threshold:
                            # the first time monitor statistics can be computed and set as a threshold
                            monitor_wrapper.monitor_statistics_threshold = .9  # * monitor_wrapper.monitor_statistics

                        # evaluate network performance
                        indices = []
                        for i, label_i in enumerate(data_test_monitor.ground_truths()):
                            if label_i in class_label_map.known_labels():
                                indices.append(i)
                        data_test_network = data_test_monitor.filter(filter=indices,
                                                                     copy=True)
                        # Precision-score for the network based on how well it can predict known classes
                        monitor_wrapper.network_statistics = precision_score(
                            y_pred=np.array(history_test.predictions)[indices],
                            y_true=data_test_network.ground_truths(),
                            average='weighted')
                        # print("Network Precision Score is {}".format(monitor_wrapper.network_statistics))
                        # performance on all classes
                        network_score_all = precision_score(
                            y_pred=history_test.predictions,
                            y_true=data_test_monitor.ground_truths(),
                            average='weighted')
                        if monitor_wrapper.network_statistics and \
                                monitor_wrapper.network_statistics < monitor_wrapper.network_statistics_threshold:
                            status, _ = evaluator.evaluate(monitor_wrapper=monitor_wrapper,
                                                           new_labels=class_label_map.known_labels(),
                                                           authority=authority,
                                                           n_initial_samples=n_initial_samples,
                                                           n_initial_classes=n_initial_classes,
                                                           status=status)
                        elif monitor_wrapper.monitor_statistics < monitor_wrapper.monitor_statistics_threshold:
                            """Adapt monitor"""
                            monitor_wrapper.adapt(data_new=data,
                                                  data_train_old=data_train,
                                                  history=history_known,
                                                  sample_threshold=authority.threshold(n_initial_samples,
                                                                                       n_initial_classes),
                                                  statistics=statistics)
                            monitor_iter += 1

                else:
                    all_labels_in_the_stream += list(history_run.predictions)

                # merge layer values for plotting
                if monitor_wrapper.skip_computations():
                    layer2values_plot_combined = None
                else:
                    layer2values_plot_combined = monitor_manager._combine_layer2values(history_plot.layer2values,
                                                                                       history_run.layer2values)
                # combine predictions and results for the monitor with previous ones
                predictions_plot_combined = history_plot.predictions + history_run.predictions
                if not history_plot.monitor2results:
                    results_plot_combined = history_run.monitor2results[monitor.id()]
                else:
                    results_plot_combined = history_plot.monitor2results[monitor.id()] + history_run.monitor2results[
                        monitor.id()]
                # merge history for plotting
                history_plot.update_history(layer2values_plot_combined,
                                            np.append(history_plot.ground_truths, all_labels_in_the_stream),
                                            predictions_plot_combined,
                                            monitor.id(),
                                            results_plot_combined)
                history_plot.add_statistics(new_history=history_run)

                # merge layer values for meta-analysis with monitor
                layer2values_monitor_combined = monitor_manager._combine_layer2values(
                    history_meta_test_with_monitor.layer2values,
                    history_run.layer2values)
                # combine predictions and results for the monitor with previous ones
                predictions_monitor_combined = history_meta_test_with_monitor.predictions + history_run.predictions
                if not history_meta_test_with_monitor.monitor2results:
                    results_monitor_combined = history_run.monitor2results[monitor.id()]
                else:
                    results_monitor_combined = history_meta_test_with_monitor.monitor2results[monitor.id()] + \
                                               history_run.monitor2results[
                                                   monitor.id()]
                # merge history for meta-analysis with monitor
                history_meta_test_with_monitor.update_history(layer2values_monitor_combined,
                                                              np.append(history_meta_test_with_monitor.ground_truths,
                                                                        all_labels_in_the_stream),
                                                              predictions_monitor_combined,
                                                              monitor.id(),
                                                              results_monitor_combined)
                history_meta_test_with_monitor.add_statistics(new_history=history_run)

                """Adaptation mode"""
                if status == STATUS_ONLINE:
                    # no further measures taken
                    pass
                elif status == STATUS_RETRAIN_NETWORK:
                    """Adapt model"""
                    adapt_iter += 1

                    # plot
                    """
                    _plot(monitor_wrapper=monitor_wrapper, history_plot=history_plot,
                          class_label_map=class_label_map, all_labels=data_train.classes,
                          title='Current abstraction and novelties for known classes {} monitor adaptation {}'.format(
                              class_label_map.known_labels(), monitor_iter))
                    """

                    # train network for fairer comparison
                    # if len(class_label_map.known_labels()) == 99 and \
                    #        not other_labels in class_label_map.known_labels():
                    #    _, _, _, _ = network_builder.retrain_network(class_label_map=class_label_map,
                    #                                                 data_train=data_train,
                    #                                                 other_labels=other_labels,
                    #                                                 statistics=statistics)

                    # retrain network
                    cloned_network, data_monitor_combined, data_train_monitor, data_test_monitor = \
                        network_builder.retrain_network(class_label_map=class_label_map,
                                                        data_train=data_train,
                                                        other_labels=other_labels,
                                                        statistics=statistics)
                    # reset samples counter for the newly trained classes
                    for label in other_labels:
                        monitor_wrapper.n_samples[label] = 0
                        # TODO: add check for new accumulated samples even when all classes are known!
                    class_label_map.add_labels(other_labels)
                    monitor_manager.transfer = True  # TODO CS: this is never set to False again

                    # retrain monitor
                    monitor_wrapper.retrain_monitor_from_scratch(data_train=data_monitor_combined,#data_train_monitor,
                                                                 data_test=data_test_monitor,
                                                                 network=cloned_network,
                                                                 known_labels=other_labels,#class_label_map.known_labels(),
                                                                 statistics=statistics)
                    collect_stats(data_name,
                                  history_test,
                                  data_test_monitor,
                                  history_meta_test,
                                  history_meta_test_retrained,
                                  history_meta_test_with_monitor,
                                  history_meta_from_scratch,
                                  test_history_meta_test,
                                  all_test_history_meta_test,
                                  test_history_meta_test_with_monitor,
                                  all_test_history_meta_test_with_monitor,
                                  test_history_meta_from_scratch,
                                  class_label_map,
                                  authority_requests,
                                  processed_inputs,
                                  options=options)

                    # data was split into training and test set for network training; recombine them again
                    data_train_monitor = data_monitor_combined
                    # reset monitor adaptation counter
                    monitor_iter = 0

                    # backtrack
                    n_backtrack = n_batch - data_i - 1
                    data_stream.backtrack(n_backtrack)
                    history_meta_test.cut_off(n_backtrack)
                    history_meta_from_scratch.cut_off(n_backtrack)
                    history_meta_test_retrained.cut_off(n_backtrack)
                    # history_meta_test_with_monitor.cut_off(n_backtrack)
                    break_loop = True
                    if timer_online_until_all_classes_known_runs and \
                            statistics.all_classes_known(n_classes=len(class_label_map)):
                        statistics.timer_online_until_all_classes_known.stop()
                        timer_online_until_all_classes_known_runs = False
                    break
            timer_online_batch.stop()
            if break_loop:
                break


def _csv_name(prefix, data_name, options):
    return prefix + data_name + "_{}_{:d}_{:d}_{:d}.csv".format(options.monitor_name, len(options.classes_initial),
                                                                options.i_model_instance, options.i_run)


def start_log(model_name, monitor_name, n_classes_initial, i_model_instance, i_run):
    name = "log_online_{}_{}_{}_{:d}_{:d}.txt".format(model_name, monitor_name, n_classes_initial, i_model_instance,
                                                      i_run)
    logger = Logger.start(name)
    return logger


def _update_training_history(cloned_network, data_test, data_train_monitor, data_test_monitor,
                             monitor_wrapper: MonitorWrapper,
                             class_label_map, history_test_accumulated, adapt_iter):
    # updating all training history
    history_plot = History()
    monitor_manager = monitor_wrapper.monitor_manager
    monitor = monitor_manager.monitor()
    if monitor_wrapper.skip_computations():
        layer2values_train = None
    else:
        layer2values_train, _, _ = obtain_predictions(model=cloned_network,
                                                      data=data_train_monitor,
                                                      class_label_map=class_label_map,
                                                      layers=monitor_manager.layers(),
                                                      transfer=monitor_manager.transfer)
    if (monitor_manager.layer2n_components is not None) and (monitor_wrapper.alpha_threshold <= 0):
        layer2values_train, layer2components = \
            reduce_dimension(layer2data=layer2values_train,
                             layers=monitor_manager.layers(),
                             layer2n_components=monitor_manager.layer2n_components,
                             layer2components=monitor_manager.layer2components)
    history_plot.update_history(layer2values_train,
                                data_train_monitor.ground_truths())
    # history for all data of known classes we have seen so far
    history_known = History()
    history_known.update_history(layer2values_train,
                                 data_train_monitor.ground_truths())
    all_labels = data_train_monitor.classes

    # history for test data
    history_test = monitor_manager.run(model=cloned_network, data=data_test_monitor, statistics=Statistics())
    history_test.add_statistics(history_test_accumulated)
    #monitor_wrapper.update_history_results(monitor=monitor, history=history_test)

    data_test_model = data_test.filter_by_classes(classes=[ind for ind in data_test.classes
                                                           if ind in class_label_map.known_labels()], copy=True)
    test_history_meta_test_with_monitor = monitor_manager.run(model=cloned_network, data=data_test_model,
                                                              statistics=Statistics())
    monitor_wrapper.update_history_results(monitor=monitor, history=test_history_meta_test_with_monitor)

    # plot
    """
    _plot(monitor_wrapper=monitor_wrapper, history_plot=history_plot,
          class_label_map=class_label_map, all_labels=all_labels,
          title='Current abstraction for known classes {} model retraining {}'.format(class_label_map.known_labels(),
                                                                                      adapt_iter))
    """
    return history_plot, history_known, history_test, \
           test_history_meta_test_with_monitor, \
           all_labels


def _plot(monitor_wrapper, history_plot, class_label_map, all_labels, title):
    if monitor_wrapper.skip_computations():
        return  # skip plotting in alpha-threshold mode

    monitor = monitor_wrapper.monitor_manager.monitor()
    for layer in monitor.layers():
        plot_2d_projection(history=history_plot, monitor=monitor,
                           layer=layer, all_classes=all_labels,
                           class_label_map=class_label_map,
                           category_title=title,
                           dimensions=[0, 1],
                           distance_thresholds=monitor_wrapper.score_thresholds)
        plt.show()
