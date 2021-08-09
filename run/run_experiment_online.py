from math import inf

from data import get_data_loader
from run.experiment_helper import *
from monitoring.online import *
from monitoring.online.OnlineFramework import _csv_name, start_log

RUN_TIME_STATS_FILE = "run_time_"


def run_experiment_online_defaults():
    # global options
    do_reduce_dimension = True
    random_monitor_acceptance_probability = 0  # if > 0, use a monitor that randomly raises a warning
    use_distance_monitor = True  # flag to use a distance-based monitor (True) or an abstraction-based monitor (False)
    alpha_threshold = 0  # if > 0, use a monitor based on "alpha threshold" instead (abstraction/distance is ignored)
    interaction_limit = inf  # number of interactions with the authority (see the explanation below)
    # - x >= 1 (in particular: 'inf'): interpret the number as the limit
    # - 0 < x < 1: interpret the number as a factor and compute the limit relative to the size of the data stream
    run_range = range(200, 201)  # indices of runs (with corresponding random seed) on the same instance

    # instance options
    instances = [
        # entries: instance constructor, box_abstraction, epsilon, clustering threshold, initial classes, model instance
        (instance_MNIST, box_abstraction_MNIST, 0.0, 0.3, 5, 1),
        (instance_F_MNIST, box_abstraction_F_MNIST, 0.0, 0.07, 5, 1),
        (instance_CIFAR10, box_abstraction_CIFAR10, 0.0, 0.3, 5, 1),
        (instance_GTSRB, box_abstraction_GTSRB, 0.0, 0.3, 22, 1),
        (instance_EMNIST, box_abstraction_EMNIST, 0.0, 0.3, 24, 1),

        #(instance_CIFAR100, box_abstraction_CIFAR100, 0.0, 0.3, 50, -1),
        # (instance_MNIST, box_abstraction_MNIST, 0.0, 0.3, -1, 1),
        # (instance_F_MNIST, box_abstraction_F_MNIST, 0.0, 0.07, -1, 1),
        # (instance_CIFAR10, box_abstraction_CIFAR10, 0.0, 0.3, -1, 1),
        # (instance_GTSRB, box_abstraction_GTSRB, 0.0, 0.3, -1, 1),
        # (instance_Doom, box_abstraction_Doom, 0.0, 0.07, 3, 1),
        # (instance_MELMAN, box_abstraction_MELMAN, 0.0, 0.07, 2, 1)
    ]

    return run_experiment_online(do_reduce_dimension=do_reduce_dimension, use_distance_monitor=use_distance_monitor,
                                 alpha_threshold=alpha_threshold,
                                 random_monitor_acceptance_probability=random_monitor_acceptance_probability,
                                 instances=instances, run_range=run_range, interaction_limit=interaction_limit)


def run_experiment_online(do_reduce_dimension, use_distance_monitor, alpha_threshold,
                          random_monitor_acceptance_probability, instances, run_range, interaction_limit=inf,
                          authority_threshold_percentage=AUTHORITY_THRESHOLD_PERCENTAGE,
                          reduction_method=REDUCTION_METHOD, initial_distance_threshold=INITIAL_DISTANCE_THRESHOLD,
                          adapt_score_thresholds=True):
    # sanity check
    if random_monitor_acceptance_probability > 0 and alpha_threshold > 0:
        raise (ValueError("Can only choose either random monitor or alpha-threshold monitor"))

    for instance_function, monitor_constructor, epsilon, clustering_threshold, classes_initial, i_model_instance in \
            instances:
        model_name, data_name, stored_network_name, n_total_classes, flatten_layer, optimizer = instance_function(
            transfer=True)
        if isinstance(classes_initial, int):
            n_classes_initial = classes_initial
            classes_initial = [k for k in range(classes_initial)]
        elif isinstance(classes_initial, list):
            n_classes_initial = len(classes_initial)
        else:
            raise(ValueError("Illegal value", classes_initial))
        for i_run in run_range:
            seed = i_run
            # create (fresh) monitors
            monitor, monitor_name = _construct_monitor_prelude(monitor_constructor=monitor_constructor,
                                                               use_distance_monitor=use_distance_monitor,
                                                               alpha_threshold=alpha_threshold,
                                                               random_probability=random_monitor_acceptance_probability,
                                                               epsilon=epsilon)
            monitors = [monitor]

            logger = start_log(model_name=model_name, monitor_name=monitor_name, n_classes_initial=n_classes_initial,
                               i_model_instance=i_model_instance, i_run=i_run)
            print("\n--- run {:d} on {} dataset ---\n".format(i_run, data_name))
            statistics = OnlineStatistics(total_classes=n_total_classes)

            # load instance
            data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, \
                _, _ = load_instance(classes_initial, n_total_classes, stored_network_name)

            model_constructor = get_model_loader(model_name, model_path)
            full_model_path = MODEL_INSTANCE_PATH + model_path
            network_builder = NetworkBuilder(model_path=full_model_path, model_constructor=model_constructor,
                                             model_name=model_name, data_name=data_name,
                                             flatten_layer=flatten_layer, optimizer=optimizer)

            if do_reduce_dimension and alpha_threshold <= 0 and random_monitor_acceptance_probability <= 0:
                layer2n_components = dict()
            else:
                layer2n_components = None
            monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold,
                                             layer2n_components=layer2n_components,
                                             skip_confidence=True,  # important for abstraction-based monitor
                                             fit_distribution_method=None,
                                             alpha_thresholding=alpha_threshold,
                                             reduction_method=reduction_method)
            # train the initial monitor
            model, _, class_label_map = \
                construct_monitor(seed=seed, data_name=data_name, data_train_model=data_train_model,
                                  data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                                  data_test_monitor=data_test_monitor, data_run=data_run, model_name=model_name,
                                  model_path=model_path, monitor_manager=monitor_manager,
                                  n_classes_total=n_total_classes, statistics=statistics)

            # prepare for online mode
            monitor_wrapper = MonitorWrapper(monitor_manager=monitor_manager, alpha_threshold=alpha_threshold,
                                             initial_distance_threshold=initial_distance_threshold,
                                             adapt_score_thresholds=adapt_score_thresholds)
            # define threshold distances for the monitor based on the initial abstraction
            for class_label in class_label_map.l2c:
                monitor_wrapper.score_thresholds[class_label] = monitor_wrapper.initial_distance_threshold

            # use run data for known classes
            data_run_known = data_run.filter_by_classes(classes=[ind for ind in data_run.classes
                                                                 if ind in class_label_map.known_labels()],
                                                        copy=True)
            # plus train data for unknown classes
            data_combined = DataSpec(randomize=False,
                                     classes=data_run_known.classes,
                                     inputs=data_run_known._inputs,
                                     labels=data_run_known._labels)
            # load instance for streaming
            data_train_stream, data_test_stream, data_train_stream_monitor, data_test_stream_monitor, \
            data_run_stream, model_path_stream, transfer_model_path_stream, _ = \
                load_instance(n_total_classes, n_total_classes, stored_network_name)
            # load data for streaming
            _, _ = get_data_loader(data_name)(
                data_train_model=data_train_stream,
                data_test_model=data_test_stream,
                data_train_monitor=data_train_stream_monitor,
                data_test_monitor=data_test_stream_monitor,
                data_run=data_run_stream,
                adversarial_data_suffix=None)

            # data_combined.merge_inplace(
            #    other=data_train_stream.filter_by_classes(classes=[ind for ind in data_run.classes
            #                                                       if ind not in class_label_map.known_labels()],
            #                                             copy=True))
            # data_combined = data_train_stream.filter_by_classes(classes=[ind for ind in data_run.classes
            #                                                             if ind not in class_label_map.known_labels()],
            #                                                    copy=True)
            # data_combined.shuffle()
            # data_stream = DataStream(data_combined)
            data_train_stream.shuffle()
            data_stream = DataStream(data_train_stream)
            authority = Authority(threshold_percentage=authority_threshold_percentage)
            evaluator = Evaluator()
            if interaction_limit >= 1:
                pass
            elif interaction_limit > 0:
                interaction_limit *= data_train_stream.n()
            else:
                raise(ValueError("interaction limit {} is not a valid input".format(interaction_limit)))
            print("authority is limited to {} interactions".format(interaction_limit))
            options = OnlineFrameworkOptions(classes_initial=classes_initial, n_classes_total=n_total_classes,
                                             batch_size=DATA_BATCH_SIZE, i_run=i_run, i_model_instance=i_model_instance,
                                             monitor_name=monitor_name, interaction_limit=interaction_limit)
            total_classes_string = classes2string(data_train_stream.classes)  # classes2string(data_combined.classes)

            # baseline network for comparison
            from_scratch_network, class_label_map_all = _baseline_network(model_constructor=model_constructor,
                                                                          instance_function=instance_function,
                                                                          total_classes_string=total_classes_string,
                                                                          data=data_train_stream)
            # data=data_combined)
            # run online mode
            online_loop(original_network=model, statistics=statistics, monitor_wrapper=monitor_wrapper,
                        network_builder=network_builder, data_train=data_train_model,
                        data_test_monitor=data_test_monitor, data_name=model_name,
                        from_scratch_network=from_scratch_network, class_label_map_all=class_label_map_all,
                        data_stream=data_stream, data_test=data_test_stream, data_train_monitor=data_train_monitor,
                        class_label_map=class_label_map,
                        authority=authority, evaluator=evaluator, options=options)

            # store results
            statistics.assert_termination()
            print(statistics)
            with open(_csv_name(RUN_TIME_STATS_FILE, model_name, options), "w", newline="") as file:
                writer = csv.writer(file)
                statistics.write_csv(writer)
            logger.stop()


def _baseline_network(model_constructor, instance_function, total_classes_string, data):
    if len(data.inputs().shape) > 2:
        # images
        input_shape = get_image_shape(data.inputs())
    else:
        # sequential data
        input_shape = int(data.inputs().shape[1])
    _, _, from_scratch_stored_network_name, _, _, _ = instance_function(transfer=False)
    from_scratch_model_path = "{}{}_{}.h5".format(MODEL_INSTANCE_PATH,
                                                  from_scratch_stored_network_name,
                                                  total_classes_string)
    class_label_map_all = class_label_map_from_labels(data.classes)
    # clone model trained on all classes
    from_scratch_network = model_constructor(weights=None, classes=len(class_label_map_all), input_shape=input_shape)
    # load pre-trained weights
    from_scratch_network.load_weights(from_scratch_model_path)
    return from_scratch_network, class_label_map_all


def _construct_monitor_prelude(monitor_constructor, use_distance_monitor, alpha_threshold,
                               random_probability, epsilon):
    raw_monitor = monitor_constructor(epsilon)
    if random_probability > 0:
        print("Choosing a randomized monitor")
        monitor = RandomMonitor(raw_monitor, random_probability)
        monitor_name = "random"
    elif alpha_threshold > 0:
        print("Choosing a (alpha) threshold monitor")
        monitor = raw_monitor
        monitor_name = "alpha"
    elif use_distance_monitor:
        distance_fun = None
        monitor_name = "distance"
        for layer in raw_monitor.layers():
            abstraction = raw_monitor.abstraction(layer)
            if isinstance(abstraction, BoxAbstraction):
                print("Choosing box distance in the monitor")
                distance_fun = box_distance_parameter
                monitor_name = "box-distance"
            elif isinstance(abstraction, MeanBallAbstraction):
                print("Choosing Euclidean distance in the monitor")
                distance_fun = euclidean_distance_parameter
                monitor_name = "euclidean-distance"
            else:
                raise (ValueError("Could not choose a distance function!"))
            break
        if distance_fun is None:
            raise (ValueError("Could not choose a distance function!"))
        monitor = DistanceMonitor(raw_monitor, distance_fun)
    else:
        print("Choosing abstraction monitor")
        monitor = raw_monitor
        monitor_name = "abstraction"
    return monitor, monitor_name


def plot_experiment_online():
    # instance options
    instances = [
        # ("MNIST", 8, None),
        # ("F_MNIST", 8, None),
        # ("CIFAR10", 8, None),
        # ("GTSRB", 41, None),
        # ("Doom", 3, None),
        # ("MELMAN", 3, None)
    ]

    for data_name, n_ticks, n_bars in instances:
        filename_prefix = "novelty_" + data_name
        storage_all = load_core_statistics("monitor", filename_prefix=filename_prefix)

        plot_false_decisions_given_all_lists(storage_all, n_ticks=n_ticks, name=filename_prefix, n_bars=n_bars)

    plt.show()
    save_all_figures(close=True)


def run_experiment_online_all():
    run_experiment_online_defaults()
    # plot_experiment_online()


if __name__ == "__main__":
    run_experiment_online_all()
