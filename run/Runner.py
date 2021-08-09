from utils import *
from data import *


def run(seed, data_name, data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run,
        model_trainer, model_name, model_path, n_epochs, batch_size, monitor_manager,
        confidence_thresholds=None, skip_image_plotting=False, show_statistics=True):
    if confidence_thresholds is None:
        confidence_thresholds = [0.0]

    # set random seed
    set_random_seed(seed)

    # construct statistics wrapper
    statistics = Statistics()

    # load data
    class_label_map, all_labels = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    print("Number of data points:", data_train_model.n(), "(model training),", data_test_model.n(), "(model test),",
          data_train_monitor.n(), "(monitor training),", data_test_monitor.n(), "(monitor test),", data_run.n(),
          "(run)")
    print("Chosen classes from the original data set:", data_train_model.classes, "(model training),",
          data_test_model.classes, "(model test),", data_train_monitor.classes, "(monitor training),",
          data_test_monitor.classes, "(monitor test),", data_run.classes, "(run)")

    # load network model or create and train it
    model, history_model = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                                     data_test=data_test_model, class_label_map=class_label_map,
                                     model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                                     statistics=statistics)

    # plot history
    plot_model_history(history_model)

    # normalize and initialize monitors
    n_classes_total = len(all_labels)
    monitor_manager.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=n_classes_total)

    # train monitors
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=statistics)

    # run monitors
    history_run = monitor_manager.run(model=model, data=data_run, statistics=statistics)

    print("\n--- session finished ---\n")

    if show_statistics:
        _show_statistics(confidence_thresholds, data_run, data_test_monitor, data_train_model, data_train_monitor,
                         history_run, class_label_map, all_labels,
                         monitor_manager, n_epochs, skip_image_plotting, statistics)
    else:
        return model, history_run, class_label_map, all_labels, statistics


def _show_statistics(confidence_thresholds, data_run, data_test_monitor, data_train_model, data_train_monitor,
                     history_run, class_label_map, all_labels, monitor_manager,
                     n_epochs, skip_image_plotting, statistics):
    # collect novelties
    novelty_wrapper = history_run.novelties(data_run, class_label_map=class_label_map, all_labels=all_labels)
    # print statistics
    print_statistics(statistics=statistics, monitor_manager=monitor_manager,
                     n_train_model=data_train_model.n(), n_train_monitor=data_train_monitor.n(),
                     n_test_monitor=data_test_monitor.n(), n_run=data_run.n(), epochs=n_epochs,
                     novelty_wrapper=novelty_wrapper, history=history_run, confidence_thresholds=confidence_thresholds)
    # plot histograms
    plot_histograms(monitor_manager=monitor_manager, data_train_monitor=data_train_monitor,
                    layer2all_trained_values=None)
    # plot monitor performance
    plot_novelty_detection(monitors=monitor_manager.monitors(), novelty_wrapper=novelty_wrapper,
                           confidence_thresholds=confidence_thresholds)
    plot_false_decisions(monitors=monitor_manager.monitors(), history=history_run,
                         confidence_thresholds=confidence_thresholds)
    # plot projection of running data
    for monitor in monitor_manager.monitors():
        for layer in monitor.layers():
            plot_2d_projection(history=history_run, monitor=monitor, layer=layer, category_title="final run",
                               all_classes=all_labels, class_label_map=class_label_map)
    # plot warning and novelty images
    if not skip_image_plotting:
        for monitor in monitor_manager.monitors():
            warn_images = history_run.warnings(monitor, data_run)
            plot_images(images=warn_images, labels=all_labels, classes=data_run.classes, iswarning=True,
                        monitor_id=monitor.id())
            novel_images = novelty_wrapper.evaluate_detection(monitor.id())
            plot_images(images=novel_images, labels=all_labels, classes=data_run.classes, iswarning=False,
                        monitor_id=monitor.id())
    print("\nDone! In order to keep the plots alive, this program does not terminate until they are closed.")
    plt.show()
