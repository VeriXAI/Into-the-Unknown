from monitoring import *
from utils import *
from data import *
from trainers import *


def construct_monitor(model_name, model_path, data_name, data_train_model, data_test_model, data_train_monitor,
                      data_test_monitor, data_run, monitor_manager: MonitorManager, n_classes_total, alphas=None,
                      model_trainer=StandardTrainer(), seed=0, n_epochs=-1, batch_size=-1,
                      adversarial_data_suffix=None, statistics=Statistics()):
    # set random seed
    set_random_seed(seed)

    # load data
    class_label_map, _ = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, adversarial_data_suffix=adversarial_data_suffix)

    # load network model or create and train it
    model, history_model = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                                     data_test=data_test_model, class_label_map=class_label_map,
                                     model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                                     statistics=statistics)

    print_data_information(data_train_monitor, data_test_monitor, data_run)

    # normalize and initialize monitors
    monitor_manager.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=n_classes_total)

    # train monitors
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=statistics)
    return model, statistics, class_label_map
