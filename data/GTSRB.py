from utils import DataSpec, load_data, DATA_PATH


def load_GTSRB(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
               data_test_monitor: DataSpec, data_run: DataSpec, adversarial_data_suffix=None):
    # names of the data files
    data_train_model.file = DATA_PATH + "GTSRB/train.p"
    data_test_model.file = DATA_PATH + "GTSRB/test.p"
    data_train_monitor.file = data_train_model.file  # use training data for training
    data_test_monitor.file = data_test_model.file  # use testing data for running
    if adversarial_data_suffix is not None:
        data_run.file = DATA_PATH + "GTSRB/adversarial{}".format(adversarial_data_suffix)
    else:
        data_run.file = data_test_model.file  # use testing data for running

    pixel_depth = 255.0

    class_label_map, all_labels = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth,
        is_adversarial_data=(adversarial_data_suffix is not None))

    # labels (not used anymore)
    # labels_all = ['label' + str(i) for i in range(43)]  # dummy names, TODO add correct names
    # labels_all[0] = "20 km/h"
    # labels_all[1] = "30 km/h"
    # labels_all[2] = "50 km/h"
    # labels_all[10] = "no passing"

    return class_label_map, all_labels
