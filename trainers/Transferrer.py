from utils import *
from data import *
from trainers import StandardTrainer


def run_transferring(data_name, data_train_model, data_test_model, model_name, model_path, transfer_model_path,
                     n_epochs, batch_size,
                 model_trainer=StandardTrainer(), seed=0, plot_name=""):
    # set random seed
    set_random_seed(seed)

    # construct statistics wrapper
    statistics = Statistics()

    # load data
    data_train_monitor = DataSpec()
    data_test_monitor = DataSpec()
    data_run = DataSpec()
    class_label_map, _ = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    print("Data: classes {} with {:d} inputs (model training), classes {} with {:d} inputs (model test)".format(
        classes2string(data_train_model.classes), data_train_model.n(), classes2string(data_test_model.classes),
        data_test_model.n()))

    # create and train network model
    all_known_labels = class_label_map.known_labels()
    del all_known_labels[-1]
    class_label_map_reduced = class_label_map_from_labels(all_known_labels)
    model, history_model = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                                     data_test=data_test_model, class_label_map=class_label_map_reduced,
                                     model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                                     statistics=statistics, transfer=True, transfer_model_name=transfer_model_path)

    # plot history
    if plot_name is not None:
        plot_model_history(history_model)
        if plot_name != "":
            figs = [plt.figure(n) for n in plt.get_fignums()]
            if len(figs) == 1:
                figs[0].savefig("../transferring_{}.pdf".format(plot_name))
                plt.close()
            else:
                save_all_figures(figs)
                plt.close("all")

    return model, statistics, history_model
