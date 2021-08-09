from math import inf


class OnlineFrameworkOptions(object):
    def __init__(self, classes_initial, n_classes_total, batch_size, i_run, i_model_instance, monitor_name,
                 interaction_limit=inf):
        self.classes_initial = classes_initial  # starting number of known classes
        self.n_classes_total = n_classes_total  # total number of classes
        self.batch_size = batch_size  # batch size
        self.i_run = i_run  # run instance
        self.i_model_instance = i_model_instance  # model instance
        self.monitor_name = monitor_name  # identifier of the monitor (for logs and file names)
        self.interaction_limit = interaction_limit
