from utils import *


class OnlineStatistics(Statistics):
    def __init__(self, total_classes):
        super().__init__()
        self.total_classes = total_classes
        self.timer_online_total = StopWatch()
        self.timer_online_until_all_classes_known = StopWatch()
        self.timer_online_retraining_networks = StopWatch()
        self.timer_online_retraining_monitors = StopWatch()
        self.timer_online_adapting_monitors = StopWatch()
        self.timers_online_batch = []

    def __str__(self):
        string = "FINAL ONLINE STATISTICS"
        string += "\n- total time: " + str(self.timer_online_total.total())
        if self.timer_online_until_all_classes_known.runs():
            string += "\n- not all classes learned"
        else:
            string += "\n- time until all classes were known: " + str(self.timer_online_until_all_classes_known.total())
        string += "\n- time retraining networks: " + str(self.timer_online_retraining_networks.total())
        string += "\n- time retraining monitors: " + str(self.timer_online_retraining_monitors.total())
        string += "\n- time adapting monitors: " + str(self.timer_online_adapting_monitors.total())
        string += "\n- time without retraining or adapting: " + str(self.time_online_without_retraining_or_adapting())
        string += "\n- time batches:"
        for i, timer_i in enumerate(self.timers_online_batch):
            string += "\n - {:d}: ".format(i) + str(timer_i.total())
        return string

    def time_online_without_retraining_or_adapting(self): \
        return self.timer_online_total.total() \
               - self.timer_online_retraining_networks.total() \
               - self.timer_online_retraining_monitors.total() \
               - self.timer_online_adapting_monitors.total()

    def all_classes_known(self, n_classes):
        return n_classes == self.total_classes

    def assert_termination(self):
        assert not self.timer_online_total.runs()
        # assert not self.timer_online_until_all_classes_known.runs()  # this one is allowed to not terminate
        assert not self.timer_online_retraining_networks.runs()
        assert not self.timer_online_retraining_monitors.runs()
        assert not self.timer_online_adapting_monitors.runs()
        for timer_batch in self.timers_online_batch:
            assert not timer_batch.runs()

    def write_csv(self, writer):
        row = list()
        row.append(self.timer_online_total.total())
        if self.timer_online_until_all_classes_known.runs():
            row.append("---")
        else:
            row.append(self.timer_online_until_all_classes_known.total())
        row.append(self.timer_online_retraining_networks.total())
        row.append(self.timer_online_retraining_monitors.total())
        row.append(self.timer_online_adapting_monitors.total())
        row.append(self.time_online_without_retraining_or_adapting())
        writer.writerow(row)
        row = list()
        for i, timer_i in enumerate(self.timers_online_batch):
            row.append(timer_i.total())
        writer.writerow(row)
