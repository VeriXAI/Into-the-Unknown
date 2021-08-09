from time import time


class StopWatch(object):
    def __init__(self):
        self._sum = 0.0
        self._start = -1.0

    def __str__(self):
        return "{} watch (sum={:f})".format("running" if self.runs() else "stopped", self._sum)


    def start(self):
        assert not self.runs(), "Stop watch is still running"
        self._start = time()

    def stop(self):
        assert self.runs(), "Stop watch is not running"
        self._sum += self.lap()
        self._start = -1.0

    def lap(self):
        assert self.runs(), "Stop watch is not running"
        return time() - self._start

    def sum(self):
        return self._sum

    def total(self):
        if self.runs():
            return self.sum() + self.lap()
        else:
            return self.sum()

    def runs(self):
        return self._start >= 0.0
