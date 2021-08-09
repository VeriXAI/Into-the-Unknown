from copy import copy

from utils import DataSpec


class DataStream(object):
    def __init__(self, data: DataSpec):
        self._data = data
        self._index = 0

    def get(self, n=1):
        lo = self._index
        self._index = min(self._index + n, self._data.n())
        hi = self._index
        if hi <= lo:
            return None  # no data remaining
        data = self._data.filter_range(lo=lo, hi=hi, copy=True)
        return data

    def backtrack(self, n):
        assert n >= 0
        print("backtracking {:d} data points in stream".format(n))
        self._index -= n
        assert 0 <= self._index < self._data.n()
