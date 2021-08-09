class AuthorityDataThreshold(object):
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper

    def can_train(self, n):
        return n >= self._lower

    def must_train(self, n):
        return n >= self._upper
