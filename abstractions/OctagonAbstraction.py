from .SetBasedAbstraction import SetBasedAbstraction
from abstractions.sets.Octagon import Octagon
from utils.Options import *


class OctagonAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=USE_EPSILON_RELATIVE):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)

    def name(self):
        return "Octagon"

    def set_type(self):
        return Octagon
