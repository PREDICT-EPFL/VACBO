import numpy as np
import GPy
import safeopt
import vabo
import copy
from matplotlib import pyplot as plt
"""
Define and implement the class of optimization problem.
"""


class ContextualOptimizationProblem(vabo.OptimizationProblem):

    def __init__(self, config):
        super(ContextualOptimizationProblem, self).__init__(config)
        self.get_context = config['get_context'] 
