"""
Implement constrained Bayesian optimizer for our test.
"""
import numpy as np
import vabo
from scipy.stats import norm
from .base_contextual_optimizer import BaseContextualBO


class ConstrainedContextualBO(BaseContextualBO):

    def __init__(self, opt_problem, constrained_contextual_BO_config):
        # optimization problem and measurement noise
        super(ConstrainedContextualBO, self).__init__(
            constrained_contextual_BO_config)
        self.constrained_bo = vabo.ConstrainedBO(
            opt_problem, constrained_contextual_BO_config)

    def make_step(self, contextual_vars):
        y_obj, constr_vals = \
            self.non_context_bo_make_step(
                contextual_vars, self.constrained_bo)
        return y_obj, constr_vals
