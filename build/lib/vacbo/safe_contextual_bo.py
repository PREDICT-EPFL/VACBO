"""
Implement constrained Bayesian optimizer for our test.
"""
import numpy as np
import vabo
from scipy.stats import norm
from .base_contextual_optimizer import BaseContextualBO


class SafeContextualBO(BaseContextualBO):

    def __init__(self, opt_problem, safe_contextual_BO_config):
        # optimization problem and measurement noise
        super(SafeContextualBO, self).__init__(
            safe_contextual_BO_config)
        self.safe_bo = vabo.SafeBO(
            opt_problem, safe_contextual_BO_config)

    def make_step(self, contextual_vars):
        y_obj, constr_vals = \
            self.non_context_bo_make_step(
                contextual_vars, self.safe_bo)
        return y_obj, constr_vals
