"""
Implement constrained Bayesian optimizer for our test.
"""
import vabo
from .base_contextual_optimizer import BaseContextualBO


class ViolationAwareContextualBO(BaseContextualBO):

    def __init__(self, opt_problem, violation_aware_contextual_BO_config):
        # optimization problem and measurement noise

        # for contextual optimization, use LCB acquisition function instead
        # of CEI acquisition
        violation_aware_contextual_BO_config['acq_func_type'] = 'LCB'
        super(ViolationAwareContextualBO, self).__init__(
            violation_aware_contextual_BO_config)
        self.violation_aware_bo = vabo.ViolationAwareBO(
            opt_problem, violation_aware_contextual_BO_config)

    def make_step(self, contextual_vars):
        y_obj, constr_vals = \
            self.non_context_bo_make_step(
                contextual_vars, self.violation_aware_bo)
        return y_obj, constr_vals
