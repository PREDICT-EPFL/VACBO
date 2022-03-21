"""
Implement optimizer base class.
"""
import numpy as np
import safeopt
import GPy
import vabo


class BaseContextualBO:

    def __init__(self, base_contextual_config, reverse_meas=False):
        self.contextual_var_ids = base_contextual_config[
            'contextual_var_ids']
        self.contextual_var_num = len(self.contextual_var_ids)

    def modify_contextual_vars(self, contextual_vars, non_contextual_bo):
        # assert type(contextual_vars) == np.ndarray
        # assert contextual_vars.ndim == 1

        # modify the contextual variables
        contextual_var_ids = self.contextual_var_ids
        non_contextual_bo.parameter_set[:, contextual_var_ids] = \
            contextual_vars

    def non_context_bo_make_step(self, contextual_vars, non_contextual_bo):
        self.modify_contextual_vars(contextual_vars, non_contextual_bo)
        y_obj, constr_vals = non_contextual_bo.make_step()
        return y_obj, constr_vals


