from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule
import numpy as np
import logging
import pdb


class RootNodeFeatureExtractor(Branchrule):

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.count = 0

    def relative_fractional(self, value):
        fractional_part = self.model.frac(value)
        return fractional_part if fractional_part <= 0.5 else 1 - fractional_part

    def extract_features(self):
        """
        Function for extracting features from the pyscipopt model.
        We only take statis features that do not depend on the LP solution.

        Returns:
            The instance embedding
        """

        # get the variable types, constraint types, (max / average row density), average objective density,

        # We first cycle through the variables (columns).
        cols = self.model.getLPColsData()
        n_cols = self.model.getNLPCols()
        n_ints = 0
        n_binaries = 0
        n_continuous = 0
        n_non_zero_obj_coefficients = 0
        for i in range(n_cols):
            t_var = cols[i].getVar()
            var_type = t_var.vtype()
            assert var_type in ['BINARY', 'INTEGER', 'CONTINUOUS', 'IMPLINT']
            if var_type == "BINARY":
                n_binaries += 1
            if var_type in ["INTEGER", "IMPLINT"]:
                n_ints += 1
            if var_type == "CONTINUOUS":
                n_continuous += 1
            # Get the objective coefficient
            objective_coefficient = cols[i].getObjCoeff()
            if not self.model.isZero(objective_coefficient):
                n_non_zero_obj_coefficients += 1

        assert n_ints + n_binaries + n_continuous == self.model.getNVars()

        self.features['ratio_integers'] = n_ints / n_cols
        self.features['ratio_continuous'] = n_continuous / n_cols
        self.features['ratio_binary'] = n_binaries / n_cols
        self.features['objective_density'] = n_non_zero_obj_coefficients / n_cols

        # Now we cycle through the constraints (rows)
        rows = self.model.getLPRowsData()
        n_rows = self.model.getNLPRows()
        max_density = 0
        sum_density = 0
        n_linear = 0
        n_logicor = 0
        n_knapsack = 0
        n_setppc = 0
        n_varbound = 0
        n_xor = 0
        n_orbitope = 0
        n_and = 0
        for i in range(n_rows):
            # Get the constraint handler responsible for the row
            cons_type_row = rows[i].getConsOriginConshdlrtype()
            assert cons_type_row in ['linear', 'logicor', 'knapsack', 'setppc', 'varbound', 'xor', 'orbitope', 'and']
            if cons_type_row == "linear":
                n_linear += 1
            if cons_type_row == "logicor":
                n_logicor += 1
            if cons_type_row == "knapsack":
                n_knapsack += 1
            if cons_type_row == "setppc":
                n_setppc += 1
            if cons_type_row == "varbound":
                n_varbound += 1
            if cons_type_row == "xor":
                n_xor += 1
            if cons_type_row == "orbitope":
                n_orbitope += 1
            if cons_type_row == "and":
                n_and +=1

            # Get the density of the row
            row_vals = rows[i].getVals()
            sum_density += len(row_vals) / n_cols
            if len(row_vals) / n_cols > max_density:
                max_density = len(row_vals) / n_cols

        self.features['ratio_linear'] = n_linear / n_rows
        self.features['ratio_logicor'] = n_logicor / n_rows
        self.features['ratio_knapsack'] = n_knapsack / n_rows
        self.features['ratio_setppc'] = n_setppc / n_rows
        self.features['ratio_varbound'] = n_varbound / n_rows
        self.features['ratio_xor'] = n_xor / n_rows
        self.features['ratio_oribtope'] = n_orbitope / n_rows
        self.features['ratio_and'] = n_and / n_rows
        self.features['average_row_density'] = sum_density / n_rows
        self.features['max_row_density'] = max_density

        return

    def branchexeclp(self, allowaddcons):
        self.count += 1
        if self.count >= 2:
            logging.error('Dummy branch rule is called after root node and its first child')
            quit()
        assert allowaddcons

        # Assert that the model is not doing anything funny
        assert not self.model.inRepropagation()
        assert not self.model.inProbing()

        # Extract the features of the model()
        self.extract_features()

        # Interrupt the solve. We only wanted features from this, we never wanted to actually branch.
        self.model.interruptSolve()

        # Make a dummy child. This branch rule should only be used at the root node!
        self.model.createChild(1, 1)
        return {"result": SCIP_RESULT.BRANCHED}