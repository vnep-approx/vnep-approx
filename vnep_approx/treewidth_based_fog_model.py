# MIT License
#
# Copyright (c) 2019 Balazs Nemeth
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import time
from gurobipy import GRB

import treewidth_model as twm
from alib import datamodel


class RandRoundSepLPOptDynVMPCollectionForFogModelResult(twm.RandRoundSepLPOptDynVMPCollectionResult):

    def __init__(self, scenario, lp_computation_information):
        # TODO (NB): use it, even if the same info should be stored??
        super(RandRoundSepLPOptDynVMPCollectionForFogModelResult, self).__init__(scenario, lp_computation_information)


class RandRoundSepLPOptDynVMPCollectionForFogModel(twm.RandRoundSepLPOptDynVMPCollection):

    ALGORITHM_ID = "RandRoundSepLPOptDynVMPCollectionForFogModel"

    def __init__(self,
                 scenario,
                 rounding_order_list,
                 lp_recomputation_mode_list,
                 lp_relative_quality,
                 rounding_samples_per_lp_recomputation_mode,
                 number_initial_mappings_to_compute,
                 number_further_mappings_to_add,
                 gurobi_settings=None,
                 logger=None):
        super(RandRoundSepLPOptDynVMPCollectionForFogModel, self).__init__(scenario,
                                                                             rounding_order_list,
                                                                             lp_recomputation_mode_list,
                                                                             lp_relative_quality,
                                                                             rounding_samples_per_lp_recomputation_mode,
                                                                             number_initial_mappings_to_compute,
                                                                             number_further_mappings_to_add,
                                                                             gurobi_settings,
                                                                             logger)

    def check_supported_objective(self):
        '''Raised ValueError if a not supported objective is found in the input scenario
        :return:
        '''
        if self.objective == datamodel.Objective.MAX_PROFIT:
            raise ValueError("The separation LP for algorithm Fog model can only handle min-cost instances.")
        elif self.objective == datamodel.Objective.MIN_COST:
            pass
        else:
            raise ValueError("The separation LP for algorithm Fog model can only handle min-cost instances.")

    def create_empty_objective(self):
        # TODO (NB): according to Matthias' thesis, this should be inicialized by solving a profit version with all profits set to 1.
        self.model.setObjective(10, GRB.MINIMIZE)

    def update_dual_costs_and_reinit_dynvmps(self):
        #update dual costs
        # TODO (NB): this should be done differently
        for snode in self.snodes:
            self.dual_costs_node_resources[snode] = self.capacity_constraints[snode].Pi
        for sedge in self.sedges:
            self.dual_costs_edge_resources[sedge] = self.capacity_constraints[sedge].Pi
        for req in self.requests:
            self.dual_costs_requests[req] = self.embedding_bound[req].Pi

        #reinit dynvmps
        for req in self.requests:
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_reinit_time = time.time()
            dynvmp_instance.reinitialize(new_node_costs=self.dual_costs_node_resources,
                                         new_edge_costs=self.dual_costs_edge_resources)
            self.dynvmp_runtimes_initialization[req].append(time.time() - single_dynvmp_reinit_time)

    def perform_separation_and_introduce_new_columns(self, current_objective, ignore_requests=[]):
        new_columns_generated = False
        total_dual_violations = 0
        for req in self.requests:
            if req in ignore_requests:
                continue
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                dual_violation_of_req = (opt_cost - req.profit + self.dual_costs_requests[req])
                if dual_violation_of_req < 0:
                    total_dual_violations += (-dual_violation_of_req)

        self._current_obj_bound = current_objective + total_dual_violations

        self._current_solution_quality = None
        if abs(current_objective) > 0.00001:
            self._current_solution_quality = total_dual_violations / current_objective
        else:
            self.logger.info("WARNING: Current objective is very close to zero; treating it as such.")
            self._current_solution_quality = total_dual_violations

        self.logger.info("\nCurrent LP solution value is {:10.5f}\n"
                           "Current dual upper bound is  {:10.5f}\n"
                         "Accordingly, current solution is at least {}-optimal".format(current_objective, current_objective+total_dual_violations, 1+self._current_solution_quality))

        if self._current_solution_quality < self.lp_relative_quality:
            self.logger.info("Ending LP computation as found solution is {}-near optimal, which lies below threshold of {}".format(self._current_solution_quality, self.lp_relative_quality))
            return False
        else:

            for req in self.requests:
                if req in ignore_requests:
                    continue
                dynvmp_instance = self.dynvmp_instances[req]
                opt_cost = dynvmp_instance.get_optimal_solution_cost()
                # TODO (NB): this should be done differently
                if opt_cost is not None and opt_cost < 0.999*(req.profit - self.dual_costs_requests[req]):
                    self.introduce_new_columns(req,
                                               maximum_number_of_columns_to_introduce=self.number_further_mappings_to_add,
                                               cutoff = req.profit-self.dual_costs_requests[req])
                    new_columns_generated = True

            return new_columns_generated

    def introduce_new_columns(self, req, maximum_number_of_columns_to_introduce=None, cutoff=twm.INFINITY):
        dynvmp_instance = self.dynvmp_instances[req]
        opt_cost = dynvmp_instance.get_optimal_solution_cost()
        current_new_allocations = []
        current_new_variables = []
        self.logger.debug("Objective when introucding new columns was {}".format(opt_cost))
        (costs, indices) = dynvmp_instance.get_ordered_root_solution_costs_and_mapping_indices(maximum_number_of_solutions_to_return=maximum_number_of_columns_to_introduce)
        mapping_list = dynvmp_instance.recover_list_of_mappings(indices)
        self.logger.debug("Will iterate mapping list {}".format(req.name))
        for index, mapping in enumerate(mapping_list):
            if costs[index] > cutoff:
                break
            #store mapping
            # TODO (NB): this should be done differently
            varname = "f_req[{}]_k[{}]".format(req.name, index+len(self.mappings_of_requests[req]))
            new_var = self.model.addVar(lb=0.0,
                                        ub=1.0,
                                        obj=req.profit,
                                        vtype=GRB.CONTINUOUS,
                                        name=varname)
            current_new_variables.append(new_var)

            #compute corresponding substrate allocation and store it
            mapping_allocations = self._compute_allocations(req, mapping)
            current_new_allocations.append(mapping_allocations)

        #make variables accessible
        self.model.update()
        for index, new_var in enumerate(current_new_variables):
            #handle allocations
            corresponding_allocation = current_new_allocations[index]
            for sres, alloc in corresponding_allocation.iteritems():
                if sres not in self.capacity_constraints.keys():
                    continue
                constr = self.capacity_constraints[sres]
                self.model.chgCoeff(constr, new_var, alloc)

            self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)

        self.mappings_of_requests[req].extend(mapping_list[:len(current_new_variables)])
        self.allocations_of_mappings[req].extend(current_new_allocations[:len(current_new_variables)])
        self.mapping_variables[req].extend(current_new_variables)
        self.logger.debug("Introduced {} new mappings for {}".format(len(current_new_variables), req.name))