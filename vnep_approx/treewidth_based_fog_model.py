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
import copy
import math

from gurobipy import GRB

import treewidth_model as twm
from alib import datamodel

EPSILON = 0.00001

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
        # create a profit variant from the base class
        scenario_with_unit_profit = copy.deepcopy(scenario)
        scenario_with_unit_profit.objective = datamodel.Objective.MAX_PROFIT
        for req in scenario_with_unit_profit.requests:
            req.profit = 1.0
            if len(req.types) != 1:
                raise ValueError("Cost minimizing separation LP is not prepared for multiple NF and/or resource types")
        # we do not need the randomized rounding in this case, so a simple separation LP is enough
        self.profit_variant_algorithm_instance = twm.SeparationLP_OptDynVMP(scenario_with_unit_profit,
                                                                             gurobi_settings,
                                                                             logger.getChild("ProfitVariantForInitialization"))
        self.profit_variant_algorithm_instance.init_model_creator()

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
        # TODO (NB): according to Matthias' thesis, this should be inicialized by solving a profit version with all profits set to 1. and
        # calculate the initial objective value based on the found initial embedding of the requests.
        # TODO: (is there easier way?): this initialization might be completly ignored due to the actions taken in
        # get_first_mappings_for_requests, this builds the variables which are involved in the objective.
        self.model.setObjective(0, GRB.MINIMIZE)

    def create_empty_request_embedding_bound_constraints(self):
        self.embedding_bound = {req : None for req in self.requests}
        for req in self.requests:
            # all request needs to be embedded, so equality is needed!
            self.embedding_bound[req] = self.model.addConstr(0, GRB.EQUAL, 1)

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

    def get_first_mappings_for_requests(self):
        """
        For the cost variant the first request mappings come from solving a profit variant with universal 1.0 cost for each request.
        Secondarily this is a feasibility test: if the achieved profit equals the number of requests, then this scenario is feasible.

        :return:
        """
        self.profit_variant_algorithm_instance.compute_solution()
        # this feasibility checking is needed!
        if not self.profit_variant_algorithm_instance.status.hasFeasibleStatus() or\
          math.fabs(self.profit_variant_algorithm_instance.status.getObjectiveValue() - len(self.scenario.requests)) > EPSILON:
            raise ValueError("Scenario is unfeasible, not all requests can be mapped at the same time!")
        # Add all mapping variables for each req's each valid mapping found by the initial optimization
        universal_node_type = self.substrate.types.copy().pop()
        for req, req_profit_variant in zip(self.requests, self.profit_variant_algorithm_instance.requests):
            for index, gurobi_var  in enumerate(self.profit_variant_algorithm_instance.mapping_variables[req_profit_variant]):
                valid_mappings_of_req = self.profit_variant_algorithm_instance.mappings_of_requests[req_profit_variant]
                allocations_of_sinlge_valid_mapping = self._compute_allocations(req, valid_mappings_of_req[index])
                # save allocations for usage at the constraints (so we wont have to model.update() after each variable addition)
                self.allocations_of_mappings[req].append(allocations_of_sinlge_valid_mapping)
                sum_cost_of_valid_mapping = 0.0
                for sres, alloc in allocations_of_sinlge_valid_mapping.iteritems():
                    res_cost = None
                    if type(sres) is tuple and universal_node_type not in sres:
                        res_cost = self.substrate.edge[sres]['cost']
                    elif type(sres) is str:
                        res_cost = self.substrate.node[sres]['cost'][universal_node_type]
                    if res_cost is not None:
                        sum_cost_of_valid_mapping += res_cost * alloc
                # keep the variable name convention used before
                new_var = self.model.addVar(lb=0.0, ub=1.0,
                                              obj=sum_cost_of_valid_mapping,
                                              vtype=GRB.CONTINUOUS, name=gurobi_var.VarName)
                self.mapping_variables[req].append(new_var)
                # update the valid embedding constraint for each corresponding variable (initially it is invalid 0=1)
                self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)
                # mappings can be safely reused, just redirect the references to the current algorithm's object
                for mapping in valid_mappings_of_req:
                    mapping.request = req
                    mapping.substrate = self.substrate
                    self.mappings_of_requests[req] = mapping
        # make added variables avaialbe
        self.model.update()
        for req in self.requests:
            for index, gurobi_var in enumerate(self.mapping_variables[req]):
                for sres, gurobi_constr in self.capacity_constraints.iteritems():
                    allocations_of_sinlge_valid_mapping = self.allocations_of_mappings[req][index]
                    # if this valid mapping has no allocations for the 'sres' substrate resource, we can skip it
                    if sres not in allocations_of_sinlge_valid_mapping:
                        continue
                    self.model.chgCoeff(gurobi_constr, gurobi_var, allocations_of_sinlge_valid_mapping[sres])
        # apply coefficient changes
        self.model.update()

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
                # TODO (NB): this needs to be set differently, due to the different type of constraint violations
                dual_violation_of_req = (opt_cost - req.profit + self.dual_costs_requests[req])
                if dual_violation_of_req < 0:
                    total_dual_violations += (-dual_violation_of_req)

        # Gives objective bound following from the weak duality: scaling the dual variables (interpreted as resource costs) by a factor
        # of 1 + eps (where eps == dual_violation_of_req) gives a feasible dual solution increasing the objective value by at most 1 + eps
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
            # TODO (NB): this should be done differently, because this is the introduction of the new variable to the objective
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
        # capacity constraints and the allocation calculations are the same at the cost variant, so this should not change
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