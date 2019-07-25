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
import random

from gurobipy import GRB

import treewidth_model as twm
from alib import datamodel, solutions


EPSILON = 0.00001


class RandRoundSepLPOptDynVMPCollectionResultForCostVariant(twm.RandRoundSepLPOptDynVMPCollectionResult):

    def __init__(self, scenario, lp_computation_information, overall_feasible):
        super(RandRoundSepLPOptDynVMPCollectionResultForCostVariant, self).__init__(scenario, lp_computation_information, overall_feasible)

    def get_solution(self):
        if not self.overall_feasible:
            # the solutions dictionary is going to be empty
            pass
        return super(RandRoundSepLPOptDynVMPCollectionResultForCostVariant, self).get_solution()

    def _cleanup_references_raw(self, original_scenario):
        for identifier in self.solutions.keys():
            #search for best solution and remove mapping information of all other solutions
            list_of_solutions = self.solutions[identifier]
            if len(list_of_solutions) > 0:
                best_solution = min(list_of_solutions, key= lambda x: x.cost)
                new_list_of_solutions = []

                for solution in list_of_solutions:
                    if solution == best_solution:
                        new_list_of_solutions.append(self._actual_cleanup_of_references_raw(original_scenario, solution))
                    else:
                        new_list_of_solutions.append(twm.RandomizedRoundingSolution(solution=None,
                                                                                profit=solution.profit,
                                                                                cost=solution.cost,
                                                                                max_node_load=solution.max_node_load,
                                                                                max_edge_load=solution.max_edge_load,
                                                                                time_to_round_solution=solution.time_to_round_solution))

                self.solutions[identifier] = new_list_of_solutions
        #lastly: adapt the collection's scenario
        self.scenario = original_scenario

    def _get_solution_overview(self):
        if not self.overall_feasible:
            result = "\nNo feasible solution found\n"
        else:
            result = "\n\t{:^10s} | {:^40s} {:^20s} | {:^8s}\n".format("COST", "LP Recomputation Mode", "Rounding Order", "INDEX")
            for identifier in self.solutions.keys():
                for solution_index, solution in enumerate(self.solutions[identifier]):
                    result += "\t" + "{:^10.5f} | {:^40s} {:^20s} | {:<8d}\n".format(solution.cost,
                                                                                     identifier[0].value,
                                                                                     identifier[1].value,
                                                                                     solution_index)
        return result


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
                 rounding_solution_quality=2.0,
                 link_capacity_violation_ratio=None,
                 node_capacity_violation_ratio=None,
                 gurobi_settings=None,
                 logger=None,
                 allow_resource_capacity_violations=True):
        super(RandRoundSepLPOptDynVMPCollectionForFogModel, self).__init__(scenario,
                                                                             rounding_order_list,
                                                                             lp_recomputation_mode_list,
                                                                             lp_relative_quality,
                                                                             rounding_samples_per_lp_recomputation_mode,
                                                                             number_initial_mappings_to_compute,
                                                                             number_further_mappings_to_add,
                                                                             gurobi_settings,
                                                                             logger,
                                                                           allow_resource_capacity_violations=allow_resource_capacity_violations,
                                                                           skip_zero_profit_reqs_at_rounding_iteration=False,
                                                                           calculate_cost_of_integral_solutions=True)
        # create a profit variant from the base class
        scenario_with_unit_profit = copy.deepcopy(scenario)
        scenario_with_unit_profit.objective = datamodel.Objective.MAX_PROFIT
        for req in scenario_with_unit_profit.requests:
            req.profit = 1.0
            if len(req.types) != 1:
                raise ValueError("Cost minimizing separation LP is not prepared for multiple NF and/or resource types")

        # Add all mapping variables for each req's each valid mapping found by the initial optimization
        self.universal_node_type = self.substrate.types.copy().pop()
        # we do not need the randomized rounding in this case, so a simple separation LP is enough
        self.profit_variant_algorithm_instance = twm.SeparationLP_OptDynVMP(scenario_with_unit_profit,
                                                                             gurobi_settings,
                                                                             logger.getChild("ProfitVariantForInitialization"),
                                                                            number_further_mappings_to_add=self.number_further_mappings_to_add)
        self.profit_variant_algorithm_instance.init_model_creator()
        # prevent rounding operation to discard requests without profit (and maintian compatibility with the original rounding scheme).

        if self.allow_resource_capacity_violations and \
                (len(self.lp_recomputation_list) > 1 or len(self.rounding_order_list) > 1):
            self.logger.warn("Some LP recomputation modes and rounding orders are ignored...")

        if self.allow_resource_capacity_violations and \
                (twm.LPRecomputationMode.NONE not in self.lp_recomputation_list or twm.RoundingOrder.NONE not in self.rounding_order_list):
            raise ValueError("Allowing capacity violations can only run with NONE rounding order and NONE LP computation settings")

        if self.allow_resource_capacity_violations and \
                (node_capacity_violation_ratio is None or link_capacity_violation_ratio is None):
            raise ValueError("If allow_resource_capacity_violations is True, the violation ratios must be given!")

        # a.k.a. beta in the publications
        self.node_capacity_violation_ratio = float(node_capacity_violation_ratio)
        # a.k.a. gamma in the publications
        self.link_capacity_violation_ratio = float(link_capacity_violation_ratio)
        self.rounding_solution_quality = float(rounding_solution_quality)

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
        # According to Matthias' thesis, this should be inicialized by solving a profit version with all profits set to 1. and
        # calculate the initial objective value based on the found initial embedding of the requests.
        # This procedure is implemented in get_first_mappings_for_requests
        self.model.setObjective(0, GRB.MINIMIZE)

    def create_empty_request_embedding_bound_constraints(self):
        self.embedding_bound = {req : None for req in self.requests}
        for req in self.requests:
            # all request needs to be embedded, so equality is needed!
            self.embedding_bound[req] = self.model.addConstr(0, GRB.EQUAL, 1)

    def update_dual_costs_and_reinit_dynvmps(self):
        #update dual costs
        for snode in self.snodes:
            # the dual variables' value must be non positive
            if self.capacity_constraints[snode].Pi > 0:
                raise ValueError("Found positive dual value for node resource cost!")
            self.dual_costs_node_resources[snode] = self.substrate.node[snode]['cost'][self.universal_node_type] -\
                                                                                    self.capacity_constraints[snode].Pi
        for sedge in self.sedges:
            if self.capacity_constraints[sedge].Pi > 0:
                raise ValueError("Found positive dual value for edge resource cost!")
            self.dual_costs_edge_resources[sedge] = self.substrate.edge[sedge]['cost'] - self.capacity_constraints[sedge].Pi
        # its meaning is unchanged compared to the profit variant, but the domain is different, which affects the constraint separation
        # decisions
        for req in self.requests:
            self.dual_costs_requests[req] = self.embedding_bound[req].Pi

        # reinit dynvmps the same way as in profit variant
        for req in self.requests:
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_reinit_time = time.time()
            dynvmp_instance.reinitialize(new_node_costs=self.dual_costs_node_resources,
                                         new_edge_costs=self.dual_costs_edge_resources)
            self.dynvmp_runtimes_initialization[req].append(time.time() - single_dynvmp_reinit_time)

    def calculate_total_cost_of_single_valid_mapping(self, allocations_of_sinlge_valid_mapping):
        """
        Calculates the embedding cost of a single valid embedding, whose allocations are given as the input.

        :param allocations_of_sinlge_valid_mapping:
        :return:
        """
        sum_cost_of_valid_mapping = 0.0
        for sres, alloc in allocations_of_sinlge_valid_mapping.iteritems():
            res_cost = None
            if type(sres) is tuple and self.universal_node_type not in sres:
                res_cost = self.substrate.edge[sres]['cost']
            elif type(sres) is str:
                res_cost = self.substrate.node[sres]['cost'][self.universal_node_type]
            if res_cost is not None:
                sum_cost_of_valid_mapping += res_cost * alloc
        return sum_cost_of_valid_mapping

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
            self.logger.warn("Scenario is unfeasible, not all requests can be mapped at the same time!")
            return False
        for req, req_profit_variant in zip(self.requests, self.profit_variant_algorithm_instance.requests):
            for index, gurobi_var  in enumerate(self.profit_variant_algorithm_instance.mapping_variables[req_profit_variant]):
                valid_mappings_of_req = self.profit_variant_algorithm_instance.mappings_of_requests[req_profit_variant]
                allocations_of_sinlge_valid_mapping = self._compute_allocations(req, valid_mappings_of_req[index])
                # save allocations for usage at the constraints (so we wont have to model.update() after each variable addition)
                self.allocations_of_mappings[req].append(allocations_of_sinlge_valid_mapping)
                sum_cost_of_valid_mapping = self.calculate_total_cost_of_single_valid_mapping(allocations_of_sinlge_valid_mapping)
                # keep the variable name convention used before
                new_var = self.model.addVar(lb=0.0, ub=1.0,
                                              obj=sum_cost_of_valid_mapping,
                                              vtype=GRB.CONTINUOUS, name=gurobi_var.VarName)
                self.mapping_variables[req].append(new_var)
                # update the valid embedding constraint for each corresponding variable (initially it is invalid 0=1)
                self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)
                # mappings can be safely reused, we just redirect the references to the current algorithm's object
                reused_valid_mapping_objects = []
                for mapping in valid_mappings_of_req:
                    mapping.request = req
                    mapping.substrate = self.substrate
                    reused_valid_mapping_objects.append(mapping)
                self.mappings_of_requests[req] = reused_valid_mapping_objects
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
        # if we passed through the initial checking the scenario shuold be feasible
        return True

    def construct_results_record(self, scenario, sep_lp_solution, overall_feasible):
        """
        Must create the self.result variable.

        :param scenario:
        :param sep_lp_solution:
        :param overall_feasible:
        :return:
        """
        self.result = RandRoundSepLPOptDynVMPCollectionResultForCostVariant(scenario, sep_lp_solution, overall_feasible)

    def perform_separation_and_introduce_new_columns(self, current_objective, ignore_requests=[]):
        """
        Overridden function to slightly modify the constraint separation procedure.

        :param current_objective:
        :param ignore_requests:
        :return:
        """
        new_columns_generated = False
        max_violation = 0
        for req in self.requests:
            if req in ignore_requests:
                continue
            # execute algorithm
            # Dual costs are already initialized
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                # the cost variant is violated if this is negative
                dual_violation_of_req = (opt_cost - self.dual_costs_requests[req])
                if dual_violation_of_req < 0 and max_violation < (-dual_violation_of_req):
                    max_violation = (-dual_violation_of_req)

        # Gives objective bound following from the weak duality: scaling the dual variables (interpreted as resource costs) by a factor
        # of 1 + eps (where eps == dual_violation_of_req) gives a feasible dual solution increasing the objective value by at most 1 + eps
        self._current_obj_bound = current_objective + max_violation

        self._current_solution_quality = None
        if abs(current_objective) > 0.00001:
            # TODO: is this correct?
            self._current_solution_quality = max_violation / current_objective
        else:
            self.logger.info("WARNING: Current objective is very close to zero; treating it as such.")
            self._current_solution_quality = max_violation

        self.logger.info("\nCurrent LP solution value is {:10.5f}\n"
                           "Current dual lower bound is  {:10.5f}\n"
                         "Accordingly, current solution is at least {}-optimal".
                         format(current_objective, self._current_obj_bound, 1+self._current_solution_quality))

        if self._current_solution_quality < self.lp_relative_quality:
            self.logger.info("Ending LP computation as found solution is {}-near optimal, which lies below threshold of {}".format(self._current_solution_quality, self.lp_relative_quality))
            return False
        else:
            for req in self.requests:
                if req in ignore_requests:
                    continue
                dynvmp_instance = self.dynvmp_instances[req]
                opt_cost = dynvmp_instance.get_optimal_solution_cost()
                # the violation criteria is changed
                if opt_cost is not None and opt_cost < self.dual_costs_requests[req]:
                    # cutoff is set to not intorduce more expensive valid mappings then our current upper bound
                    self.introduce_new_columns(req,
                                               maximum_number_of_columns_to_introduce=self.number_further_mappings_to_add,
                                               cutoff = self.dual_costs_requests[req])
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
            # compute corresponding substrate allocation
            mapping_allocations = self._compute_allocations(req, mapping)
            # the original substrate graph costs must be used because we are adding variables to the primal's objective
            total_cost_of_mapping = self.calculate_total_cost_of_single_valid_mapping(mapping_allocations)
            varname = "f_req[{}]_k[{}]".format(req.name, len(self.mappings_of_requests[req]) + index)
            new_var = self.model.addVar(lb=0.0, ub=1.0,
                                        obj=total_cost_of_mapping,
                                        vtype=GRB.CONTINUOUS,
                                        name=varname)
            current_new_variables.append(new_var)
            # store the substrate allocation
            current_new_allocations.append(mapping_allocations)

        # make variables accessible
        self.model.update()
        # capacity constraints and the allocation calculations are the same at the cost variant, so this should not change
        for index, new_var in enumerate(current_new_variables):
            # handle allocations
            corresponding_allocation = current_new_allocations[index]
            for sres, alloc in corresponding_allocation.iteritems():
                if sres not in self.capacity_constraints.keys():
                    continue
                constr = self.capacity_constraints[sres]
                self.model.chgCoeff(constr, new_var, alloc)

            #the new variable corresponding to its request's embedding bound
            self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)

        # add only the first len(current_new_variables) mappings from the ordered list
        self.mappings_of_requests[req].extend(mapping_list[:len(current_new_variables)])
        self.allocations_of_mappings[req].extend(current_new_allocations[:len(current_new_variables)])
        self.mapping_variables[req].extend(current_new_variables)
        self.logger.debug("Introduced {} new mappings for {}".format(len(current_new_variables), req.name))

    def rounding_iteration_with_violations(self, filtered_mappings_of_requests):
        """
        Chooses a mapping for each of the requests, calculates capacity violation ratios of the resulting overall solution

        :param total_allocations:
        :param filtered_mappings_of_requests:
        :return:
        """
        total_allocations = {}
        self._initialize_allocations_dict(total_allocations)
        scenario_solution = solutions.IntegralScenarioSolution(name="", scenario=self.scenario)
        # it is 0, but let's administer it for possible later extensions, and framework compatibility
        total_profit = 0.0
        for req, list_of_mappings_with_probs in filtered_mappings_of_requests.iteritems():
            # NOTE: without random module inicialization it might give different mappings if ran 2x on the same setting
            p = random.random()
            total_p = 0.0
            chosen_mapping = None
            # order of the list does not matter here, because we only check the resource violations on a complete mapping
            for X, original_mapping_idx, mapping in list_of_mappings_with_probs:
                total_p += X
                if p < total_p:
                    chosen_mapping = (original_mapping_idx, mapping)
                    break
            if chosen_mapping is None:
                raise ValueError("No mapping is chosen, embedding variable values MUST add up to 1.0 at this point...")
            # administer resource reservations of the mapping
            for res in self.substrate_resources:
                if res in self.allocations_of_mappings[req][chosen_mapping[0]].keys():
                    total_allocations[res] += self.allocations_of_mappings[req][chosen_mapping[0]][res]
            total_profit += req.profit
            scenario_solution.add_mapping(req, chosen_mapping[1])

        max_node_load, max_edge_load = self.calc_max_loads(total_allocations)
        total_cost = self.calc_cost_of_integral_solution(scenario_solution, total_allocations)
        return scenario_solution, total_profit, total_cost, max_node_load, max_edge_load

    def filter_costly_mappings_of_requests(self):
        """
        Discards all mapping for each request which are higher than a factor of weighted_average_cost_of_req.
        Normalizes the remaining weights.

        :return:
        """
        filtered_mappings_of_requests = {req: list() for req in self.requests}
        for req in self.requests:
            # we have to ensure this value is not zero, otherwise 0 cost optimal solutions are filtered
            weighted_average_cost_of_req = 0.00001
            for mapping_index, mapping in enumerate(self.mappings_of_requests[req]):
                allocations_of_sinlge_valid_mapping = self._compute_allocations(req, mapping)
                total_cost_of_mapping = self.calculate_total_cost_of_single_valid_mapping(allocations_of_sinlge_valid_mapping)
                weighted_average_cost_of_req += self.mapping_variables[req][mapping_index].X * total_cost_of_mapping
            # filter costly mappings
            self.logger.debug("Filtering mapping for request {} which are more expensive "
                              "than solution quaility: {} x weighted average cost {}".
                              format(req.name, self.rounding_solution_quality, weighted_average_cost_of_req))
            for mapping_index, mapping in enumerate(self.mappings_of_requests[req]):
                # according to experiments current implementation is memory intensive, rather recalculate it than store it
                allocations_of_sinlge_valid_mapping = self._compute_allocations(req, mapping)
                total_cost_of_mapping = self.calculate_total_cost_of_single_valid_mapping(allocations_of_sinlge_valid_mapping)
                if total_cost_of_mapping < self.rounding_solution_quality * weighted_average_cost_of_req:
                    # the gurobi variable is not needed, just its value
                    filtered_mappings_of_requests[req].append((self.mapping_variables[req][mapping_index].X, mapping_index, mapping))
            if len(filtered_mappings_of_requests[req]) > 0:
                sum_of_remaining_mappings = sum(map(lambda t: t[0], filtered_mappings_of_requests[req]))
                # normalize the weights so they add up to 1.0
                filtered_mappings_of_requests[req] = map(lambda t: (t[0] / float(sum_of_remaining_mappings), t[1], t[2]),
                                                         filtered_mappings_of_requests[req])
            else:
                # other (currently non-existing) RR heuristics might find solution, Infeasibility is set by
                # validate_randomized_rounding_solutions.
                self.logger.info("All valid mappings of request {} are considered costly, solution cannot be feasible with "
                                 "current randomized rounding heuristic".format(req.name))
                return None
            self.logger.debug("Filtered {} costly valid mappings for request {}".format(
                len(self.mapping_variables[req]) - len(filtered_mappings_of_requests[req]), req.name))
        return filtered_mappings_of_requests

    def round_solution_with_violations(self, lp_computation_mode, rounding_order):
        """
        Returns list of integers solutions of the given randomized rounding setting for the cost variant.

        :param lp_computation_mode:
        :param rounding_order:
        :return:
        """
        result_list = []
        self.logger.debug("Executing rounding (without allowing capacity violations) according to settings: {} {}".
                          format(lp_computation_mode.value, rounding_order.value))
        # this function is only prepared for this combination
        if lp_computation_mode is not twm.LPRecomputationMode.NONE or rounding_order is not twm.RoundingOrder.NONE:
            self.logger.warn("Returning empty randomized roudning result list as this setting is not supported.")
            return result_list

        self.model.optimize()
        solutionCount = self.model.getAttr("SolCount")
        if solutionCount > 0:
            # if there is a solution to the fractional variant, its objective value is our current lower bound for the integral solution
            lower_cost_bound = self.model.getAttr("ObjVal")
            self.logger.info("Best fractional solution cost value is {}".format(lower_cost_bound))
        else:
            self.logger.info("No solution found by the Separation LP, randomized rounding heuristic cannot start, returning no results.")
            return result_list

        filtered_mappings_of_requests = self.filter_costly_mappings_of_requests()
        if filtered_mappings_of_requests is None:
            # if for some reason the filtering was not successful
            return result_list

        best_solution_cost = None
        for q in xrange(self.rounding_samples_per_lp_recomputation_mode[lp_computation_mode]):
            time_rr0 = time.time()

            solution, profit, cost, max_node_load, max_edge_load = \
                self.rounding_iteration_with_violations(filtered_mappings_of_requests)
            if len(solution.request_mapping) != len(self.requests):
                self.logger.warn("Not all requests has a mapping in this rounding iteration! "
                                 "This should never happen as the probabilities add up to 1.0")
                # skip adding this bad solution (in case it would happen)
            else:
                # Only allow solutions which meet the capacity violation ratios
                if max_node_load < self.node_capacity_violation_ratio and \
                        max_edge_load < self.link_capacity_violation_ratio:
                    self.logger.debug("Found integer solution with cost value {} and meets allowed capacity violations.".format(cost))
                    if best_solution_cost is not None:
                        # NOTE: administering the best solution is not necessary
                        if cost < best_solution_cost:
                            best_solution_cost = cost
                            current_solution_quality = best_solution_cost / float(lower_cost_bound)
                            self.logger.debug("Found integer solution with meeting approximation bounds, "
                                              "so solution is at least {}-optimal.".format(current_solution_quality))

                    else:
                        # this is only at the first iteration, so we wont have to guess an initial value
                        best_solution_cost = cost
                    time_rr = time.time() - time_rr0
                    solution_tuple = twm.RandomizedRoundingSolution(solution=solution,
                                                                    profit=profit,
                                                                    cost=cost,
                                                                    max_node_load=max_node_load,
                                                                    max_edge_load=max_edge_load,
                                                                    time_to_round_solution=time_rr)
                    result_list.append(solution_tuple)
                else:
                    self.logger.debug("Discarding solution because it violates capacities more than "
                                      "allowed: node {}; link {}".format(max_node_load, max_edge_load))
        return result_list

    def validate_randomized_rounding_solutions(self):
        """
        Checks if all requests are mapped after the randomized rounding as required by the constraints.
        If not, then set the solution infeasible. (example occurance: the chosen mapping for the single request does not meet the capacity
        constraints and new mapping is not chosen for it. So the solution would have 0 cost and seem feasible, but does not meet the
        constraint of mapping all requests.)

        :return:
        """
        # remove integral solution which does not have all requests mapped, and set the result infeasible if there are no
        # integral solutions left
        for identifier in self.result.solutions.keys():
            list_of_valid_solutions = []
            for rr_solution in self.result.solutions[identifier]:
                if len(rr_solution.solution.request_mapping) == len(self.requests):
                    list_of_valid_solutions.append(rr_solution)
            self.result.solutions[identifier] = list_of_valid_solutions
        is_result_still_feasible = False
        for identifier, list_of_valid_solutions in self.result.solutions.iteritems():
            if len(list_of_valid_solutions) > 0:
                is_result_still_feasible = True
        if not is_result_still_feasible:
            self.logger.info("Setting result infeasible, because none of the integral solutions after the randomized rounding maps all requests.")
        self.result.overall_feasible = is_result_still_feasible

