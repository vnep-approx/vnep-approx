# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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
#

import time
from collections import namedtuple
from random import Random

from gurobipy import GRB, LinExpr

from alib import modelcreator
from . import modelcreator_ecg_decomposition

random = Random("randomized_rounding")


class RandomizedRoundingError(Exception): pass


RandomizedRoundingMetaData = namedtuple("RandomizedRoundingMetaData", ["time_preprocessing", "time_optimization", "time_postprocessing",
                                                                       "lost_flow_in_decomposition", "temporal_log", "status"])
RandomizedRoundingSolutionData = namedtuple("RandomizedRoundingSolutionData", ["profit",
                                                                               "max_node_load",
                                                                               "max_edge_load",
                                                                               "time_to_round_solution"])

class RandomizedRoundingTriumvirateResult(modelcreator.AlgorithmResult):
    def __init__(self, meta_data, collection_of_samples_with_violations, result_wo_violations, mdk_result, mdk_meta_data):
        super(RandomizedRoundingTriumvirateResult, self).__init__()
        self.meta_data = meta_data
        self.collection_of_samples_with_violations = collection_of_samples_with_violations
        self.result_wo_violations = result_wo_violations
        self.mdk_result = mdk_result
        self.mdk_meta_data = mdk_meta_data

    def __str__(self):
        return "RandomizedRoundingResult: {}".format(self.meta_data.status)

    def get_solution(self):
        return self

    def cleanup_references(self, original_scenario):
        pass

    def __str__(self):
        output_string = "global meta_data:                {}".format(self.meta_data)
        output_string += "\ncollection_of_samples (excerpt): {} [...]".format(self.collection_of_samples_with_violations[0:10])
        output_string += "\nresult w/o violation:            {}".format(self.result_wo_violations)
        output_string += "\nmdk_meta_data:                   {}".format(self.mdk_meta_data)
        output_string += "\nmdk_solution:                    {}".format(self.mdk_result)
        return output_string


class RandomizedRoundingTriumvirate(object):
    ''' This class implements 3 different randomized rounding heuristics for obtaining integral mappings.
        The first class of heuristics applies rounding directly. Either the solution minimizing resource violations or the
        one maximizing the profit is returned.
        The second one does round but discards any (chosen) mappings whose addition would violate resource capacities.
        The last one simply computes the optimal solution of the corresponding Multi-Dimensional Knapsack (MDK).
        Specifically, given the decomposed solution for each mapping a binary variable is introduced such that capacities
        are not exceeded and for each request at most one mapping is selected.

    '''

    ALGORITHM_ID = "RandomizedRoundingTriumvirate"

    def __init__(self, scenario, gurobi_settings=None, logger=None, number_of_solutions_to_round=1000, mdk_gurobi_parameters=None, write_lp_file_format=None, decomposition_epsilon=1e-10, relative_decomposition_abortion_epsilon=1e-3, absolute_decomposition_abortion_epsilon=1e-6):
        self.scenario = scenario
        # This is the (relaxation) of the novel decomposable LP for cactus graphs.
        logger.info("Starting randomized rounding trivate for scenario {}".format(scenario.name))

        lp_output_file = None
        if write_lp_file_format is not None:
            lp_output_file = scenario.name + write_lp_file_format

        self.mc = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(self.scenario,
                                                                                 gurobi_settings=gurobi_settings,
                                                                                 logger=logger,
                                                                                 lp_output_file=lp_output_file,
                                                                                 absolute_decomposition_abortion_epsilon=float(
                                                                                     absolute_decomposition_abortion_epsilon),
                                                                                 relative_decomposition_abortion_epsilon=float(
                                                                                     relative_decomposition_abortion_epsilon),  #conversion to float if parameter was given as string
                                                                                 decomposition_epsilon=float(decomposition_epsilon))                                      #conversion to float if parameter was given as string
        self.temporal_log = self.mc.temporal_log
        self._fractional_solution = None
        self.solution = None
        self.logger = logger

        self.substrate_resources = list(self.scenario.substrate.edges)
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
                self.substrate_resources.append((ntype, snode))

        self.number_of_solutions_to_round = number_of_solutions_to_round
        if mdk_gurobi_parameters is None:
            #default settings according to IFIP 2018 evaluation
            self.mdk_gurobi_settings = modelcreator.GurobiSettings(timelimit=120,
                                                                   threads=1,
                                                                   logtoconsole=0)
        else:
            if isinstance(mdk_gurobi_parameters, modelcreator.GurobiSettings):
                self.mdk_gurobi_settings = mdk_gurobi_parameters
            elif isinstance(mdk_gurobi_parameters, tuple):
                gurobisettings_dict = {}
                key = None
                if len(mdk_gurobi_parameters) % 2 != 0:
                    raise ValueError("MDK Parameter Settings could not be converted to dictionary...")
                for index, elem in enumerate(mdk_gurobi_parameters):
                    if index % 2 == 0:
                        key = elem
                    if index % 2 == 1:
                        gurobisettings_dict[key] = elem
                self.mdk_gurobi_settings = modelcreator.GurobiSettings(**gurobisettings_dict)




    def init_model_creator(self):
        self.mc.init_model_creator()

    def compute_integral_solution(self):

        self._fractional_solution = self.mc.compute_fractional_solution()
        if self._fractional_solution is None:
            return None

        meta_data = RandomizedRoundingMetaData(time_preprocessing=self.mc.time_preprocess,
                                               time_optimization=self.mc.time_optimization,
                                               time_postprocessing=self.mc.time_postprocessing,
                                               lost_flow_in_decomposition=self.mc.lost_flow_in_the_decomposition,
                                               temporal_log=self.temporal_log,
                                               status=self.mc.status)
        # First rounding heuristic. It might violate constraints, It is a collection of rounded solutions
        collection_of_samples_with_violations = self.collect_X_randomized_rounding_samples_with_potential_violations(self.number_of_solutions_to_round)

        # Second rounding heuristic. It doesn't violate constraints. Gathers all requests for a while, until they don't violate constraints.
        result_wo_violations = self.round_solution_without_violations(self.number_of_solutions_to_round)

        # Third rounding
        mdk = DecompositionMDK(self.scenario, self._fractional_solution, logger=self.logger, gurobi_settings=self.mdk_gurobi_settings)
        mdk.init_model_creator()
        mdk_solution = mdk.compute_integral_solution()

        mdk_meta_data = RandomizedRoundingMetaData(time_preprocessing=mdk.time_preprocess,
                                                   time_optimization=mdk.time_optimization,
                                                   time_postprocessing=mdk.time_postprocessing,
                                                   lost_flow_in_decomposition=0.0,
                                                   temporal_log=mdk.temporal_log,
                                                   status=mdk.status)

        self.solution = RandomizedRoundingTriumvirateResult(meta_data=meta_data,
                                                           collection_of_samples_with_violations=collection_of_samples_with_violations,
                                                           result_wo_violations=result_wo_violations,
                                                           mdk_result=mdk_solution,
                                                           mdk_meta_data=mdk_meta_data
                                                           )
        return self.solution

    def collect_X_randomized_rounding_samples_with_potential_violations(self, number_of_samples):
        L = {}
        data = []
        for q in xrange(number_of_samples):
            time_rr0 = time.clock()

            # this is an embedding (decision on which requests to embed) with rounded values -- profit and min/max load for this solution.
            profit, max_node_load, max_edge_load = self.rounding_iteration_violations_allowed_sampling_max_violations(L)

            time_rr = time.clock() - time_rr0

            solution_tuple = RandomizedRoundingSolutionData(profit=profit,
                                                            max_node_load=max_node_load,
                                                            max_edge_load=max_edge_load,
                                                            time_to_round_solution=time_rr)
            # collection of many (violating) solution options
            data.append(solution_tuple)

        return data

    def rounding_iteration_violations_allowed_sampling_max_violations(self, L):
        r_prime = set()
        B = 0.0
        self._initialize_load_dict(L)
        for req in self.scenario.get_requests():
            p = random.random()
            total_flow = 0.0
            fractional_mapping = None
            found_mapping = False

            if req in self._fractional_solution.request_mapping:
                for frac_mapping in self._fractional_solution.request_mapping[req]:
                    total_flow += self._fractional_solution.mapping_flows[frac_mapping.name]
                    if p < total_flow:
                        found_mapping = True
                        fractional_mapping = frac_mapping
                        break

            if fractional_mapping is not None:
                r_prime.add(req)
                B += req.profit
                for res in self.substrate_resources:
                    L[res] += self._fractional_solution.mapping_loads[fractional_mapping.name][res]

        max_node_load, max_edge_load = self.calc_max_loads(L)

        return B, max_node_load, max_edge_load

    def round_solution_without_violations(self, number_of_samples):

        L = {}

        best_solution_tuple = None

        for q in xrange(number_of_samples):
            time_rr0 = time.clock()

            profit, max_node_load, max_edge_load = self.rounding_iteration_violations_without_violations(L, outer_tries=5)

            time_rr = time.clock() - time_rr0

            solution_tuple = RandomizedRoundingSolutionData(profit=profit,
                                                            max_node_load=max_node_load,
                                                            max_edge_load=max_edge_load,
                                                            time_to_round_solution=time_rr)

            if best_solution_tuple is None or best_solution_tuple.profit < solution_tuple.profit:
                best_solution_tuple = solution_tuple

        return best_solution_tuple

    def rounding_iteration_violations_without_violations(self, L, outer_tries):
        r_prime = set()
        B = 0.0
        self._initialize_load_dict(L)

        req_list = [req for req in self.scenario.get_requests()]
        random.shuffle(req_list)

        for i in range(0, outer_tries + 1):

            for req in self.scenario.get_requests():

                if req not in self._fractional_solution.request_mapping or req in r_prime:
                    continue

                p = random.random()
                total_flow = 0.0
                fractional_mapping = None

                for frac_mapping in self._fractional_solution.request_mapping[req]:
                    total_flow += self._fractional_solution.mapping_flows[frac_mapping.name]
                    if p < total_flow:
                        fractional_mapping = frac_mapping
                        break

                if fractional_mapping is not None:

                    if self.check_whether_mapping_would_obey_resource_violations(L, self._fractional_solution.mapping_loads[fractional_mapping.name]):
                        r_prime.add(req)
                        B += req.profit
                        for res in self.substrate_resources:
                            L[res] += self._fractional_solution.mapping_loads[fractional_mapping.name][res]

            random.shuffle(req_list)

        max_node_load, max_edge_load = self.calc_max_loads(L)
        return B, max_node_load, max_edge_load

    def check_whether_mapping_would_obey_resource_violations(self, L, mapping_loads):
        result = True

        for (ntype, snode) in self.substrate_node_resources:
            if L[(ntype, snode)] + mapping_loads[(ntype, snode)] > self.scenario.substrate.node[snode]["capacity"][ntype]:
                result = False
                break

        for (u, v) in self.substrate_edge_resources:
            if L[(u, v)] + mapping_loads[(u, v)] > self.scenario.substrate.edge[(u, v)]["capacity"]:
                result = False
                break

        return result

    def _initialize_load_dict(self, L):
        sub = self.scenario.substrate
        for snode in sub.nodes:
            for ntype in sub.node[snode]["capacity"]:
                L[(ntype, snode)] = 0.0
        for u, v in sub.edges:
            L[(u, v)] = 0.0

    def calc_max_loads(self, L):
        max_node_load = 0
        max_edge_load = 0
        for (ntype, snode) in self.substrate_node_resources:
            ratio = L[(ntype, snode)] / float(self.scenario.substrate.node[snode]["capacity"][ntype])
            if ratio > max_node_load:
                max_node_load = ratio
        for (u, v) in self.substrate_edge_resources:
            ratio = L[(u, v)] / float(self.scenario.substrate.edge[(u, v)]["capacity"])
            if ratio > max_edge_load:
                max_edge_load = ratio
        return (max_node_load, max_edge_load)


class DecompositionMDK(modelcreator.AbstractModelCreator):
    """
    Given the decomposition into convex combinations of valid mappings for each request,
    the MDK computes the optimal rounding given all mapping possibilities found.
    """
    def __init__(self,
                 scenario,
                 fractional_solution,
                 gurobi_settings=None,
                 optimization_callback=modelcreator.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(DecompositionMDK, self).__init__(gurobi_settings=gurobi_settings,
                                               optimization_callback=optimization_callback,
                                               lp_output_file=lp_output_file,
                                               potential_iis_filename=potential_iis_filename,
                                               logger=logger)

        self.scenario = scenario
        self.fractional_solution = fractional_solution

        self.substrate_resources = list(self.scenario.substrate.edges)
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
                self.substrate_resources.append((ntype, snode))
        self.var_embedding_variable = {}

    def create_variables(self):

        for req in self.fractional_solution.request_mapping:
            self.var_embedding_variable[req] = {}
            counter = 1
            for fractional_mapping in self.fractional_solution.request_mapping[req]:
                var_name = "embedding_variable_{}_{}_{}".format(req.name, counter, fractional_mapping.name)
                self.var_embedding_variable[req][fractional_mapping] = self.model.addVar(lb=0.0,
                                                                                         ub=1.0,
                                                                                         obj=0.0,
                                                                                         vtype=GRB.BINARY,
                                                                                         name=var_name)
                counter += 1

    def create_constraints(self):

        # bound resources..
        for (ntype, snode) in self.substrate_node_resources:
            constr_name = "obey_capacity_nodes_{}".format((ntype, snode))
            load_expr = LinExpr()
            for req in self.var_embedding_variable:
                for fractional_mapping in self.var_embedding_variable[req]:
                    if self.fractional_solution.mapping_loads[fractional_mapping.name][(ntype, snode)] > 0.001:
                        load_expr.addTerms(self.fractional_solution.mapping_loads[fractional_mapping.name][(ntype, snode)], self.var_embedding_variable[req][fractional_mapping])
            self.model.addConstr(load_expr, GRB.LESS_EQUAL, float(self.scenario.substrate.node[snode]["capacity"][ntype]), name=constr_name)

        for (u, v) in self.substrate_edge_resources:
            constr_name = "obey_capacity_edges_{}".format((u, v))
            load_expr = LinExpr()
            for req in self.var_embedding_variable:
                for fractional_mapping in self.var_embedding_variable[req]:
                    if self.fractional_solution.mapping_loads[fractional_mapping.name][(u, v)] > 0.001:
                        load_expr.addTerms(self.fractional_solution.mapping_loads[fractional_mapping.name][(u, v)], self.var_embedding_variable[req][fractional_mapping])
            self.model.addConstr(load_expr, GRB.LESS_EQUAL, float(self.scenario.substrate.edge[(u, v)]["capacity"]), name=constr_name)

        for req in self.var_embedding_variable:
            constr_name = "embed_at_most_one_decomposition_{}".format(req.name)
            at_most_one_decomp_expr = LinExpr()
            for fractional_mapping in self.var_embedding_variable[req]:
                at_most_one_decomp_expr.addTerms(1.0, self.var_embedding_variable[req][fractional_mapping])
            self.model.addConstr(at_most_one_decomp_expr, GRB.LESS_EQUAL, 1.0, name=constr_name)

        return

    def create_objective(self):
        # the original objective of the LP which produced the decomposable fractional solutions
        objective = LinExpr()
        for req in self.var_embedding_variable:
            for fractional_mapping in self.var_embedding_variable[req]:
                objective.addTerms(req.profit, self.var_embedding_variable[req][fractional_mapping])
        self.model.setObjective(objective, GRB.MAXIMIZE)


    def preprocess_input(self):
        pass

    # simply returns the same format as the others...
    def recover_integral_solution_from_variables(self):
        B = 0
        L = {}
        self._initialize_load_dict(L)
        for req in self.var_embedding_variable:
            was_embedded = False
            for fractional_mapping in self.var_embedding_variable[req]:
                if self.var_embedding_variable[req][fractional_mapping].X > 0.5:
                    if was_embedded:
                        raise Exception("can only embed a single decomposition!")
                    was_embedded = True
                    B += req.profit
                    for res in self.substrate_resources:
                        L[res] += self.fractional_solution.mapping_loads[fractional_mapping.name][res]

        max_node_load, max_edge_load = self.calc_max_loads(L)

        # note that we are in the post_processing stage... hence, we cannot add self.time_postprocessing ...
        result = RandomizedRoundingSolutionData(profit=B,
                                                max_node_load=max_node_load,
                                                max_edge_load=max_edge_load,
                                                time_to_round_solution=self.time_preprocess + self.time_optimization)
        return result

    def post_process_integral_computation(self):
        return self.solution

    def _initialize_load_dict(self, L):
        sub = self.scenario.substrate
        for snode in sub.nodes:
            for ntype in sub.node[snode]["capacity"]:
                L[(ntype, snode)] = 0.0
        for u, v in sub.edges:
            L[(u, v)] = 0.0

    def calc_max_loads(self, L):
        max_node_load = 0
        max_edge_load = 0
        for (ntype, snode) in self.substrate_node_resources:
            ratio = L[(ntype, snode)] / float(self.scenario.substrate.node[snode]["capacity"][ntype])
            if ratio > max_node_load:
                max_node_load = ratio
        for (u, v) in self.substrate_edge_resources:
            ratio = L[(u, v)] / float(self.scenario.substrate.edge[(u, v)]["capacity"])
            if ratio > max_edge_load:
                max_edge_load = ratio
        return (max_node_load, max_edge_load)
