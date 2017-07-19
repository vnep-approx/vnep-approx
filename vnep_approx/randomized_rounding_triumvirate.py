# MIT License
#
# Copyright (c) 2016-2017 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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


NUMBER_OF_ITERATIONS = 250

RandomizedRoundingMetaData = namedtuple("RandomizedRoundingMetaData", ["time_preprocessing", "time_optimization", "time_postprocessing",
                                                                       "lost_flow_in_decomposition", "temporal_log", "status"])
RandomizedRoundingSolutionData = namedtuple("RandomizedRoundingSolutionData", ["profit",
                                                                               "max_node_load",
                                                                               "max_edge_load",
                                                                               "time_to_round_solution"])

RandomizedRoundingTriumviratReducedResult_January_2017 = namedtuple(
    "RandomizedRoundingTriumviratReducedResult_January_2017",
    ["meta_data", "best_feasible", ]
)


class RandomizedRoundingTriumviratResult(modelcreator.AlgorithmResult):
    def __init__(self, meta_data, collection_of_samples_with_violations, result_wo_violations, mdk_result, mdk_meta_data):
        super(RandomizedRoundingTriumviratResult, self).__init__()
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


class RandomizedRoundingTriumvirat(object):
    ALGORITHM_ID = "RandomizedRoundingTriumvirat"

    def __init__(self, scenario, gurobi_settings=None, logger=None):
        self.scenario = scenario
        self.mc = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(self.scenario,
                                                                                 gurobi_settings=gurobi_settings,
                                                                                 logger=logger)
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

    def init_model_creator(self):
        self.mc.init_model_creator()

    def compute_integral_solution(self, onlyloads=True):

        self.mc.model.setParam("Method", 2)
        self.mc.model.setParam("NumericFocus", 2)
        self.mc.model.setParam("BarConvTol", 0.000000000000001)
        self.mc.model.setParam("OptimalityTol", 0.001)

        self._fractional_solution = self.mc.compute_fractional_solution()
        if self._fractional_solution is None:
            return None

        meta_data = RandomizedRoundingMetaData(time_preprocessing=self.mc.time_preprocess,
                                               time_optimization=self.mc.time_optimization,
                                               time_postprocessing=self.mc.time_postprocessing,
                                               lost_flow_in_decomposition=self.mc.lost_flow_in_the_decomposition,
                                               temporal_log=self.temporal_log,
                                               status=self.mc.status)

        collection_of_samples_with_violations = self.collect_X_randomized_rounding_samples_with_potential_violations(NUMBER_OF_ITERATIONS)

        result_wo_violations = self.round_solution_without_violations(NUMBER_OF_ITERATIONS)

        mdk = DecompositionMDK(self.scenario, self._fractional_solution, logger=self.logger)
        mdk.init_model_creator()
        mdk_solution = mdk.compute_integral_solution()

        mdk_meta_data = RandomizedRoundingMetaData(time_preprocessing=mdk.time_preprocess,
                                                   time_optimization=mdk.time_optimization,
                                                   time_postprocessing=mdk.time_postprocessing,
                                                   lost_flow_in_decomposition=0.0,
                                                   temporal_log=mdk.temporal_log,
                                                   status=mdk.status)

        self.solution = RandomizedRoundingTriumviratResult(meta_data=meta_data,
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

            profit, max_node_load, max_edge_load = self.rounding_iteration_violations_allowed_sampling_max_violations(L)

            time_rr = time.clock() - time_rr0

            solution_tuple = RandomizedRoundingSolutionData(profit=profit,
                                                            max_node_load=max_node_load,
                                                            max_edge_load=max_edge_load,
                                                            time_to_round_solution=time_rr)

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
        # NO IDEA WHY THIS DOES NOT WORK BUT THE ABVOE DOES

        for (ntype, snode) in self.substrate_node_resources:
            constr_name = "obey_capacity_nodes_{}".format((ntype, snode))
            load_expr = LinExpr([(self.fractional_solution.mapping_loads[fractional_mapping.name][(ntype, snode)], self.var_embedding_variable[req][fractional_mapping])] for req in self.var_embedding_variable for fractional_mapping in self.var_embedding_variable[req] if self.fractional_solution.mapping_loads[fractional_mapping.name][(ntype, snode)] > 0.001)
            self.model.addConstr(load_expr, GRB.LESS_EQUAL, float(self.scenario.substrate.node[snode]["capacity"][ntype]), name=constr_name)

        for (u, v) in self.substrate_edge_resources:
            constr_name = "obey_capacity_edges_{}".format((u, v))
            load_expr = LinExpr([(self.fractional_solution.mapping_loads[fractional_mapping.name][(u, v)], self.var_embedding_variable[req][fractional_mapping])] for req in self.var_embedding_variable for fractional_mapping in self.var_embedding_variable[req] if self.fractional_solution.mapping_loads[fractional_mapping.name][(u, v)] > 0.001)
            self.model.addConstr(load_expr, GRB.LESS_EQUAL, float(self.scenario.substrate.edge[(u, v)]["capacity"]), name=constr_name)

    def create_objective(self):
        objective = LinExpr()
        for req in self.var_embedding_variable:
            for fractional_mapping in self.var_embedding_variable[req]:
                objective.addTerms(req.profit, self.var_embedding_variable[req][fractional_mapping])
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def compute_integral_solution(self):
        # hard coded.. TODO
        self.model.setParam("LogToConsole", 0)
        self.model.setParam("TimeLimit", 120)
        self.model.setParam("Threads", 1)

        return super(DecompositionMDK, self).compute_integral_solution()

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
