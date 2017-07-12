import logging
import time
from collections import namedtuple
from random import Random

from alib import modelcreator, solutions
from . import modelcreator_ecg_decomposition

logger = logging.getLogger(__name__)

random = Random("randomized_rounding")


class RandomizedRoundingError(Exception): pass


# ALPHA, BETA, GAMMA = 0.0, 99999999.0, 99999999.0
NUMBER_OF_ITERATIONS = 100

RandomizedRoundingSolutionExtract = namedtuple("RandomizedRoundingSolutionExtract", "meta_data, solution_list")
RandomizedRoundingMetaData = namedtuple("RandomizedRoundingMetaData", "substrate_node_resource_names substrate_edge_resource_names time_preprocessing time_optimization time_postprocessing")
RandomizedRoundingSolutionData = namedtuple("RandomizedRoundingSolutionData", "profit loadfactor_substrate_nodes loadfactor_substrate_edges time_to_round_solution lost_flow_in_decomposition")


class RandomizedRoundingResult(modelcreator.AlgorithmResult):
    def __init__(self, solution, temporal_log, status):
        super(RandomizedRoundingResult, self).__init__()
        self.solution = solution
        self.temporal_log = temporal_log
        self.status = status

    def __str__(self):
        return "RandomizedRoundingResult: {}".format(self.status)

    def get_solution(self):
        return self

    def cleanup_references(self, original_scenario):
        pass


class RandomizedRounding(object):
    ALGORITHM_ID = "RandomizedRounding"

    def __init__(self, scenario, gurobi_settings=None, logger=None):
        self.scenario = scenario
        self.mc = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(self.scenario,
                                                                                 gurobi_settings=gurobi_settings,
                                                                                 logger=logger)
        self.temporal_log = self.mc.temporal_log
        self._fractional_solution = None
        self.solution = None

    def init_model_creator(self):
        self.mc.init_model_creator()

    def compute_integral_solution(self, onlyloads=True):
        # todo: Put some logging code etc here
        if onlyloads:
            self.solution = self._run_alg_only_loads()
            return self.solution
        else:
            self.solution = self._run_alg()
            return self.solution

    def _run_alg_only_loads(self):

        self.mc.model.setParam("Method", 2)
        # self.mc.model.setParam("Crossover", 0)

        self._fractional_solution = self.mc.compute_fractional_solution()
        if self._fractional_solution is None:
            return None
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
        # for meta

        substrate_edges = list(self.scenario.substrate.edges)
        substrate_nodes_res = list(self.substrate_node_resources)

        meta_data = RandomizedRoundingMetaData(substrate_node_resource_names=substrate_nodes_res,
                                               substrate_edge_resource_names=substrate_edges,
                                               time_preprocessing=self.mc.time_preprocess,
                                               time_optimization=self.mc.time_optimization,
                                               time_postprocessing=self.mc.time_postprocessing)

        meta = (substrate_nodes_res, substrate_edges, self.mc.time_optimization,
                self.mc.time_postprocessing)
        self.substrate_resources = self.substrate_edge_resources + self.substrate_node_resources
        data = []
        for q in xrange(NUMBER_OF_ITERATIONS):
            time_rr0 = time.clock()
            (loadfactor_substrate_resources,
             loadfactor_substrate_edges), profit = self._rounding_iteration_only_loads()
            time_rr = time.clock() - time_rr0

            solution_tuple = RandomizedRoundingSolutionData(profit=profit,
                                                            loadfactor_substrate_nodes=loadfactor_substrate_resources,
                                                            loadfactor_substrate_edges=loadfactor_substrate_edges,
                                                            time_to_round_solution=time_rr,
                                                            lost_flow_in_decomposition=self.mc.lost_flow_in_the_decomposition)

            data.append(solution_tuple)

        result = RandomizedRoundingSolutionExtract(meta_data=meta_data,
                                                   solution_list=data)
        print result.meta_data
        for solution_data in result.solution_list:
            print solution_data

        return RandomizedRoundingResult(result, self.temporal_log, self.mc.status)

    def _run_alg(self):
        self._fractional_solution = self.mc.compute_fractional_solution()
        if self._fractional_solution is None:
            return None
        solution = None
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
        self.substrate_resources = self.substrate_edge_resources + self.substrate_node_resources
        for q in xrange(NUMBER_OF_ITERATIONS):
            solution = self._rounding_iteration()
            if solution is not None:
                break
        return solution

    def _rounding_iteration_only_loads(self):
        solution = solutions.IntegralScenarioSolution(self.scenario.name + "_solution", self.scenario)
        r_prime = set()
        B = 0.0
        L = self._initialize_load_dict()
        for req in self.scenario.get_requests():
            p = random.random()
            total_flow = 0.0
            fractional_mapping = None
            found_mapping = False

            if req in self._fractional_solution.request_mapping:
                for fractional_mapping in self._fractional_solution.request_mapping[req]:
                    total_flow += self._fractional_solution.mapping_flows[fractional_mapping.name]
                    if p < total_flow:
                        found_mapping = True
                        break
            if fractional_mapping is None or not found_mapping:
                mapping = solutions.Mapping(req.name + "_mapping", req, self.scenario.substrate, False)
            else:
                r_prime.add(req)
                B += req.profit
                mapping = solutions.Mapping(req.name + "_mapping", req, self.scenario.substrate, True)
                for rnode in req.nodes:
                    snode = fractional_mapping.get_mapping_of_node(rnode)
                    mapping.map_node(rnode, snode)
                for redge in req.edges:
                    sedge = fractional_mapping.mapping_edges[redge]
                    mapping.map_edge(redge, sedge)
                for res in self.substrate_resources:
                    L[res] += self._fractional_solution.mapping_loads[fractional_mapping.name][res]

            solution.add_mapping(req, mapping)
        loads = self._calc_loadfactor(B, L)
        return loads, B

    def _rounding_iteration(self):
        solution = solutions.IntegralScenarioSolution(self.scenario.name + "_solution", self.scenario)
        r_prime = set()
        B = 0.0
        L = self._initialize_load_dict()
        for req in self.scenario.get_requests():
            p = random.random()
            total_flow = 0.0
            fractional_mapping = None
            found_mapping = False
            if req in self._fractional_solution.request_mapping:
                for fractional_mapping in self._fractional_solution.request_mapping[req]:
                    total_flow += self._fractional_solution.mapping_flows[fractional_mapping.name]
                    if p < total_flow:
                        found_mapping = True
                        break
            if fractional_mapping is None or not found_mapping:
                raise RandomizedRoundingError("Random draw failed!")
            r_prime.add(req)
            B += req.profit
            mapping = solutions.Mapping(req.name + "_mapping", req, self.scenario.substrate, True)
            for rnode in req.nodes:
                snode = fractional_mapping.get_mapping_of_node(rnode)
                mapping.map_node(rnode, snode)
            for redge in req.edges:
                sedge = fractional_mapping.mapping_edges[redge]
                mapping.map_edge(redge, sedge)
            solution.add_mapping(req, mapping)
            for res in self.substrate_resources:
                L[res] += self._fractional_solution.mapping_loads[fractional_mapping.name][res]

        if self._check_validity(B, L):
            return solution

    def _check_validity(self, B, L):
        if B < ALPHA * self.mc.status.objValue:
            print "profit too low: {} < {}".format(B, ALPHA * self.mc.status.objValue)
            return False
        for (ntype, snode) in self.substrate_node_resources:
            max_node_load = (1.0 + BETA) * self.scenario.substrate.node[snode]["capacity"][ntype]
            if L[(ntype, snode)] > max_node_load:
                print "node violation: {} > {}".format(L[(ntype, snode)], max_node_load)
                return False
        for (u, v) in self.substrate_edge_resources:
            max_edge_load = (1.0 + GAMMA) * self.scenario.substrate.edge[(u, v)]["capacity"]
            if L[(u, v)] > max_edge_load:
                print "edge violation: {} > {}".format(L[(u, v)], max_edge_load)
                return False
        return True

    def _calc_loadfactor(self, B, L):
        loadfactor_substrate_resources = []
        loadfactor_substrate_edges = []
        for (ntype, snode) in self.substrate_node_resources:
            max_node_load = self.scenario.substrate.node[snode]["capacity"][ntype]
            ratio = L[(ntype, snode)] / max_node_load
            loadfactor_substrate_resources.append(ratio)
        for (u, v) in self.substrate_edge_resources:
            max_edge_load = self.scenario.substrate.edge[(u, v)]["capacity"]
            ratio = L[(u, v)] / max_edge_load
            loadfactor_substrate_edges.append(ratio)
        return (loadfactor_substrate_resources, loadfactor_substrate_edges)

    def _initialize_load_dict(self):
        L = {}
        sub = self.scenario.substrate
        for snode in sub.nodes:
            for ntype in sub.node[snode]["capacity"]:
                L[(ntype, snode)] = 0.0
        for u, v in sub.edges:
            L[(u, v)] = 0.0

        return L
