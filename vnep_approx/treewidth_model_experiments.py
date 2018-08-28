import multiprocessing as mp
import os
import random
import time

import treewidth_model as twm
import yaml

from alib import datamodel, modelcreator, util, scenariogeneration

try:
    import cPickle as pickle
except ImportError:
    import pickle

random.seed(0)

"""Scenariogeneration: Mostly copied from alib to simplify it. Reuses the ScenarioParameterContainer from the alib"""


def generate_pickle_from_yml(parameter_file, scenario_output_file, threads, scenario_index_offset=0):
    param_space = yaml.load(parameter_file)
    sg = SimpleTreeDecompositionExperimentScenarioGenerator(threads)
    repetition = 1
    if 'scenario_repetition' in param_space:
        repetition = param_space['scenario_repetition']
        del param_space['scenario_repetition']
    sg.generate_scenarios(param_space, repetition, scenario_index_offset=scenario_index_offset)
    container = sg.scenario_parameter_container
    out = os.path.abspath(os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                       scenario_output_file))
    with open(out, "wb") as f:
        pickle.dump(container, f)


class SimpleTreeDecompositionExperimentScenarioGenerator(object):
    """Mostly copied from alib.scenariogeneration, but uses the build_scenario_simple function defined below instead."""

    def __init__(self, threads):
        self.threads = threads

    def generate_scenarios(self, scenario_parameter_space, repetition=1, scenario_index_offset=0):
        self.repetition = repetition
        self.scenario_parameter_container = scenariogeneration.ScenarioParameterContainer(scenario_parameter_space, scenario_index_offset=scenario_index_offset)
        self.scenario_parameter_container.generate_all_scenario_parameter_combinations(repetition)
        scenario_parameter_combination_list = self.scenario_parameter_container.scenario_parameter_combination_list
        self.scenario_parameter_container.init_reverselookup_dict()
        if self.threads > 1:
            iterator = self._multiprocessed(scenario_parameter_combination_list, scenario_index_offset=scenario_index_offset)
        else:
            iterator = self._singleprocessed(scenario_parameter_combination_list, scenario_index_offset=scenario_index_offset)
        for i, scenario, sp in iterator:
            self.scenario_parameter_container.fill_reverselookup_dict(sp, i)
            self.scenario_parameter_container.scenario_list.append(scenario)
            self.scenario_parameter_container.scenario_triple[i] = (sp, scenario)
        return self.scenario_parameter_container.scenario_triple

    def _singleprocessed(self, scenario_parameter_combination_list, scenario_index_offset=0):
        for i, sp in enumerate(scenario_parameter_combination_list, scenario_index_offset):
            yield build_scenario_simple((i, sp))

    def _multiprocessed(self, scenario_parameter_combination_list, scenario_index_offset=0):
        proc_pool = mp.Pool(processes=self.threads, maxtasksperchild=100)
        for out in proc_pool.map(build_scenario_simple, list(enumerate(scenario_parameter_combination_list, scenario_index_offset))):
            yield out
        proc_pool.close()
        proc_pool.join()


def build_scenario_simple(index_parameter_tuple):
    """
    A simpler version of the alib.scenariogeneration.build_scenario function, which
    only performs request graph generation, ignoring the more complex alib scenariogeneration
    framework.

    :param index_parameter_tuple:
    :return:
    """
    i, parameters = index_parameter_tuple
    dummy_substrate = None
    graph_generator = SimpleRandomGraphGenerator()
    requests = None
    assert len(parameters["request_generation"]) == 1

    strategy, class_param_dict = parameters["request_generation"].items()[0]
    assert len(class_param_dict) == 1
    generation_class, raw_parameters = class_param_dict.items()[0]
    requests = graph_generator.generate_request_list(raw_parameters)

    scenario = datamodel.Scenario(
        name="scenario_{}_rep_{}".format(i / parameters['maxrepetition'], parameters['repetition']),
        substrate=dummy_substrate,
        requests=requests,
        objective=datamodel.Objective.MIN_COST,
    )

    return i, scenario, parameters


class SimpleRandomGraphGenerator(object):
    """
    Mostly copied from alib.scenariogeneration, since much of the original code adds unnecessary complexity (costs, demands, etc)
    """

    EXPECTED_PARAMETERS = [
        "number_of_nodes",
        "probability"
    ]

    def __init__(self):
        pass

    def generate_request_list(self, raw_parameters):
        number_nodes = raw_parameters["number_of_nodes"]
        connection_probability = raw_parameters["probability"]
        graphs = []
        graph = self.generate_graph(number_nodes, connection_probability)
        graphs.append(graph)
        return graphs

    def generate_graph(self, number_of_nodes, connection_probability):
        name = "req"
        req = datamodel.Graph(name)

        # create nodes
        for i in xrange(1, number_of_nodes + 1):
            req.add_node(str(i))

        # create edges
        for i in req.nodes:
            for j in req.nodes:
                if i == j:
                    continue
                if random.random() <= connection_probability:
                    req.add_edge(i, j)
        return req


"""
Execution-related code: Mostly reuses the alib framework, by imitating the structure of a LP/MIP-based
modelcreator from the alib

"""


class TreeDecompositionAlgorithmResult(modelcreator.AlgorithmResult):

    def __init__(
            self,
            scenario,
            tree_decompositions,
            runtime_preprocessing,
            runtime_algorithm,
            runtime_postprocessing,
            individual_runtimes,
    ):
        self.scenario = scenario
        self.runtime_preprocessing = runtime_preprocessing
        self.runtime_algorithm = runtime_algorithm
        self.runtime_postprocessing = runtime_postprocessing
        self.tree_decompositions = tree_decompositions
        self.individual_runtimes = individual_runtimes

    def get_solution(self):
        return self.tree_decompositions

    def cleanup_references(self, original_scenario):
        pass

    def __str__(self):
        from pprint import pprint
        return pprint.pformat(vars(self))


class EvaluateTreeDecomposition(object):
    ALGORITHM_ID = "EvaluateTreeDecomposition"

    def __init__(self, scenario, gurobi_settings=None, logger=None):
        self.scenario = scenario
        self.logger = logger
        # self.parameters = parameters

        self.runtime_preprocessing = None
        self.runtime_algorithm = None
        self.runtime_postprocessing = None

        self.individual_execution_times = None

    def init_model_creator(self):
        time_preprocess_start = time.clock()

        # TODO: Any Preprocessing goes here
        self.individual_execution_times = []

        self.runtime_preprocessing = time.clock() - time_preprocess_start

    def compute_integral_solution(self):
        time_optimization_start = time.clock()
        tree_decompositions = []
        for g in self.scenario.requests:
            start_time = time.clock()
            tree_decomposition = twm.compute_tree_decomposition(g)
            end_time = time.clock()
            self.individual_execution_times.append(end_time - start_time)

            tree_decompositions.append(tree_decomposition)

        self.runtime_algorithm = time.clock() - time_optimization_start

        # do the postprocessing
        time_postprocess_start = time.clock()

        # TODO: Any Postprocessing

        self.runtime_postprocessing = time.clock() - time_postprocess_start

        result = TreeDecompositionAlgorithmResult(
            self.scenario,
            tree_decompositions,
            self.runtime_preprocessing,
            self.runtime_algorithm,
            self.runtime_postprocessing,
            self.individual_execution_times,
        )

        return result

    def cleanup_references(self, original_scenario):
        pass
