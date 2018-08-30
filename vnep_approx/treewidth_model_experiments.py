import itertools
import multiprocessing as mp
import os
import random
import time
import logging

import treewidth_model as twm
import yaml

from alib import datamodel, util

try:
    import cPickle as pickle
except ImportError:
    import pickle

random.seed(0)
logger = logging.getLogger(__name__)


def run_experiment_from_yaml(parameter_file, output_file_base_name, threads):
    param_space = yaml.load(parameter_file)
    sg = SimpleTreeDecompositionExperiment(threads, output_file_base_name)
    sg.start_experiments(param_space)


class SimpleTreeDecompositionExperiment(object):
    """Mostly copied from alib.scenariogeneration, but uses the build_scenario_simple function defined below instead."""

    def __init__(self, threads, output_file_base_name):
        self.threads = threads
        self.output_file_base_name = output_file_base_name
        self.output_files = [
            self.output_file_base_name.format(process_index=process_index)
            for process_index in range(self.threads)
        ]

    def start_experiments(self, scenario_parameter_space):
        repetition = 1
        if 'scenario_repetition' in scenario_parameter_space:
            repetition = scenario_parameter_space['scenario_repetition']
            del scenario_parameter_space['scenario_repetition']

        random_seed_base = 0
        if 'random_seed_base' in scenario_parameter_space:
            random_seed_base = scenario_parameter_space['random_seed_base']
            del scenario_parameter_space['random_seed_base']

        processes = [mp.Process(
            target=execute_single_experiment,
            name="worker_{}".format(process_index),
            args=(
                process_index,
                self.threads,
                scenario_parameter_space,
                random_seed_base + process_index,
                repetition,
                self.output_files[process_index],
            )) for process_index in range(self.threads)]

        for p in processes:
            logger.info("Starting process {}".format(p))
            p.start()

        for p in processes:
            p.join()

        self.combine_results_to_overall_pickle()

    def combine_results_to_overall_pickle(self):
        logger.info("Combining results")
        result_dict = {}
        for fname in self.output_files:
            with open(fname, "r") as f:
                try:
                    while True:
                        result = pickle.load(f)
                        if result.num_nodes not in result_dict:
                            result_dict[result.num_nodes] = {}
                        if result.probability not in result_dict[result.num_nodes]:
                            result_dict[result.num_nodes][result.probability] = []
                        result_dict[result.num_nodes][result.probability].append(result)
                except EOFError:
                    pass


        pickle_file = self.output_file_base_name.format(process_index="aggregated")
        logger.info("Writing combined Pickle to {}".format(pickle_file))
        with open(pickle_file, "w") as f:
            pickle.dump(result_dict, f)


def execute_single_experiment(process_index, num_processes, parameter_space, random_seed, repetitions, out_file):
    random.seed(random_seed)
    num_nodes_list = parameter_space["number_of_nodes"]
    connection_probabilities_list = parameter_space["probability"]

    graph_generator = SimpleRandomGraphGenerator()

    logger = util.get_logger("worker_{}_pid_{}".format(process_index, os.getpid()), propagate=False, make_file=True)

    for param_index, params in enumerate(itertools.product(
            num_nodes_list,
            connection_probabilities_list,
            range(repetitions)
    )):
        if param_index % num_processes == process_index:
            num_nodes, prob, repetition_index = params
            logger.info("Processing graph {} with {} nodes and {} prob, rep {}".format(param_index, num_nodes, prob, repetition_index))
            gen_time_start = time.time()
            graph = graph_generator.generate_graph(num_nodes, prob)
            gen_time = time.time() - gen_time_start

            algorithm_time_start = time.time()
            tree_decomp = twm.compute_tree_decomposition(graph)
            algorithm_time = time.time() - algorithm_time_start
            assert tree_decomp.is_tree_decomposition(graph)

            result = TreeDecompositionAlgorithmResult(
                parameter_index=param_index,
                num_nodes=num_nodes,
                probability=prob,
                num_edges=len(graph.edges),
                treewidth=tree_decomp.width,
                runtime_algorithm=algorithm_time,
            )
            logger.info("Result: {}".format(result))

            del graph
            del tree_decomp

            with open(out_file, "a") as f:
                pickle.dump(result, f)


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


class TreeDecompositionAlgorithmResult(object):
    def __init__(
            self,
            parameter_index,
            num_nodes,
            probability,
            num_edges,
            treewidth,
            runtime_algorithm,
    ):
        self.parameter_index = parameter_index
        self.treewidth = treewidth
        self.probability = probability
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.runtime_algorithm = runtime_algorithm

    def __str__(self):
        return "Result {}: num_nodes {} , probability {}, {} edges, Treewidth {}, Decomposition runtime {} s".format(
            self.parameter_index,
            self.num_nodes,
            self.probability,
            self.num_edges,
            self.treewidth,
            self.runtime_algorithm,
        )
