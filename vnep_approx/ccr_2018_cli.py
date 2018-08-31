# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost
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


import os
import sys

import click
import pickle
import treewidth_model_experiments
from collections import namedtuple

import alib.cli
from alib import run_experiment, util
from . import modelcreator_ecg_decomposition, randomized_rounding_triumvirate, treewidth_model
import evaluation_ifip_networking_2018

from . import ccr_2018_eval

@click.group()
def cli():
    pass


@cli.command()
@click.argument('scenario_pickle')
@click.argument('reduced_randround_pickle')
@click.argument('output_pickle')
@click.option('--number_of_instances', default=5)
def extract_ifip_scenarios(scenario_pickle,
                           reduced_randround_pickle,
                           output_pickle,
                           number_of_instances):
    print("Input is {} {} {}".format(scenario_pickle, reduced_randround_pickle, output_pickle))

    print("Reading reduced randround pickle{}..".format(reduced_randround_pickle))
    rrr_pickle = None
    with open(reduced_randround_pickle, "r") as f:
        rrr_pickle = pickle.load(f)
    print("Done: {}".format(rrr_pickle))

    rrr_sol_dict = rrr_pickle.algorithm_scenario_solution_dictionary['RandomizedRoundingTriumvirate']
    list_of_scenario_indices_with_runtime = []
    for scenario_index, value in rrr_sol_dict.iteritems():
        rand_round_solution = rrr_sol_dict[scenario_index][0]
        total_runtime = rand_round_solution.meta_data.time_preprocessing + \
                        rand_round_solution.meta_data.time_optimization + \
                        rand_round_solution.meta_data.time_postprocessing
        LPobjValue = rrr_sol_dict[scenario_index][0].meta_data.status.objValue
        print("Runtime of scenario with index {} is {}.".format(scenario_index, total_runtime))
        list_of_scenario_indices_with_runtime.append((scenario_index, total_runtime, LPobjValue))

    sorted_list_of_scenarios = sorted(list_of_scenario_indices_with_runtime, key= lambda x: x[1])
    print sorted_list_of_scenarios

    easy_instance_information = sorted_list_of_scenarios[:number_of_instances]
    hard_instance_information = sorted_list_of_scenarios[-number_of_instances:]

    print("Reading scenario pickle {}..".format(scenario_pickle))
    scenario_storage = None
    with open(scenario_pickle, "r") as f:
        scenario_storage = pickle.load(f)

    print("Done: {}".format(scenario_storage))

    def collect_data_to_pickle(instance_list):
        result_list = []
        for (scenario_index, total_runtime, LPobjValue) in instance_list:
            scenario = scenario_storage.scenario_list[scenario_index]
            rr_solution = rrr_sol_dict[scenario_index][0]
            result_list.append(ccr_2018_eval.CCREvaluationInstance(scenario, scenario_index, rr_solution, total_runtime, LPobjValue))
        return result_list

    easy_instances_to_pickle = collect_data_to_pickle(easy_instance_information)
    hard_instances_to_pickle = collect_data_to_pickle(hard_instance_information)

    with open("easy" + output_pickle, "w") as f:
        pickle.dump(easy_instances_to_pickle, f)
    with open("hard" + output_pickle, "w") as f:
        pickle.dump(hard_instances_to_pickle, f)


@cli.command()
@click.argument('scenario_pickle')
@click.argument('index', type=click.INT)
def execute_single_ccr(scenario_pickle, index):
    pickle_contents = None
    with open(scenario_pickle, "r") as f:
        pickle_contents = pickle.load(f)
    print pickle_contents
    print pickle_contents[index]

    scenario = pickle_contents[index]

    util.initialize_root_logger("debug.log", allow_override=True)

    alg = ccr_2018_eval.SeparationLP_DynVMP(ccr_eval_instance=scenario)
    alg.init_model_creator()
    alg.compute_solution()



if __name__ == '__main__':
    cli()