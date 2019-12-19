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

import os

import click

import alib.cli
from alib import run_experiment, util
from . import modelcreator_ecg_decomposition, randomized_rounding_triumvirate
from . import treewidth_model
from . import vine
import logging



@click.group()
def cli():
    pass

def initialize_logger(filename, log_level_print, log_level_file, allow_override=False):
    log_level_print = logging._levelNames[log_level_print.upper()]
    log_level_file = logging._levelNames[log_level_file.upper()]
    util.initialize_root_logger(filename, log_level_print, log_level_file, allow_override=allow_override)


@cli.command()
@click.argument('scenario_output_file')
@click.argument('parameters', type=click.File('r'))
@click.option('--threads', default=1)
def generate_scenarios(scenario_output_file, parameters, threads):
    alib.cli.f_generate_scenarios(scenario_output_file, parameters, threads)


@cli.command()
@click.argument('experiment_yaml', type=click.File('r'))
@click.argument('min_scenario_index', type=click.INT)
@click.argument('max_scenario_index', type=click.INT)
@click.option('--concurrent', default=1, help="number of processes to be used in parallel")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
@click.option('--shuffle_instances/--original_order', default=True, help="shall instances be shuffled or ordered according to their ids (ascendingly)")
@click.option('--overwrite_existing_temporary_scenarios/--use_existing_temporary_scenarios', default=False, help="shall existing temporary scenario files be overwritten or used?")
@click.option('--overwrite_existing_intermediate_solutions/--use_existing_intermediate_solutions', default=False, help="shall existing intermediate solution files be overwritten or used?")
@click.option('--remove_temporary_scenarios/--keep_temporary_scenarios', is_flag=True, default=False, help="shall temporary scenario files be removed after execution?")
@click.option('--remove_intermediate_solutions/--keep_intermediate_solutions', is_flag=True, default=False, help="shall intermediate solutions be removed after execution?")
def start_experiment(experiment_yaml,
                     min_scenario_index,
                     max_scenario_index,
                     concurrent,
                     log_level_print,
                     log_level_file,
                     shuffle_instances,
                     overwrite_existing_temporary_scenarios,
                     overwrite_existing_intermediate_solutions,
                     remove_temporary_scenarios,
                     remove_intermediate_solutions
                     ):
    click.echo('Start Experiment')
    f_start_experiment(experiment_yaml.name,
                       min_scenario_index,
                       max_scenario_index,
                       concurrent,
                       log_level_print,
                       log_level_file,
                       shuffle_instances,
                       overwrite_existing_temporary_scenarios,
                       overwrite_existing_intermediate_solutions,
                       remove_temporary_scenarios,
                       remove_intermediate_solutions
                       )


def f_start_experiment(experiment_yaml,
                       min_scenario_index,
                       max_scenario_index,
                       concurrent,
                       log_level_print,
                       log_level_file,
                       shuffle_instances=True,
                       overwrite_existing_temporary_scenarios=False,
                       overwrite_existing_intermediate_solutions=False,
                       remove_temporary_scenarios=False,
                       remove_intermediate_solutions=False
                       ):
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(experiment_yaml).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_experiment_execution.log".format(file_basename))

    initialize_logger(log_file, log_level_print, log_level_file)

    run_experiment.register_algorithm(
        modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition.ALGORITHM_ID,
        modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition
    )

    run_experiment.register_algorithm(
        treewidth_model.SeparationLP_OptDynVMP.ALGORITHM_ID,
        treewidth_model.SeparationLP_OptDynVMP
    )

    run_experiment.register_algorithm(
        randomized_rounding_triumvirate.RandomizedRoundingTriumvirate.ALGORITHM_ID,
        randomized_rounding_triumvirate.RandomizedRoundingTriumvirate
    )


    run_experiment.register_algorithm(
        vine.OfflineViNEAlgorithmCollection.ALGORITHM_ID,
        vine.OfflineViNEAlgorithmCollection
    )


    run_experiment.register_algorithm(
        treewidth_model.RandRoundSepLPOptDynVMPCollection.ALGORITHM_ID,
        treewidth_model.RandRoundSepLPOptDynVMPCollection
    )
    with open(experiment_yaml, "r") as actual_experiment_yaml:
        run_experiment.run_experiment(
            actual_experiment_yaml,
            min_scenario_index,
            max_scenario_index,
            concurrent,
            shuffle_instances,
            overwrite_existing_temporary_scenarios,
            overwrite_existing_intermediate_solutions,
            remove_temporary_scenarios,
            remove_intermediate_solutions
        )


if __name__ == '__main__':
    cli()
