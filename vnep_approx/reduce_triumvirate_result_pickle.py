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

import os

from alib import util, solutions
from . import randomized_rounding_triumvirate

try:
    import cPickle as pickle
except ImportError:
    import pickle

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions


def compress_pickles():
    util.ExperimentPathHandler.initialize()

    with open(os.path.join(util.ExperimentPathHandler.INPUT_DIR,
                           "sigmetrics_redone_solutions_triumvirat.pickle"),
              "r") as f:
        sss = pickle.load(f)

    print sss
    print "\n".join(str(x) for x in sss.algorithm_scenario_solution_dictionary.values()[0].iteritems())

    sss.scenario_parameter_container.scenario_list = None
    sss.scenario_parameter_container.scenario_triple = None

    for alg, scenario_solution_dict in sss.algorithm_scenario_solution_dictionary.iteritems():
        for sc_id, ex_param_solution_dict in scenario_solution_dict.iteritems():
            for ex_id, solution in ex_param_solution_dict.iteritems():
                compressed = reduce_single_solution(solution)
                ex_param_solution_dict[ex_id] = compressed

    with open(os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                           "sigmetrics_redone_solutions_triumvirat_reduced.pickle"),
              "w") as f:
        pickle.dump(sss, f)


def reduce_single_solution(solution):
    if solution is None:
        return None
    avg_runtime = get_avg_runtime(solution)
    best_feasible = get_best_feasible_or_least_violating_solution(solution)
    best_objective = get_highest_obj_sol(solution)
    del solution.collection_of_samples_with_violations[:]

    # set the time of both to avg_runtime
    best_feasible = best_feasible._asdict()
    best_feasible["time_to_round_solution"] = avg_runtime
    best_feasible = randomized_rounding_triumvirate.RandomizedRoundingSolutionData(**best_feasible)

    best_objective = best_objective._asdict()
    best_objective["time_to_round_solution"] = avg_runtime
    best_objective = randomized_rounding_triumvirate.RandomizedRoundingSolutionData(**best_objective)

    solution.collection_of_samples_with_violations.append(best_feasible)
    solution.collection_of_samples_with_violations.append(best_objective)
    return solution


def get_avg_runtime(full_solution):
    t = 0.0
    for sample in full_solution.collection_of_samples_with_violations:
        if sample is None:
            continue
        t += sample.time_to_round_solution

    return t / len(full_solution.collection_of_samples_with_violations)


def get_best_feasible_or_least_violating_solution(full_solution):
    print "Getting the best feasible solution"
    best_sample = None
    best_max_load = None
    best_obj = None
    for sample in full_solution.collection_of_samples_with_violations:
        if sample is None:
            continue
        sample_max_load = max(sample.max_node_load, sample.max_edge_load)
        sample_obj = sample.profit

        replace_best = False
        if best_sample is None:  # initialize with any sample
            replace_best = True
        elif best_max_load > 1.0:
            if sample_max_load < best_max_load:
                # current best result is violating the cap. and we found one with smaller violation
                replace_best = True
        elif best_max_load <= 1.0:
            if sample_max_load <= 1.0 and sample_obj > best_obj:
                # current best result is feasible but we found a
                # feasible solution with a better objective
                replace_best = True

        if replace_best:
            print "    Replacing:"
            print "          Old:", best_sample
            print "          New:", sample
            best_sample = sample
            best_max_load = sample_max_load
            best_obj = sample_obj
        else:
            print "      Discard:", sample
    print "Best feasible with obj {} and max load {}: {}".format(best_obj, best_max_load, best_sample)
    return best_sample


def get_highest_obj_sol(full_sol):
    print "Getting the solution with the greatest objective"
    best_sample = None
    best_max_load = None
    best_obj = None
    for sample in full_sol.collection_of_samples_with_violations:
        if sample is None:
            continue
        sample_max_load = max(sample.max_node_load, sample.max_edge_load)
        sample_obj = sample.profit

        replace_best = False
        if best_sample is None:  # initialize with any sample
            replace_best = True
        elif abs(sample_obj - best_obj) < 0.0001:  # approx equal objective
            replace_best = (sample_max_load < best_max_load)  # replace if smaller load
        elif sample_obj > best_obj:  # significantly better objective
            replace_best = True

        if replace_best:
            print "    Replacing:"
            print "          Old:", best_sample
            print "          New:", sample
            best_sample = sample
            best_max_load = sample_max_load
            best_obj = sample_obj
        else:
            print "      Discard:", sample
    print "Best objective with obj {} and max load {}: {}".format(best_obj, best_max_load, best_sample)
    return best_sample


if __name__ == '__main__':
    compress_pickles()
