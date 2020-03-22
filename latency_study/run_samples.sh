#!/bin/bash

#either the $PYTHONPATH must be set to include alib, vnep_approx and evaluation_ifip_networking_2018 or 
#you execute this from within the virtual environment in which these packages were installed

# export ALIB_EXPERIMENT_HOME=$(pwd)

# mkdir -p log/ && mkdir -p input && mkdir -p output


set -e
shopt -s nullglob

function deletelog() {
  rm -r log
  mkdir log
}

export ALIB_EXPERIMENT_HOME="."

function move_logs_and_output() {
    # echo "moving files"
	mv $ALIB_EXPERIMENT_HOME/output/* $ALIB_EXPERIMENT_HOME/input/
	deletelog
}


NEW_SCENARIOS=false
RUN_APPROX=false
RUN_BASELINE=false
EVAL=true

EXCLUDE_EXEC_PARAMS="{'latency_approximation_factor': 0.1}"
#{'latency_approximation_type': 'strict', 'latency_approximation_limit': 10, 'latency_approximation_factor': 0.1}" # , 'latency_approximation_limit': 10

FILTER_GENERATION_PARAMS="None"#"['edge_resource_factor', 'node_resource_factor']"
#, 'edge_resource_factor', 'node_resource_factor']" 'topology',
#"None"['topology']"['number_of_requests', 'edge_resource_factor', 'node_resource_factor']"


function new_scenarios() {
    echo "Generate Scenarios"
    python -m vnep_approx.cli generate_scenarios scenarios.pickle latency_scenarios.yml
    move_logs_and_output
}

function run_baseline() {
    echo "Run Baseline"
    python -m vnep_approx.cli start_experiment baseline_execution.yml 0 10000 --concurrent 2  --remove_temporary_scenarios --remove_intermediate_solutions
    move_logs_and_output
}

function run_approx() {
    echo "Run Approx"
    python -m vnep_approx.cli start_experiment latency_execution.yml 0 10000 --concurrent 8 --remove_temporary_scenarios --remove_intermediate_solutions
    move_logs_and_output
}

function reduce_baseline() {
    echo "Reduce Baseline"
    python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp baseline_results.pickle
    move_logs_and_output
}
function reduce_approx() {
    echo "Reduce Approx"
    python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp latency_study_results.pickle
    move_logs_and_output
}

rm -r log
#rm -r input
rm -r output
mkdir -p log input output

#new_scenarios

run_approx

run_baseline

reduce_approx

reduce_baseline

#generate plots in folder ./plots
#mkdir -p ./plots
#
#if [ "$EVAL" = true ]
#    then eval
#    else echo "Skipping Evaluation"
#fi

rm gurobi.log