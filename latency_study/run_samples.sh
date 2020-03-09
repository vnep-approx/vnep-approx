#!/bin/bash

#either the $PYTHONPATH must be set to include alib, vnep_approx and evaluation_ifip_networking_2018 or 
#you execute this from within the virtual environment in which these packages were installed

# export ALIB_EXPERIMENT_HOME=$(pwd)

# mkdir -p log/ && mkdir -p input && mkdir -p output


set -e
shopt -s nullglob

function move_logs_and_output() {
    # echo "moving files"
	for file in $ALIB_EXPERIMENT_HOME/output/*; 	do mv $file $ALIB_EXPERIMENT_HOME/input/; done
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
    python -m vnep_approx.cli generate_scenarios scenarios2.pickle latency_scenarios.yml
    move_logs_and_output
}

function run_baseline() {
    echo "Run Baseline"
    python -m vnep_approx.cli start_experiment baseline_execution.yml 0 10000 --concurrent 2  --remove_temporary_scenarios --remove_intermediate_solutions
    move_logs_and_output
}

function run_approx() {
    echo "Run Approx"
    python -m vnep_approx.cli start_experiment latency_execution.yml 0 10000 --concurrent 2 --remove_temporary_scenarios --remove_intermediate_solutions
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
function eval() {
    echo "Evaluate"
    python -m evaluation_acm_ccr_2019.cli evaluate_separation_with_latencies baseline_results_reduced.pickle latency_study_results_reduced.pickle ./plots/ --filter_parameter_keys "$FILTER_GENERATION_PARAMS" --filter_exec_params "$EXCLUDE_EXEC_PARAMS" --output_filetype png
    move_logs_and_output
}


deletelog

if [ "$NEW_SCENARIOS" = true ]
    then new_scenarios
    else echo "Skipping Scenario generation"
fi

if [ "$RUN_APPROX" = true ]
    then run_approx
    else echo "Skipping Approx execution"
fi

if [ "$RUN_BASELINE" = true ]
    then run_baseline
    else echo "Skipping Baseline execution"
fi

if [ "$RUN_APPROX" = true ]
    then reduce_approx
    else echo "Skipping Approx reduction"
fi

if [ "$RUN_BASELINE" = true ]
    then reduce_baseline
    else echo "Skipping Baseline reduction"
fi


#generate plots in folder ./plots
mkdir -p ./plots

if [ "$EVAL" = true ]
    then eval
    else echo "Skipping Evaluation"
fi

rm gurobi.log