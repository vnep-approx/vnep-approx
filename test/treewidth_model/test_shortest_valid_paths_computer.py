import pytest
from vnep_approx import treewidth_model

import numpy as np

from alib import mip
from alib import datamodel as dm
from alib import util
from test_data.request_test_data import create_test_request
from test_data.substrate_test_data import create_test_substrate_topology_zoo
from test_data.substrate_test_data import create_test_substrate_topology_zoo

import random
import time
import logging
import itertools


logger = util.get_logger(__name__, make_file=False, propagate=True)

random.seed(0)

# TEST Valid Shortest Path Computer
@pytest.mark.parametrize("substrate_id", ["BtAsiaPac", "DeutscheTelekom", "Geant2012", "Surfnet", "Dfn"])
@pytest.mark.parametrize("cost_spread", [-1, 0.5, 1.0, 2.0, 4.0, 8.0])  #cost spread of -1 will test uniform costs
def test_shortest_valid_paths_computer_no_latencies(substrate_id, cost_spread):
    req = create_test_request("single edge", set_allowed_nodes=False)
    sub = create_test_substrate_topology_zoo(substrate_id, include_latencies=False)
    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    if cost_spread == -1:
        # uniform edge costs
        edge_costs = {sedge: 1.0 for sedge in sub.edges}
        svpc_dijkstra = treewidth_model.ShortestValidPathsComputer.createSVPC(
            treewidth_model.ShortestValidPathsComputer.Approx_NoLatencies, sub, vmrc, edge_costs)
        svpc_dijkstra.compute()

        for reqedge in req.edges:
            for snode_source in sub.nodes:
                for snode_target in sub.nodes:
                    if snode_source == snode_target:
                        assert svpc_dijkstra.valid_sedge_costs[reqedge][(snode_source, snode_target)] == 0
                    else:
                        assert svpc_dijkstra.valid_sedge_costs[reqedge][(snode_source, snode_target)] >= 1
    else:
        # random edge costs

        edge_costs = {sedge: max(1, 1000.0 * random.random()) for sedge in sub.edges}
        for sedge in sub.edges:
            sub.edge[sedge]['cost'] = edge_costs[sedge]

        bellman_ford_time = time.time()
        sub.initialize_shortest_paths_costs()
        bellman_ford_time = time.time() - bellman_ford_time

        svpc_dijkstra = treewidth_model.ShortestValidPathsComputer.createSVPC(
            treewidth_model.ShortestValidPathsComputer.Approx_NoLatencies, sub, vmrc, edge_costs)
        dijkstra_time = time.time()
        svpc_dijkstra.compute()
        dijkstra_time = time.time() - dijkstra_time

        for reqedge in req.edges:
            for snode_source in sub.nodes:
                for snode_target in sub.nodes:
                    # print svpc.valid_sedge_costs[reqedge][(snode_source, snode_target)]
                    # print sub.get_shortest_paths_cost(snode_source, snode_target)
                    assert svpc_dijkstra.valid_sedge_costs[reqedge][(snode_source, snode_target)] == pytest.approx(
                        sub.get_shortest_paths_cost(snode_source, snode_target))

        logger.info(
            "\nComputation times were:\n\tBellman-Ford: {:2.4f}\n"
            "\tDijkstra:     {:2.4f}\n"
            "\tSpeedup by using Dijkstra over Bellman: {:2.2f} (<1 is bad)\n".format(
                bellman_ford_time, dijkstra_time, (bellman_ford_time / dijkstra_time)))




@pytest.mark.parametrize("substrate_id", ["BtAsiaPac"])#, "DeutscheTelekom", "Geant2012", "Surfnet", "Dfn"])
@pytest.mark.parametrize("cost_spread", [0.5, 1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("epsilon", [1.0, 0.5, 0.1, 0.01])
@pytest.mark.parametrize("limit_factor", [8.0, 4.0, 2.0, 1.0, 0.5])
def test_shortest_valid_paths_with_latencies(substrate_id, cost_spread, epsilon, limit_factor):
    return
    req = create_test_request("single edge", set_allowed_nodes=False)
    sub = create_test_substrate_topology_zoo(substrate_id, include_latencies=True)
    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    edge_costs = {sedge: cost_spread*random.random()+1.0 for sedge in sub.edges}
    for sedge in sub.edges:
        sub.edge[sedge]['cost'] = edge_costs[sedge]

    maximal_latency_upper_bound = sum([sub.edge[sedge]["latency"] for sedge in sub.edges])
    minimum_edge_cost = min([sub.edge[sedge]["cost"] for sedge in sub.edges])
    average_latency = maximal_latency_upper_bound / len(sub.edges)

    edge_costs = {sedge: sub.edge[sedge]['cost'] for sedge in sub.edges}
    edge_latencies = {sedge: sub.edge[sedge]['latency'] for sedge in sub.edges}
    limit = average_latency * limit_factor

    runtime_exact_mip = 0.0
    runtime_approx_mip = 0.0
    runtime_strict = 0.0
    runtime_flex = 0.0

    def time_computation(spvc):
        start_time = time.time()
        spvc.compute()
        return time.time() - start_time

    def compute_latency_of_path(sedge_path):
        if sedge_path is None:
            return 0.0
        return sum([sub.edge[sedge]["latency"] for sedge in sedge_path])

    def nan_to_negative_value(value):
        if np.isnan(value):
            # this guarantees that this cost is ..
            # a) negative and
            # b) the absolute value of the returned cost is smaller than the minimum cost value
            return -minimum_edge_cost / (10*max(epsilon, 1/epsilon))
        return value

    svpc_exact_mip = treewidth_model.ShortestValidPathsComputer.createSVPC(
        treewidth_model.ShortestValidPathsComputer.Approx_Exact_MIP,
        sub,
        vmrc,
        edge_costs,
        edge_latencies=edge_latencies,
        limit=limit,
        epsilon=0.0)

    svpc_approximate_mip = treewidth_model.ShortestValidPathsComputer.createSVPC(
        treewidth_model.ShortestValidPathsComputer.Approx_Exact_MIP,
        sub, vmrc, edge_costs,
        edge_latencies=edge_latencies,
        limit=limit,
        epsilon=epsilon)

    svpc_strict = treewidth_model.ShortestValidPathsComputer.createSVPC(
        treewidth_model.ShortestValidPathsComputer.Approx_Strict, sub, vmrc, edge_costs,
        edge_latencies=edge_latencies,
        limit=limit,
        epsilon=epsilon)

    svpc_flex = treewidth_model.ShortestValidPathsComputer.createSVPC(
        treewidth_model.ShortestValidPathsComputer.Approx_Flex, sub, vmrc, edge_costs,
        edge_latencies=edge_latencies,
        limit=limit,
        epsilon=epsilon)

    logger.info("\n\n========================================================================================================\n\n"
                "Considering now a latency limit of {} (average latency is {}), an epsilon of {} and a cost spread of {}\n\n"
                "========================================================================================================\n\n".format(limit, average_latency, epsilon, cost_spread))


    logger.info("\n\nStarting exact MIP...\n\n")
    runtime_exact_mip += time_computation(svpc_exact_mip)
    logger.info("\n\nStarting approximate MIP...\n\n")
    runtime_approx_mip += time_computation(svpc_approximate_mip)
    logger.info("\n\nStarting strict...\n\n")
    runtime_strict += time_computation(svpc_strict)
    logger.info("\n\nStarting flex ...\n\n")
    runtime_flex += time_computation(svpc_flex)

    logger.info(
        "\t{:^6s} | {:^6s} || {:^15s} | {:^15s} | {:^15s} | {:^15s} || {:^15s} | {:^15s} || {:^15s} | {:^15s} | {:^15s} | {:^15s}".format("Source", "Target", "c(Flex)", "c(Exact-MIP)",
                                                                           "c(Approx-MIP)", "c(Strict)", "epsilon", "latency_bound", "l(Flex)", "l(Exact-MIP)",
                                                                           "l(Approx-MIP)", "l(Strict)"))

    failure_counts = {alg: {"cost": 0, "lat": 0} for alg in ["exact_mip", "approx_mip", "flex", "strict"]}


    for reqedge in req.edges:
        for snode_source in sub.nodes:
            for snode_target in sub.nodes:
                if snode_source == snode_target:
                    assert svpc_exact_mip.get_valid_sedge_costs_for_reqedge(reqedge,
                                                                            (snode_source, snode_target)) == 0.0
                    assert svpc_approximate_mip.get_valid_sedge_costs_for_reqedge(reqedge,
                                                                            (snode_source, snode_target)) == 0.0
                    assert svpc_strict.get_valid_sedge_costs_for_reqedge(reqedge,
                                                                            (snode_source, snode_target)) == 0.0
                    assert svpc_flex.get_valid_sedge_costs_for_reqedge(reqedge,
                                                                            (snode_source, snode_target)) == 0.0
                else:
                    cost_flex = nan_to_negative_value(svpc_flex.get_valid_sedge_costs_for_reqedge(reqedge, (snode_source, snode_target)))
                    cost_exact_mip = nan_to_negative_value(svpc_exact_mip.get_valid_sedge_costs_for_reqedge(reqedge, (snode_source, snode_target)))
                    cost_approx_mip = nan_to_negative_value(svpc_approximate_mip.get_valid_sedge_costs_for_reqedge(reqedge, (snode_source, snode_target)))
                    cost_strict =  nan_to_negative_value(svpc_strict.get_valid_sedge_costs_for_reqedge(reqedge, (snode_source, snode_target)))

                    path_flex = svpc_flex.get_valid_sedge_path(reqedge, snode_source, snode_target)
                    path_exact_mip = svpc_exact_mip.get_valid_sedge_path(reqedge, snode_source, snode_target)
                    path_approx_mip = svpc_approximate_mip.get_valid_sedge_path(reqedge, snode_source, snode_target)
                    path_strict = svpc_strict.get_valid_sedge_path(reqedge, snode_source, snode_target)


                    lat_flex = compute_latency_of_path(path_flex)
                    lat_exact_mip = compute_latency_of_path(path_exact_mip)
                    lat_approx_mip = compute_latency_of_path(path_approx_mip)
                    lat_strict = compute_latency_of_path(path_strict)

                    failure_dict = {alg : {"cost": False, "lat": False} for alg in ["exact_mip", "approx_mip", "flex", "strict"]}

                    def value_lies_outside_of_range(value, reference_value, lower_factor, upper_factor):
                        result = False
                        result |= (abs(value) < abs(reference_value) * lower_factor)
                        result |= (abs(value) > abs(reference_value) * upper_factor)
                        return result

                    def bool_to_failure_output(boolean_value):
                        if boolean_value:
                            return "FAILED"
                        else:
                            return "PASSED"

                    failure_dict["approx_mip"]["cost"] |= value_lies_outside_of_range(cost_approx_mip, cost_exact_mip, 0.999, 1.001 + epsilon)
                    failure_dict["strict"]["cost"] |= value_lies_outside_of_range(cost_strict, cost_exact_mip, 0.999, 1.001 + epsilon)
                    failure_dict["flex"]["cost"] |= value_lies_outside_of_range(cost_flex, cost_exact_mip, 0.0, 1.001)

                    failure_dict["exact_mip"]["lat"] |= value_lies_outside_of_range(lat_exact_mip, limit, 0.0, 1.001)
                    failure_dict["approx_mip"]["lat"] |= value_lies_outside_of_range(lat_approx_mip, limit, 0.0, 1.001)
                    failure_dict["strict"]["lat"] |= value_lies_outside_of_range(lat_strict, limit, 0.0, 1.001)
                    failure_dict["flex"]["lat"] |= value_lies_outside_of_range(lat_exact_mip, limit, 0.0, 1.001 + epsilon)

                    failure_found = any([failure_dict[alg][type] for alg in failure_dict for type in failure_dict[alg]])
                    failure_message = None

                    output_message = "\t{:^6s} | {:^6s} || {:^15.4f} | {:^15.4f} | {:^15.4f} | {:^15.4f} || {:^15.4f} | {:^15.4f} || {:^15.4f} | {:^15.4f} | {:^15.4f} | {:^15.4f} ".format(
                        snode_source,
                        snode_target,
                        cost_flex,
                        cost_exact_mip,
                        cost_approx_mip,
                        cost_strict,
                        epsilon,
                        limit,
                        lat_flex,
                        lat_exact_mip,
                        lat_approx_mip,
                        lat_strict
                    )
                    if failure_found:
                        failure_message = "\t{:^6s} | {:^6s} || {:^15s} | {:^15s} | {:^15s} | {:^15s} || {:^15.4f} | {:^15.4f} || {:^15s} | {:^15s} | {:^15s} | {:^15s} ".format(
                            snode_source,
                            snode_target,
                            bool_to_failure_output(failure_dict["flex"]["cost"]),
                            bool_to_failure_output(failure_dict["exact_mip"]["cost"]),
                            bool_to_failure_output(failure_dict["approx_mip"]["cost"]),
                            bool_to_failure_output(failure_dict["strict"]["cost"]),
                            epsilon,
                            limit,
                            bool_to_failure_output(failure_dict["flex"]["lat"]),
                            bool_to_failure_output(failure_dict["exact_mip"]["lat"]),
                            bool_to_failure_output(failure_dict["approx_mip"]["lat"]),
                            bool_to_failure_output(failure_dict["strict"]["lat"])
                        )

                    if failure_found:
                        logger.error(output_message)
                        logger.error(failure_message)
                    else:
                        logger.debug(output_message)

                    for alg in failure_dict:
                        for type in failure_dict[alg]:
                            if failure_dict[alg][type]:
                                failure_counts[alg][type] += 1

    logger.info("Runtimes are \n"
                "\tExact-MIP:    {:10.4f}\n"
                "\tApprox-MIP:   {:10.4f}\n"
                "\tStrict:       {:10.4f}\n"
                "\tFlex:         {:10.4f}\n\n\n".format(runtime_exact_mip,
                                                        runtime_approx_mip,
                                                        runtime_strict,
                                                        runtime_flex))

    number_of_failed_tests = sum([failure_counts[alg][type] for alg in failure_counts for type in failure_counts[alg]])
    logger.info("Total number of failures: {}\n".format(number_of_failed_tests))

    number_of_node_combinations = len(sub.nodes) * len(sub.nodes)
    for alg in failure_counts:
        for type in failure_counts[alg]:
            if failure_counts[alg][type] > 0:
                logger.error("\tSummary\t{:^15s} {:^15s}: {:4d} failed of {:4d} ({:6.3f}%)".format(alg, type, failure_counts[alg][type], number_of_node_combinations, 100.0*failure_counts[alg][type]/float(number_of_node_combinations)))
            else:
                logger.info(
                    "\t\Summary\t{:^15s} {:^15s}: {:4d} failed of {:4d} ({:6.3f}%)".format(alg, type, failure_counts[alg][type],
                                                                                 number_of_node_combinations,
                                                                                 100.0 * failure_counts[alg][type] / float(
                                                                                     number_of_node_combinations)))

    assert number_of_failed_tests == 0