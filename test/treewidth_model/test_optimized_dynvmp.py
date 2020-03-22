import pytest
from vnep_approx import treewidth_model

import numpy as np

from alib import mip
from alib import datamodel as dm
from test_data.request_test_data import create_test_request, example_requests, example_requests_small
from test_data.substrate_test_data import create_test_substrate_topology_zoo

import random
import time

random.seed(0)


# TEST Valid Mapping Restriction Computer
@pytest.mark.parametrize("request_id",
                         example_requests)
def test_valid_mapping_restriction_computer(request_id):
    req = create_test_request(request_id, set_allowed_nodes=False)
    sub = create_test_substrate_topology_zoo()
    # print(req)
    # print(sub)
    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    # assert that all nodes and all edges may be mapped anywhere...
    substrate_node_set = set(sub.nodes)
    substrate_edge_set = set(sub.edges)

    for reqnode in req.nodes:
        allowed_nodes = vmrc.get_allowed_snode_list(reqnode)
        assert set(allowed_nodes) == substrate_node_set

    for reqedge in req.edges:
        allowed_edges = vmrc.get_allowed_sedge_set(reqedge)
        assert allowed_edges == substrate_edge_set

    # set substrate node and edge capacities randomly
    for snode in sub.nodes:
        sub.node[snode]['capacity'] = 2.0 * random.random()

    for sedge in sub.edges:
        sub.edge[sedge]['capacity'] = 2.0 * random.random()

    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    for reqnode in req.nodes:
        allowed_nodes = vmrc.get_allowed_snode_list(reqnode)
        snodes_with_not_enough_capacity = set([snode for snode in sub.nodes
                                               if sub.get_node_capacity(snode) < req.get_node_demand(reqnode)])
        assert snodes_with_not_enough_capacity.union(set(allowed_nodes)) == substrate_node_set

    for reqedge in req.edges:
        allowed_edges = vmrc.get_allowed_sedge_set(reqedge)
        sedges_with_not_enough_capacity = set([sedge for sedge in sub.edges
                                               if sub.get_edge_capacity(sedge) < req.get_edge_demand(reqedge)])
        assert sedges_with_not_enough_capacity.union(set(allowed_edges)) == substrate_edge_set

    snode_list = list(sub.nodes)
    sedge_list = list(sub.edges)

    # now additionally also introduce some mapping restrictions
    for reqnode in req.nodes:
        random.shuffle(snode_list)
        req.set_allowed_nodes(reqnode, snode_list[0:random.randint(1, len(snode_list))])
    for reqedge in req.edges:
        random.shuffle(sedge_list)
        req.set_allowed_edges(reqedge, sedge_list[0:random.randint(1, len(sedge_list))])

    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    for reqnode in req.nodes:
        allowed_nodes = vmrc.get_allowed_snode_list(reqnode)
        snodes_with_not_enough_capacity = set([snode for snode in sub.nodes
                                               if sub.get_node_capacity(snode) < req.get_node_demand(reqnode)])
        forbidden_snodes = set([snode for snode in sub.nodes if snode not in allowed_nodes])
        assert snodes_with_not_enough_capacity.union(forbidden_snodes).union(set(allowed_nodes)) == substrate_node_set

    for reqedge in req.edges:
        allowed_edges = vmrc.get_allowed_sedge_set(reqedge)
        sedges_with_not_enough_capacity = set([sedge for sedge in sub.edges
                                               if sub.get_edge_capacity(sedge) < req.get_edge_demand(reqedge)])
        forbidden_sedges = set([sedge for sedge in sub.edges if sedge not in allowed_edges])
        assert sedges_with_not_enough_capacity.union(forbidden_sedges).union(set(allowed_edges)) == substrate_edge_set


# TEST OptimizedDynVMP
@pytest.mark.parametrize("request_id",
                         example_requests)
def test_opt_dynvmp(request_id):
    req = create_test_request(request_id, set_allowed_nodes=False)
    sub = create_test_substrate_topology_zoo()
    # assert that all nodes and all edges may be mapped anywhere...
    substrate_node_set = set(sub.nodes)
    substrate_edge_set = set(sub.edges)

    # set substrate node and edge capacities high
    for snode in sub.nodes:
        for type in sub.node[snode]['capacity'].keys():
            sub.node[snode]['capacity'][type] = 100

    for sedge in sub.edges:
        sub.edge[sedge]['capacity'] = 100

    snode_list = list(sub.nodes)
    sedge_list = list(sub.edges)
    valid_reqnode_list = [snode for snode in sub.nodes if sub.get_node_capacity(snode) > 1.01]

    # # now additionally also introduce some mapping restrictions
    for reqnode in req.nodes:
        random.shuffle(valid_reqnode_list)

        selected_nodes = []
        while len(selected_nodes) == 0:
            print "selected nodes are {}".format(selected_nodes)
            selected_nodes = valid_reqnode_list[0:random.randint(10, len(valid_reqnode_list))]
        req.set_allowed_nodes(reqnode, selected_nodes)
        print "Allowd nodes for {} are {}.".format(reqnode, req.get_allowed_nodes(reqnode))
    for reqedge in req.edges:
        random.shuffle(sedge_list)
        req.set_allowed_edges(reqedge, sedge_list[0:random.randint(20, len(sedge_list))])
        print "Allowd edges for {} are {}.".format(reqedge, req.get_allowed_edges(reqedge))

    # random edge costs
    edge_costs = {sedge: max(1, 1000.0 * random.random()) for sedge in sub.edges}
    for sedge in sub.edges:
        sub.edge[sedge]['cost'] = 1  # edge_costs[sedge]
    for snode in sub.nodes:
        for type in sub.node[snode]['cost'].keys():
            sub.node[snode]['cost'][type] = 1

    scenario = dm.Scenario("test", sub, [req])
    gurobi = mip.ClassicMCFModel(scenario)
    gurobi.init_model_creator()
    gurobi.model.setParam("LogToConsole", 1)
    gurobi.compute_integral_solution()
    gurobi.model.write("foo.lp")

    td_comp = treewidth_model.TreeDecompositionComputation(req)
    tree_decomp = td_comp.compute_tree_decomposition()
    assert tree_decomp.is_tree_decomposition(req)
    sntd = treewidth_model.SmallSemiNiceTDArb(tree_decomp, req)
    assert sntd.is_tree_decomposition(req)

    opt_dynvmp = treewidth_model.OptimizedDynVMP(sub,
                                                 req,
                                                 sntd)
    opt_dynvmp.initialize_data_structures()
    opt_dynvmp.compute_solution()

    obj = opt_dynvmp.get_optimal_solution_cost()
    if obj is not None:
        costs, mapping_indices = opt_dynvmp.get_ordered_root_solution_costs_and_mapping_indices(maximum_number_of_solutions_to_return=100)
        for index, cost in np.ndenumerate(costs):
            if index == 0:
                assert cost == obj
            print "{}-th best cost is: {} and index is {}".format(index, cost, mapping_indices[index])
        corresponding_mappings = opt_dynvmp.recover_list_of_mappings(mapping_indices)
        print "Number of obtained mappings: {}, mappings: {}".format(len(corresponding_mappings), corresponding_mappings)

    result_mapping = opt_dynvmp.recover_mapping()
    print "Returned mapping! Namely, the following: {}".format(result_mapping)

    # change costs
    snode_costs = opt_dynvmp.snode_costs
    sedge_costs = opt_dynvmp.sedge_costs
    for snode in snode_costs.keys():
        snode_costs[snode] *= 2
    for sedge in sedge_costs.keys():
        sedge_costs[sedge] *= 2

    opt_dynvmp.reinitialize(snode_costs, sedge_costs)
    opt_dynvmp.compute_solution()
    result_mapping = opt_dynvmp.recover_mapping()
    print "Returned mapping! Namely, the following: {}".format(result_mapping)


@pytest.mark.parametrize("request_id", example_requests)
@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize("allowed_nodes_ratio", [0.1, 0.3, 0.5])  # avoid too large values because of memory footprint
@pytest.mark.parametrize("allowed_edges_ratio", [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
def test_opt_dynvmp_and_classic_mcf_agree_for_unambiguous_scenario(
        request_id, random_seed, allowed_nodes_ratio, allowed_edges_ratio
):
    random.seed(random_seed)
    print "allowed nodes: {} %, allowed edges: {} %".format(100 * allowed_nodes_ratio, 100 * allowed_edges_ratio)
    req = create_test_request(request_id, set_allowed_nodes=False)
    node_type = "test_type"
    sub = create_test_substrate_topology_zoo(node_types=[node_type])
    assert set(sub.get_types()) == {node_type}

    num_allowed_nodes = int(allowed_nodes_ratio * len(sub.nodes))
    for i in req.nodes:
        assert req.get_type(i) == node_type
        req.node[i]["allowed_nodes"] = random.sample(list(sub.nodes), num_allowed_nodes)

    num_allowed_edges = int(allowed_edges_ratio * len(sub.edges))
    for ij in req.edges:
        req.edge[ij]["allowed_edges"] = random.sample(list(sub.edges), num_allowed_edges)

    snode_costs = {}
    sedge_costs = {}

    def randomize_substrate_costs():
        # randomize all costs to force unambiguous cost-optimal embedding
        for u in sub.nodes:
            for t in sub.get_supported_node_types(u):
                sub.node[u]["cost"][t] = random.random()
                snode_costs[u] = sub.node[u]["cost"][t]
        for uv in sub.edges:
            sub.edge[uv]["cost"] = random.random()
            sedge_costs[uv] = sub.edge[uv]["cost"]

    randomize_substrate_costs()

    td_comp = treewidth_model.TreeDecompositionComputation(req)
    tree_decomp = td_comp.compute_tree_decomposition()
    sntd = treewidth_model.SmallSemiNiceTDArb(tree_decomp, req)
    opt_dynvmp = treewidth_model.OptimizedDynVMP(sub, req, sntd, snode_costs, sedge_costs)
    times_dynvmp = []
    start_time = time.time()
    opt_dynvmp.initialize_data_structures()
    opt_dynvmp.compute_solution()
    root_cost, mapping = opt_dynvmp.recover_mapping()
    times_dynvmp.append(time.time() - start_time)

    number_of_tests = 10

    times_gurobi = []

    for x in range(number_of_tests):
        def compare_solutions():
            if root_cost is not None:  # solution exists
                assert gurobi_solution is not None
                gurobi_obj = gurobi_solution.status.objValue
                assert abs(root_cost - gurobi_obj) <= 0.0001
                assert mapping.mapping_nodes == gurobi_solution.solution.request_mapping[req].mapping_nodes
                assert mapping.mapping_edges == gurobi_solution.solution.request_mapping[req].mapping_edges
            else:
                assert gurobi_solution is None

        start_time = time.time()
        gurobi = mip.ClassicMCFModel(dm.Scenario("test", sub, [req], objective=dm.Objective.MIN_COST))
        gurobi.init_model_creator()
        gurobi_solution = gurobi.compute_integral_solution()
        times_gurobi.append(time.time()-start_time)

        compare_solutions()

        randomize_substrate_costs()

        start_time = time.time()
        opt_dynvmp.reinitialize(snode_costs, sedge_costs)
        opt_dynvmp.compute_solution()
        root_cost, mapping = opt_dynvmp.recover_mapping()
        times_dynvmp.append(time.time() - start_time)

    gurobi = mip.ClassicMCFModel(dm.Scenario("test", sub, [req], objective=dm.Objective.MIN_COST))
    gurobi.init_model_creator()
    gurobi_solution = gurobi.compute_integral_solution()

    compare_solutions()

    initial_computation_time = times_dynvmp[0]
    other_computation_times = times_dynvmp[1:]
    average_of_others = sum(other_computation_times) / float(len(other_computation_times))
    print("        Request graph size (|V|,|E|):  ({},{})\n"
          "             Width of the request is:  {}\n"
          "      Runtime of initial computation:  {:02.4f} s\n"
          "Runtime of later computations (avg.):  {:02.4f} s\n"
          "               This is a speed-up of:  {:02.2f} times\n"
          "               Total time in Dyn-VMP:  {:02.4f} s\n"
          "                Total time in Gurobi:  {:02.4f} s\n"
          "         Speedup Dyn-VMP over Gurobi:  {:02.2f} times (>1 is good)\n".format(len(req.nodes),
                                                                           len(req.edges),
                                                                           sntd.width,
                                                                           initial_computation_time,
                                                                           average_of_others,
                                                                           initial_computation_time / average_of_others,
                                                                           sum(times_dynvmp),
                                                                           sum(times_gurobi),
                                                                           sum(times_gurobi)/sum(times_dynvmp)))
