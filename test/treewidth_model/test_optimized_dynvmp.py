import pytest
from vnep_approx import treewidth_model

from alib import datamodel
from test_data.request_test_data import create_test_request, example_requests, example_requests_small, create_test_substrate_large
from test_data.substrate_test_data import create_test_substrate

import random
import time

random.seed(0)

# TEST Valid Mapping Restriction Computer

@pytest.mark.parametrize("request_id",
                         example_requests)
def test_valid_mapping_restriction_computer(request_id):
    req = create_test_request(request_id, set_allowed_nodes=False)
    sub = create_test_substrate_large()
    #print(req)
    #print(sub)
    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()

    #assert that all nodes and all edges may be mapped anywhere...
    substrate_node_set = set(sub.nodes)
    substrate_edge_set = set(sub.edges)

    for reqnode in req.nodes:
        allowed_nodes = vmrc.get_allowed_snode_list(reqnode)
        assert set(allowed_nodes) == substrate_node_set

    for reqedge in req.edges:
        allowed_edges = vmrc.get_allowed_sedge_set(reqedge)
        assert allowed_edges == substrate_edge_set

    #set substrate node and edge capacities randomly
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

    #now additionally also introduce some mapping restrictions
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


# TEST Valid Shortest Path Computer

@pytest.mark.parametrize("request_id",
                         example_requests)
def test_shortest_valid_paths_computer(request_id):
    req = create_test_request(request_id, set_allowed_nodes=False)
    sub = create_test_substrate_large()
    #print(req)
    #print(sub)
    vmrc = treewidth_model.ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()


    #uniform edge costs

    edge_costs = {sedge : 1.0 for sedge in sub.edges}

    svpc = treewidth_model.ShortestValidPathsComputer(sub, req, vmrc, edge_costs)

    svpc.compute()

    for reqedge in req.edges:
        for snode_source in sub.nodes:
            for snode_target in sub.nodes:
                if snode_source == snode_target:
                    assert svpc.valid_sedge_costs[reqedge][(snode_source, snode_target)] == 0
                else:
                    assert svpc.valid_sedge_costs[reqedge][(snode_source, snode_target)] >= 1

    #random edge costs

    edge_costs = {sedge : max(1,1000.0 * random.random()) for sedge in sub.edges}
    for sedge in sub.edges:
        sub.edge[sedge]['cost'] = edge_costs[sedge]

    bellman_ford_time = time.time()
    sub.initialize_shortest_paths_costs()
    bellman_ford_time = time.time() - bellman_ford_time


    svpc = treewidth_model.ShortestValidPathsComputer(sub, req, vmrc, edge_costs)
    dijkstra_time = time.time()
    svpc.compute()
    dijkstra_time = time.time() - dijkstra_time

    for reqedge in req.edges:
        for snode_source in sub.nodes:
            for snode_target in sub.nodes:
                #print svpc.valid_sedge_costs[reqedge][(snode_source, snode_target)]
                #print sub.get_shortest_paths_cost(snode_source, snode_target)
                assert svpc.valid_sedge_costs[reqedge][(snode_source, snode_target)] == pytest.approx(sub.get_shortest_paths_cost(snode_source, snode_target))


    print("\nComputation times were:\n\tBellman-Ford: {}\n\tDijkstra:     {}\n\n\tSpeedup by using Dijkstra: {:2.2f} (<1 is bad)".format(bellman_ford_time, dijkstra_time, (bellman_ford_time/dijkstra_time)))


# TEST OptimizedDynVMP

@pytest.mark.parametrize("request_id",
                         example_requests_small)
def test_opt_dynvmp(request_id):
    req = create_test_request(request_id, set_allowed_nodes=False)
    sub = create_test_substrate_large()

    print sub.node

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
            selected_nodes = valid_reqnode_list[0:random.randint(30*len(valid_reqnode_list)/40, len(valid_reqnode_list))]
        req.set_allowed_nodes(reqnode, selected_nodes)
        print "Allowd nodes for {} are {}.".format(reqnode, req.get_allowed_nodes(reqnode))
    for reqedge in req.edges:
        random.shuffle(sedge_list)
        req.set_allowed_edges(reqedge,sedge_list[0:random.randint(30*len(sedge_list)/40, len(sedge_list))])
        print "Allowd edges for {} are {}.".format(reqedge, req.get_allowed_edges(reqedge))

    #random edge costs
    edge_costs = {sedge : max(1,1000.0 * random.random()) for sedge in sub.edges}
    for sedge in sub.edges:
        sub.edge[sedge]['cost'] = 1#edge_costs[sedge]
    for snode in sub.nodes:
        for type in sub.node[snode]['cost'].keys():
            sub.node[snode]['cost'][type] = 1


    from alib import mip
    from alib import datamodel as dm
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
    print "Really usable edges are..."
    for reqedge in req.edges:
        "\t allowed edges of {} are {}".format(reqedge, opt_dynvmp.vmrc.get_allowed_sedge_set(reqedge))
    opt_dynvmp.compute_solution()
    result_mapping = opt_dynvmp.recover_mapping()
    print "Returned mapping! Namely, the following: {}".format(result_mapping)






