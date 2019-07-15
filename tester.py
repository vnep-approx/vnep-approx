import random
import time
import numpy as np
import cPickle as pickle

from alib import datamodel
from test.treewidth_model.test_data.substrate_test_data import create_test_substrate
from test.treewidth_model.test_data.request_test_data import example_requests, create_test_request
from vnep_approx.treewidth_model import ValidMappingRestrictionComputer
# from vnep_approx.treewidth_model import ShortestValidPathsComputer as SVPC_goel
from vnep_approx.treewidth_model import ShortestValidPathsComputerWithoutLatencies as SVPC_given
from vnep_approx.backup.lorenz import ShortestValidPathsComputerLORENZ as SVPC_Lorenz
from vnep_approx.backup.lorenz_optimized import ShortestValidPathsComputerLORENZ as SVPC_LORENZ_OPT

from vnep_approx.backup.goel_optimized import ShortestValidPathsComputer as SVPC_goel_OPT
from vnep_approx.backup.goel import ShortestValidPathsComputer as SVPC_goel

# from vnep_approx.deferred.extendedgraph import ExtendedGraph
# from vnep_approx.deferred.extended_graph_visualizer import ExtendedGraphVisualizer

from vnep_approx.test_results import verify_correct_result


class LatencyParameters:
    pass
    # TODO



class GraphCreatorAarnet:
    def __init__(self, lat_parameters):
        self.lat_parameters = lat_parameters
        self.parameters = {
            "node_types": ["t1", "t2"],
            "topology": "Aarnet",
            "node_cost_factor": 1.0,
            "node_capacity": 10000.0,
            "edge_cost": 1.0,
            "edge_capacity": 10000.0,
            "number_of_requests": 1,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 6,
            "node_type_distribution": 0.2,
            "probability": 0.0,
            "node_resource_factor": 0.02,
            "edge_resource_factor": 50.0,
            "profit_factor": 1,
        }

        self.sub_reader = scenariogeneration.TopologyZooReader()
        self.sub = self.sub_reader.read_substrate(self.parameters)
        self.sub = create_test_substrate()

    def _add_latencies(self):
        for e in self.sub.edges:
            self.sub.edge[e]["latency"] = get_latency(self.lat_parameters)


def create_large_substrate(num_nodes, edge_res_factor):
    # if num_nodes > 26:
    #     num_nodes = 26

    import string

    names = []
    i = 1
    while num_nodes > 26:
        names.extend([ i * c for c in string.ascii_lowercase ])
        num_nodes -= 26
        i += 1
    names.extend([ i * c for c in string.ascii_lowercase[:num_nodes] ])

    sub = datamodel.Substrate("test_sub_"+str(num_nodes)+"_nodes")

    for letter in names:
        sub.add_node(letter, types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    for i, start in enumerate(names):
        for end in names[i+1:]:
            if random.random() <= edge_res_factor:
                sub.add_edge(start, end, capacity=100.0, cost=1.0, bidirected=random.random() <= 0.25)
    return sub


def create_large_request(num_nodes, sub, topology):
    if num_nodes > 26:
        num_nodes = 26

    reverse_edges = set()
    request_dict = example_requests[topology]
    request = datamodel.Request("{}_req".format(topology.replace(" ", "_")))
    for node in request_dict["nodes"]:
        allowed = sub.nodes
        if "assumed_allowed_nodes" in request_dict:
            allowed = request_dict["assumed_allowed_nodes"][node]
        request.add_node(node, 1, "test_type", allowed_nodes=allowed)
    for edge in request_dict["edges"]:
        if edge in reverse_edges:
            edge = edge[1], edge[0]
        request.add_edge(edge[0], edge[1], 1)
    return request




def get_latency(lat_pars):
    return random.randint(lat_pars["min_value"], lat_pars["max_value"] + 1)

def get_cost(cost_pars=None):
    if cost_pars is None:
        return float(random.randint(0, 10) + 1)
    else:
        return float(random.randint(cost_pars["min_value"],  cost_pars["max_value"] + 1))


def run_test():

    recompute_pars = True
    save_pars = False

    # if recompute_pars:
    #
    #     # graph_creator = GraphCreatorAarnet(lat_pars)
    #     # sub = graph_creator.sub
    #     # req = create_random_test_request(sub, **{"edge_resource_factor": 200})
    #
    # sub = create_large_substrate(8, 0.5)
    #     # sub = create_test_substrate()
    #
    #     # req = create_test_request("simple path")
    #     # req.node["i1"]["allowed_nodes"] = ["u", "v", "w"]
    #     # req.node["i2"]["allowed_nodes"] = ["v", "u", "w"]
    #     # req.node["i3"]["allowed_nodes"] = ["u", "w", "v"]
    #
    # req = create_large_request(0, sub, "dragon 3")
    #
    # lat_pars = {"min_value": 50, "max_value": 400}
    # edge_costs = {e: 1.0 for e in sub.edges}
    # edge_latencies = {e: get_latency(lat_pars) for e in sub.edges}
    #     save_resources(sub, req, edge_costs, edge_latencies)
    # else:
    #     sub, req, edge_costs, edge_latencies = load_resources()



    if recompute_pars:

        # """ large, random """
        # sub = create_large_substrate(40, 0.8)
        # req = create_large_request(0, sub, "dragon 3")
        #
        # lat_pars = {"min_value": 50, "max_value": 400}
        # edge_costs = {e: get_cost() for e in sub.edges}
        # edge_latencies = {e: get_latency(lat_pars) for e in sub.edges}


        # """ triangle """
        # sub = build_triangle()
        # req = create_test_request("simple path")
        # edge_costs = {("u", "v"): 1, ("v", "w"): 2, ("u", "w"): 5, ("w", "x"): 2}
        # edge_latencies = {("u", "v"): 22, ("v", "w"): 22, ("u", "w"): 22, ("w", "x"): 22}
        # # edge_latencies = {("u", "v"): 352, ("v", "w"): 373, ("u", "w"): 106, ("w", "x"): 155}


        sub, req = build_substrate_and_request()

        edge_latencies = {}
        edge_costs = {}
        for sedge in sub.edges:
            edge_latencies[sedge] = int(sub.edge[sedge]["latency"]) #'* (10 ** 5)) + 40
            edge_costs[sedge] = sub.edge[sedge]["cost"]


        if save_pars:
            save_resources(sub, req, edge_costs, edge_latencies)
    else:
        sub, req, edge_costs, edge_latencies = load_resources()

    # sub.nodes = list(sorted(sub.nodes, key=lambda x: int(x)))
    # sub.edges = list(sorted(sub.edges, key=lambda (x,y): (int(x), int(y))))

    print req
    print sub


    print "edge_latencies: ", edge_latencies
    print "costs: ", edge_costs

    vmrc = ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()
    # print vmrc.get_number_of_different_edge_sets()
    # print(vmrc.get_reqedge_to_edgeset_id_mapping())
    print(vmrc.get_edge_set_mapping())


    print "\n\n-- run --"

    print "\n\n--------- mine ----------"
    svpcwl_goel = SVPC_goel(sub, vmrc, edge_costs, edge_latencies, epsilon=0.5, limit=4000)
    start_mine = time.time()
    # svpcwl_goel.compute()
    my_time = time.time() - start_mine

    # print svpcwl_goel.valid_sedge_paths

    # print svpcwl.valid_sedge_costs
    # print svpcwl.valid_sedge_pred

    # print "\n", svpcwl.valid_sedge_latencies
    # print edge_latencies


    print "\n\n--------- my Lorenz ----------"

    svpcwl_lorenz = SVPC_Lorenz(sub, vmrc, edge_costs, edge_latencies, epsilon=0.1, limit=1000)
    start_mine2 = time.time()
    # svpcwl_lorenz.compute()
    my_time2 = time.time() - start_mine2

    # print my_time2

    print "\n\n--------- his ----------"
    svpc_given = SVPC_given(sub, req, vmrc, edge_costs)
    start_his = time.time()
    svpc_given.compute()
    his_time = time.time() - start_his


    # print svpc_given.valid_sedge_costs
    # print svpc_given.valid_sedge_pred


    print "\n\n--------- sum ----------"

    print "costs equal: \t", cmp(svpcwl_goel.valid_sedge_costs, svpc_given.valid_sedge_costs) == 0
    # print "preds equal:\t", cmp(svpcwl.valid_sedge_pred, svpc_given.valid_sedge_pred) == 0
    # print "my time: ", my_time, "\t his time: ", his_time, "\t my time Lorenz:  ", my_time2
    # print "my time: ", my_time, "\t my time Lorenz:  ", my_time2
    print "my time: ", my_time, "\t my time Lorenz:  ", my_time2, "\t his time:  ", his_time
    # print (float(my_time2) / my_time) if my_time > 0 else np.inf, "times faster\n\n"
    # print "Lor:\t", svpcwl_lorenz.valid_sedge_paths , "\nGoel:\t", svpcwl_goel.valid_sedge_paths

    # print check_approximation_guarantee(svpcwl.valid_sedge_costs, svpc_given.valid_sedge_costs, svpcwl.epsilon)



    # check_if_all_paths_valid(svpcwl, sub)

    # ext_graph = ExtendedGraph(req, sub)
    # visualizer = ExtendedGraphVisualizer()
    # visualizer.visualize(ext_graph, substrate=sub, output_path="./out/test_example.png")

    print "exit debug"


def check_edges_in_substrate(svpc, sub):
    inv_edges = set()
    for edgeset in range(svpc.number_of_valid_edge_sets):
        for start_node in sub.nodes:
            for target_node in sub.nodes:
                path = svpc.valid_sedge_paths[edgeset][start_node][target_node]

                split_path = set()


                if not path <= sub.edges:
                    print "ERROR!!"
                    inv_edges.add(path)

    if inv_edges:
        print "Invalid Paths:\n", inv_edges
    else:
        print "all edges good!"

def check_if_all_paths_valid(svpcwl, sub):

    inv_edges = set()
    for snode_source in sub.nodes:
        num_source_node = svpcwl.snode_id_to_num_id[snode_source]
        for snode_target in sub.nodes:
            n = svpcwl.snode_id_to_num_id[snode_target]
            total_latencies = 0
            path = [snode_target]
            while n != num_source_node:
                end = svpcwl.num_id_to_snode_id[n]
                n = svpcwl.valid_sedge_pred[0][snode_source].get(end, None)

                if n is None:
                    break

                path = [n] + path
                n = svpcwl.snode_id_to_num_id[n]

                sedge = (svpcwl.num_id_to_snode_id[n], end)

                if sedge not in sub.edge:
                    inv_edges.add(sedge)
                    break

                total_latencies += svpcwl.edge_latencies[sedge]

            if total_latencies > svpcwl.limit:
                print "ERROR: latency limit overstepped: {} -> {} by {},  \tusing {}".format(num_source_node, svpcwl.snode_id_to_num_id[snode_target], total_latencies - svpcwl.limit, path)

    for sedge in inv_edges:
        print "ERROR: invalid edge: {}".format(sedge)

    print "valid path check done"


def check_approximation_guarantee(approximated_costs, actual_costs, epsilon):

    errors = []
    max_factor = -1
    for edge_set_index, edge_set_dict in approximated_costs.items():
        for (start_node, end_node), costs in edge_set_dict.items():

            act_costs = actual_costs[edge_set_index][(start_node, end_node)]

            if np.isnan(act_costs) and np.isnan(costs):
                approx_correct = True
            else:
                approx_correct = (costs <= act_costs * (1 + epsilon))

                if act_costs > 0:
                    factor = float(costs) / act_costs
                    if factor > max_factor:
                        max_factor = factor

            if not approx_correct:
                errors.append((edge_set_index, start_node, end_node, costs, act_costs, factor))

            # print "Costs from {} to {}:\n\tmine\t{}\n\this:\t{}\n\t\t\t\t->  {}"\
            #     .format(start_node, end_node, costs, act_costs, approx_correct)

        return max_factor, errors




def save_resources(sub, req, costs, lat):
    with open('pickles/costs.p', 'wb') as handle:
        pickle.dump(costs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/lat.p', 'wb') as handle:
        pickle.dump(lat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/sub.p', 'wb') as handle:
        pickle.dump(sub, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/req.p', 'wb') as handle:
        pickle.dump(req, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_resources():
    with open('pickles/lat.p', 'rb') as handle:
        b = pickle.load(handle)
    with open('pickles/costs.p', 'rb') as handle:
        c = pickle.load(handle)
    with open('pickles/sub.p', 'rb') as handle:
        a = pickle.load(handle)
    with open('pickles/req.p', 'rb') as handle:
        d = pickle.load(handle)
    return a, d, c, b


def build_triangle():
    sub = datamodel.Substrate("test_sub_triangle")
    sub.add_node("u", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("v", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("w", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("x", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    sub.add_edge("u", "v", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("v", "w", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("u", "w", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("w", "x", capacity=100.0, cost=1.0, bidirected=False)

    return sub



def build_substrate_and_request():
    with open('latency_study/pickles/sub.p', 'rb') as handle:
        a = pickle.load(handle)
    with open('latency_study/pickles/req.p', 'rb') as handle:
        d = pickle.load(handle)
    return a, d




def inspect_svpc():

    # svpc = None
    # with open('latency_study/pickles/svpc.p', 'rb') as handle:
    #     svpc = pickle.load(handle)
    #
    # # svpc.compute()
    #
    #
    # svpcwl = SVPC(svpc.substrate, svpc.valid_mapping_restriction_computer, svpc.edge_costs, svpc.edge_latencies, 1, svpc.limit)

    # svpcwl.limit = 4000

    sub, req = build_substrate_and_request()

    vmrc = ValidMappingRestrictionComputer(sub, req)

    svpcwl = SVPC_goel(sub, vmrc, )

    svpcwl.compute()

    with open('pickles/paths.p', 'wb') as handle:
        pickle.dump(svpcwl.valid_sedge_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    check_edges_in_substrate(svpcwl, svpcwl.substrate)


    print "done"



def inpsect_pickles():

    svpc = None

    with open('latency_study/pickles/before/vnet_2_mappings_0.p', 'rb') as handle:
        before = pickle.load(handle)

    with open('latency_study/pickles/after/vnet_2_mappings_0.p', 'rb') as handle:
        after = pickle.load(handle)

    print "loaded"

    before.valid_mapping_restriction_computer.compute()

    before.limit = 10
    before.epsilon = 1

    # for key,value in before.edge_latencies.iteritems():
    #     if value == 0:
    #         before.edge_latencies[key] = 0.1

    start_time = time.time()
    before.compute()
    print "time: ", time.time() - start_time
    print "done"

def check_substrates():

    with open('latency_study/pickles/before/vnet_1_mappings_0.p', 'rb') as handle:
        svpc = pickle.load(handle)

    with open('latency_study/pickles/before/vnet_2_mappings_0.p', 'rb') as handle:
        svpc2 = pickle.load(handle)

    print svpc.substrate == svpc2.substrate


def compare():
    with open('latency_study/pickles/before/vnet_1_mappings_0.p', 'rb') as handle:
        lorenz = pickle.load(handle)

    print lorenz.limit

    goel = SVPC(lorenz.substrate, lorenz.valid_mapping_restriction_computer,
                lorenz.edge_costs, lorenz.edge_latencies, lorenz.epsilon, 10)

    print "loaded"

    lorenz.epsilon = 1
    goel.epsilon = 1

    start_time = time.time()
    goel.compute()
    print "time GOEL:\t\t ", time.time() - start_time

    start_time = time.time()
    lorenz.compute()
    print "time LORENZ:\t ", time.time() - start_time

    print "done"

def check_optimization_lorenz():

    limit, epsilon = 10, 1

    with open('latency_study/pickles/before/vnet_1_mappings_0.p', 'rb') as handle:
        svpc = pickle.load(handle)

    lorenz = SVPC_Lorenz(svpc.substrate, svpc.valid_mapping_restriction_computer, svpc.edge_costs, svpc.edge_latencies, epsilon, limit)
    optimized = SVPC_LORENZ_OPT(svpc.substrate, svpc.valid_mapping_restriction_computer, svpc.edge_costs, svpc.edge_latencies, epsilon, limit)

    print "setup"

    start_time = time.time()
    lorenz.compute()
    print "time LORENZ:\t\t ", time.time() - start_time

    start_time = time.time()
    optimized.compute()
    print "time OPTIMIZED:\t ", time.time() - start_time


    for attr in ["paths", "costs"]:
        first = getattr(optimized, "valid_sedge_" + attr)
        second = getattr(lorenz, "valid_sedge_" + attr)
        dict_differences(first, second, False)

    print "done"



def check_optimization_goel(num_reps=10, substrate=None):

    # limit, epsilon = 20, 0.00001
    #
    # with open('latency_study/pickles/before/vnet_1_mappings_0.p', 'rb') as handle:
    #     svpc = pickle.load(handle)
    #
    # goel =      SVPC_goel       (svpc.substrate, svpc.valid_mapping_restriction_computer, svpc.edge_costs, svpc.edge_latencies, epsilon, limit)
    # optimized = SVPC_goel_OPT   (svpc.substrate, svpc.valid_mapping_restriction_computer, svpc.edge_costs, svpc.edge_latencies, epsilon, limit)
    # plain =     SVPC_given      (svpc.substrate, None, svpc.valid_mapping_restriction_computer, svpc.edge_costs)


    limit, epsilon = 10, 0.01
    substrate = create_large_substrate(30, 1)
    request = create_large_request(0, substrate, "dragon 3")

    lat_pars = {"min_value": 50, "max_value": 400}
    cost_pars = {"min_value": 50, "max_value": 400}
    edge_costs = {e: get_cost(cost_pars) / 200 for e in substrate.edges}
    edge_latencies = {e: get_latency(lat_pars) / 200 for e in substrate.edges}

    vmrc = ValidMappingRestrictionComputer(substrate, request)
    vmrc.compute()


    goel =      SVPC_goel       (substrate, vmrc, edge_costs, edge_latencies, epsilon, limit)
    optimized = SVPC_goel_OPT   (substrate, vmrc, edge_costs, edge_latencies, epsilon, limit)
    plain =     SVPC_given      (substrate, None, vmrc, edge_costs)


    print "setup substrate on ", substrate.get_number_of_nodes()\
                                 , " nodes and ", substrate.get_number_of_edges(), " edges"

    # ---------------------         170 s           ---------------

    total_goel, total_opt = 0, 0
    time_plain = 0

    for i in range(num_reps):
        print " - - - - - - - - - - - - - - - - -"
        start_time = time.time()
        goel.compute()
        time_goel = time.time() - start_time
        total_goel += time_goel
        print "time GOEL:\t\t ", time_goel
        #
        #
        # with open('pickles/lorenz.p', 'wb') as handle:
        #     pickle.dump(lorenz, handle, protocol=pickle.HIGHEST_PROTOCOL)

        start_time = time.time()
        optimized.compute()
        time_optimized = time.time() - start_time
        total_opt += time_optimized
        print "time OPTIMIZED:\t ", time_optimized


        start_time = time.time()
        plain.compute()
        time_plain = time.time() - start_time
        print "time PLAIN:\t\t ", time.time() - start_time

    print "----------------------------------"

    print " avg time GOEL:\t\t", total_goel / num_reps
    print " avg time OPTIIZED:\t", total_opt / num_reps
    print " limit overstepped" if optimized.latency_limit_overstepped else " limit held"


    print "\t\t ->\t", (total_goel / num_reps) / time_plain, "\n\t\t\t", (total_opt / num_reps) / time_plain,\
        "\n\t\t\t", 1

    verify_correct_result(goel)
    verify_correct_result(optimized)


    # evaluate perforance

    # for attr in ["costs", "paths"]:
    #     first = getattr(optimized, "valid_sedge_" + attr)
    #     second = getattr(goel, "valid_sedge_" + attr)
    #     dict_differences(attr, first, second, edge_latencies, (1 + epsilon) * limit, False)


    print "done"

def dict_differences(attr, first, second, metric, bound, output=True):
    errors_found = False
    for key, edgesets in first.iteritems():
        for key2, value2 in edgesets.iteritems():

            if attr == "paths":
                for key3, value3 in value2.iteritems():
                    if output:
                        print "  checking ", second[key][key2][key3], " == ", value3
                    # if second[key][key2][key3] != value3:
                    if second.get_valid_sedge_path(key, key2, key3) != value3:
                        # print "ERROR: opt[", key2, ", ", key3, "] = \n\t", value3, " != \n\t", second[key][key2][key3], " = lor[", key, "]"
                        # errors_found = True

                        second_costs = 0
                        for e in second[key][key2][key3]:
                            second_costs += metric[e]

                        first_costs = 0
                        for e in value3:
                            first_costs += metric[e]

                        if first_costs > bound:
                            print "ERROR: opt[", key2, ", ", key3, "] = \n\t", first_costs, " != \n\t", second_costs, " = lor[", key, "]"
                            errors_found = True

            elif attr == "costs":
                if output:
                    print "  checking ", second[key][key2], " == ", value2
                if second[key][key2] < value2 and not (np.isnan(second[key][key2]) and np.isnan(value2)):
                    print "ERROR: opt[", key2, "] = \n\t", value2, " != \n\t", second[key][key2], " = lor[", key, "]"
                    errors_found = True
    print attr, "check done" + (", no errors!" if not errors_found else "\n  DIFFERENCES FOUND!!!")


def check_result(svpc, req, sub):
    for e in req.edges:
        for start in sub.nodes:
            for end in sub.nodes:
                path = svpc.get_valid_sedge_path(e, start, end)
                print path


if __name__ == "__main__":
    # run_test()
    # inspect_svpc()
    # inpsect_pickles()
    # compare()
    # check_substrates()
    check_optimization_goel()
    # check_optimization_lorenz()