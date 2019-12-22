import random
import time
import numpy as np

from alib import datamodel
from test.treewidth_model.test_data.request_test_data import example_requests
from vnep_approx.treewidth_model import ValidMappingRestrictionComputer
from vnep_approx.treewidth_model import ShortestValidPathsComputer_NoLatencies as SVPC_given
from vnep_approx.latency_files.lorenz import ShortestValidPathsComputerLORENZ as SVPC_Lorenz
from vnep_approx.latency_files.goel import ShortestValidPathsComputer as SVPC_goel
from vnep_approx.latency_files.SVPC_all import ShortestValidPathsComputer as SVPC_all

from vnep_approx.latency_files.test_results import verify_correct_result



""" main function for running and comparing SVPC """
def run_SVPC_comparison():

    """ setup variables """
    limit               = 10000 # 7000 # 3500
    epsilon             = 0.02
    number_nodes        = 26
    edge_res_factor     = 0.8

    variance            = 1000
    mean                = 1000
    num_reps            = 2


    """ create graphs """
    substrate = create_large_substrate(number_nodes, edge_res_factor)
    request = create_large_request(0, substrate, "dragon 3")

    edge_costs      = dict(zip(substrate.edges, np.random.normal(mean, variance, substrate.get_number_of_edges())))
    edge_latencies  = dict(zip(substrate.edges, np.random.normal(mean, variance, substrate.get_number_of_edges())))

    for key, val in edge_costs.iteritems():
        if val < 0:
            edge_costs[key] = - val

    for key, val in edge_latencies.iteritems():
        if val < 0:
            edge_latencies[key] = - val


    vmrc = ValidMappingRestrictionComputer(substrate, request)
    vmrc.compute()

    svpc_args = [substrate, vmrc, edge_costs, edge_latencies, epsilon, limit]

    goel    = SVPC_all.createSVPC(SVPC_all.Approx_Flex,         *svpc_args)
    lorenz  = SVPC_all.createSVPC(SVPC_all.Approx_Strict,       *svpc_args)
    plain   = SVPC_all.createSVPC(SVPC_all.Approx_NoLatencies,  *svpc_args)
    exact   = SVPC_all.createSVPC(SVPC_all.Approx_Exact,        *svpc_args)



    """ set candidates """

    candidates = [goel, lorenz, plain]


    names = {goel: "GOEL\t", lorenz: "LORENZ\t", plain: "PLAIN  ", exact: "EXACT:\t"}
    times = [list() for _ in range(len(candidates))]


    print "setup substrate on ", substrate.get_number_of_nodes()\
                                 , " nodes and ", substrate.get_number_of_edges(), " edges"

    # ------------------------------------

    for i in range(num_reps):
        print " - - - - - - - - - - - - - - - - -"
        for ix, cand in enumerate(candidates):
            start_time = time.time()
            cand.compute()
            time_needed = time.time() - start_time
            times[ix].append(time_needed)
            print "time " + names[cand] + "\t\t ", time_needed

    print "----------------------------------"

    averages = list()
    time_plain = 0
    for ix, cand in enumerate(candidates):
        avg = np.mean(times[ix])
        averages.append(avg)
        print " avg time "+ names[cand] +":\t\t", avg
        if cand == plain:
            time_plain = avg
        else:
            print "\t\t" + ("limit overstepped" if cand.latency_limit_overstepped else "limit held")

    print " "

    if time_plain > 0:
        if plain in candidates and len(candidates) > 1:
            for ix, cand in enumerate(candidates):
                if cand == plain:
                    continue
                print "\t\t ->\t", averages[ix] / time_plain
            print "\t\t ->\t1\n"

    for cand in candidates:
        if cand == plain:
            continue

        print "Testing " + names[cand] + " .."
        verify_correct_result(cand, verify_optimality=(number_nodes <= 15))

    print "done"




def create_large_substrate(num_nodes, edge_res_factor):
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
                sub.add_edge(start, end, capacity=100.0, cost=1.0, bidirected=False)
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


def check_edges_in_substrate(svpc, sub):
    inv_edges = set()
    for edgeset in range(svpc.number_of_valid_edge_sets):
        for start_node in sub.nodes:
            for target_node in sub.nodes:
                path = svpc.valid_sedge_paths[edgeset][start_node][target_node]

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

        return max_factor, errors


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

def construct_sptp_substrate(): # to check shortest path tree property
    sub = datamodel.Substrate("test_sub_sptp")
    sub.add_node("u", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("v", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("t", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("s", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    sub.add_edge("u", "v", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("v", "t", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("s", "u", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("s", "v", capacity=100.0, cost=1.0, bidirected=False)

    edge_costs = {
        ("u", "v") : 3,
        ("s", "v") : 1,
        ("v", "t") : 5,
        ("s", "u") : 1,
    }
    edge_latencies = {
        ("u", "v") : 2,
        ("s", "v") : 9,
        ("v", "t") : 2,
        ("s", "u") : 4,
    }

    return sub, edge_costs, edge_latencies


if __name__ == "__main__":
    run_SVPC_comparison()