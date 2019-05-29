import random
import time
import cPickle as pickle

from alib import datamodel
from test.treewidth_model.test_data.substrate_test_data import create_test_substrate
from test.treewidth_model.test_data.request_test_data import example_requests, create_test_request
from vnep_approx.treewidth_model import ValidMappingRestrictionComputer
from vnep_approx.latencies_extension import ShortestValidPathsComputerWithLatencies as SVPC
from vnep_approx.treewidth_model import ShortestValidPathsComputer as SVPC_given

# from vnep_approx.deferred.extendedgraph import ExtendedGraph
# from vnep_approx.deferred.extended_graph_visualizer import ExtendedGraphVisualizer


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
    if num_nodes > 26:
        num_nodes = 26

    import string

    sub = datamodel.Substrate("test_sub_"+str(num_nodes)+"_nodes")

    for letter in string.ascii_lowercase[:num_nodes]:
        sub.add_node(letter, types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    for i, start in enumerate(string.ascii_lowercase[:num_nodes]):
        for end in string.ascii_lowercase[i+1:num_nodes]:
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

def get_cost():
    return random.randint(0, 10) + 1


def run_test():

    recompute_pars = False

    # if recompute_pars:
    #
    #     # graph_creator = GraphCreatorAarnet(lat_pars)
    #     # sub = graph_creator.sub
    #     # req = create_random_test_request(sub, **{"edge_resource_factor": 200})
    #
    #     sub = create_large_substrate(8, 0.5)
    #     # sub = create_test_substrate()
    #
    #     # req = create_test_request("simple path")
    #     # req.node["i1"]["allowed_nodes"] = ["u", "v", "w"]
    #     # req.node["i2"]["allowed_nodes"] = ["v", "u", "w"]
    #     # req.node["i3"]["allowed_nodes"] = ["u", "w", "v"]
    #
    #     req = create_large_request(0, sub, "dragon 3")
    #
    #     lat_pars = {"min_value": 50, "max_value": 400}
    #     edge_costs = {e: 1.0 for e in sub.edges}
    #     edge_latencies = {e: get_latency(lat_pars) for e in sub.edges}
    #     save_resources(sub, req, edge_costs, edge_latencies)
    # else:
    #     sub, req, edge_costs, edge_latencies = load_resources()

    sub = build_triangle()
    req = create_test_request("simple path")
    edge_costs = {("u", "v"): 3, ("v", "w"): 1, ("u", "w"): 5}
    edge_latencies = {("u", "v"): 20, ("v", "w"): 30, ("u", "w"): 30}

    print req
    print sub


    print "edge_latencies: ", edge_latencies

    vmrc = ValidMappingRestrictionComputer(sub, req)
    vmrc.compute()
    print vmrc.get_number_of_different_edge_sets()
    print(vmrc.get_reqedge_to_edgeset_id_mapping())
    print(vmrc.get_edge_set_mapping())


    print "\n\n-- run --"

    start_mine = time.time()
    svpcwl = SVPC(sub, vmrc, edge_costs, edge_latencies, epsilon=0.1, limit=700)
    svpcwl.compute()
    my_time = time.time() - start_mine

    print "\n\n--------- mine ----------"
    print svpcwl.valid_sedge_costs
    print svpcwl.valid_sedge_pred

    # print "\n", svpcwl.valid_sedge_latencies
    # print edge_latencies

    print "\n\n--------- his ----------"
    start_his = time.time()
    svpcwl2 = SVPC_given(sub, req, vmrc, edge_costs)
    svpcwl2.compute()
    his_time = time.time() - start_his


    print svpcwl2.valid_sedge_costs
    print svpcwl2.valid_sedge_pred


    print "\n\n--------- sum ----------"

    print "costs equal: \t", cmp(svpcwl.valid_sedge_costs, svpcwl2.valid_sedge_costs) == 0
    print "preds equal:\t", cmp(svpcwl.valid_sedge_pred, svpcwl2.valid_sedge_pred) == 0
    print "my time: ", my_time, "\t his time: ", his_time



    # ext_graph = ExtendedGraph(req, sub)
    # visualizer = ExtendedGraphVisualizer()
    # visualizer.visualize(ext_graph, substrate=sub, output_path="./out/test_example.png")



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

    sub.add_edge("u", "v", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("v", "w", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("u", "w", capacity=100.0, cost=1.0, bidirected=False)

    return sub



if __name__ == "__main__":
    run_test()