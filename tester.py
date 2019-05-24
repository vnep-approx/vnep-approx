import random
from alib import scenariogeneration
from test.treewidth_model.test_data.substrate_test_data import create_test_substrate
from test.treewidth_model.test_data.request_test_data import create_random_test_request, create_test_request
from vnep_approx.treewidth_model import ValidMappingRestrictionComputer
from vnep_approx.latencies_extension import ShortestValidPathsComputerWithLatencies as SVPC


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
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
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
            self.sub.edge[e]["latency"] = random.randint(self.lat_parameters["min_value"],
                                                         self.lat_parameters["max_value"] + 1)


def get_latency(lat_pars):
    return random.randint(lat_pars["min_value"], lat_pars["max_value"] + 1)


def run_test():
    lat_pars = {"min_value": 50, "max_value": 400}
    graph_creator = GraphCreatorAarnet(lat_pars)
    sub = graph_creator.sub
    # req = create_random_test_request(sub, **{"edge_resource_factor": 200})

    req = create_test_request("simple path")
    req.node["i1"]["allowed_nodes"] = ["u", "v"]
    req.node["i2"]["allowed_nodes"] = ["w"]
    req.node["i3"]["allowed_nodes"] = None

    print req
    print sub

    edge_costs = {e: 1 for e in sub.edges}
    edge_latencies = {e: get_latency(lat_pars) for e in sub.edges}

    vmrc = ValidMappingRestrictionComputer(sub, req)

    svpcwl = SVPC(sub, vmrc, edge_costs, edge_latencies, 0.1, 4000)

    svpcwl.compute()

    print svpcwl.valid_sedge_costs
    print svpcwl.valid_sedge_latencies
    print svpcwl.valid_sedge_pred



if __name__ == "__main__":
    run_test()