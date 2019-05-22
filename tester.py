import random
from alib import scenariogeneration
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

    def _add_latencies(self):
        for e in self.sub.edges:
            self.sub.edge[e]["latency"] = random.randint(self.lat_parameters["min_value"],
                                                         self.lat_parameters["max_value"] + 1)

def run_test():
    lat_pars = {"min_value": 50, "max_value": 400}
    graph_creator = GraphCreatorAarnet(lat_pars)
    sub = graph_creator.sub

    svpcwl = SVPC(sub, )


    print svpcwl.approx_latencies(sub, 4000, 0.1, '0', '12')

    print sub


if __name__ == "__main__":
    run_test()