from alib import datamodel, scenariogeneration


def create_test_substrate_topology_zoo():
    sub = datamodel.Substrate("test_sub")
    sub.add_node("u", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("v", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("w", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    sub.add_edge("u", "v", capacity=100.0, cost=1.0, bidirected=False, latency=1.0)
    sub.add_edge("v", "w", capacity=100.0, cost=1.0, bidirected=False, latency=1.0)
    sub.add_edge("w", "u", capacity=100.0, cost=1.0, bidirected=False, latency=2.0)
    return sub


def create_test_substrate_topology_zoo(substrate_name="Geant2012", include_latencies=False, **kwargs):
    raw_parameters = dict(
        topology=substrate_name,
        node_types=["test_type"],
        edge_capacity=10.0,
        node_cost_factor=1.0,
        node_capacity=10.0,
        node_type_distribution=1,
        include_latencies=include_latencies
    )
    raw_parameters.update(kwargs)
    return scenariogeneration.TopologyZooReader().read_substrate(
        raw_parameters,
    )
