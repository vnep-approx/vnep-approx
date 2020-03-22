from alib import datamodel, scenariogeneration, solutions, util


def create_test_substrate(substrate_id,
                          node_capacities=None,
                          node_costs=None,
                          edge_capacities=None,
                          edge_costs=None,
                          bidirected_edges=False):
    """ Node types are implicitly defined by the dictionary keys of node_capacities. (default ["t"]) """
    if node_capacities is None:
        node_capacities = {}
    if node_costs is None:
        node_costs = {}
    if edge_capacities is None:
        edge_capacities = {}
    if edge_costs is None:
        edge_costs = {}

    sub_nodes = SUBSTRATE_TOPOLOGIES[substrate_id]["nodes"]
    sub_edges = SUBSTRATE_TOPOLOGIES[substrate_id]["edges"]

    sub = datamodel.Substrate("{}_sub".format(substrate_id.replace(" ", "_")))
    for u in sub_nodes:
        u_types = node_capacities.get(u).keys() if u in node_capacities else ["t"]
        u_capacities = node_capacities.get(u) if u in node_capacities else dict(t=1.0)
        u_costs = node_costs.get(u) if u in node_costs else {t: 1.0 for t in u_capacities}
        sub.add_node(u, types=u_types, capacity=u_capacities, cost=u_costs)
    for uv in sub_edges:
        uv_capacity = edge_capacities.get(uv, 1.0)
        uv_cost = edge_costs.get(uv, 1.0)
        u, v = uv
        sub.add_edge(u, v, capacity=uv_capacity, cost=uv_cost, bidirected=bidirected_edges)

    return sub


def create_test_request(request_id,
                        test_substrate,
                        reversed_edges=None,
                        allowed_nodes=None,
                        allowed_edges=None,
                        node_types=None,
                        node_demands=None,
                        edge_demands=None):
    if reversed_edges is None:
        reversed_edges = set()
    if allowed_nodes is None:
        allowed_nodes = {}
    if allowed_edges is None:
        allowed_edges = {}
    if node_types is None:
        node_types = {}
    if node_demands is None:
        node_demands = {}
    if edge_demands is None:
        edge_demands = {}

    req_nodes = REQUEST_TOPOLOGIES[request_id]["nodes"]
    req_edges = REQUEST_TOPOLOGIES[request_id]["edges"]

    request = datamodel.Request("{}_req".format(request_id.replace(" ", "_")))
    for i in req_nodes:
        i_type = node_types.get(i, "t")
        i_demand = node_demands.get(i, 1.0)
        i_allowed_nodes = allowed_nodes.get(i, set(test_substrate.nodes))
        request.add_node(i, demand=i_demand, ntype=i_type, allowed_nodes=i_allowed_nodes)
    for ij in req_edges:
        if ij in reversed_edges:
            ij = ij[1], ij[0]
        ij_demand = edge_demands.get(ij, 1.0)
        ij_allowed = allowed_edges.get(ij)
        request.add_edge(ij[0], ij[1], ij_demand, allowed_edges=ij_allowed)

    return request


def get_test_scenario(test_data_dict):
    substrate = create_test_substrate(
        test_data_dict["substrate_id"],
        **test_data_dict.get("substrate_kwargs", {})
    )
    request = create_test_request(
        test_data_dict["request_id"],
        substrate,
        **test_data_dict.get("request_kwargs", {})
    )
    return datamodel.Scenario(
        name="test_scen",
        requests=[request],
        substrate=substrate,
        objective=test_data_dict.get("objective", datamodel.Objective.MAX_PROFIT),
    )


REQUEST_TOPOLOGIES = dict(
    single_edge_request=dict(
        nodes=["i1", "i2"],
        edges=[("i1", "i2")],
    ),
    confluence_request=dict(
        nodes=["i1", "i2", "i3"],
        edges=[("i1", "i2"), ("i2", "i3"), ("i1", "i3")],
    ),
)

SUBSTRATE_TOPOLOGIES = dict(
    single_edge_substrate=dict(
        nodes=["u1", "u2"],
        edges=[("u1", "u2")],
    ),
    confluence_substrate=dict(
        nodes=["u1", "u2", "u3"],
        edges=[("u1", "u2"), ("u2", "u3"), ("u1", "u3")],
    ),
)

SHORTEST_PATH_TEST_CASES = [
    "single_edge_request_and_substrate",
    "single_edge_request_confluence_substrate_prefer_cheaper_node_cost",
    "single_edge_request_confluence_substrate_split_flow_to_use_cheaper_edge",
    "single_edge_request_confluence_substrate_force_splitted_flow",
    "respect_edge_mapping_restrictions",
]

SPLITTABLE_TEST_CASES = [
    "single_edge_request_and_substrate",
    "single_edge_request_confluence_substrate_split_flow_to_use_cheaper_edge",
    "single_edge_request_confluence_substrate_force_splitted_flow",
    "respect_edge_mapping_restrictions",
]

# These tests assume that the algorithm considers resource costs.
# They are ignored when the "pure" load balancing objective is used
COST_SPECIFIC_TEST_CASES = {
    "single_edge_request_confluence_substrate_prefer_cheaper_node_cost",
    "single_edge_request_confluence_substrate_split_flow_to_use_cheaper_edge",
}

SINGLE_REQUEST_EMBEDDING_TEST_CASES = dict(
    single_edge_request_and_substrate=dict(
        request_id="single_edge_request",
        substrate_id="single_edge_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u1"],
                i2=["u2"],
            ),
        ),
        expected_integer_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u2",
            ),
            mapping_edges={
                ("i1", "i2"): [("u1", "u2")]
            },
        ),
        expected_fractional_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u2",
            ),
            mapping_edges={
                ("i1", "i2"): {("u1", "u2"): 1.0}
            },
        ),
    ),
    single_edge_request_confluence_substrate_prefer_cheaper_node_cost=dict(
        request_id="single_edge_request",
        substrate_id="confluence_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u1", "u2"],
                i2=["u3"],
            ),
        ),
        substrate_kwargs=dict(
            node_costs=dict(u1=dict(t=0.5), u2=dict(t=200))
        ),
        expected_integer_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): [("u1", "u3")]
            },
        ),
    ),
    single_edge_request_confluence_substrate_split_flow_to_use_cheaper_edge=dict(
        request_id="single_edge_request",
        substrate_id="confluence_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u1"],
                i2=["u3"],
            ),
        ),
        substrate_kwargs=dict(
            edge_capacities={
                ("u1", "u2"): 1.0,
                ("u1", "u3"): 0.25,
            },
            edge_costs={
                ("u1", "u2"): 20.0,
                ("u1", "u3"): 1.0,
            }
        ),
        expected_integer_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): [("u1", "u2"), ("u2", "u3")]
            },
        ),
        expected_fractional_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): {("u1", "u2"): 0.75, ("u2", "u3"): 0.75, ("u1", "u3"): 0.25}
            },
        ),
    ),
    single_edge_request_confluence_substrate_force_splitted_flow=dict(
        request_id="single_edge_request",
        substrate_id="confluence_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u1"],
                i2=["u3"],
            ),
        ),
        substrate_kwargs=dict(
            edge_capacities={
                ("u1", "u2"): 0.75,
                ("u1", "u3"): 0.25,
            }
        ),
        expected_integer_solution=None,
        expected_fractional_solution=dict(
            mapping_nodes=dict(
                i1="u1",
                i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): {("u1", "u2"): 0.75, ("u2", "u3"): 0.75, ("u1", "u3"): 0.25}
            },
        ),
    ),
    respect_edge_mapping_restrictions=dict(
        request_id="single_edge_request",
        substrate_id="confluence_substrate",

        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u1"],
                i2=["u3"],
            ),
            allowed_edges={
                ("i1", "i2"): {("u1", "u2"), ("u2", "u3")},
            },
        ),
        substrate_kwargs=dict(
            # make the allowed edges so expensive that the algorithm would never choose them without edge placement restrictions
            edge_costs={
                ("u1", "u2"): 10000000,
                ("u2", "u3"): 10000000,
                ("u1", "u3"): 1,
            },
        ),
        expected_integer_solution=dict(
            mapping_nodes=dict(
                i1="u1", i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): [("u1", "u2"), ("u2", "u3")]
            },
        ),
        expected_fractional_solution=dict(
            mapping_nodes=dict(
                i1="u1", i2="u3",
            ),
            mapping_edges={
                ("i1", "i2"): {("u1", "u2"): 1.0, ("u2", "u3"): 1.0}
            },
        ),
    ),
)

SINGLE_REQUEST_REJECT_EMBEDDING_TEST_CASES = dict(
    insufficient_node_capacity=dict(
        request_id="single_edge_request",
        substrate_id="single_edge_substrate",
        request_kwargs=dict(
            node_demands=dict(
                i1=1000.0,
            ),
        ),
    ),
    insufficient_node_capacity_on_forced_embedding_target=dict(
        request_id="single_edge_request",
        substrate_id="single_edge_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(
                i1=["u2"],
            ),
            node_demands=dict(
                i1=100,
            ),
        ),
        substrate_kwargs=dict(
            node_capacities=dict(
                u=dict(t=1),
                v=dict(t=1000),
            )
        ),
    ),
    insufficient_edge_capacity=dict(
        request_id="single_edge_request",
        substrate_id="single_edge_substrate",
        request_kwargs=dict(
            allowed_nodes=dict(  # forbid colocation for this test case
                i1=["u1"],
                i2=["u2"],
            ),
            edge_demands={
                ("i1", "i2"): 1000.0,
            },
        )
    ),
)
