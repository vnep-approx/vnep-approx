from alib import datamodel, scenariogeneration


def create_test_request(request_id, reverse_edges=None, set_allowed_nodes=True):
    """
    :param request_id:
    :param reverse_edges: set of edges that should be reversed if they are contained in the original request.
    :return:
    """
    if reverse_edges is None:
        reverse_edges = set()
    request_dict = example_requests[request_id]
    request = datamodel.Request("{}_req".format(request_id.replace(" ", "_")))
    for node in request_dict["nodes"]:
        allowed = ["u", "v", "w"]
        if "assumed_allowed_nodes" in request_dict:
            allowed = request_dict["assumed_allowed_nodes"][node]
        if set_allowed_nodes:
            request.add_node(node, 1, "test_type", allowed_nodes=allowed)
        else:
            request.add_node(node, 1, "test_type")
    for edge in request_dict["edges"]:
        if edge in reverse_edges:
            edge = edge[1], edge[0]
        request.add_edge(edge[0], edge[1], 1)
    return request


def create_random_test_request(substrate, **kwargs):
    req_gen = scenariogeneration.UniformRequestGenerator()
    raw_parameters = dict(
        number_of_requests=1,
        min_number_of_nodes=5,
        max_number_of_nodes=10,
        probability=0.2,
        variability=0.1,
        node_resource_factor=1,
        edge_resource_factor=1,
        normalize=False,
    )
    raw_parameters.update(kwargs)
    return req_gen.generate_request(
        "test_req_random",
        raw_parameters,
        substrate,
    )


SINGLE_EDGE = {
    "nodes": [
        "i1", "i2"
    ],
    "edges": [
        ("i1", "i2")
    ],
}

SIMPLE_PATH = {
    "nodes": [
        "i1", "i2", "i3"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
    ],
}

SIMPLE_CYCLE_1 = {
    "nodes": [
        "i1", "i2", "i3"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
    ],
}

SIMPLE_CYCLE_2 = {
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
    ],
}

# i4 should not propagate to i1
CYCLE_WITH_NODE_BEFORE_START_1 = {
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
    ],
}

CYCLE_WITH_NODE_BEFORE_START_2 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
        ("i2", "i6"),
        ("i3", "i4"),
        ("i4", "i5"),
        ("i5", "i6"),
    ],
}

CYCLE_WITH_NODE_AFTER_END = {
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
        ("i3", "i4"),
    ],
}

# i4 should propagate to all cycle nodes i2, i3, i4
CYCLE_WITH_NODES_BEFORE_AND_AFTER = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
        ("i4", "i5"),
    ],
}

# i4 should not propagate to i5
CYCLE_WITH_BRANCH = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
        ("i3", "i5"),
    ],
}

# i3 should propagate to i1, i2, i3
# i6 should propagate to i4, i5, i6
TWO_CYCLES_CONNECTED_WITH_SINGLE_EDGE = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
        ("i3", "i4"),
        ("i4", "i5"),
        ("i4", "i6"),
        ("i5", "i6"),
    ],
}

# i3 should propagate to i1, i2, i3
# i5 should propagate to i3, i4, i5
TWO_CYCLES_CONNECTED_DIRECTLY = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
        ("i3", "i4"),
        ("i3", "i5"),
        ("i4", "i5"),
    ],

}

# i4 should propagate to i1, i2, i3, i4
# i7 should propagate to i4, i5, i6, i7
# i10 should propagate to i4, i8, i9, i10
THREE_CYCLES_CONNECTED_DIRECTLY = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"
    ],
    "edges": [
        # circle 1
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
        # circle 2
        ("i4", "i5"),
        ("i4", "i6"),
        ("i5", "i7"),
        ("i6", "i7"),
        # circle 3
        ("i4", "i8"),
        ("i4", "i9"),
        ("i8", "i10"),
        ("i9", "i10"),
    ],
}

# i4 should propagate to i1, i2, i3, i4
# i7 should propagate to i4, i5, i6, i7
# i10 should propagate to i7, i8, i9, i10
THREE_CYCLES_CONNECTED_CHAINED = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"
    ],
    "edges": [
        # circle 1
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
        # circle 2
        ("i4", "i5"),
        ("i4", "i6"),
        ("i5", "i7"),
        ("i6", "i7"),
        # circle 3
        ("i7", "i8"),
        ("i7", "i9"),
        ("i8", "i10"),
        ("i9", "i10"),
    ],
}

CYCLE_ON_CYCLE_1 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i6"),
        ("i3", "i4"),
        ("i3", "i5"),
        ("i4", "i5"),
        ("i5", "i6"),
    ],
}

CYCLE_ON_CYCLE_2 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i5"),
        ("i4", "i5"),
        ("i4", "i6"),
        ("i5", "i6"),
    ],
}

CYCLE_ON_CYCLE_3 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i6"),
        ("i4", "i5"),
        ("i4", "i6"),
        ("i5", "i6"),
    ],
}

CYCLE_ON_CYCLE_4 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i8"),
        ("i3", "i4"),
        ("i3", "i5"),
        ("i4", "i5"),
        ("i5", "i6"),
        ("i5", "i7"),
        ("i6", "i7"),
        ("i7", "i8"),
    ],
}

CYCLES_CROSSING = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i2", "i5"),
        ("i3", "i4"),
        ("i3", "i5"),
        ("i4", "i6"),
        ("i5", "i6"),
    ],
}

THREE_BRANCHES = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i1", "i4"),
        ("i2", "i5"),
        ("i3", "i5"),
        ("i4", "i5"),
    ],
}

DRAGON_1 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i1", "i4"),
        ("i1", "i5"),
        ("i2", "i4"),
        ("i2", "i6"),
        ("i3", "i5"),
        ("i3", "i6"),
    ],
}

DRAGON_2 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i1", "i4"),
        ("i1", "i5"),
        ("i1", "i6"),
        ("i2", "i4"),
        ("i2", "i6"),
        ("i3", "i5"),
        ("i3", "i6"),
    ],
}

DRAGON_3 = {
    "nodes": [
        "i1", "i2", "i3", "i4", "i5", "i6", "i7"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i5"),
        ("i1", "i6"),
        ("i2", "i3"),
        ("i2", "i4"),
        ("i3", "i5"),
        ("i3", "i7"),
        ("i4", "i6"),
        ("i4", "i7"),
    ],
}

example_requests_small = {
    "cycle with branch": CYCLE_ON_CYCLE_1
}


example_requests = {
    "single edge": SINGLE_EDGE,
    "simple path": SIMPLE_PATH,
    "simple cycle 1": SIMPLE_CYCLE_1,
    "simple cycle 2": SIMPLE_CYCLE_2,
    "cycle with node before start 1": CYCLE_WITH_NODE_BEFORE_START_1,
    "cycle with node before start 2": CYCLE_WITH_NODE_BEFORE_START_2,
    "cycle with node after end": CYCLE_WITH_NODE_AFTER_END,
    "cycle with nodes before and after": CYCLE_WITH_NODES_BEFORE_AND_AFTER,
    "cycle with branch": CYCLE_WITH_BRANCH,
    "two cycles connected with single edge": TWO_CYCLES_CONNECTED_WITH_SINGLE_EDGE,
    "two cycles connected directly": TWO_CYCLES_CONNECTED_DIRECTLY,
    "three cycles connected directly": THREE_CYCLES_CONNECTED_DIRECTLY,
    "three cycles chained": THREE_CYCLES_CONNECTED_CHAINED,
    "cycle on cycle 1": CYCLE_ON_CYCLE_1,
    "cycle on cycle 2": CYCLE_ON_CYCLE_2,
    "cycle on cycle 3": CYCLE_ON_CYCLE_3,
    "cycle on cycle 4": CYCLE_ON_CYCLE_4,
    "cycles crossing": CYCLES_CROSSING,
    "three branches": THREE_BRANCHES,
    "dragon 1": DRAGON_1,
    "dragon 2": DRAGON_2,
    "dragon 3": DRAGON_3,
}

example_requests_1 = {
    "simple path": SIMPLE_PATH
}
