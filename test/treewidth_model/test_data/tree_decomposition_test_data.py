"""
Some valid tree decompositions for example requests.

"""
from vnep_approx import treewidth_model
from . import request_test_data


def create_test_tree_decomposition(td_dict):
    req_id_no_space = td_dict.get("request_id", "unknown").replace(" ", "_")
    tree_decomp = treewidth_model.TreeDecomposition("{}_TD".format(req_id_no_space))

    for node, bag in td_dict["bags"].items():
        tree_decomp.add_node(node, node_bag=bag)
    for t1, t2 in td_dict["edges"]:
        tree_decomp.add_edge(t1, t2)
    return tree_decomp


VALID_TREE_DECOMPOSITIONS = [
    dict(
        request_id="simple path",
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3"]),
        },
        edges=[
            ("bag_1", "bag_2"),
        ],
    ),
    dict(
        request_id="cycle with nodes before and after",
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3", "i4"]),
            "bag_3": frozenset(["i4", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
        ],

    ),
    dict(
        request_id="two cycles connected with single edge",
        bags={
            "bag_1": frozenset(["i1", "i2", "i3"]),
            "bag_2": frozenset(["i3", "i4"]),
            "bag_3": frozenset(["i4", "i5", "i6"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),

        ],
    ),
    dict(
        request_id="three cycles connected directly",
        bags={
            "bag_1": frozenset(["i1", "i2", "i3", "i4"]),
            "bag_2": frozenset(["i4", "i5", "i6", "i7"]),
            "bag_3": frozenset(["i4", "i8", "i9", "i10"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_1", "bag_3"),
        ],
    ),
    dict(  # derived by treedecomposition.com webservice
        request_id="cycle on cycle 4",
        bags={
            "bag_1": frozenset(["i2", "i5", "i7"]),
            "bag_2": frozenset(["i2", "i3", "i5"]),
            "bag_3": frozenset(["i1", "i2", "i3"]),
            "bag_4": frozenset(["i2", "i7", "i8"]),
            "bag_5": frozenset(["i5", "i6", "i7"]),
            "bag_6": frozenset(["i3", "i4", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_1", "bag_4"),
            ("bag_1", "bag_5"),

            ("bag_2", "bag_3"),
            ("bag_2", "bag_6"),
        ],
    ),
    dict(  # derived by treedecomposition.com webservice
        request_id="cycles crossing",
        bags={
            "bag_1": frozenset(["i4", "i5", "i6"]),
            "bag_2": frozenset(["i2", "i3", "i4", "i5"]),
            "bag_3": frozenset(["i1", "i2", "i3", ]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
        ],
    ),
    dict(  # derived by treedecomposition.com webservice
        request_id="three branches",
        bags={
            "bag_1": frozenset(["i1", "i4", "i5"]),
            "bag_2": frozenset(["i1", "i2", "i5"]),
            "bag_3": frozenset(["i1", "i3", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_1", "bag_3"),
        ],
    ),
    dict(  # derived by treedecomposition.com webservice
        request_id="dragon 1",
        bags={
            "bag_1": frozenset(["i2", "i3", "i6"]),
            "bag_2": frozenset(["i1", "i2", "i3"]),
            "bag_3": frozenset(["i1", "i2", "i4"]),
            "bag_4": frozenset(["i1", "i3", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
            ("bag_2", "bag_4"),
        ],
    ),
    dict(  # derived by treedecomposition.com webservice
        request_id="dragon 3",
        bags={
            "bag_1": frozenset(["i1", "i4", "i6"]),
            "bag_2": frozenset(["i1", "i2", "i3", "i4"]),
            "bag_3": frozenset(["i3", "i4", "i7"]),
            "bag_4": frozenset(["i1", "i3", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
            ("bag_2", "bag_4"),
        ],
    ),
]

INVALID_TREE_DECOMPOSITION_INTERSECTION_PROPERTY = [
    dict(
        # Re-introduce node i1 from earlier bag
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3"]),
            "bag_3": frozenset(["i3", "i4"]),
            "bag_4": frozenset(["i1", "i4", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
            ("bag_3", "bag_4"),
        ],
    ),
]

INVALID_TREE_DECOMPOSITIONS_NOT_A_TREE = [
    dict(
        # contains cycle
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3"]),
            "bag_3": frozenset(["i3", "i1"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_2", "bag_3"),
            ("bag_3", "bag_1"),
        ],
    ),
    dict(
        # disconnected
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3"]),
            "bag_3": frozenset(["i3", "i4"]),
            "bag_4": frozenset(["i4", "i5"]),
        },
        edges=[
            ("bag_1", "bag_2"),
            ("bag_3", "bag_4"),
        ],
    ),
]

INVALID_TREE_DECOMPOSITIONS = INVALID_TREE_DECOMPOSITION_INTERSECTION_PROPERTY + INVALID_TREE_DECOMPOSITIONS_NOT_A_TREE

PACE_INPUT_FORMAT = {
    "simple path": "p tw 3 2\n1 2\n2 3",
    "two cycles connected directly": "p tw 5 6\n1 2\n1 3\n2 3\n3 4\n3 5\n4 5",
}

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
