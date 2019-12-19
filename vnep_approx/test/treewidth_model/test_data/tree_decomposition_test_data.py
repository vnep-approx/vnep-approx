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


VALID_TREE_DECOMPOSITIONS = {
    "simple path": dict(
        bags={
            "bag_1": frozenset(["i1", "i2"]),
            "bag_2": frozenset(["i2", "i3"]),
        },
        edges=[
            ("bag_1", "bag_2"),
        ],
    ),
    "cycle with nodes before and after": dict(
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
    "two cycles connected with single edge": dict(
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
    "three cycles connected directly": dict(
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
    "cycle on cycle 4": dict(
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
    "cycles crossing": dict(
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
    "three branches": dict(
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
    "dragon 1": dict(
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
    "dragon 3": dict(
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
}

NICE_TREE_DECOMPOSITIONS = {
    "simple path": dict(
        root="leaf_1",
        bags={
            "leaf_1": frozenset(["i1"]),
            "intro_1": frozenset(["i1", "i2"]),
            "forget_1": frozenset(["i2"]),
            "intro_2": frozenset(["i2", "i3"]),
            "leaf_2": frozenset(["i3"]),
        },
        edges=[
            ("leaf_1", "intro_1"),
            ("intro_1", "forget_1"),
            ("forget_1", "intro_2"),
            ("intro_2", "leaf_2"),
        ],
    ),
    "three cycles connected directly": dict(
        root="leaf_1",
        bags={
            "leaf_1": frozenset(["i1"]),
            "intro_1": frozenset(["i1", "i2"]),
            "intro_2": frozenset(["i1", "i2", "i3"]),
            "intro_3": frozenset(["i1", "i2", "i3", "i4"]),
            "forget_1": frozenset(["i2", "i3", "i4"]),
            "forget_2": frozenset(["i3", "i4"]),
            "join_1": frozenset(["i4"]),
            "join_2": frozenset(["i4"]),
            "intro_4": frozenset(["i4", "i5"]),
            "intro_5": frozenset(["i4", "i5", "i6"]),
            "intro_6": frozenset(["i4", "i5", "i6", "i7"]),
            "forget_4": frozenset(["i5", "i6", "i7"]),
            "forget_5": frozenset(["i6", "i7"]),
            "leaf_2": frozenset(["i7"]),
            "join_3": frozenset(["i4"]),
            "intro_8": frozenset(["i4", "i8"]),
            "intro_9": frozenset(["i4", "i8", "i9"]),
            "intro_10": frozenset(["i4", "i8", "i9", "i10"]),
            "forget_6": frozenset(["i8", "i9", "i10"]),
            "forget_7": frozenset(["i9", "i10"]),
            "leaf_3": frozenset(["i10"]),
        },
        edges=[
            ("leaf_1", "intro_1"),
            ("intro_1", "intro_2"),
            ("intro_2", "intro_3"),
            ("intro_3", "forget_1"),
            ("forget_1", "forget_2"),
            ("forget_2", "join_1"),
            ("join_1", "join_2"),
            ("join_2", "intro_4"),
            ("intro_4", "intro_5"),
            ("intro_5", "intro_6"),
            ("intro_6", "forget_4"),
            ("forget_4", "forget_5"),
            ("forget_5", "leaf_2"),
            ("join_1", "join_3"),
            ("join_3", "intro_8"),
            ("intro_8", "intro_9"),
            ("intro_9", "intro_10"),
            ("intro_10", "forget_6"),
            ("forget_6", "forget_7"),
            ("forget_7", "leaf_3"),
        ],
    ),
}

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

POST_ORDER_TRAVERSALS = [
    dict(
        request_id="simple path",
        root="bag_1",
        order=["bag_2", "bag_1"],
    ),
    dict(
        request_id="cycle on cycle 4",
        root="bag_1",
        order=["bag_3", "bag_6", "bag_2", "bag_4", "bag_5", "bag_1"],
    ),
    dict(
        request_id="cycle on cycle 4",
        root="bag_3",
        order=["bag_4", "bag_5", "bag_1", "bag_6", "bag_2", "bag_3"],
    ),
]

CHECK_COMPATIBLE_MAPPINGS_VALID_EXAMPLES = [
    ({1, 2, 3, 4}, ("a", "b", "c", "d"), {4, 3}, ("c", "d")),
    ({1, 2, 3, 4}, ("a", "b", "c", "d"), {4, 3, 7}, ("c", "d", "e")),
    ({1, 2}, ("a", "b"), {2, 1}, ("a", "b")),
    ({1, 2, 3, 4}, ("a", "b", "c", "d"), {4, 3}, ("c", "d")),
]

CHECK_COMPATIBLE_MAPPINGS_INVALID_EXAMPLES = [
    ({1, 2}, ("a", "b"), {2, 1}, ("b", "a",)),
]
