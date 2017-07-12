from collections import OrderedDict

from alib import datamodel


def filter_requests_by_tags(*tag_groups):
    """
    Filter request data by tags. Tags can be grouped.
    A request id is returned if it includes all tags of at least one tag group.

    :param tag_groups: A tag group is either a single string or a set of strings
    :return:
    """

    tag_groups = [{tags} if isinstance(tags, str) else set(tags) for tags in tag_groups]
    if not tag_groups:
        # default: tags_args is empty: yield everything
        tag_groups = [set()]
    for request_id, req_data in example_requests.items():
        if any(tags <= req_data["tags"] for tags in tag_groups):
            yield request_id


def create_request(request_id, reverse_edges=None):
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
        request.add_node(node, 1, "t1", allowed_nodes=allowed)
    for edge in request_dict["edges"]:
        if edge in reverse_edges:
            edge = edge[1], edge[0]
        request.add_edge(edge[0], edge[1], 1)
    request.graph["root"] = next(i for i in request.get_nodes() if not request.get_in_neighbors(i))
    return request


SIMPLE_PATH = {
    "tags": {"cactus", "bags", "tree"},
    "nodes": [
        "i1", "i2", "i3"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
    ],
    "labels": {
        "i1": set(),
        "i2": set(),
        "i3": set(),
    },
    "bags": {
        "i1": {frozenset(): {("i1", "i2")}},
        "i2": {frozenset(): {("i2", "i3")}},
        "i3": {frozenset(): set()},
    }
}

SIMPLE_CYCLE_1 = {
    "tags": {"cactus", "bags"},
    "nodes": [
        "i1", "i2", "i3"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
    ],
    "labels": {
        "i1": {"i3"},
        "i2": {"i3"},
        "i3": {"i3"},
    },
    "bags": {
        "i1": {frozenset(["i3"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i3"]): {("i2", "i3")}},
        "i3": {frozenset(): set()},
    }
}

SIMPLE_CYCLE_2 = {
    "tags": {"cactus"},
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
    ],
    "labels": {
        "i1": {"i4"},
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4"},
    }
}

# i4 should not propagate to i1
CYCLE_WITH_NODE_BEFORE_START_1 = {
    "tags": {"cactus", "flow_preservation_constraints", "node_mapping_constraints", "decomposition"},
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i2", "i3"),
        ("i2", "i4"),
        ("i3", "i4"),
    ],
    "labels": {
        "i1": set(),
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4"},
    },
    "assumed_root": "i2",
    "assumed_allowed_nodes": {
        "i1": ["v"],
        "i2": ["u"],
        "i3": ["u", "v"],
        "i4": ["u", "v"],
    },
    "constraints": {
        "flow_preservation": {
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i2','i1')]_comm_index[]", (
                (-1.0, "edge_var", (('i2', 'i1'), frozenset(), ("v", "u"))),
                (-1.0, "source_var", (('i2', 'i1'), frozenset(), "u")),
                (1.0, "edge_var", (('i2', 'i1'), frozenset(), ("u", "v"))),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i2','i1')]_comm_index[]", (
                (-1.0, "edge_var", (('i2', 'i1'), frozenset(), ("u", "v"))),
                (1.0, "edge_var", (('i2', 'i1'), frozenset(), ("v", "u"))),
                (1.0, "sink_var", (('i2', 'i1'), frozenset(), "v")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i2','i3')]_comm_index[i4_u]", (
                (-1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "u")]), ("v", "u"))),
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i4", "u")]), "u")),
                (1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "u")]), "u")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i2','i3')]_comm_index[i4_v]", (
                (-1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "v")]), ("v", "u"))),
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i4", "v")]), "u")),
                (1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "v")]), ("u", "v"))),
                (1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "v")]), "u")),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i2','i3')]_comm_index[i4_u]", (
                (-1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "u")]), ("v", "u"))),
                (1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "u")]), "v")),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i2','i3')]_comm_index[i4_v]", (
                (-1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "v")]), ("u", "v"))),
                (1.0, "edge_var", (("i2", "i3"), frozenset([("i4", "v")]), ("v", "u"))),
                (1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "v")]), "v")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i3','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "u")]), "u")),
                (-1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "u")]), ("v", "u"))),
                (1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "sink_var", (("i3", "i4"), frozenset([("i4", "u")]), "u")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i3','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "v")]), "u")),
                (-1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "v")]), ("v", "u"))),
                (1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "v")]), ("u", "v"))),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i3','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "u")]), "v")),
                (-1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "u")]), ("v", "u"))),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i3','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "v")]), "v")),
                (-1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "v")]), ("u", "v"))),
                (1.0, "edge_var", (("i3", "i4"), frozenset([("i4", "v")]), ("v", "u"))),
                (1.0, "sink_var", (("i3", "i4"), frozenset([("i4", "v")]), "v")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i2','i4')]_comm_index[i4_u]", (
                (-1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "u")]), ("v", "u"))),
                (-1.0, "source_var", (("i2", "i4"), frozenset([("i4", "u")]), "u")),
                (1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "sink_var", (("i2", "i4"), frozenset([("i4", "u")]), "u")),
            )),
            ("flow_preservation_req[{req_name}]_snode[u]_vedge[('i2','i4')]_comm_index[i4_v]", (
                (-1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "v")]), ("v", "u"))),
                (-1.0, "source_var", (("i2", "i4"), frozenset([("i4", "v")]), "u")),
                (1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "v")]), ("u", "v"))),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i2','i4')]_comm_index[i4_u]", (
                (-1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "u")]), ("u", "v"))),
                (1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "u")]), ("v", "u"))),
            )),
            ("flow_preservation_req[{req_name}]_snode[v]_vedge[('i2','i4')]_comm_index[i4_v]", (
                (-1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "v")]), ("u", "v"))),
                (1.0, "edge_var", (("i2", "i4"), frozenset([("i4", "v")]), ("v", "u"))),
                (1.0, "sink_var", (("i2", "i4"), frozenset([("i4", "v")]), "v")),
            ))
        },
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i1", "v")),
                (1.0, "node_mapping_var", ("i1", frozenset(), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i2", "u")),
                (1.0, "node_mapping_var", ("i2", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[u]_bag[i4]", (
                (-1.0, "node_agg_var", ("i2", "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i4]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[v]_bag[i4]", (
                (-1.0, "node_agg_var", ("i3", "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i4", "u")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i4", "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[u]_vedge[('i2','i1')]_comm_index[]", (
                (-1.0, "source_var", (("i2", "i1"), frozenset(), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i1]_snode[v]_vedge[('i2','i1')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i2", "i1"), frozenset(), "v")),
                (1.0, "node_mapping_var", ("i1", frozenset(), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[u]_vedge[('i2','i3')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[u]_vedge[('i2','i3')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[u]_vedge[('i2','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i2", "i4"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[u]_vedge[('i2','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i2", "i4"), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i2','i3')]_comm_index[i4_u]_bag[i4]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i2','i3')]_comm_index[i4_v]_bag[i4]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i2','i3')]_comm_index[i4_u]_bag[i4]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i2','i3')]_comm_index[i4_v]_bag[i4]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i3','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i3','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i2','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i2", "i4"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i2", "i4"), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i3','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i3", "i4"), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i3','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i3", "i4"), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
        }
    },
    "mappings": [
        {
            "flow": 0.25,
            "expected": {
                "nodes": {
                    "i1": "v",
                    "i2": "u",
                    "i3": "u",
                    "i4": "u",
                },
                "edges": {
                    ("i1", "i2"): [("v", "u")],
                    ("i2", "i3"): [],
                    ("i2", "i4"): [],
                    ("i3", "i4"): [],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset(), "v"),
                ("node_mapping_var", "i2", frozenset(), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "u")]), "u"),
                ("node_mapping_var", "i3", frozenset([("i4", "u")]), "u"),
                ("node_mapping_var", "i4", frozenset(), "u"),

                ("source_var", ("i2", "i1"), frozenset(), "u"),
                ("edge_var", ("i2", "i1"), frozenset(), ("u", "v")),
                ("sink_var", ("i2", "i1"), frozenset(), "v"),

                ("source_var", ("i2", "i3"), frozenset([("i4", "u")]), "u"),
                ("sink_var", ("i2", "i3"), frozenset([("i4", "u")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "u")]), "u"),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "u")]), "u"),

                ("source_var", ("i3", "i4"), frozenset([("i4", "u")]), "u"),
                ("sink_var", ("i3", "i4"), frozenset([("i4", "u")]), "u"),
            }
        },
        {
            "flow": 0.5,
            "expected": {
                "nodes": {
                    "i1": "v",
                    "i2": "u",
                    "i3": "u",
                    "i4": "v",
                },
                "edges": {
                    ("i1", "i2"): [("v", "u")],
                    ("i2", "i3"): [],
                    ("i2", "i4"): [("u", "v")],
                    ("i3", "i4"): [("u", "w"), ("w", "v")],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset(), "v"),
                ("node_mapping_var", "i2", frozenset(), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "v")]), "u"),
                ("node_mapping_var", "i3", frozenset([("i4", "v")]), "u"),
                ("node_mapping_var", "i4", frozenset(), "v"),

                ("source_var", ("i2", "i1"), frozenset(), "u"),
                ("edge_var", ("i2", "i1"), frozenset(), ("u", "v")),
                ("sink_var", ("i2", "i1"), frozenset(), "v"),

                ("source_var", ("i2", "i3"), frozenset([("i4", "v")]), "u"),
                ("sink_var", ("i2", "i3"), frozenset([("i4", "v")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "v")]), "u"),
                ("edge_var", ("i2", "i4"), frozenset([("i4", "v")]), ("u", "v")),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "v")]), "v"),

                ("source_var", ("i3", "i4"), frozenset([("i4", "v")]), "u"),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "v")]), ("u", "w")),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "v")]), ("w", "v")),
                ("sink_var", ("i3", "i4"), frozenset([("i4", "v")]), "v"),
            }
        }
    ]
}

# unbalanced version; i6 should not propagate to i1
CYCLE_WITH_NODE_BEFORE_START_2 = {
    "tags": {"cactus"},
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
    "labels": {
        "i1": set(),
        "i2": {"i6"},
        "i3": {"i6"},
        "i4": {"i6"},
        "i5": {"i6"},
        "i6": {"i6"},
    }
}

# only i3 should propagate to i1, i2, i3
CYCLE_WITH_NODE_AFTER_END = {
    "tags": {"cactus", "node_mapping_constraints"},
    "nodes": [
        "i1", "i2", "i3", "i4"
    ],
    "edges": [
        ("i1", "i2"),
        ("i1", "i3"),
        ("i2", "i3"),
        ("i3", "i4"),
    ],
    "labels": {
        "i1": {"i3"},
        "i2": {"i3"},
        "i3": {"i3"},
        "i4": set(),
    },
    "assumed_root": "i1",
    "assumed_allowed_nodes": {
        "i1": ["u"],
        "i2": ["v"],
        "i3": ["u", "v"],
        "i4": ["w"],
    },
    "constraints": {
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[u]_bag[i3]", (
                (-1.0, "node_agg_var", ("i1", "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i3]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "v")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i3", "v")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i4", "w")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i3_u]", (
                (-1.0, "source_var", (("i1", "i2"), frozenset([("i3", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i3_v]", (
                (-1.0, "source_var", (("i1", "i2"), frozenset([("i3", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i3_u]", (
                (-1.0, "source_var", (("i1", "i3"), frozenset([("i3", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i3_v]", (
                (-1.0, "source_var", (("i1", "i3"), frozenset([("i3", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i3", "v")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i3_u]_bag[i3]", (
                (-1.0, "sink_var", (("i1", "i2"), frozenset([("i3", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i3_v]_bag[i3]", (
                (-1.0, "sink_var", (("i1", "i2"), frozenset([("i3", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i3')]_comm_index[i3_u]", (
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i3", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i3')]_comm_index[i3_v]", (
                (-1.0, "source_var", (("i2", "i3"), frozenset([("i3", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i3", "v")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i1", "i3"), frozenset([("i3", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i1','i3')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i1", "i3"), frozenset([("i3", "v")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i2','i3')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i3", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i2','i3')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i2", "i3"), frozenset([("i3", "v")]), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset(), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[v]_vedge[('i3','i4')]_comm_index[]", (
                (-1.0, "source_var", (("i3", "i4"), frozenset(), "v")),
                (1.0, "node_mapping_var", ("i3", frozenset(), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (("i3", "i4"), frozenset(), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "w")),
            )),
        }
    }
}

# i4 should propagate to all cycle nodes i2, i3, i4
CYCLE_WITH_NODES_BEFORE_AND_AFTER = {
    "tags": {"cactus"},
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
    "labels": {
        "i1": set(),
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4"},
        "i5": set(),
    }
}

# i4 should not propagate to i5
CYCLE_WITH_BRANCH = {
    "tags": {"cactus", "bags"},
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
    "labels": {
        "i1": {"i4"},
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4"},
        "i5": set(),
    },
    "bags": {
        "i1": {frozenset(["i4"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i4"]): {("i2", "i4")}},
        "i3": {frozenset(["i4"]): {("i3", "i4")},
               frozenset(): {("i3", "i5")}, },
        "i4": {frozenset(): set()},
        "i5": {frozenset(): set()},
    },
}

# i3 should propagate to i1, i2, i3
# i6 should propagate to i4, i5, i6
TWO_CYCLES_CONNECTED_WITH_SINGLE_EDGE = {
    "tags": {"cactus"},
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
    "labels": {
        "i1": {"i3"},
        "i2": {"i3"},
        "i3": {"i3"},
        "i4": {"i6"},
        "i5": {"i6"},
        "i6": {"i6"},
    }
}

# i3 should propagate to i1, i2, i3
# i5 should propagate to i3, i4, i5
TWO_CYCLES_CONNECTED_DIRECTLY = {
    "tags": {"cactus", "bags"},
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
    "labels": {
        "i1": {"i3"},
        "i2": {"i3"},
        "i3": {"i3", "i5"},
        "i4": {"i5"},
        "i5": {"i5"},
    },
    "bags": {
        "i1": {frozenset(["i3"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i3"]): {("i2", "i3")}},
        "i3": {frozenset(["i5"]): {("i3", "i4"), ("i3", "i5")}},
        "i4": {frozenset(["i5"]): {("i4", "i5")}},
        "i5": {frozenset(): set()},
    },
}

# i4 should propagate to i1, i2, i3, i4
# i7 should propagate to i4, i5, i6, i7
# i10 should propagate to i4, i8, i9, i10
THREE_CYCLES_CONNECTED_DIRECTLY = {
    "tags": {"cactus", "bags", "node_mapping_constraints"},
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
    "labels": {
        "i1": {"i4"},
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4", "i7", "i10"},
        "i5": {"i7"},
        "i6": {"i7"},
        "i7": {"i7"},
        "i8": {"i10"},
        "i9": {"i10"},
        "i10": {"i10"},
    },
    "bags": {
        "i1": {frozenset(["i4"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i4"]): {("i2", "i4")}},
        "i3": {frozenset(["i4"]): {("i3", "i4")}},
        "i4": {frozenset(["i7"]): {("i4", "i5"), ("i4", "i6")},
               frozenset(["i10"]): {("i4", "i8"), ("i4", "i9")}},
        "i5": {frozenset(["i7"]): {("i5", "i7")}},
        "i6": {frozenset(["i7"]): {("i6", "i7")}},
        "i7": {frozenset(): set()},
        "i8": {frozenset(["i10"]): {("i8", "i10")}},
        "i9": {frozenset(["i10"]): {("i9", "i10")}},
        "i10": {frozenset(): set()},
    },
    "assumed_root": "i1",
    "ignore_bfs": True,
    "assumed_allowed_nodes": {
        "i1": ["u"],
        "i2": ["v"],
        "i3": ["u"],
        "i4": ["u", "v"],
        "i5": ["u"],
        "i6": ["u"],
        "i7": ["u", "w"],
        "i8": ["w"],
        "i9": ["v"],
        "i10": ["v", "w"],
    },
    "constraints": {
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[u]_bag[i4]", (
                (-1.0, "node_agg_var", ("i1", "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i4]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i4]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[u]_bag[i7]", (
                (-1.0, "node_agg_var", ("i4", "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[v]_bag[i7]", (
                (-1.0, "node_agg_var", ("i4", "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[u]_bag[i10]", (
                (-1.0, "node_agg_var", ("i4", "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[v]_bag[i10]", (
                (-1.0, "node_agg_var", ("i4", "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[u]_bag[i7]", (
                (-1.0, "node_agg_var", ("i5", "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[u]_bag[i7]", (
                (-1.0, "node_agg_var", ("i6", "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i7]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i7", "u")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i7]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i7", "w")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "w")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i8]_snode[w]_bag[i10]", (
                (-1.0, "node_agg_var", ("i8", "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "v")]), "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "w")]), "w")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i9]_snode[v]_bag[i10]", (
                (-1.0, "node_agg_var", ("i9", "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i10]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i10", "v")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i10]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i10", "w")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "w")),
            )),

            #####



            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v")]), "u")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_u]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_u]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_u]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v")]), "u")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i2','i4')]_comm_index[]_bag[i7]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[]_bag[i7]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i2','i4')]_comm_index[]_bag[i10]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[]_bag[i10]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i3','i4')]_comm_index[]_bag[i7]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i3','i4')]_comm_index[]_bag[i7]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i3','i4')]_comm_index[]_bag[i10]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i3','i4')]_comm_index[]_bag[i10]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i5')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i5')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i6')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i6')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i8')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i4', 'i8'), frozenset([("i10", "v")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i8')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i4', 'i8'), frozenset([("i10", "w")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i9')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i4', 'i9'), frozenset([("i10", "v")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[u]_vedge[('i4','i9')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i4', 'i9'), frozenset([("i10", "w")]), "u")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i5')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i7", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i5')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i7", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i6')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i7", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i6')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i7", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i7", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i8')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i4', 'i8'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i8')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i4', 'i8'), frozenset([("i10", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i9')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i4', 'i9'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i9')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i4', 'i9'), frozenset([("i10", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i10", "w")]), "v")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i4','i5')]_comm_index[i7_u]_bag[i7]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i4','i5')]_comm_index[i7_w]_bag[i7]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i7')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i5', 'i7'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i7')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i5', 'i7'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i7", "w")]), "u")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i4','i6')]_comm_index[i7_u]_bag[i7]", (
                (-1.0, "sink_var", (('i4', 'i6'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i4','i6')]_comm_index[i7_w]_bag[i7]", (
                (-1.0, "sink_var", (('i4', 'i6'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i6','i7')]_comm_index[i7_u]", (
                (-1.0, "source_var", (('i6', 'i7'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i6','i7')]_comm_index[i7_w]", (
                (-1.0, "source_var", (('i6', 'i7'), frozenset([("i7", "w")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset([("i7", "w")]), "u")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i7]_snode[u]_vedge[('i5','i7')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i7'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i7]_snode[w]_vedge[('i5','i7')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i7'), frozenset([("i7", "w")]), "w")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i7]_snode[u]_vedge[('i6','i7')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i6', 'i7'), frozenset([("i7", "u")]), "u")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i7]_snode[w]_vedge[('i6','i7')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i6', 'i7'), frozenset([("i7", "w")]), "w")),
                (1.0, "node_mapping_var", ("i7", frozenset(), "w")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i8]_snode[w]_vedge[('i4','i8')]_comm_index[i10_v]_bag[i10]", (
                (-1.0, "sink_var", (('i4', 'i8'), frozenset([("i10", "v")]), "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "v")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i8]_snode[w]_vedge[('i4','i8')]_comm_index[i10_w]_bag[i10]", (
                (-1.0, "sink_var", (('i4', 'i8'), frozenset([("i10", "w")]), "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "w")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i8]_snode[w]_vedge[('i8','i10')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i8', 'i10'), frozenset([("i10", "v")]), "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "v")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i8]_snode[w]_vedge[('i8','i10')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i8', 'i10'), frozenset([("i10", "w")]), "w")),
                (1.0, "node_mapping_var", ("i8", frozenset([("i10", "w")]), "w")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i9]_snode[v]_vedge[('i4','i9')]_comm_index[i10_v]_bag[i10]", (
                (-1.0, "sink_var", (('i4', 'i9'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "v")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i9]_snode[v]_vedge[('i4','i9')]_comm_index[i10_w]_bag[i10]", (
                (-1.0, "sink_var", (('i4', 'i9'), frozenset([("i10", "w")]), "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i9]_snode[v]_vedge[('i9','i10')]_comm_index[i10_v]", (
                (-1.0, "source_var", (('i9', 'i10'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i9]_snode[v]_vedge[('i9','i10')]_comm_index[i10_w]", (
                (-1.0, "source_var", (('i9', 'i10'), frozenset([("i10", "w")]), "v")),
                (1.0, "node_mapping_var", ("i9", frozenset([("i10", "w")]), "v")),
            )),

            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i10]_snode[v]_vedge[('i8','i10')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i8', 'i10'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i10]_snode[w]_vedge[('i8','i10')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i8', 'i10'), frozenset([("i10", "w")]), "w")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i10]_snode[v]_vedge[('i9','i10')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i9', 'i10'), frozenset([("i10", "v")]), "v")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i10]_snode[w]_vedge[('i9','i10')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i9', 'i10'), frozenset([("i10", "w")]), "w")),
                (1.0, "node_mapping_var", ("i10", frozenset(), "w")),
            )),

        }
    }
}

# i4 should propagate to i1, i2, i3, i4
# i7 should propagate to i4, i5, i6, i7
# i10 should propagate to i7, i8, i9, i10
THREE_CYCLES_CONNECTED_CHAINED = {
    "tags": {"cactus", "bags"},
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
    "labels": {
        "i1": {"i4"},
        "i2": {"i4"},
        "i3": {"i4"},
        "i4": {"i4", "i7"},
        "i5": {"i7"},
        "i6": {"i7"},
        "i7": {"i7", "i10"},
        "i8": {"i10"},
        "i9": {"i10"},
        "i10": {"i10"},
    },
    "bags": {
        "i1": {frozenset(["i4"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i4"]): {("i2", "i4")}},
        "i3": {frozenset(["i4"]): {("i3", "i4")}},
        "i4": {frozenset(["i7"]): {("i4", "i5"), ("i4", "i6")}},
        "i5": {frozenset(["i7"]): {("i5", "i7")}},
        "i6": {frozenset(["i7"]): {("i6", "i7")}},
        "i7": {frozenset(["i10"]): {("i7", "i8"), ("i7", "i9")}},
        "i8": {frozenset(["i10"]): {("i8", "i10")}},
        "i9": {frozenset(["i10"]): {("i9", "i10")}},
        "i10": {frozenset(): set()},
    },
}

CYCLE_ON_CYCLE_1 = {
    "tags": {"multiple_labels", "bags", "node_mapping_constraints"},
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
    "labels": {
        "i1": {"i6"},
        "i2": {"i6"},
        "i3": {"i5", "i6"},
        "i4": {"i5", "i6"},
        "i5": {"i5", "i6"},
        "i6": {"i6"},
    },
    "bags": {
        "i1": {frozenset(["i6"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i6"]): {("i2", "i6")}},
        "i3": {frozenset(["i5", "i6"]): {("i3", "i4"), ("i3", "i5")}},
        "i4": {frozenset(["i5", "i6"]): {("i4", "i5")}},
        "i5": {frozenset(["i6"]): {("i5", "i6")}},
        "i6": {frozenset(): set()},
    },
    "assumed_root": "i1",
    "ignore_bfs": True,
    "assumed_allowed_nodes": {
        "i1": ["u"],
        "i2": ["v"],
        "i3": ["u"],
        "i4": ["w"],
        "i5": ["u", "v"],
        "i6": ["u", "w"],
    },
    "constraints": {
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[u]_bag[i6]", (
                (-1.0, "node_agg_var", ("i1", "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i6]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i5_i6]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[w]_bag[i5_i6]", (
                (-1.0, "node_agg_var", ("i4", "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "w")]), "w")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[u]_bag[i6]", (
                (-1.0, "node_agg_var", ("i5", "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[v]_bag[i6]", (
                (-1.0, "node_agg_var", ("i5", "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i2', 'i6'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i2', 'i6'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i6_u]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i6_w]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i5_u__i6_u]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i5", "u"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "u")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i5_u__i6_w]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i5", "u"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "w")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i5_v__i6_u]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i5", "v"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "u")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i5_v__i6_w]_bag[i5_i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i5", "v"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "w")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i5')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "u")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i5')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "u"), ("i6", "w")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i5')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "u")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i5')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i4', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i5", "v"), ("i6", "w")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i3','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i3','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i3','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i3','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i4','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i4','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i4','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i4','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i4', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i5','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i5','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i2','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i2','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i5','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i5','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
        }
    }
}

CYCLE_ON_CYCLE_2 = {
    "tags": {"multiple_labels", "bags"},
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
    "labels": {
        "i1": {"i5", "i6"},
        "i2": {"i5", "i6"},
        "i3": {"i5", "i6"},
        "i4": {"i5", "i6"},
        "i5": {"i5", "i6"},
        "i6": {"i6"},
    },
    "bags": {
        "i1": {frozenset(["i5", "i6"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i5", "i6"]): {("i2", "i4")}},
        "i3": {frozenset(["i5", "i6"]): {("i3", "i5")}},
        "i4": {frozenset(["i5", "i6"]): {("i4", "i5"), ("i4", "i6")}},
        "i5": {frozenset(["i6"]): {("i5", "i6")}},
        "i6": {frozenset(): set()},
    },
}

CYCLE_ON_CYCLE_3 = {
    "tags": set(),
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
    "labels": {
        "i1": {"i6"},
        "i2": {"i6"},
        "i3": {"i6"},
        "i4": {"i6"},
        "i5": {"i6"},
        "i6": {"i6"},
    }
}

CYCLE_ON_CYCLE_4 = {
    "tags": {"bags"},
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
    "labels": {
        "i1": {"i8"},
        "i2": {"i8"},
        "i3": {"i5", "i8"},
        "i4": {"i5", "i8"},
        "i5": {"i5", "i7", "i8"},
        "i6": {"i7", "i8"},
        "i7": {"i7", "i8"},
        "i8": {"i8"},
    },
    "bags": {
        "i1": {frozenset(["i8"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i8"]): {("i2", "i8")}},
        "i3": {frozenset(["i5", "i8"]): {("i3", "i4"), ("i3", "i5")}},
        "i4": {frozenset(["i5", "i8"]): {("i4", "i5")}},
        "i5": {frozenset(["i7", "i8"]): {("i5", "i6"), ("i5", "i7")}},
        "i6": {frozenset(["i7", "i8"]): {("i6", "i7")}},
        "i7": {frozenset(["i8"]): {("i7", "i8")}},
        "i8": {frozenset(): set()},
    },
}

CYCLES_CROSSING = {
    "tags": {"multiple_labels", "bags", "node_mapping_constraints", "decomposition"},
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
    "labels": {
        "i1": {"i4", "i5", "i6"},
        "i2": {"i4", "i5", "i6"},
        "i3": {"i4", "i5", "i6"},
        "i4": {"i4", "i6"},
        "i5": {"i5", "i6"},
        "i6": {"i6"},
    },
    "bags": {
        "i1": {frozenset(["i4", "i5", "i6"]): {("i1", "i2"), ("i1", "i3")}},
        "i2": {frozenset(["i4", "i5", "i6"]): {("i2", "i4"), ("i2", "i5")}},
        "i3": {frozenset(["i4", "i5", "i6"]): {("i3", "i4"), ("i3", "i5")}},
        "i4": {frozenset(["i6"]): {("i4", "i6")}},
        "i5": {frozenset(["i6"]): {("i5", "i6")}},
        "i6": {frozenset(): set()},
    },
    "assumed_root": "i1",
    "ignore_bfs": True,
    "assumed_allowed_nodes": {
        "i1": ["u"],
        "i2": ["v"],
        "i3": ["u"],
        "i4": ["v", "w"],
        "i5": ["u", "v"],
        "i6": ["u", "w"],
    },
    "constraints": {
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[u]_bag[i4_i5_i6]", (
                (-1.0, "node_agg_var", ("i1", "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i4_i5_i6]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i4_i5_i6]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[v]_bag[i6]", (
                (-1.0, "node_agg_var", ("i4", "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[w]_bag[i6]", (
                (-1.0, "node_agg_var", ("i4", "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "w")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[u]_bag[i6]", (
                (-1.0, "node_agg_var", ("i5", "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[v]_bag[i6]", (
                (-1.0, "node_agg_var", ("i5", "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            # i1, u, i1 -> i2
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i5_u__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i5_u__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i5_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i5_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i5_u__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i5_u__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i5_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i5_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i1, u, i1 -> i3
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_u__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_u__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_u__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_u__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),

            # i2, v, i1 -> i2
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v__i5_u__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v__i5_u__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v__i5_v__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v__i5_v__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_w__i5_u__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_w__i5_u__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_w__i5_v__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_w__i5_v__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "v")),
            )),
            # i2, v, i2 -> i4
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_v__i6_u]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_v__i6_w]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_w__i6_u]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "w"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_w__i6_w]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "w"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "v")),
            )),
            # i2, v, i2 -> i5
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i5')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i2', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i5')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i2', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i5')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i2', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i5')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i2', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "v")),
            )),

            # i3, u, i1 -> i3
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_u__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_u__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_v__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_v__i5_v__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_u__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_u__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_v__i6_u]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i4_w__i5_v__i6_w]_bag[i4_i5_i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i3, u, i3 -> i4
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_v__i6_u]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_v__i6_w]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_w__i6_u]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "w"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i4')]_comm_index[i4_w__i6_w]", (
                (-1.0, "source_var", (('i3', 'i4'), frozenset([("i4", "w"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i3, u, i3 -> i5
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),

            # i4, v, i2 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "v")),
            )),
            # i4, w, i2 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i2','i4')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "w"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i2','i4')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "w"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "w")),
            )),
            # i4, v, i3 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i3','i4')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i3','i4')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "v")),
            )),
            # i4, w, i3 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "w"), ("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "w")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i3','i4')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i4'), frozenset([("i4", "w"), ("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "w")),
            )),
            # i4, v, i4 -> i6
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i4','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "v")),
            )),
            # i4, w, i4 -> i6
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i6", "u")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "u")]), "w")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i4','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i4', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset([("i6", "w")]), "w")),
            )),

            # i5, u, i2 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i2','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i2','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            # i5, v, i2 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i2','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i2','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i2', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            # i5, u, i3 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i3','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i3','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            # i5, v, i3 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i3','i5')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i3','i5')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),
            # i5, u, i5 -> i6
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i5','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "u")),
            )),
            # i5, v, i5 -> i6
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i5','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i5','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i5', 'i6'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset([("i6", "w")]), "v")),
            )),

            # i6, u, i4 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i4','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i4', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            # i6, w, i4 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i4','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i4', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            # i6, u, i5 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i5','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            # i6, w, i5 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i5','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i5', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
        }
    },
    "mappings": [
        {
            "flow": 0.125,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "w",
                    "i5": "v",
                    "i6": "u",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i2", "i4"): [("v", "w")],
                    ("i2", "i5"): [],
                    ("i3", "i4"): [("u", "w")],
                    ("i3", "i5"): [("u", "w"), ("w", "v")],
                    ("i4", "i6"): [("w", "u")],
                    ("i5", "i6"): [("v", "u")],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i4", frozenset([("i6", "u")]), "w"),
                ("node_mapping_var", "i5", frozenset([("i6", "u")]), "v"),
                ("node_mapping_var", "i6", frozenset(), "u"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "w"), ("i6", "u")]), "v"),
                ("edge_var", ("i2", "i4"), frozenset([("i4", "w"), ("i6", "u")]), ("v", "w")),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "w"), ("i6", "u")]), "w"),

                ("source_var", ("i2", "i5"), frozenset([("i5", "v"), ("i6", "u")]), "v"),
                ("sink_var", ("i2", "i5"), frozenset([("i5", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i3", "i4"), frozenset([("i4", "w"), ("i6", "u")]), "u"),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "w"), ("i6", "u")]), ("u", "w")),
                ("sink_var", ("i3", "i4"), frozenset([("i4", "w"), ("i6", "u")]), "w"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "v"), ("i6", "u")]), "u"),
                ("edge_var", ("i3", "i5"), frozenset([("i5", "v"), ("i6", "u")]), ("u", "w")),
                ("edge_var", ("i3", "i5"), frozenset([("i5", "v"), ("i6", "u")]), ("w", "v")),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i4", "i6"), frozenset([("i6", "u")]), "w"),
                ("edge_var", ("i4", "i6"), frozenset([("i6", "u")]), ("w", "u")),
                ("sink_var", ("i4", "i6"), frozenset([("i6", "u")]), "u"),

                ("source_var", ("i5", "i6"), frozenset([("i6", "u")]), "v"),
                ("edge_var", ("i5", "i6"), frozenset([("i6", "u")]), ("v", "u")),
                ("sink_var", ("i5", "i6"), frozenset([("i6", "u")]), "u"),
            }
        },
        {
            "flow": 0.25,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "v",
                    "i5": "u",
                    "i6": "u",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i2", "i4"): [],
                    ("i2", "i5"): [("v", "u")],
                    ("i3", "i4"): [("u", "v")],
                    ("i3", "i5"): [],
                    ("i4", "i6"): [("v", "u")],
                    ("i5", "i6"): [],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i4", frozenset([("i6", "u")]), "v"),
                ("node_mapping_var", "i5", frozenset([("i6", "u")]), "u"),
                ("node_mapping_var", "i6", frozenset(), "u"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "v"),
                ("edge_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), ("v", "u")),
                ("sink_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "u"),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), ("u", "v")),
                ("sink_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i4", "i6"), frozenset([("i6", "u")]), "v"),
                ("edge_var", ("i4", "i6"), frozenset([("i6", "u")]), ("v", "u")),
                ("sink_var", ("i4", "i6"), frozenset([("i6", "u")]), "u"),

                ("source_var", ("i5", "i6"), frozenset([("i6", "u")]), "u"),
                ("sink_var", ("i5", "i6"), frozenset([("i6", "u")]), "u"),
            }
        },
        # This is the same as above, except (i3, i4) -> [(u, w), (w, v)] instead of [(u, v)]
        {
            "flow": 0.5,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "v",
                    "i5": "u",
                    "i6": "u",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i2", "i4"): [],
                    ("i2", "i5"): [("v", "u")],
                    ("i3", "i4"): [("u", "w"), ("w", "v")],
                    ("i3", "i5"): [],
                    ("i4", "i6"): [("v", "u")],
                    ("i5", "i6"): [],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i4", frozenset([("i6", "u")]), "v"),
                ("node_mapping_var", "i5", frozenset([("i6", "u")]), "u"),
                ("node_mapping_var", "i6", frozenset(), "u"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "v"),
                ("edge_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), ("v", "u")),
                ("sink_var", ("i2", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "u"),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), ("u", "w")),
                ("edge_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), ("w", "v")),
                ("sink_var", ("i3", "i4"), frozenset([("i4", "v"), ("i6", "u")]), "v"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "u"), ("i6", "u")]), "u"),

                ("source_var", ("i4", "i6"), frozenset([("i6", "u")]), "v"),
                ("edge_var", ("i4", "i6"), frozenset([("i6", "u")]), ("v", "u")),
                ("sink_var", ("i4", "i6"), frozenset([("i6", "u")]), "u"),

                ("source_var", ("i5", "i6"), frozenset([("i6", "u")]), "u"),
                ("sink_var", ("i5", "i6"), frozenset([("i6", "u")]), "u"),
            }
        }
    ]
}

THREE_BRANCHES = {
    "tags": set(),
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
    "labels": {
        "i1": {"i5"},
        "i2": {"i5"},
        "i3": {"i5"},
        "i4": {"i5"},
        "i5": {"i5"},
    }
}

DRAGON_1 = {
    "tags": {"dragon", "bags", "multiple_labels", "decomposition"},  # note: we only use this graph to test decomposition of unembedded request!
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
    "labels": {
        "i1": {"i4", "i5", "i6"},
        "i2": {"i4", "i6"},
        "i3": {"i5", "i6"},
        "i4": {"i4"},
        "i5": {"i5"},
        "i6": {"i6"},
    },
    "bags": {
        "i1": {frozenset(["i4", "i5", "i6"]): {("i1", "i2"), ("i1", "i3"), ("i1", "i4"), ("i1", "i5")}},
        "i2": {frozenset(["i4"]): {("i2", "i4")},
               frozenset(["i6"]): {("i2", "i6")}},
        "i3": {frozenset(["i5"]): {("i3", "i5")},
               frozenset(["i6"]): {("i3", "i6")}},
        "i4": {frozenset(): set()},
        "i5": {frozenset(): set()},
        "i6": {frozenset(): set()},
    },
    "assumed_root": "i1",
    "ignore_bfs": True,
    "mappings": []
}

DRAGON_2 = {
    "tags": {"dragon", "bags", "multiple_labels", "node_mapping_constraints", "decomposition"},
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
    "labels": {
        "i1": {"i4", "i5", "i6"},
        "i2": {"i4", "i6"},
        "i3": {"i5", "i6"},
        "i4": {"i4"},
        "i5": {"i5"},
        "i6": {"i6"},
    },
    "bags": {
        "i1": {frozenset(["i4", "i5", "i6"]): {("i1", "i2"), ("i1", "i3"), ("i1", "i4"), ("i1", "i5"), ("i1", "i6")}},
        "i2": {frozenset(["i4"]): {("i2", "i4")},
               frozenset(["i6"]): {("i2", "i6")}},
        "i3": {frozenset(["i5"]): {("i3", "i5")},
               frozenset(["i6"]): {("i3", "i6")}},
        "i4": {frozenset(): set()},
        "i5": {frozenset(): set()},
        "i6": {frozenset(): set()},
    },
    "assumed_root": "i1",
    "ignore_bfs": True,
    "assumed_allowed_nodes": {
        "i1": ["u"],
        "i2": ["v"],
        "i3": ["u"],
        "i4": ["v", "w"],
        "i5": ["u", "v"],
        "i6": ["u", "w"],
    },
    "constraints": {
        "node_mapping": {
            ("node_mapping_aggregation_req[{req_name}]_vnode[i1]_snode[u]_bag[i4_i5_i6]", (
                (-1.0, "node_agg_var", ("i1", "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i4]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i2]_snode[v]_bag[i6]", (
                (-1.0, "node_agg_var", ("i2", "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i5]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i3]_snode[u]_bag[i6]", (
                (-1.0, "node_agg_var", ("i3", "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "w")]), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i4", "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i4]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i4", "w")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "w")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i5", "u")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i5]_snode[v]_bag[]", (
                (-1.0, "node_agg_var", ("i5", "v")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "v")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[u]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            ("node_mapping_aggregation_req[{req_name}]_vnode[i6]_snode[w]_bag[]", (
                (-1.0, "node_agg_var", ("i6", "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),

            # i1, u, i1 -> i2
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i6_u]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i2')]_comm_index[i4_w__i6_w]", (
                (-1.0, "source_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i1, u, i1 -> i3
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i5_u__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i5_u__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i5_v__i6_u]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i3')]_comm_index[i5_v__i6_w]", (
                (-1.0, "source_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i1, u, i1 -> i4
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i1', 'i4'), frozenset([("i4", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i4')]_comm_index[i4_w]", (
                (-1.0, "source_var", (('i1', 'i4'), frozenset([("i4", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i1, u, i1 -> i5
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i5')]_comm_index[i5_u]", (
                (-1.0, "source_var", (('i1', 'i5'), frozenset([("i5", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i5')]_comm_index[i5_v]", (
                (-1.0, "source_var", (('i1', 'i5'), frozenset([("i5", "v")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),
            # i1, u, i1 -> i6
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i1', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i1]_snode[u]_vedge[('i1','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i1', 'i6'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "v"), ("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "w")]), "u")),
            )),

            # i2, v, i1 -> i2
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_v]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "u")]), "v")),
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i4_w]_bag[i4]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "u")]), "v")),
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "u")]), "v")),
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i1','i2')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "v"), ("i6", "w")]), "v")),
                (-1.0, "sink_var", (('i1', 'i2'), frozenset([("i4", "w"), ("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),
            # i2, v, i2 -> i4
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_v]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "v")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i4')]_comm_index[i4_w]", (
                (-1.0, "source_var", (('i2', 'i4'), frozenset([("i4", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i4", "w")]), "v")),
            )),
            # i2, v, i2 -> i6
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i2', 'i6'), frozenset([("i6", "u")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "u")]), "v")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i2]_snode[v]_vedge[('i2','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i2', 'i6'), frozenset([("i6", "w")]), "v")),
                (1.0, "node_mapping_var", ("i2", frozenset([("i6", "w")]), "v")),
            )),

            # i3, u, i1 -> i3
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i5_u]_bag[i5]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i5_v]_bag[i5]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i6_u]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "u")]), "u")),
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i1','i3')]_comm_index[i6_w]_bag[i6]", (
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "u"), ("i6", "w")]), "u")),
                (-1.0, "sink_var", (('i1', 'i3'), frozenset([("i5", "v"), ("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "w")]), "u")),
            )),
            # i3, u, i3 -> i5
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_u]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i5')]_comm_index[i5_v]", (
                (-1.0, "source_var", (('i3', 'i5'), frozenset([("i5", "v")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i5", "v")]), "u")),
            )),
            # i3, u, i3 -> i6
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i6')]_comm_index[i6_u]", (
                (-1.0, "source_var", (('i3', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "u")]), "u")),
            )),
            ("node_mapping_req[{req_name}]_vnode[i3]_snode[u]_vedge[('i3','i6')]_comm_index[i6_w]", (
                (-1.0, "source_var", (('i3', 'i6'), frozenset([("i6", "w")]), "u")),
                (1.0, "node_mapping_var", ("i3", frozenset([("i6", "w")]), "u")),
            )),

            # i4, v, i1 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i1','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
            # i4, w, i1 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i1','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i4'), frozenset([("i4", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "w")),
            )),
            # i4, v, i2 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[v]_vedge[('i2','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "v")]), "v")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "v")),
            )),
            # i4, w, i2 -> i4
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i4]_snode[w]_vedge[('i2','i4')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i4'), frozenset([("i4", "w")]), "w")),
                (1.0, "node_mapping_var", ("i4", frozenset(), "w")),
            )),

            # i5, u, i1 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i1','i5')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i5'), frozenset([("i5", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "u")),
            )),
            # i5, v, i1 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i1','i5')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i5'), frozenset([("i5", "v")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "v")),
            )),
            # i5, u, i3 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[u]_vedge[('i3','i5')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "u")]), "u")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "u")),
            )),
            # i5, v, i3 -> i5
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i5]_snode[v]_vedge[('i3','i5')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i3', 'i5'), frozenset([("i5", "v")]), "v")),
                (1.0, "node_mapping_var", ("i5", frozenset(), "v")),
            )),

            # i6, u, i1 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i1','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            # i6, w, i1 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i1','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i1', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            # i6, u, i2 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i2','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            # i6, w, i2 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i2','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i2', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),
            # i6, u, i3 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[u]_vedge[('i3','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i3', 'i6'), frozenset([("i6", "u")]), "u")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "u")),
            )),
            # i6, w, i3 -> i6
            ("node_mapping_in_edge_to_bag_req[{req_name}]_vnode[i6]_snode[w]_vedge[('i3','i6')]_comm_index[]_bag[]", (
                (-1.0, "sink_var", (('i3', 'i6'), frozenset([("i6", "w")]), "w")),
                (1.0, "node_mapping_var", ("i6", frozenset(), "w")),
            )),

        }
    },
    "mappings": [
        {
            "flow": 0.375,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "v",
                    "i5": "u",
                    "i6": "w",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i1", "i4"): [("u", "w"), ("w", "v")],
                    ("i1", "i5"): [],
                    ("i1", "i6"): [("u", "w")],
                    ("i2", "i4"): [],
                    ("i2", "i6"): [("v", "w")],
                    ("i3", "i5"): [],
                    ("i3", "i6"): [("u", "w")],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "v")]), "v"),
                ("node_mapping_var", "i2", frozenset([("i6", "w")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i5", "u")]), "u"),
                ("node_mapping_var", "i3", frozenset([("i6", "w")]), "u"),
                ("node_mapping_var", "i4", frozenset(), "v"),
                ("node_mapping_var", "i5", frozenset(), "u"),
                ("node_mapping_var", "i6", frozenset(), "w"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i5", "u"), ("i6", "w")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i5", "u"), ("i6", "w")]), "u"),

                ("source_var", ("i1", "i4"), frozenset([("i4", "v")]), "u"),
                ("edge_var", ("i1", "i4"), frozenset([("i4", "v")]), ("u", "w")),
                ("edge_var", ("i1", "i4"), frozenset([("i4", "v")]), ("w", "v")),
                ("sink_var", ("i1", "i4"), frozenset([("i4", "v")]), "v"),

                ("source_var", ("i1", "i5"), frozenset([("i5", "u")]), "u"),
                ("sink_var", ("i1", "i5"), frozenset([("i5", "u")]), "u"),

                ("source_var", ("i1", "i6"), frozenset([("i6", "w")]), "u"),
                ("edge_var", ("i1", "i6"), frozenset([("i6", "w")]), ("u", "w")),
                ("sink_var", ("i1", "i6"), frozenset([("i6", "w")]), "w"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "v")]), "v"),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "v")]), "v"),

                ("source_var", ("i2", "i6"), frozenset([("i6", "w")]), "v"),
                ("edge_var", ("i2", "i6"), frozenset([("i6", "w")]), ("v", "w")),
                ("sink_var", ("i2", "i6"), frozenset([("i6", "w")]), "w"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "u")]), "u"),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "u")]), "u"),

                ("source_var", ("i3", "i6"), frozenset([("i6", "w")]), "u"),
                ("edge_var", ("i3", "i6"), frozenset([("i6", "w")]), ("u", "w")),
                ("sink_var", ("i3", "i6"), frozenset([("i6", "w")]), "w"),
            }
        },
        {
            "flow": 0.0625,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "v",
                    "i5": "u",
                    "i6": "w",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i1", "i4"): [("u", "w"), ("w", "v")],
                    ("i1", "i5"): [],
                    ("i1", "i6"): [("u", "v"), ("v", "w")],
                    ("i2", "i4"): [],
                    ("i2", "i6"): [("v", "w")],
                    ("i3", "i5"): [],
                    ("i3", "i6"): [("u", "w")],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "v"), ("i5", "u"), ("i6", "w")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "v")]), "v"),
                ("node_mapping_var", "i2", frozenset([("i6", "w")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i5", "u")]), "u"),
                ("node_mapping_var", "i3", frozenset([("i6", "w")]), "u"),
                ("node_mapping_var", "i4", frozenset(), "v"),
                ("node_mapping_var", "i5", frozenset(), "u"),
                ("node_mapping_var", "i6", frozenset(), "w"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "v"), ("i6", "w")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i5", "u"), ("i6", "w")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i5", "u"), ("i6", "w")]), "u"),

                ("source_var", ("i1", "i4"), frozenset([("i4", "v")]), "u"),
                ("edge_var", ("i1", "i4"), frozenset([("i4", "v")]), ("u", "w")),
                ("edge_var", ("i1", "i4"), frozenset([("i4", "v")]), ("w", "v")),
                ("sink_var", ("i1", "i4"), frozenset([("i4", "v")]), "v"),

                ("source_var", ("i1", "i5"), frozenset([("i5", "u")]), "u"),
                ("sink_var", ("i1", "i5"), frozenset([("i5", "u")]), "u"),

                ("source_var", ("i1", "i6"), frozenset([("i6", "w")]), "u"),
                ("edge_var", ("i1", "i6"), frozenset([("i6", "w")]), ("u", "v")),
                ("edge_var", ("i1", "i6"), frozenset([("i6", "w")]), ("v", "w")),
                ("sink_var", ("i1", "i6"), frozenset([("i6", "w")]), "w"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "v")]), "v"),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "v")]), "v"),

                ("source_var", ("i2", "i6"), frozenset([("i6", "w")]), "v"),
                ("edge_var", ("i2", "i6"), frozenset([("i6", "w")]), ("v", "w")),
                ("sink_var", ("i2", "i6"), frozenset([("i6", "w")]), "w"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "u")]), "u"),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "u")]), "u"),

                ("source_var", ("i3", "i6"), frozenset([("i6", "w")]), "u"),
                ("edge_var", ("i3", "i6"), frozenset([("i6", "w")]), ("u", "w")),
                ("sink_var", ("i3", "i6"), frozenset([("i6", "w")]), "w"),
            }
        },
        {
            "flow": 0.125,
            "expected": {
                "nodes": {
                    "i1": "u",
                    "i2": "v",
                    "i3": "u",
                    "i4": "w",
                    "i5": "v",
                    "i6": "u",
                },
                "edges": {
                    ("i1", "i2"): [("u", "v")],
                    ("i1", "i3"): [],
                    ("i1", "i4"): [("u", "w")],
                    ("i1", "i5"): [("u", "v")],
                    ("i1", "i6"): [],
                    ("i2", "i4"): [("v", "w")],
                    ("i2", "i6"): [("v", "u")],
                    ("i3", "i5"): [("u", "v")],
                    ("i3", "i6"): [],
                },
            },
            "variables": {
                ("node_mapping_var", "i1", frozenset([("i4", "w"), ("i5", "v"), ("i6", "u")]), "u"),
                ("node_mapping_var", "i2", frozenset([("i4", "w")]), "v"),
                ("node_mapping_var", "i2", frozenset([("i6", "u")]), "v"),
                ("node_mapping_var", "i3", frozenset([("i5", "v")]), "u"),
                ("node_mapping_var", "i3", frozenset([("i6", "u")]), "u"),
                ("node_mapping_var", "i4", frozenset(), "w"),
                ("node_mapping_var", "i5", frozenset(), "v"),
                ("node_mapping_var", "i6", frozenset(), "u"),

                ("source_var", ("i1", "i2"), frozenset([("i4", "w"), ("i6", "u")]), "u"),
                ("edge_var", ("i1", "i2"), frozenset([("i4", "w"), ("i6", "u")]), ("u", "v")),
                ("sink_var", ("i1", "i2"), frozenset([("i4", "w"), ("i6", "u")]), "v"),

                ("source_var", ("i1", "i3"), frozenset([("i5", "v"), ("i6", "u")]), "u"),
                ("sink_var", ("i1", "i3"), frozenset([("i5", "v"), ("i6", "u")]), "u"),

                ("source_var", ("i1", "i4"), frozenset([("i4", "w")]), "u"),
                ("edge_var", ("i1", "i4"), frozenset([("i4", "w")]), ("u", "w")),
                ("sink_var", ("i1", "i4"), frozenset([("i4", "w")]), "w"),

                ("source_var", ("i1", "i5"), frozenset([("i5", "v")]), "u"),
                ("edge_var", ("i1", "i5"), frozenset([("i5", "v")]), ("u", "v")),
                ("sink_var", ("i1", "i5"), frozenset([("i5", "v")]), "v"),

                ("source_var", ("i1", "i6"), frozenset([("i6", "u")]), "u"),
                ("sink_var", ("i1", "i6"), frozenset([("i6", "u")]), "u"),

                ("source_var", ("i2", "i4"), frozenset([("i4", "w")]), "v"),
                ("edge_var", ("i2", "i4"), frozenset([("i4", "w")]), ("v", "w")),
                ("sink_var", ("i2", "i4"), frozenset([("i4", "w")]), "w"),

                ("source_var", ("i2", "i6"), frozenset([("i6", "u")]), "v"),
                ("edge_var", ("i2", "i6"), frozenset([("i6", "u")]), ("v", "u")),
                ("sink_var", ("i2", "i6"), frozenset([("i6", "u")]), "u"),

                ("source_var", ("i3", "i5"), frozenset([("i5", "v")]), "u"),
                ("edge_var", ("i3", "i5"), frozenset([("i5", "v")]), ("u", "v")),
                ("sink_var", ("i3", "i5"), frozenset([("i5", "v")]), "v"),

                ("source_var", ("i3", "i6"), frozenset([("i6", "u")]), "u"),
                ("sink_var", ("i3", "i6"), frozenset([("i6", "u")]), "u"),
            }
        },
    ],
}

DRAGON_3 = {
    "tags": {"dragon", "bags", "multiple_labels"},
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
    "labels": {
        "i1": {"i5", "i6"},
        "i2": {"i5", "i6", "i7"},
        "i3": {"i5", "i7"},
        "i4": {"i6", "i7"},
        "i5": {"i5"},
        "i6": {"i6"},
        "i7": {"i7"},
    },
    "bags": {
        "i1": {frozenset(["i5", "i6"]): {("i1", "i2"), ("i1", "i5"), ("i1", "i6")}},
        "i2": {frozenset(["i5", "i6", "i7"]): {("i2", "i3"), ("i2", "i4")}},
        "i3": {frozenset(["i5"]): {("i3", "i5")},
               frozenset(["i7"]): {("i3", "i7")}},
        "i4": {frozenset(["i6"]): {("i4", "i6")},
               frozenset(["i7"]): {("i4", "i7")}},
        "i5": {frozenset(): set()},
        "i6": {frozenset(): set()},
        "i7": {frozenset(): set()},
    },
}

example_requests = OrderedDict([
    ("simple path", SIMPLE_PATH),
    ("simple cycle 1", SIMPLE_CYCLE_1),
    ("simple cycle 2", SIMPLE_CYCLE_2),
    ("cycle with node before start 1", CYCLE_WITH_NODE_BEFORE_START_1),
    ("cycle with node before start 2", CYCLE_WITH_NODE_BEFORE_START_2),
    ("cycle with node after end", CYCLE_WITH_NODE_AFTER_END),
    ("cycle with nodes before and after", CYCLE_WITH_NODES_BEFORE_AND_AFTER),
    ("cycle with branch", CYCLE_WITH_BRANCH),
    ("two cycles connected with single edge", TWO_CYCLES_CONNECTED_WITH_SINGLE_EDGE),
    ("two cycles connected directly", TWO_CYCLES_CONNECTED_DIRECTLY),
    ("three cycles connected directly", THREE_CYCLES_CONNECTED_DIRECTLY),
    ("three cycles chained", THREE_CYCLES_CONNECTED_CHAINED),
    ("cycle on cycle 1", CYCLE_ON_CYCLE_1),
    ("cycle on cycle 2", CYCLE_ON_CYCLE_2),
    ("cycle on cycle 3", CYCLE_ON_CYCLE_3),
    ("cycle on cycle 4", CYCLE_ON_CYCLE_4),
    ("cycles crossing", CYCLES_CROSSING),
    ("three branches", THREE_BRANCHES),
    ("dragon 1", DRAGON_1),
    ("dragon 2", DRAGON_2),
    ("dragon 3", DRAGON_3)
])
