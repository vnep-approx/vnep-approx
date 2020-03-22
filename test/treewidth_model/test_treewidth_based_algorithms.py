import pytest
from vnep_approx import treewidth_model

from alib import datamodel
from test_data.request_test_data import create_test_request, example_requests, create_random_test_request
from test_data.substrate_test_data import create_test_substrate_topology_zoo

import random

random.seed(0)


################################ Dyn-VMP

@pytest.mark.parametrize("request_id",
                         example_requests)
def test_dyn_vmp(request_id):
    req = create_test_request(request_id)
    sub = get_test_substrate()
    alg = treewidth_model.DynVMPAlgorithm(req, sub)

    alg.preprocess_input()
    m = alg.compute_valid_mapping()
    assert set(m.mapping_nodes.keys()) == set(req.nodes)
    assert set(m.mapping_edges.keys()) == set(req.edges)


@pytest.mark.parametrize("request_id",
                         example_requests)
def test_dyn_vmp_with_allowed_nodes_restrictions(request_id):
    req = create_test_request(request_id)

    sub = get_test_substrate()

    # add some restrictions
    for i in req.nodes:
        num_allowed = random.randint(1, 2)
        req.node[i]["allowed_nodes"] = random.sample(list(sub.nodes), num_allowed)

    alg = treewidth_model.DynVMPAlgorithm(req, sub)

    alg.preprocess_input()
    m = alg.compute_valid_mapping()
    assert set(m.mapping_nodes.keys()) == set(req.nodes)
    assert set(m.mapping_edges.keys()) == set(req.edges)


def test_dyn_vmp_with_larger_scenario():
    sub = create_test_substrate_topology_zoo()

    req = create_random_test_request(sub,
                                     min_number_nodes=8,
                                     max_number_nodes=12,
                                     probability=0.25,
                                     edge_resource_factor=10,
                                     node_resource_factor=0.1)
    # add some restrictions
    for i in req.nodes:
        num_allowed = random.randint(1, 2)
        req.node[i]["allowed_nodes"] = random.sample(list(sub.nodes), num_allowed)

    alg = treewidth_model.DynVMPAlgorithm(req, sub)

    alg.preprocess_input()
    m = alg.compute_valid_mapping()
    assert set(m.mapping_nodes.keys()) == set(req.nodes)
    assert set(m.mapping_edges.keys()) == set(req.edges)


################################ "Classic" LP formulation


@pytest.mark.parametrize("request_id", example_requests)
def test_tw_formulation_precomputed_scenario(request_id):
    # build test scenario
    req = create_test_request(request_id)
    sub = create_test_substrate_topology_zoo()
    scenario = datamodel.Scenario(name="test_scen", substrate=sub, requests=[req])

    tw_mc = treewidth_model._TreewidthModelCreator(scenario)
    tw_mc.init_model_creator()


def test_mapping_space_works():
    sub = get_test_substrate()
    req = create_test_request("simple path")
    req.node["i1"]["allowed_nodes"] = ["u", "v"]
    req.node["i2"]["allowed_nodes"] = ["w"]
    req.node["i3"]["allowed_nodes"] = None

    assert set(treewidth_model.mapping_space(req, sub, ["i1", "i2", "i3"])) == {
        ("u", "w", "u"),
        ("u", "w", "v"),
        ("u", "w", "w"),
        ("v", "w", "u"),
        ("v", "w", "v"),
        ("v", "w", "w"),
    }


def get_test_substrate(types=None):
    sub = datamodel.Substrate("foo")
    if types is None:
        types = ["test_type"]
    sub.add_node(
        "u",
        types=types,
        capacity={t: 2.0 for t in types},
        cost={t: 5 * random.random() for t in types},
    )
    sub.add_node(
        "v",
        types=types,
        capacity={t: 2.0 for t in types},
        cost={t: 5 * random.random() for t in types},
    )
    sub.add_node(
        "w",
        types=types,
        capacity={t: 2.0 for t in types},
        cost={t: 5 * random.random() for t in types},
    )
    sub.add_edge(
        "u", "v", capacity=2.0, cost=random.random()
    )
    sub.add_edge(
        "v", "w", capacity=2.0, cost=random.random()
    )
    sub.add_edge(
        "w", "u", capacity=2.0, cost=random.random()
    )
    return sub
