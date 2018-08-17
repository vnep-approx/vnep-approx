import pytest
from vnep_approx import treewidth_model

from alib import datamodel
from test_data.request_test_data import create_test_request
from test_data.substrate_test_data import create_test_substrate
from test_data.tree_decomposition_test_data import (
    create_test_tree_decomposition,
    VALID_TREE_DECOMPOSITIONS,
)


@pytest.mark.parametrize("td_dict", VALID_TREE_DECOMPOSITIONS)
def test_tw_formulation_precomputed_scenario(td_dict):
    # build test scenario
    req = create_test_request(td_dict["request_id"])
    sub = create_test_substrate()
    td = create_test_tree_decomposition(td_dict)
    scenario = datamodel.Scenario(name="test_scen", substrate=sub, requests=[req])

    tw_mc = treewidth_model.TreewidthModelCreator(scenario, precomputed_tree_decompositions={req.name: td})
    tw_mc.init_model_creator()


def test_mapping_space_works():
    sub = create_test_substrate()
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
