import pytest

from vnep_approx import treewidth_model
from test_data.request_test_data import create_test_request, example_requests

from test_data.tree_decomposition_test_data import PACE_INPUT_FORMAT


@pytest.mark.parametrize("test_data", PACE_INPUT_FORMAT.items())
def test_conversion_to_PACE_format_works(test_data):
    req_id, expected = test_data
    req = create_test_request(req_id)
    td_comp = treewidth_model.TreeDecompositionComputation(req)
    assert td_comp._convert_graph_to_td_input_format() == expected


@pytest.mark.parametrize("req_id", example_requests)
def test_tree_decomposition_computation_returns_valid_tree_decompositions(req_id):
    req = create_test_request(req_id)
    td_comp = treewidth_model.TreeDecompositionComputation(req)
    tree_decomp = td_comp.compute_tree_decomposition()
    assert tree_decomp.is_tree_decomposition(req)
