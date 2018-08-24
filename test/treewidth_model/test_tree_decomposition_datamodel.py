import pytest
from test_data.request_test_data import create_test_request, example_requests
from test_data.tree_decomposition_test_data import (
    create_test_tree_decomposition,
    VALID_TREE_DECOMPOSITIONS,
    INVALID_TREE_DECOMPOSITION_INTERSECTION_PROPERTY,
    INVALID_TREE_DECOMPOSITIONS_NOT_A_TREE,
    INVALID_TREE_DECOMPOSITIONS,
    POST_ORDER_TRAVERSALS,
    CHECK_COMPATIBLE_MAPPINGS_VALID_EXAMPLES,
    CHECK_COMPATIBLE_MAPPINGS_INVALID_EXAMPLES,
    NICE_TREE_DECOMPOSITIONS,
)
from vnep_approx import treewidth_model


@pytest.mark.parametrize("request_id",
                         example_requests)
def test_trivial_decomposition_is_always_valid(request_id):
    req = create_test_request(request_id)
    tree_decomp = treewidth_model.TreeDecomposition("test")
    tree_decomp.add_node("bag_1", frozenset(req.nodes))
    assert tree_decomp.is_tree_decomposition(req)


@pytest.mark.parametrize("request_id",
                         VALID_TREE_DECOMPOSITIONS)
def test_hardcoded_decompositions_are_valid(request_id):
    req = create_test_request(request_id=request_id)
    tree_decomp = create_test_tree_decomposition(VALID_TREE_DECOMPOSITIONS[request_id])
    assert tree_decomp.is_tree_decomposition(req)


@pytest.mark.parametrize("request_id",
                         NICE_TREE_DECOMPOSITIONS)
def test_nice_tree_decompositions_are_recognized(request_id):
    req = create_test_request(request_id=request_id)
    tree_decomp = create_test_tree_decomposition(NICE_TREE_DECOMPOSITIONS[request_id])
    arborescence = tree_decomp.convert_to_arborescence(NICE_TREE_DECOMPOSITIONS[request_id]["root"])

    assert tree_decomp.is_tree_decomposition(req)
    assert treewidth_model.is_nice_tree_decomposition(tree_decomp, arborescence)


@pytest.mark.parametrize("request_id",
                         ["simple path"])
def test_nice_tree_decomposition_conversion(request_id):
    req = create_test_request(request_id=request_id)

    initial_td = treewidth_model.compute_tree_decomposition(req)
    nice_td_generator = treewidth_model.NiceTDConversion(req, initial_td=initial_td)
    root = sorted(initial_td.nodes)[0]
    nice_td_generator.initialize(root=root)
    nice_td = nice_td_generator.make_nice_tree_decomposition()
    assert nice_td.is_tree_decomposition(req)
    assert treewidth_model.is_nice_tree_decomposition(nice_td, nice_td.convert_to_arborescence(root))


@pytest.mark.parametrize("tree_decomposition_dict",
                         INVALID_TREE_DECOMPOSITION_INTERSECTION_PROPERTY)
def test_invalid_decompositions_intersection_property(tree_decomposition_dict):
    tree_decomp = create_test_tree_decomposition(tree_decomposition_dict)
    assert not tree_decomp._verify_intersection_property()


@pytest.mark.parametrize("tree_decomposition_dict",
                         INVALID_TREE_DECOMPOSITIONS_NOT_A_TREE)
def test_invalid_decompositions_not_a_tree(tree_decomposition_dict):
    tree_decomp = create_test_tree_decomposition(tree_decomposition_dict)
    assert not tree_decomp._is_tree()


@pytest.mark.parametrize("request_id",
                         VALID_TREE_DECOMPOSITIONS)
def test_hardcoded_decompositions_can_convert_to_arborescence(request_id):
    tree_decomp = create_test_tree_decomposition(VALID_TREE_DECOMPOSITIONS[request_id])
    arborescence = tree_decomp.convert_to_arborescence()

    # check node & edge sets
    assert arborescence.root in arborescence.nodes
    assert arborescence.nodes == tree_decomp.nodes
    for edge in arborescence.edges:
        assert frozenset(edge) in tree_decomp.edges
    for edge in tree_decomp.edges:
        t1, t2 = edge
        assert (t1, t2) in arborescence.edges or (t2, t1) in arborescence.edges

    # verify reachability from root
    q = {arborescence.root}
    visited = set()
    while q:
        t1 = q.pop()
        visited.add(t1)
        neighbors = arborescence.get_out_neighbors(t1)
        assert len(set(neighbors) & visited) <= 1  # Should be tree-like: at most one ancestor
        for t2 in neighbors:
            if t2 in visited:
                continue
            assert t2 not in q
            q.add(t2)
    return visited == set(arborescence.nodes)


@pytest.mark.parametrize("tree_decomposition_dict",
                         INVALID_TREE_DECOMPOSITIONS)
def test_converting_invalid_decomposition_to_arborescence_raises_valueerror(tree_decomposition_dict):
    tree_decomp = create_test_tree_decomposition(tree_decomposition_dict)
    with pytest.raises(ValueError):
        tree_decomp.convert_to_arborescence()


@pytest.mark.parametrize("traversal",
                         POST_ORDER_TRAVERSALS)
def test_post_order_traversal(traversal):
    tree_decomp = create_test_tree_decomposition(VALID_TREE_DECOMPOSITIONS[traversal["request_id"]])
    arborescence = tree_decomp.convert_to_arborescence(root=traversal["root"])

    assert list(arborescence.post_order_traversal()) == traversal["order"]


@pytest.mark.parametrize("mapping_pair_tuple",
                         CHECK_COMPATIBLE_MAPPINGS_VALID_EXAMPLES)
def test_check_compatible_mappings_valid_examples(mapping_pair_tuple):
    assert treewidth_model.check_if_mappings_are_compatible(*mapping_pair_tuple)


@pytest.mark.parametrize("mapping_pair_tuple",
                         CHECK_COMPATIBLE_MAPPINGS_INVALID_EXAMPLES)
def test_check_compatible_mappings_invalid_examples(mapping_pair_tuple):
    assert not treewidth_model.check_if_mappings_are_compatible(*mapping_pair_tuple)
