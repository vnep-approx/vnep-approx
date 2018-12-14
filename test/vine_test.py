import pytest
import vine_test_data as vtd
from vnep_approx import vine

from alib import datamodel, solutions


@pytest.mark.parametrize("use_load_balancing", [True, False])
@pytest.mark.parametrize("use_costs_with_lb_objective", [True, False])
@pytest.mark.parametrize("test_case", vtd.SHORTEST_PATH_TEST_CASES)
def test_single_request_embeddings_shortest_path_method(use_load_balancing, use_costs_with_lb_objective, test_case):
    test_data = vtd.SINGLE_REQUEST_EMBEDDING_TEST_CASES[test_case]

    scenario = vtd.get_test_scenario(test_data)

    v = vine.WiNESingleWindow(scenario,
                              edge_mapping_method=vine.EdgeMappingMethod.SHORTEST_PATH,
                              use_costs_with_lb_objective=use_costs_with_lb_objective,
                              use_load_balancing_objective=use_load_balancing)
    sol = v.compute_integral_solution()
    req, m = next(sol.request_mapping.iteritems())  # there should only be one...

    expected_mapping = test_data["expected_integer_solution"]
    if expected_mapping is None:
        assert m is None
    else:
        assert isinstance(m, solutions.Mapping)
        assert m.mapping_nodes == expected_mapping["mapping_nodes"]
        assert m.mapping_edges == expected_mapping["mapping_edges"]


@pytest.mark.parametrize("use_load_balancing", [True, False])
@pytest.mark.parametrize("use_costs_with_lb_objective", [True, False])
@pytest.mark.parametrize("test_case", vtd.SPLITTABLE_TEST_CASES)
def test_single_request_embeddings_splittable_path_method(use_load_balancing, use_costs_with_lb_objective, test_case):
    if (use_load_balancing
            and not use_costs_with_lb_objective
            and test_case in vtd.COST_SPECIFIC_TEST_CASES):
        # Skip tests that assume that costs are relevant
        return

    test_data = vtd.SINGLE_REQUEST_EMBEDDING_TEST_CASES[test_case]

    scenario = vtd.get_test_scenario(test_data)

    v = vine.WiNESingleWindow(scenario,
                              edge_mapping_method=vine.EdgeMappingMethod.SPLITTABLE,
                              use_load_balancing_objective=use_load_balancing,
                              use_costs_with_lb_objective=use_costs_with_lb_objective)
    sol = v.compute_integral_solution()
    req, m = next(sol.request_mapping.iteritems())  # there should only be one...

    expected_mapping = test_data["expected_fractional_solution"]
    if expected_mapping is None:
        assert m is None
    else:
        assert isinstance(m, vine.SplittableMapping)
        assert m.mapping_nodes == expected_mapping["mapping_nodes"]
        assert m.mapping_edges == expected_mapping["mapping_edges"]


@pytest.mark.parametrize("test_case", vtd.SINGLE_REQUEST_REJECT_EMBEDDING_TEST_CASES)
def test_single_request_rejected_embeddings(test_case):
    test_data = vtd.SINGLE_REQUEST_REJECT_EMBEDDING_TEST_CASES[test_case]
    scenario = vtd.get_test_scenario(test_data)

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()

    req, m = next(sol.request_mapping.iteritems())  # there should only be one...

    assert m is None


def test_only_one_request_is_embedded_due_to_capacity_limitations():
    sub = vtd.create_test_substrate("single_edge_substrate")
    req1 = vtd.create_test_request(
        "single_edge_request", sub,
        allowed_nodes=dict(i1=["u1"], i2=["u2"])
    )
    req2 = vtd.create_test_request(
        "single_edge_request", sub,
    )
    req1.name = "req1"
    req2.name = "req2"
    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req1, req2],
        substrate=sub,
        objective=datamodel.Objective.MAX_PROFIT,
    )

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()
    m1 = sol.request_mapping[req1]
    assert m1.mapping_nodes == dict(
        i1="u1",
        i2="u2",
    )
    assert m1.mapping_edges == {("i1", "i2"): [("u1", "u2")]}
    assert sol.request_mapping[req2] is None


def test_requests_are_processed_in_profit_order():
    sub = vtd.create_test_substrate("single_edge_substrate")
    req1 = vtd.create_test_request(
        "single_edge_request", sub,
        allowed_nodes=dict(i1=["u1"], i2=["u2"])
    )
    req2 = vtd.create_test_request(
        "single_edge_request", sub,
        allowed_nodes=dict(i1=["u1"], i2=["u2"])
    )
    req2.profit = req1.profit + 1

    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req1, req2],
        substrate=sub,
        objective=datamodel.Objective.MAX_PROFIT,
    )

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()
    m = sol.request_mapping[req2]

    # Although req2 is second in the request list, it must be processed first due to its higher profit.
    assert m.mapping_nodes == dict(
        i1="u1",
        i2="u2",
    )
    assert m.mapping_edges == {("i1", "i2"): [("u1", "u2")]}
    assert sol.request_mapping[req1] is None
