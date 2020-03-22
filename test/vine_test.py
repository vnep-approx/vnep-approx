import copy
import pytest
import warnings

import vine_test_data as vtd
from vnep_approx import vine

from alib import datamodel, solutions



@pytest.mark.parametrize("lp_objective", list(vine.ViNELPObjective))
@pytest.mark.parametrize("test_case", vtd.SHORTEST_PATH_TEST_CASES)
def test_single_request_embeddings_shortest_path_method(lp_objective, test_case):
    if lp_objective == vine.ViNELPObjective.ViNE_COSTS_DEF or lp_objective == vine.ViNELPObjective.ViNE_LB_DEF:
        warnings.warn("The objectives VINE_COSTS_DEF and VINE_LB_DEF are inherently incompatible with this test, as specific Scenario costs are set!\n" 
                      "Accordingly, this test is not executed!")
        return

    test_data = vtd.SINGLE_REQUEST_EMBEDDING_TEST_CASES[test_case]

    scenario = vtd.get_test_scenario(test_data)


    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.UNSPLITTABLE,
                                  lp_objective=lp_objective,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC)
    result = v.compute_integral_solution()
    solution = result.get_solution()
    req, m = next(solution.request_mapping.iteritems())  # there should only be one...

    expected_mapping = test_data["expected_integer_solution"]
    if expected_mapping is None:
        assert m is None
    else:
        assert isinstance(m, solutions.Mapping)
        assert m.mapping_nodes == expected_mapping["mapping_nodes"]
        assert m.mapping_edges == expected_mapping["mapping_edges"]


@pytest.mark.parametrize("lp_objective", list(vine.ViNELPObjective))
@pytest.mark.parametrize("test_case", vtd.SPLITTABLE_TEST_CASES)
def test_single_request_embeddings_splittable_path_method(lp_objective, test_case):
    if (lp_objective in [vine.ViNELPObjective.ViNE_LB_DEF, vine.ViNELPObjective.ViNE_COSTS_DEF]
            and test_case in vtd.COST_SPECIFIC_TEST_CASES):
        # Skip tests that require costs to enforce a unique mapping
        return

    test_data = vtd.SINGLE_REQUEST_EMBEDDING_TEST_CASES[test_case]

    scenario = vtd.get_test_scenario(test_data)

    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.SPLITTABLE,
                                  lp_objective=lp_objective,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC
                                  )
    result = v.compute_integral_solution()
    solution = result.get_solution()
    req, m = next(solution.request_mapping.iteritems())  # there should only be one...

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

    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.UNSPLITTABLE,
                                  lp_objective=vine.ViNELPObjective.ViNE_COSTS_DEF,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC)

    result = v.compute_integral_solution()
    solution = result.get_solution()
    req, m = next(solution.request_mapping.iteritems())  # there should only be one...

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

    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.UNSPLITTABLE,
                                  lp_objective=vine.ViNELPObjective.ViNE_COSTS_DEF,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC
                                  )
    result = v.compute_integral_solution()
    solution = result.get_solution()
    m1 = solution.request_mapping[req1]
    assert m1.mapping_nodes == dict(
        i1="u1",
        i2="u2",
    )
    assert m1.mapping_edges == {("i1", "i2"): [("u1", "u2")]}
    assert solution.request_mapping[req2] is None


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

    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.UNSPLITTABLE,
                                  lp_objective=vine.ViNELPObjective.ViNE_COSTS_DEF,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC
                                  )
    result = v.compute_integral_solution()
    solution = result.get_solution()
    m = solution.request_mapping[req2]

    # Although req2 is second in the request list, it must be processed first due to its higher profit.
    assert m.mapping_nodes == dict(
        i1="u1",
        i2="u2",
    )
    assert m.mapping_edges == {("i1", "i2"): [("u1", "u2")]}
    assert solution.request_mapping[req1] is None


def test_cleanup_references():
    sub = vtd.create_test_substrate("single_edge_substrate")
    req1 = vtd.create_test_request("single_edge_request", sub)
    req2 = vtd.create_test_request("single_edge_request", sub)
    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req1, req2],
        substrate=sub,
        objective=datamodel.Objective.MAX_PROFIT,
    )
    v = vine.OfflineViNEAlgorithm(scenario,
                                  edge_embedding_model=vine.ViNEEdgeEmbeddingModel.UNSPLITTABLE,
                                  lp_objective=vine.ViNELPObjective.ViNE_COSTS_DEF,
                                  rounding_procedure=vine.ViNERoundingProcedure.DETERMINISTIC
                                  )
    result = v.compute_integral_solution()

    scenario_copy = copy.deepcopy(scenario)

    assert result.get_solution().scenario is not scenario_copy
    for req in result.get_solution().request_mapping:
        assert req not in scenario_copy.requests
    for req in result.runtime_per_request:
        assert req not in scenario_copy.requests
    for req in result.mapping_status_per_request:
        assert req not in scenario_copy.requests

    result.cleanup_references(scenario_copy)

    solution = result.get_solution()
    assert solution.scenario is scenario_copy
    for req in solution.request_mapping:
        assert req in scenario_copy.requests
    for req in result.runtime_per_request:
        assert req in scenario_copy.requests
    for req in result.mapping_status_per_request:
        assert req in scenario_copy.requests
