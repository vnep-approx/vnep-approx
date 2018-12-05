import random

import pytest

from alib import datamodel, scenariogeneration, solutions, util
from vnep_approx import vine


def get_single_edge_req():
    req = datamodel.Request("test_req")
    req.add_node("i", 1.0, "t", ["u"])
    req.add_node("j", 1.0, "t", ["v"])
    req.add_edge("i", "j", 1.0)
    req.profit = 1
    return req


def get_single_edge_sub():
    sub = datamodel.Substrate("test_sub")
    sub.add_node("u", ["t"], {"t": 1}, {"t": 1})
    sub.add_node("v", ["t"], {"t": 1}, {"t": 1})
    sub.add_edge("u", "v", bidirected=False, capacity=1)
    return sub


def test_execute_dvine_procedure_single_request():
    req = get_single_edge_req()
    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req],
        substrate=get_single_edge_sub(),
        objective=datamodel.Objective.MAX_PROFIT,
    )

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()
    m = sol.request_mapping[req]

    assert m.mapping_nodes == dict(
        i="u",
        j="v",
    )
    assert m.mapping_edges == {("i", "j"): [("u", "v")]}


def test_only_one_request_is_embedded_due_to_capacity_limitations():
    req1 = get_single_edge_req()
    req2 = get_single_edge_req()
    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req1, req2],
        substrate=get_single_edge_sub(),
        objective=datamodel.Objective.MAX_PROFIT,
    )

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()
    m1 = sol.request_mapping[req1]
    assert m1.mapping_nodes == dict(
        i="u",
        j="v",
    )
    assert m1.mapping_edges == {("i", "j"): [("u", "v")]}
    assert sol.request_mapping[req2] is None


def test_only_requests_are_processed_in_profit_order():
    req1 = get_single_edge_req()
    req2 = get_single_edge_req()
    req2.profit = req1.profit + 1

    scenario = datamodel.Scenario(
        name="test_scen",
        requests=[req1, req2],
        substrate=get_single_edge_sub(),
        objective=datamodel.Objective.MAX_PROFIT,
    )

    v = vine.WiNESingleWindow(scenario)
    sol = v.compute_integral_solution()
    m = sol.request_mapping[req2]
    assert m.mapping_nodes == dict(
        i="u",
        j="v",
    )
    assert m.mapping_edges == {("i", "j"): [("u", "v")]}
    assert sol.request_mapping[req1] is None
