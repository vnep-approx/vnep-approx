import copy
import random

import pytest

from alib import datamodel, modelcreator
from commutativity_model_test_data import create_request, filter_requests_by_tags
from vnep_approx import commutativity_model

i, j, k = "ijk"
ij = i, j
jk = j, k
ik = i, k
u, v, w = "uvw"
uv = u, v
vu = v, u
uw = u, w
wu = w, u
vw = v, w
wv = w, v


def generate_random_request_graph(size, prob, req_name="test_req", substrate_nodes=(u, v, w)):
    req = datamodel.Request(req_name)
    for i in range(1, size + 1):
        node = "{}".format(i)
        neighbors = []
        for j in req.nodes:
            if random.random() <= prob:
                neighbors.append(j)
        req.add_node(node, 1.0, "t1", random.sample(substrate_nodes, random.randrange(1, len(substrate_nodes))))
        for j in neighbors:
            req.add_edge(j, node, 1.0)

    visited = set()
    all_nodes = set(req.nodes)
    while visited != all_nodes:
        start_node = (all_nodes - visited).pop()
        if visited:
            req.add_edge(random.choice(list(visited)), start_node, 1.0)
        visited.add(start_node)
        stack = [start_node]
        while stack:
            i = stack.pop()
            neighbors = req.get_in_neighbors(i) + req.get_out_neighbors(i)
            for j in neighbors:
                if j in visited:
                    continue
                else:
                    stack.append(j)
                    visited.add(j)
    req.graph["root"] = random.choice(list(req.nodes))
    return req


def test_commutativity_with_gurobi(triangle_request, substrate):
    triangle_request.profit = 10000
    triangle_request.add_node("l", 0.75, "t1", [v])
    triangle_request.add_edge("l", i, 0.5, None)

    scenario = datamodel.Scenario("test", substrate, [triangle_request], objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()
    assert solution is not None


@pytest.mark.parametrize("request_id", filter_requests_by_tags())
def test_solve_single_request_scenarios_with_gurobi(request_id, substrate):
    request = create_request(request_id)
    request.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, [request], objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    assert len(solution.request_mapping[request]) > 0


def test_solve_all_request_scenarios_with_gurobi(substrate):
    requests = [create_request(request_id) for request_id in filter_requests_by_tags()]
    for req in requests:
        req.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    for req in requests:
        assert len(solution.request_mapping[req]) > 0


@pytest.mark.parametrize("seed", range(10))
def test_integral_solution_random_request_with_gurobi(seed, substrate):
    random.seed(seed)
    req = generate_random_request_graph(10, 0.25)
    requests = [req]
    req.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.init_model_creator()
    solution = mc.compute_integral_solution()

    assert solution is not None
    assert req in solution.request_mapping


@pytest.mark.parametrize("seed", range(10))
def test_integral_solution_multiple_random_requests_with_gurobi(seed, substrate):
    random.seed(seed)
    requests = [generate_random_request_graph(7, 0.25, "test_req_{}".format(i))
                for i in range(1, 11)]
    for req in requests:
        req.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(
        scenario, gurobi_settings=modelcreator.GurobiSettings(threads=2, timelimit=60)
    )
    mc.init_model_creator()
    solution = mc.compute_integral_solution()

    assert solution is not None
    for req in requests:
        assert req in solution.request_mapping


@pytest.mark.parametrize("seed", range(10))
def test_fractional_solution_random_request_with_gurobi(seed, substrate):
    random.seed(seed)
    req = generate_random_request_graph(10, 0.25)
    requests = [req]
    req.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    assert len(solution.request_mapping[req]) > 0


@pytest.mark.parametrize("seed", range(10))
def test_fractional_solution_multiple_random_requests_with_gurobi(seed, substrate):
    random.seed(seed)
    requests = [generate_random_request_graph(7, 0.25, "test_req_{}".format(i))
                for i in range(1, 11)]
    for req in requests:
        req.profit = 1.0

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(
        scenario, gurobi_settings=modelcreator.GurobiSettings(threads=2, timelimit=60)
    )
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    for req in requests:
        assert len(solution.request_mapping[req]) > 0


def test_gurobi_with_insufficient_node_resources_invalid_substrate_node(substrate, triangle_request):
    # triangle_request_copy will be embeddable, while triangle_request will have no valid mapping for node i
    triangle_request_copy = copy.deepcopy(triangle_request)
    triangle_request_copy.name = "test_req_copy"
    requests = [triangle_request, triangle_request_copy]
    for req in requests:
        req.profit = 1.0

    triangle_request.node[i]["demand"] = 1.0
    triangle_request_copy.node[i]["allowed_nodes"] = {"v"}
    substrate.node[u]["capacity"]["t1"] = 0.5

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(
        scenario, gurobi_settings=modelcreator.GurobiSettings(threads=2, timelimit=60)
    )
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    assert triangle_request not in solution.request_mapping
    assert len(solution.request_mapping[triangle_request_copy]) > 0


def test_gurobi_with_insufficient_node_resources_cannot_embed_all(substrate, triangle_request):
    triangle_request_copy = copy.deepcopy(triangle_request)
    triangle_request_copy.name = "test_req_copy"
    requests = [triangle_request, triangle_request_copy]
    for req in requests:
        req.profit = 1.0

    triangle_request.node[i]["demand"] = 1.0
    triangle_request_copy.node[i]["demand"] = 1.0
    triangle_request.node[k]["allowed_nodes"] = {"w"}
    triangle_request_copy.node[k]["allowed_nodes"] = {"w"}
    substrate.node[u]["capacity"]["t1"] = 1.5

    scenario = datamodel.Scenario("test", substrate, requests, objective=datamodel.Objective.MAX_PROFIT)
    mc = commutativity_model.CommutativityModelCreator(
        scenario, gurobi_settings=modelcreator.GurobiSettings(threads=2, timelimit=60)
    )
    mc.init_model_creator()
    solution = mc.compute_fractional_solution()

    assert solution is not None
    assert len(solution.request_mapping[triangle_request]) > 0
    assert len(solution.request_mapping[triangle_request_copy]) > 0

    flow_sum = sum(solution.mapping_flows[m.name] for req in requests for m in solution.request_mapping[req])
    assert flow_sum == 1.5
