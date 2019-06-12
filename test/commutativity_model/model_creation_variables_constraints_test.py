import gurobipy
import pytest

from alib import datamodel
from commutativity_model_test_data import example_requests, create_request, filter_requests_by_tags

from vnep_approx import commutativity_model

pytestmark = pytest.mark.usefixtures("mock_gurobi")

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


def extract_var_names(var_dict):
    return {key: value.name for (key, value) in var_dict.items()}


def test_create_sub_lp_variables(tiny_substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    scenario = datamodel.Scenario("test", tiny_substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][ku_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('i','j')]_sedge[('u','v')]_comm_index[k_u]",
        vu: "edge_flow_req[test_req]_vedge[('i','j')]_sedge[('v','u')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][kv_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('i','j')]_sedge[('u','v')]_comm_index[k_v]",
        vu: "edge_flow_req[test_req]_vedge[('i','j')]_sedge[('v','u')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][ku_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('j','k')]_sedge[('u','v')]_comm_index[k_u]",
        vu: "edge_flow_req[test_req]_vedge[('j','k')]_sedge[('v','u')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][kv_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('j','k')]_sedge[('u','v')]_comm_index[k_v]",
        vu: "edge_flow_req[test_req]_vedge[('j','k')]_sedge[('v','u')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][ku_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('i','k')]_sedge[('u','v')]_comm_index[k_u]",
        vu: "edge_flow_req[test_req]_vedge[('i','k')]_sedge[('v','u')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][kv_index].var_edge_flow) == {
        uv: "edge_flow_req[test_req]_vedge[('i','k')]_sedge[('u','v')]_comm_index[k_v]",
        vu: "edge_flow_req[test_req]_vedge[('i','k')]_sedge[('v','u')]_comm_index[k_v]",
    }

    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][ku_index].var_node_flow_source) == {
        u: "node_flow_source_req[test_req]_snode[u]_vedge[('i','j')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][kv_index].var_node_flow_source) == {
        u: "node_flow_source_req[test_req]_snode[u]_vedge[('i','j')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][ku_index].var_node_flow_source) == {
        v: "node_flow_source_req[test_req]_snode[v]_vedge[('j','k')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][kv_index].var_node_flow_source) == {
        v: "node_flow_source_req[test_req]_snode[v]_vedge[('j','k')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][ku_index].var_node_flow_source) == {
        u: "node_flow_source_req[test_req]_snode[u]_vedge[('i','k')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][kv_index].var_node_flow_source) == {
        u: "node_flow_source_req[test_req]_snode[u]_vedge[('i','k')]_comm_index[k_v]",
    }

    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][ku_index].var_node_flow_sink) == {
        v: "node_flow_sink_req[test_req]_snode[v]_vedge[('i','j')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ij][kv_index].var_node_flow_sink) == {
        v: "node_flow_sink_req[test_req]_snode[v]_vedge[('i','j')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][ku_index].var_node_flow_sink) == {
        u: "node_flow_sink_req[test_req]_snode[u]_vedge[('j','k')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][jk][kv_index].var_node_flow_sink) == {
        v: "node_flow_sink_req[test_req]_snode[v]_vedge[('j','k')]_comm_index[k_v]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][ku_index].var_node_flow_sink) == {
        u: "node_flow_sink_req[test_req]_snode[u]_vedge[('i','k')]_comm_index[k_u]",
    }
    assert extract_var_names(mc.edge_sub_lp[triangle_request][ik][kv_index].var_node_flow_sink) == {
        v: "node_flow_sink_req[test_req]_snode[v]_vedge[('i','k')]_comm_index[k_v]",
    }


@pytest.mark.parametrize("request_id", filter_requests_by_tags("flow_preservation_constraints"))
def test_create_sub_lp_flow_preservation_constraints(request_id, tiny_substrate, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr

    req = create_request(request_id)
    req.graph["root"] = example_requests[request_id]["assumed_root"]
    scenario = datamodel.Scenario("test", tiny_substrate, [req])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    mc.preprocess_input()
    mc._create_sub_lp_variables()
    mc._create_sub_lp_constraints()

    def edge_var(ij, index, uv):
        return mc.edge_sub_lp[req][ij][index].var_edge_flow[uv]

    def source_var(ij, index, u):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_source[u]

    def sink_var(ij, index, v):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_sink[v]

    var_type_lookup = dict(edge_var=edge_var, source_var=source_var, sink_var=sink_var)

    expected = set()
    for constraint_data in example_requests[request_id]["constraints"]["flow_preservation"]:
        name, expr_data = constraint_data
        name = name.format(req_name=req.name)
        expr = []
        for coeff, var_type, var_keys in expr_data:
            expr.append((coeff, var_type_lookup[var_type](*var_keys)))
        expected.add(MockConstr(MockLinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name))
    assert set(mc.model.constrs) == expected


def test_create_node_mapping_variables(tiny_substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    scenario = datamodel.Scenario("test", tiny_substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    mc.preprocess_input()
    mc._create_node_mapping_variables()

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    assert extract_var_names(mc.var_node_mapping[triangle_request][i][u]) == {
        ku_index: "node_mapping_req[test_req]_vnode[i]_snode[u]_comm_index[k_u]",
        kv_index: "node_mapping_req[test_req]_vnode[i]_snode[u]_comm_index[k_v]",
    }
    assert extract_var_names(mc.var_node_mapping[triangle_request][j][v]) == {
        ku_index: "node_mapping_req[test_req]_vnode[j]_snode[v]_comm_index[k_u]",
        kv_index: "node_mapping_req[test_req]_vnode[j]_snode[v]_comm_index[k_v]",
    }
    assert extract_var_names(mc.var_node_mapping[triangle_request][k][u]) == {
        frozenset(): "node_mapping_req[test_req]_vnode[k]_snode[u]_comm_index[]",
    }
    assert extract_var_names(mc.var_node_mapping[triangle_request][k][v]) == {
        frozenset(): "node_mapping_req[test_req]_vnode[k]_snode[v]_comm_index[]",
    }


@pytest.mark.parametrize("request_id", filter_requests_by_tags("node_mapping_constraints"))
def test_create_node_mapping_constraints(request_id, substrate, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr

    req = create_request(request_id)
    req.graph["root"] = example_requests[request_id]["assumed_root"]
    scenario = datamodel.Scenario("test", substrate, [req])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    if example_requests[request_id].get("ignore_bfs", False):
        mc.dag_requests[req] = req

    mc.preprocess_input()
    mc._create_sub_lp_variables()
    mc._create_node_mapping_variables()
    mc._create_node_mapping_constraints()

    def node_agg_var(i, u):
        return mc.var_aggregated_node_mapping[req][i][u]

    def node_mapping_var(i, index, u):
        return mc.var_node_mapping[req][i][u][index]

    def source_var(ij, index, u):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_source[u]

    def sink_var(ij, index, v):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_sink[v]

    var_type_lookup = dict(node_agg_var=node_agg_var,
                           node_mapping_var=node_mapping_var,
                           source_var=source_var,
                           sink_var=sink_var)

    expected = set()
    for constraint_data in example_requests[request_id]["constraints"]["node_mapping"]:
        name, expr_data = constraint_data
        name = name.format(req_name=req.name)
        expr = []
        for coeff, var_type, var_keys in expr_data:
            expr.append((coeff, var_type_lookup[var_type](*var_keys)))
        expected.add(MockConstr(MockLinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name))

    assert set(mc.model.constrs) == expected


def test_force_embedding_constraint(tiny_substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr

    scenario = datamodel.Scenario("test", tiny_substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_force_embedding_constraint()

    assert set(mc.model.constrs) == {
        MockConstr(MockLinExpr([
            (-1.0, mc.var_embedding_decision[triangle_request]),
            (1.0, mc.var_aggregated_node_mapping[triangle_request][i][u]),
        ]), gurobipy.GRB.EQUAL, 0.0, "force_node_mapping_for_embedded_request_req[test_req]"),
    }


def test_force_embedding_constraint_multiple_root_mappings(tiny_substrate, triangle_request, monkeypatch, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr

    # Extend the root node's allowed nodes
    triangle_request.node[i]["allowed_nodes"] = set("uv")
    scenario = datamodel.Scenario("test", tiny_substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")

    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_force_embedding_constraint()

    assert set(mc.model.constrs) == {
        MockConstr(MockLinExpr([
            (-1.0, mc.var_embedding_decision[triangle_request]),
            (1.0, mc.var_aggregated_node_mapping[triangle_request][i][u]),
            (1.0, mc.var_aggregated_node_mapping[triangle_request][i][v]),
        ]), gurobipy.GRB.EQUAL, 0.0, "force_node_mapping_for_embedded_request_req[test_req]"),
    }
