from collections import namedtuple

import pytest
from gurobipy import GRB

from alib import datamodel, solutions
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


@pytest.mark.parametrize("request_id", filter_requests_by_tags())
def test_find_node_labels(request_id):
    expected = example_requests[request_id]["labels"]
    request = create_request(request_id)
    assert commutativity_model.CommutativityLabels.create_labels(request).node_labels == expected


@pytest.mark.parametrize("request_id", filter_requests_by_tags("bags"))
def test_calculate_edge_label_bags(request_id):
    request = create_request(request_id)
    labels = commutativity_model.CommutativityLabels.create_labels(request)
    expected = example_requests[request_id]["bags"]
    obtained = {i: labels.calculate_edge_label_bags(i) for i in request.nodes}
    assert obtained == expected


def test_find_node_labels_errors_two_root_nodes():
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u, v])
    req.add_node(j, 1.0, "t1", [v, w])
    req.add_node(k, 1.0, "t1", [u, w])
    req.add_node("l", 1.0, "t1", [u, w])

    req.graph["root"] = i
    req.add_edge(i, j, 1.0)
    req.add_edge(i, "l", 1.0)
    req.add_edge(k, j, 1.0)
    req.add_edge(k, "l", 1.0)

    with pytest.raises(commutativity_model.CommutativityModelError) as e:
        commutativity_model.CommutativityLabels.create_labels(req)
    assert str(e.value) == "Node k has no parents, but is not root (root is i). May have multiple components or not be rooted"


def test_find_node_labels_errors_root_with_inneighbors():
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u, v])
    req.add_node(j, 1.0, "t1", [v, w])
    req.add_node(k, 1.0, "t1", [u, w])
    req.add_node("l", 1.0, "t1", [u, w])

    req.graph["root"] = i
    req.add_edge(i, j, 1.0)
    req.add_edge("l", i, 1.0)
    req.add_edge(i, k, 1.0)
    req.add_edge(k, j, 1.0)

    with pytest.raises(commutativity_model.CommutativityModelError) as e:
        commutativity_model.CommutativityLabels.create_labels(req)
    assert str(e.value) == "Root node i has in-neighbors ['l']"


def test_find_node_labels_errors_directed_cycle():
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u, v])
    req.add_node(j, 1.0, "t1", [v, w])
    req.add_node(k, 1.0, "t1", [u, w])

    req.graph["root"] = i
    req.add_edge(i, j, 1.0)
    req.add_edge(j, k, 1.0)
    req.add_edge(k, i, 1.0)

    with pytest.raises(commutativity_model.CommutativityModelError) as e:
        commutativity_model.CommutativityLabels.create_labels(req)
    assert str(e.value) == "Request is no DAG: cycle contains i, k"


def test_initialize_dag_request_error_with_multiple_components():
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u, v])
    req.add_node(j, 1.0, "t1", [v, w])
    req.add_node(k, 1.0, "t1", [u, w])
    req.add_node("l", 1.0, "t1", [u, w])

    req.graph["root"] = i
    req.add_edge(i, j, 1.0)
    req.add_edge(k, "l", 1.0)

    with pytest.raises(commutativity_model.CommutativityModelError) as e:
        commutativity_model.CommutativityModelCreator._initialize_dag_request(req)
    assert str(e.value).startswith("Request graph may have multiple components: Nodes set(['k', 'l']) were not visited by bfs.")


def test_get_valid_substrate_nodes(substrate):
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u, v])
    req.add_node(j, 1.0, "t1", [v, w])
    req.add_node(k, 1.0, "t1", [u, w])

    _get_valid_substrate_nodes = commutativity_model.CommutativityModelCreator._get_valid_substrate_nodes.__func__

    class MockModelCreator(object):
        pass

    mmc = MockModelCreator()
    mmc.substrate = datamodel.SubstrateX(substrate)

    assert (
        _get_valid_substrate_nodes(mmc, req, {i, j, k}) == {
            i: {u, v},
            j: {v, w},
            k: {u, w},
        }
    )


def test_recover_fractional_solution_from_variables(substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    l = "l"
    li = l, i
    t1 = "t1"
    triangle_request.add_node(l, 0.75, t1, [w])
    triangle_request.add_edge(l, i, 2.0, None)

    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_sub_lp_variables()

    load_base = {uv_: 0.0 for uv_ in substrate.edges}
    load_base.update({(t1, u_): 0.0 for u_ in substrate.nodes})

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    li_sub_lp = mc.edge_sub_lp[triangle_request][(i, l)][frozenset()]

    ij_sub_lp_ku = mc.edge_sub_lp[triangle_request][ij][ku_index]
    jk_sub_lp_ku = mc.edge_sub_lp[triangle_request][jk][ku_index]
    ik_sub_lp_ku = mc.edge_sub_lp[triangle_request][ik][ku_index]

    ij_sub_lp_kv = mc.edge_sub_lp[triangle_request][ij][kv_index]
    jk_sub_lp_kv = mc.edge_sub_lp[triangle_request][jk][kv_index]
    ik_sub_lp_kv = mc.edge_sub_lp[triangle_request][ik][kv_index]

    # Mapping 1
    flow_1 = 0.25
    mc.var_embedding_decision[triangle_request].x += flow_1

    mc.var_node_mapping[triangle_request][l][w][frozenset()].x += flow_1
    mc.var_node_mapping[triangle_request][i][u][ku_index].x += flow_1
    mc.var_node_mapping[triangle_request][j][v][ku_index].x += flow_1
    mc.var_node_mapping[triangle_request][k][u][frozenset()].x += flow_1

    li_sub_lp.var_node_flow_source[u].x += flow_1
    li_sub_lp.var_edge_flow[uw].x += flow_1
    li_sub_lp.var_node_flow_sink[w].x += flow_1

    ij_sub_lp_ku.var_node_flow_source[u].x += flow_1
    ij_sub_lp_ku.var_edge_flow[uv].x += flow_1
    ij_sub_lp_ku.var_node_flow_sink[v].x += flow_1

    jk_sub_lp_ku.var_node_flow_source[v].x += flow_1
    jk_sub_lp_ku.var_edge_flow[vw].x += flow_1
    jk_sub_lp_ku.var_edge_flow[wu].x += flow_1
    jk_sub_lp_ku.var_node_flow_sink[u].x += flow_1

    ik_sub_lp_ku.var_node_flow_source[u].x += flow_1
    ik_sub_lp_ku.var_node_flow_sink[u].x += flow_1

    load_1 = load_base.copy()
    load_1.update({
        (t1, u): 1.0 + 1.0,  # i, k
        (t1, v): 1.0,  # j
        (t1, w): 0.75,  # l
        uv: 1.0,  # ij
        wu: 2.0 + 1.0,  # li (reversed), jk
        vw: 1.0,  # jk
    })

    # Mapping 2
    flow_2 = 0.125
    mc.var_embedding_decision[triangle_request].x += flow_2

    mc.var_node_mapping[triangle_request][l][w][frozenset()].x += flow_2
    mc.var_node_mapping[triangle_request][i][u][ku_index].x += flow_2
    mc.var_node_mapping[triangle_request][j][v][ku_index].x += flow_2
    mc.var_node_mapping[triangle_request][k][u][frozenset()].x += flow_2

    li_sub_lp.var_node_flow_source[u].x += flow_2
    li_sub_lp.var_edge_flow[uw].x += flow_2
    li_sub_lp.var_node_flow_sink[w].x += flow_2

    ij_sub_lp_ku.var_node_flow_source[u].x += flow_2
    ij_sub_lp_ku.var_edge_flow[uw].x += flow_2
    ij_sub_lp_ku.var_edge_flow[wv].x += flow_2
    ij_sub_lp_ku.var_node_flow_sink[v].x += flow_2

    jk_sub_lp_ku.var_node_flow_source[v].x += flow_2
    jk_sub_lp_ku.var_edge_flow[vw].x += flow_2
    jk_sub_lp_ku.var_edge_flow[wu].x += flow_2
    jk_sub_lp_ku.var_node_flow_sink[u].x += flow_2

    ik_sub_lp_ku.var_node_flow_source[u].x += flow_2
    ik_sub_lp_ku.var_node_flow_sink[u].x += flow_2

    load_2 = load_base.copy()
    load_2.update({
        (t1, u): 1.0 + 1.0,  # i, k
        (t1, v): 1.0,  # j
        (t1, w): 0.75,  # l
        uw: 1.0,  # ij
        wv: 1.0,  # ij
        wu: 2.0 + 1.0,  # li (reversed), jk
        vw: 1.0,  # jk
    })

    # Mapping 3
    flow_3 = 0.25 + 0.0625
    mc.var_embedding_decision[triangle_request].x += flow_3

    mc.var_node_mapping[triangle_request][l][w][frozenset()].x += flow_3
    mc.var_node_mapping[triangle_request][i][u][kv_index].x += flow_3
    mc.var_node_mapping[triangle_request][j][v][kv_index].x += flow_3
    mc.var_node_mapping[triangle_request][k][v][frozenset()].x += flow_3

    li_sub_lp.var_node_flow_source[u].x += flow_3
    li_sub_lp.var_edge_flow[uw].x += flow_3
    li_sub_lp.var_node_flow_sink[w].x += flow_3

    ij_sub_lp_kv.var_node_flow_source[u].x += flow_3
    ij_sub_lp_kv.var_edge_flow[uw].x += flow_3
    ij_sub_lp_kv.var_edge_flow[wv].x += flow_3
    ij_sub_lp_kv.var_node_flow_sink[v].x += flow_3

    jk_sub_lp_kv.var_node_flow_source[v].x += flow_3
    jk_sub_lp_kv.var_node_flow_sink[v].x += flow_3

    ik_sub_lp_kv.var_node_flow_source[u].x += flow_3
    ik_sub_lp_kv.var_edge_flow[uv].x += flow_3
    ik_sub_lp_kv.var_node_flow_sink[v].x += flow_3

    load_3 = load_base.copy()
    load_3.update({
        (t1, u): 1.0,  # i
        (t1, v): 1.0 + 1.0,  # j, k
        (t1, w): 0.75,  # l
        uw: 1.0,  # ij
        wu: 2.0,  # li  (reversed)
        wv: 1.0,  # ij
        uv: 1.0,  # ik
    })

    mapping_nt = namedtuple("MappingNT", "flow nodes edges load")
    edge_mapping_nt = namedtuple("EdgeMappingNT", "ij uv_list")

    expected_mappings_as_tuples = {
        mapping_nt(
            flow_1,
            frozenset({i: u, l: w, j: v, k: u}.items()),
            frozenset([
                edge_mapping_nt(li, (wu,)),
                edge_mapping_nt(ij, (uv,)),
                edge_mapping_nt(jk, (vw, wu)),
                edge_mapping_nt(ik, ())
            ]),
            frozenset(load_1.items())
        ),
        mapping_nt(
            flow_2,
            frozenset({i: u, l: w, j: v, k: u}.items()),
            frozenset([
                edge_mapping_nt(li, (wu,)),
                edge_mapping_nt(ij, (uw, wv)),
                edge_mapping_nt(jk, (vw, wu)),
                edge_mapping_nt(ik, ())
            ]),
            frozenset(load_2.items())
        ),
        mapping_nt(
            flow_3,
            frozenset({i: u, l: w, j: v, k: v}.items()),
            frozenset([
                edge_mapping_nt(li, (wu,)),
                edge_mapping_nt(ij, (uw, wv)),
                edge_mapping_nt(jk, ()),
                edge_mapping_nt(ik, (uv,))
            ]),
            frozenset(load_3.items())
        )
    }

    obtained_solution = mc.recover_fractional_solution_from_variables()
    obtained_mappings_as_tuples = {
        mapping_nt(
            obtained_solution.mapping_flows[m.name],
            frozenset(m.mapping_nodes.items()),
            frozenset(edge_mapping_nt(k_, tuple(v_))
                      for (k_, v_) in m.mapping_edges.items()),
            frozenset(obtained_solution.mapping_loads[m.name].items()),
        )
        for m in obtained_solution.request_mapping[triangle_request]
    }

    assert obtained_mappings_as_tuples == expected_mappings_as_tuples


def test_extract_request_mapping(substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])

    mc.var_embedding_decision[triangle_request].x = 0.25

    mc.var_node_mapping[triangle_request][i][u][ku_index].x = 0.25
    mc.var_node_mapping[triangle_request][j][v][ku_index].x = 0.25
    mc.var_node_mapping[triangle_request][k][u][frozenset()].x = 0.25

    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 0.25
    ij_sub_lp.var_edge_flow[uv].x = 0.25
    ij_sub_lp.var_node_flow_sink[v].x = 0.25

    jk_sub_lp = mc.edge_sub_lp[triangle_request][jk][ku_index]
    jk_sub_lp.var_node_flow_source[v].x = 0.25
    jk_sub_lp.var_edge_flow[vw].x = 0.25
    jk_sub_lp.var_edge_flow[wu].x = 0.25
    jk_sub_lp.var_node_flow_sink[u].x = 0.25

    ik_sub_lp = mc.edge_sub_lp[triangle_request][ik][ku_index]
    ik_sub_lp.var_node_flow_source[u].x = 0.25
    ik_sub_lp.var_node_flow_sink[u].x = 0.25

    mapping, flow = mc.extract_request_mapping(triangle_request, mc.request_labels[triangle_request], 1)

    assert flow == 0.25
    assert mapping.mapping_nodes == {i: u, j: v, k: u}
    assert mapping.mapping_edges == {
        ij: [uv],
        jk: [vw, wu],
        ik: [],
    }


def test_reduce_flow_in_modelcreator(substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    scenario = datamodel.Scenario("test", substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])

    mapping = solutions.Mapping("foo", triangle_request, substrate, True)
    mapping.mapping_nodes = {i: u, j: v, k: u}
    mapping.mapping_edges = {ij: [uv], jk: [vw, wu], ik: []}

    mc.reduce_flow(mapping, 0.25)
    assert mc._used_flow_embedding_decision[triangle_request] == 0.25

    assert mc._used_flow_node_mapping[triangle_request][i][u][ku_index] == 0.25
    assert mc._used_flow_node_mapping[triangle_request][j][v][ku_index] == 0.25
    assert mc._used_flow_node_mapping[triangle_request][k][u][frozenset()] == 0.25
    for ij_, uv_list in mapping.mapping_edges.items():
        i_, j_ = ij_
        sub_lp = mc.edge_sub_lp[triangle_request][ij_][ku_index]
        assert sub_lp._used_flow_source[mapping.mapping_nodes[i_]] == 0.25
        assert sub_lp._used_flow_sink[mapping.mapping_nodes[j_]] == 0.25
        for uv_ in uv_list:
            u_, v_ = uv_
            assert sub_lp._used_flow[uv_] == 0.25


def test_reduce_flow_in_modelcreator_with_reversed_edges(substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    triangle_request.graph["root"] = j
    scenario = datamodel.Scenario("test", substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_embedding_decision()
    mc._create_node_mapping_variables()
    mc._create_sub_lp_variables()

    mapping = solutions.Mapping("foo", triangle_request, substrate, True)
    mapping.mapping_nodes = {i: u, j: v, k: u}
    mapping.mapping_edges = {ij: [uv], jk: [vw, wu], ik: []}

    mc.reduce_flow(mapping, 0.25)
    assert mc._used_flow_embedding_decision[triangle_request] == 0.25

    dag_request = mc.dag_requests[triangle_request]
    cycle_end = [i_ for i_ in triangle_request.nodes if len(dag_request.get_in_neighbors(i_)) == 2][0]
    comm_index = frozenset([(cycle_end, mapping.mapping_nodes[cycle_end])])

    for i_, u_ in mapping.mapping_nodes.items():
        if i_ == cycle_end:
            assert mc._used_flow_node_mapping[triangle_request][i_][u_][frozenset()] == 0.25
        else:
            assert mc._used_flow_node_mapping[triangle_request][i_][u_][comm_index] == 0.25
    for ij_, uv_list in mapping.mapping_edges.items():
        # We only need ij_ in the orientation of the BFS request
        if ij_ not in dag_request.edges:
            ij_ = ij_[1], ij_[0]
        i_, j_ = ij_
        sub_lp = mc.edge_sub_lp[triangle_request][ij_][comm_index]
        assert sub_lp._used_flow_source[mapping.mapping_nodes[i_]] == 0.25
        assert sub_lp._used_flow_sink[mapping.mapping_nodes[j_]] == 0.25
        for uv_ in uv_list:
            u_, v_ = uv_
            if sub_lp.is_reversed_edge:
                uv_ = v_, u_
            assert sub_lp._used_flow[uv_] == 0.25


def test_sublp_extend_edge_load_constraints(substrate, triangle_request, import_gurobi_mock):
    MockModel = import_gurobi_mock.MockModel
    # add a single edge to the request that will be reversed
    l = "l"
    li = l, i
    triangle_request.add_node(l, 1.0, "t1", [v])
    triangle_request.add_edge(l, i, 2.0, None)

    # set some different demands on different edges
    triangle_request.edge[ij]["demand"] = 0.5
    triangle_request.edge[ik]["demand"] = 1.5

    scenario = datamodel.Scenario("test", substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_request_load()
    mc._create_sub_lp_variables()

    ku_index = frozenset([("k", "u")])
    kv_index = frozenset([("k", "u")])
    sub_lp_li = mc.edge_sub_lp[triangle_request][(i, l)][frozenset()]
    sub_lp_ij_ku = mc.edge_sub_lp[triangle_request][ij][ku_index]
    sub_lp_ij_kv = mc.edge_sub_lp[triangle_request][ij][kv_index]
    sub_lp_jk_ku = mc.edge_sub_lp[triangle_request][jk][ku_index]
    sub_lp_jk_kv = mc.edge_sub_lp[triangle_request][jk][kv_index]
    sub_lp_ik_ku = mc.edge_sub_lp[triangle_request][ik][ku_index]
    sub_lp_ik_kv = mc.edge_sub_lp[triangle_request][ik][kv_index]

    request_load_var_dict = mc.var_request_load[triangle_request]

    edge_load_constraint_dict = {
        sub_edge: [(-1.0, mc.var_request_load[triangle_request][sub_edge])]
        for sub_edge in substrate.edges
    }
    sub_lp_li.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_ij_ku.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_ij_kv.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_jk_ku.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_jk_kv.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_ik_ku.extend_edge_load_constraints(edge_load_constraint_dict)
    sub_lp_ik_kv.extend_edge_load_constraints(edge_load_constraint_dict)

    assert edge_load_constraint_dict == {
        uv: [(-1.0, request_load_var_dict[uv]),
             (2.0, sub_lp_li.var_edge_flow[vu]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[uv]),
             (0.5, sub_lp_ij_kv.var_edge_flow[uv]),
             (1.0, sub_lp_jk_ku.var_edge_flow[uv]),
             (1.0, sub_lp_jk_kv.var_edge_flow[uv]),
             (1.5, sub_lp_ik_ku.var_edge_flow[uv]),
             (1.5, sub_lp_ik_kv.var_edge_flow[uv])],
        vu: [(-1.0, request_load_var_dict[vu]),
             (2.0, sub_lp_li.var_edge_flow[uv]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[vu]),
             (0.5, sub_lp_ij_kv.var_edge_flow[vu]),
             (1.0, sub_lp_jk_ku.var_edge_flow[vu]),
             (1.0, sub_lp_jk_kv.var_edge_flow[vu]),
             (1.5, sub_lp_ik_ku.var_edge_flow[vu]),
             (1.5, sub_lp_ik_kv.var_edge_flow[vu])],
        vw: [(-1.0, request_load_var_dict[vw]),
             (2.0, sub_lp_li.var_edge_flow[wv]),  # substrate copy will be reversed in decomp. => sub_lp uses wv as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[vw]),
             (0.5, sub_lp_ij_kv.var_edge_flow[vw]),
             (1.0, sub_lp_jk_ku.var_edge_flow[vw]),
             (1.0, sub_lp_jk_kv.var_edge_flow[vw]),
             (1.5, sub_lp_ik_ku.var_edge_flow[vw]),
             (1.5, sub_lp_ik_kv.var_edge_flow[vw])],
        wv: [(-1.0, request_load_var_dict[wv]),
             (2.0, sub_lp_li.var_edge_flow[vw]),  # substrate copy will be reversed in decomp. => sub_lp uses wv as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[wv]),
             (0.5, sub_lp_ij_kv.var_edge_flow[wv]),
             (1.0, sub_lp_jk_ku.var_edge_flow[wv]),
             (1.0, sub_lp_jk_kv.var_edge_flow[wv]),
             (1.5, sub_lp_ik_ku.var_edge_flow[wv]),
             (1.5, sub_lp_ik_kv.var_edge_flow[wv])],
        uw: [(-1.0, request_load_var_dict[uw]),
             (2.0, sub_lp_li.var_edge_flow[wu]),  # substrate copy will be reversed in decomp. => sub_lp uses wv as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[uw]),
             (0.5, sub_lp_ij_kv.var_edge_flow[uw]),
             (1.0, sub_lp_jk_ku.var_edge_flow[uw]),
             (1.0, sub_lp_jk_kv.var_edge_flow[uw]),
             (1.5, sub_lp_ik_ku.var_edge_flow[uw]),
             (1.5, sub_lp_ik_kv.var_edge_flow[uw])],
        wu: [(-1.0, request_load_var_dict[wu]),
             (2.0, sub_lp_li.var_edge_flow[uw]),  # substrate copy will be reversed in decomp. => sub_lp uses wv as dict key for var_edge_flow
             (0.5, sub_lp_ij_ku.var_edge_flow[wu]),
             (0.5, sub_lp_ij_kv.var_edge_flow[wu]),
             (1.0, sub_lp_jk_ku.var_edge_flow[wu]),
             (1.0, sub_lp_jk_kv.var_edge_flow[wu]),
             (1.5, sub_lp_ik_ku.var_edge_flow[wu]),
             (1.5, sub_lp_ik_kv.var_edge_flow[wu])]
    }


def test_modelcreator_create_constraints_track_edge_loads(substrate, triangle_request, import_gurobi_mock):
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr
    MockModel = import_gurobi_mock.MockModel

    # add a single edge to the request that will be reversed
    l = "l"
    li = l, i
    il = i, l
    triangle_request.add_node(l, 1.0, "t1", [v])
    triangle_request.add_edge(l, i, 2.0, None)

    # set some different demands on different edges
    triangle_request.edge[ij]["demand"] = 0.5
    triangle_request.edge[ik]["demand"] = 1.5

    scenario = datamodel.Scenario("test", substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_request_load()
    mc._create_sub_lp_variables()
    mc._create_constraints_track_edge_loads()

    ku_index = frozenset([("k", "u")])
    kv_index = frozenset([("k", "v")])
    sub_lp_li = mc.edge_sub_lp[triangle_request][il][frozenset()]
    sub_lp_ij_ku = mc.edge_sub_lp[triangle_request][ij][ku_index]
    sub_lp_ij_kv = mc.edge_sub_lp[triangle_request][ij][kv_index]
    sub_lp_jk_ku = mc.edge_sub_lp[triangle_request][jk][ku_index]
    sub_lp_jk_kv = mc.edge_sub_lp[triangle_request][jk][kv_index]
    sub_lp_ik_ku = mc.edge_sub_lp[triangle_request][ik][ku_index]
    sub_lp_ik_kv = mc.edge_sub_lp[triangle_request][ik][kv_index]

    assert set(mc.model.constrs) == {
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][uv]),
                 (2.0, sub_lp_li.var_edge_flow[vu]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[uv]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[uv]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[uv]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[uv]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[uv]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[uv])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('u','v')]"
        ),
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][vu]),
                 (2.0, sub_lp_li.var_edge_flow[uv]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[vu]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[vu]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[vu]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[vu]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[vu]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[vu])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('v','u')]"
        ),
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][vw]),
                 (2.0, sub_lp_li.var_edge_flow[wv]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[vw]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[vw]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[vw]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[vw]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[vw]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[vw])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('v','w')]"
        ),
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][wv]),
                 (2.0, sub_lp_li.var_edge_flow[vw]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[wv]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[wv]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[wv]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[wv]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[wv]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[wv])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('w','v')]"
        ),
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][uw]),
                 (2.0, sub_lp_li.var_edge_flow[wu]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[uw]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[uw]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[uw]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[uw]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[uw]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[uw])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('u','w')]"
        ),
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][wu]),
                 (2.0, sub_lp_li.var_edge_flow[uw]),  # substrate will be reversed in decomp. => sub_lp uses vu as dict key for var_edge_flow
                 (0.5, sub_lp_ij_ku.var_edge_flow[wu]),
                 (0.5, sub_lp_ij_kv.var_edge_flow[wu]),
                 (1.0, sub_lp_jk_ku.var_edge_flow[wu]),
                 (1.0, sub_lp_jk_kv.var_edge_flow[wu]),
                 (1.5, sub_lp_ik_ku.var_edge_flow[wu]),
                 (1.5, sub_lp_ik_kv.var_edge_flow[wu])]
            ),
            GRB.EQUAL, 0.0, name="track_edge_load_req[test_req]_sedge[('w','u')]"
        ),
    }


def test_create_constraints_track_node_loads(monkeypatch, substrate, triangle_request, import_gurobi_mock):
    MockConstr = import_gurobi_mock.MockConstr
    MockLinExpr = import_gurobi_mock.MockLinExpr
    MockModel = import_gurobi_mock.MockModel

    # add a request node with new type
    l = "l"
    t1, t2 = "t1", "t2"
    triangle_request.add_node(l, 0.75, t2, [v])
    triangle_request.add_edge(l, i, 2.0, None)

    substrate.node[v]["supported_types"].append(t2)
    substrate.node[v]["capacity"][t2] = 100.0
    substrate.node[v]["cost"][t2] = 0.25
    substrate.types.add(t2)

    scenario = datamodel.Scenario("test", substrate, [triangle_request])
    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mock_model")
    mc.preprocess_input()
    mc.create_variables_request_load()
    mc._create_node_mapping_variables()
    mc._create_constraints_track_node_loads()

    ku_index = frozenset([("k", "u")])
    kv_index = frozenset([("k", "v")])

    assert set(mc.model.constrs) == {
        MockConstr(
            MockLinExpr(
                [(-1.0, mc.var_request_load[triangle_request][t2, v]),
                 (0.75, mc.var_aggregated_node_mapping[triangle_request][l][v])]
            ),
            GRB.EQUAL, 0.0, name="track_node_load_req[test_req]_type[t2]_snode[v]"
        ),
        MockConstr(
            MockLinExpr(
                [
                    (-1.0, mc.var_request_load[triangle_request][t1, v]),
                    (1.0, mc.var_aggregated_node_mapping[triangle_request][j][v]),
                    (1.0, mc.var_aggregated_node_mapping[triangle_request][k][v]),
                ]
            ),
            GRB.EQUAL, 0.0, name="track_node_load_req[test_req]_type[t1]_snode[v]"
        ),
        MockConstr(
            MockLinExpr(
                [
                    (-1.0, mc.var_request_load[triangle_request][t1, u]),
                    (1.0, mc.var_aggregated_node_mapping[triangle_request][i][u]),
                    (1.0, mc.var_aggregated_node_mapping[triangle_request][k][u]),
                ]
            ),
            GRB.EQUAL, 0.0, name="track_node_load_req[test_req]_type[t1]_snode[u]"
        ),
        MockConstr(
            MockLinExpr(
                [
                    (-1.0, mc.var_request_load[triangle_request][t1, w]),
                ]
            ),
            GRB.EQUAL, 0.0, name="track_node_load_req[test_req]_type[t1]_snode[w]"
        ),
    }
