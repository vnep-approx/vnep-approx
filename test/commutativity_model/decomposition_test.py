import pytest

from alib import datamodel, solutions
from commutativity_model_test_data import example_requests, create_request, filter_requests_by_tags
from gurobi_mock import MockModel
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


def test_extract_edge_mapping(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 0.25
    ij_sub_lp.var_edge_flow[uv].x = 0.25
    ij_sub_lp.var_node_flow_sink[v].x = 0.25

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uv], v, 0.25)


def test_extract_edge_mapping_long_path(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 0.5
    ij_sub_lp.var_edge_flow[uw].x = 0.5
    ij_sub_lp.var_edge_flow[wv].x = 0.5
    ij_sub_lp.var_node_flow_sink[v].x = 0.5

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uw, wv], v, 0.5)


def test_extract_edge_mapping_reversed_edge(substrate, triangle_request):
    triangle_request.graph["root"] = j

    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    kv_index = frozenset([(k, v)])

    ji_sub_lp = mc.edge_sub_lp[triangle_request][("j", "i")].values()[0]
    ji_sub_lp.var_node_flow_source[v].x = 0.5
    ji_sub_lp.var_edge_flow[wu].x = 0.5
    ji_sub_lp.var_edge_flow[vw].x = 0.5
    ji_sub_lp.var_node_flow_sink[u].x = 0.5

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(j, v)

    assert ji_sub_lp.extract_edge_mapping(partial_mapping) == ([uw, wv], u, 0.5)


def test_extract_edge_mapping_with_fixed_end_node(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    kv_index = frozenset([(k, v)])

    ik_sub_lp = mc.edge_sub_lp[triangle_request][ik][kv_index]
    ik_sub_lp.var_node_flow_source[u].x = 0.5
    ik_sub_lp.var_edge_flow[uw].x = 0.5
    ik_sub_lp.var_edge_flow[wv].x = 0.5
    ik_sub_lp.var_node_flow_sink[v].x = 0.5

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)
    partial_mapping.map_node(k, v)

    assert ik_sub_lp.extract_edge_mapping(partial_mapping) == ([uw, wv], v, 0.5)


def test_extract_edge_mapping_colocated_nodes(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    ik_sub_lp = mc.edge_sub_lp[triangle_request][ik][ku_index]
    ik_sub_lp.var_node_flow_source[u].x = 0.5
    ik_sub_lp.var_node_flow_sink[u].x = 0.5

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    assert ik_sub_lp.extract_edge_mapping(partial_mapping) == ([], u, 0.5)


def test_extract_edge_mapping_only_use_exit_with_node_sink_flow(substrate, triangle_request):
    triangle_request.node[j]["allowed_nodes"] = [v, w]
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 0.5

    ij_sub_lp.var_edge_flow[uw].x = 0.5
    ij_sub_lp.var_edge_flow[wv].x = 0.5

    ij_sub_lp.var_node_flow_sink[w].x = 0.0
    ij_sub_lp.var_node_flow_sink[v].x = 0.5

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uw, wv], v, 0.5)


def test_extract_edge_mapping_only_use_exit_with_remaining_node_sink_flow(substrate, triangle_request):
    triangle_request.node[j]["allowed_nodes"] = [v, w]
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 1.0

    # This contains 3 mappings with values 0.5, 0.25 and 0.25
    ij_sub_lp.var_edge_flow[uw].x = 0.5
    ij_sub_lp.var_edge_flow[uv].x = 0.5
    ij_sub_lp.var_edge_flow[wv].x = 0.25

    ij_sub_lp.var_node_flow_sink[w].x = 0.25
    ij_sub_lp.var_node_flow_sink[v].x = 0.75

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    ij_sub_lp._used_flow_source[u] += 0.5
    ij_sub_lp._used_flow[uv] += 0.5
    ij_sub_lp._used_flow_sink[v] += 0.5
    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uw], w, 0.25)

    ij_sub_lp._used_flow_source[u] += 0.25
    ij_sub_lp._used_flow[uw] += 0.25
    ij_sub_lp._used_flow_sink[w] += 0.25
    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uw, wv], v, 0.25)


def test_extract_edge_mapping_multiple_start_nodes(substrate, triangle_request):
    triangle_request.node[i]["allowed_nodes"] = [u, v]
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])
    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]
    ij_sub_lp.var_node_flow_source[u].x = 0.3
    ij_sub_lp.var_node_flow_source[v].x = 0.7

    ij_sub_lp.var_edge_flow[uv].x = 0.3

    ij_sub_lp.var_node_flow_sink[v].x = 1.0

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)

    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([uv], v, 0.3)

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, v)

    assert ij_sub_lp.extract_edge_mapping(partial_mapping) == ([], v, 0.7)


def test_sublp_reduce_flow(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])

    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]

    partial_mapping = solutions.Mapping("test_mapping", triangle_request, substrate, True)
    partial_mapping.map_node(i, u)
    partial_mapping.map_node(j, v)
    partial_mapping.map_edge(ij, [uw, wv])

    ij_sub_lp.reduce_flow(partial_mapping, 0.75)

    assert ij_sub_lp._used_flow[uw] == 0.75
    assert ij_sub_lp._used_flow[wv] == 0.75
    assert ij_sub_lp._used_flow_source[u] == 0.75
    assert ij_sub_lp._used_flow_sink[v] == 0.75


def test_sublp_inconsistency_causes_exception(substrate, triangle_request):
    scenario = datamodel.Scenario("test", substrate, [triangle_request])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")
    mc.preprocess_input()
    mc._create_sub_lp_variables()

    ku_index = frozenset([(k, u)])

    ij_sub_lp = mc.edge_sub_lp[triangle_request][ij][ku_index]

    ij_sub_lp.var_node_flow_source[u].x = 1.0
    partial_mapping = solutions.Mapping("", triangle_request, substrate, True)
    partial_mapping.mapping_nodes[i] = "u"
    with pytest.raises(ValueError) as e:
        ij_sub_lp.extract_edge_mapping(partial_mapping)
    assert str(e.value) == "Did not find valid edge mapping for ('i', 'j')"


@pytest.mark.parametrize("request_id", filter_requests_by_tags("decomposition"))
def test_decomposition_full_request(request_id, substrate):
    req = create_request(request_id)
    req.graph["root"] = example_requests[request_id]["assumed_root"]
    scenario = datamodel.Scenario("test", substrate, [req])

    mc = commutativity_model.CommutativityModelCreator(scenario)
    mc.model = MockModel("mockmodel")

    if example_requests[request_id].get("ignore_bfs", False):
        mc.dag_requests[req] = req

    mc.preprocess_input()
    mc.create_variables()

    def node_mapping_var(i, index, u):
        return mc.var_node_mapping[req][i][u][index]

    def source_var(ij, index, u):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_source[u]

    def sink_var(ij, index, v):
        return mc.edge_sub_lp[req][ij][index].var_node_flow_sink[v]

    def edge_var(ij, index, uv):
        return mc.edge_sub_lp[req][ij][index].var_edge_flow[uv]

    var_type_lookup = dict(node_mapping_var=node_mapping_var,
                           source_var=source_var,
                           sink_var=sink_var,
                           edge_var=edge_var)

    expected_mappings = set()
    expected_embedding_flow = 0.0
    for test_mapping in example_requests[request_id]["mappings"]:
        flow = test_mapping["flow"]
        expected_embedding_flow += flow
        vars = test_mapping["variables"]
        expected_mappings.add(
            (flow,
             frozenset(test_mapping["expected"]["nodes"].items()),
             frozenset((k, tuple(v)) for (k, v) in test_mapping["expected"]["edges"].items()))
        )
        for var_type, req_key, comm_index, substrate_key in vars:
            var = var_type_lookup[var_type](req_key, comm_index, substrate_key)
            var.x += flow
        mc.var_embedding_decision[req].x += flow

    frac_sol = mc.recover_fractional_solution_from_variables()

    obtained_mappings = set()
    for mapping_list in frac_sol.request_mapping.values():
        for mapping in mapping_list:
            obtained_mappings.add((
                frac_sol.mapping_flows[mapping.name],
                frozenset(mapping.mapping_nodes.items()),
                frozenset((k, tuple(v)) for (k, v) in mapping.mapping_edges.items())
            ))

    assert obtained_mappings == expected_mappings
    assert mc._used_flow_embedding_decision[req] == expected_embedding_flow

    expected_used_node_flow = {req: {}}
    for i, u_comm_index_dict in mc.var_node_mapping[req].items():
        expected_used_node_flow[req][i] = {}
        for u, comm_index_dict in u_comm_index_dict.items():
            expected_used_node_flow[req][i][u] = {}
            for comm_index, var in comm_index_dict.items():
                expected_used_node_flow[req][i][u][comm_index] = var.x
    assert mc._used_flow_node_mapping == expected_used_node_flow

    for ij in mc.dag_requests[req].edges:
        for comm_index, sub_lp in mc.edge_sub_lp[req][ij].iteritems():
            expected_used_source_flow = {
                u: var.x for u, var in sub_lp.var_node_flow_source.iteritems()
            }
            assert sub_lp._used_flow_source == expected_used_source_flow
            expected_used_sink_flow = {
                u: var.x for u, var in sub_lp.var_node_flow_sink.iteritems()
            }
            assert sub_lp._used_flow_sink == expected_used_sink_flow
            expected_used_edge_flow = {
                uv: var.x for uv, var in sub_lp.var_edge_flow.iteritems()
            }
            assert sub_lp._used_flow == expected_used_edge_flow
