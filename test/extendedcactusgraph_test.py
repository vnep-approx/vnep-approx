import pytest

from alib import datamodel
from vnep_approx.extendedcactusgraph import ExtendedCactusGraph, ExtendedCactusGraphError


class TestExtendedCactusGraph:
    def setup(self):
        self.substrate = datamodel.Substrate("paper_example_substrate")
        self.substrate.add_node("u", ["universal"], {"universal": 1000}, {"universal": 0.0})
        self.substrate.add_node("v", ["universal"], {"universal": 1000}, {"universal": 0.0})
        self.substrate.add_node("w", ["universal"], {"universal": 1000}, {"universal": 0.0})
        self.substrate.add_edge("u", "v", capacity=1000, bidirected=False)
        self.substrate.add_edge("v", "w", capacity=1000, bidirected=False)
        self.substrate.add_edge("w", "u", capacity=1000, bidirected=False)

        self.request = datamodel.Request("paper_example_request")
        self.request.add_node("i", 0.0, "universal", ["w"])
        self.request.add_node("j", 0.0, "universal", ["v", "w"])
        self.request.add_node("k", 0.0, "universal", ["u"])
        self.request.add_node("l", 0.0, "universal", ["u", "w"])
        self.request.add_node("m", 0.0, "universal", ["u", "v"])
        self.request.add_node("n", 0.0, "universal", ["u", "v"])
        self.request.add_node("p", 0.0, "universal", ["v"])
        self.request.add_node("q", 0.0, "universal", ["u", "w"])
        self.request.add_edge("i", "j", 0.0)
        self.request.add_edge("j", "k", 0.0)
        self.request.add_edge("k", "l", 0.0)
        self.request.add_edge("l", "m", 0.0)
        self.request.add_edge("m", "j", 0.0)
        self.request.add_edge("m", "p", 0.0)
        self.request.add_edge("p", "n", 0.0)
        self.request.add_edge("p", "q", 0.0)
        self.request.graph["root"] = "j"

        self.single_edge_sub = datamodel.Substrate("simple_substrate")
        self.single_edge_sub.add_node("u", ["universal"], {"universal": 1000}, {"universal": 0.0})
        self.single_edge_sub.add_node("v", ["universal"], {"universal": 1000}, {"universal": 0.0})
        self.single_edge_sub.add_edge("u", "v", capacity=1000, bidirected=False)

        self.simple_cycle_req = datamodel.Request("simple_cycle_request")
        self.simple_cycle_req.add_node("i", 0.0, "universal", ["u"])
        self.simple_cycle_req.add_node("j", 0.0, "universal", ["v"])
        self.simple_cycle_req.add_node("k", 0.0, "universal", ["w"])
        self.simple_cycle_req.add_edge("i", "j", 0.0)
        self.simple_cycle_req.add_edge("j", "k", 0.0)
        self.simple_cycle_req.add_edge("k", "i", 0.0)
        self.simple_cycle_req.graph["root"] = "i"

        self.single_edge_req = datamodel.Request("simple_path_request")
        self.single_edge_req.add_node("i", 0.0, "universal", ["u", "w"])
        self.single_edge_req.add_node("j", 0.0, "universal", ["w", "v"])
        self.single_edge_req.add_edge("i", "j", 0.0)
        self.single_edge_req.graph["root"] = "i"

    def test_preprocessing(self):
        ecg = ExtendedCactusGraph(self.request, self.substrate)
        reversed_edges = ecg.reversed_request_edges
        reversed_edges_theory = [("i", "j"), ("m", "j"), ("l", "m")]
        for e in reversed_edges_theory:
            assert e in reversed_edges, "{} - {}".format(reversed_edges, reversed_edges_theory)
        assert len(reversed_edges) == len(reversed_edges_theory)

        assert len(ecg._nodes_to_explore) == 0

        expected_paths = [[("j", "i")], [("m", "p")], [("p", "n")], [("p", "q")]]
        for path in expected_paths:
            assert path in ecg._paths
        assert len(expected_paths) == len(ecg._paths)

        expected_cycle = [[("j", "k"), ("k", "l")], [("j", "m"), ("m", "l")]]
        for branch in expected_cycle:
            found_cycle = ecg._cycles[0]
            left, right = found_cycle[0], found_cycle[1]
            either_left_or_right = (all(e in left for e in branch) and not any(e in right for e in branch) or
                                    all(e in right for e in branch) and not any(e in left for e in branch))
            assert either_left_or_right
        assert 1 == len(ecg._cycles)
        assert ecg.cycle_branch_nodes == {"m"}

    def test_correct_topology_for_single_edge_request(self):
        ecg = ExtendedCactusGraph(self.single_edge_req, self.substrate)

        u_i_source = ExtendedCactusGraph._super_node_name("i", "u", "source")
        w_i_source = ExtendedCactusGraph._super_node_name("i", "w", "source")
        u_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "u", "layer")
        v_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "v", "layer")
        w_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "w", "layer")
        v_j_sink = ExtendedCactusGraph._super_node_name("j", "v", "sink")
        w_j_sink = ExtendedCactusGraph._super_node_name("j", "w", "sink")
        expected_nodes = [u_i_source, w_i_source, u_ij, v_ij, w_ij, v_j_sink, w_j_sink]
        expected_edges = [
            (w_i_source, w_ij), (u_i_source, u_ij),  # inflow
            (u_ij, v_ij), (v_ij, w_ij), (w_ij, u_ij),  # layer
            (v_ij, v_j_sink), (w_ij, w_j_sink)  # outflow
        ]
        assert set(expected_nodes) == ecg.nodes
        assert set(expected_edges) == ecg.edges

        #  check that nodes are correctly mapped
        assert "i" in ecg.source_nodes
        allowed_nodes = self.single_edge_req.get_allowed_nodes("i")
        for u in allowed_nodes:
            assert u in ecg.source_nodes["i"]
        assert len(allowed_nodes) == len(ecg.source_nodes["i"])

        assert "j" in ecg.sink_nodes
        allowed_nodes = self.single_edge_req.get_allowed_nodes("j")
        for u in allowed_nodes:
            assert u in ecg.sink_nodes["j"]
        assert len(allowed_nodes) == len(ecg.sink_nodes["j"])

    def test_changing_request_edge_orientation_reverses_substrate_orientation_in_layer(self):
        #  check that the ecg rooted at i contains the substrate in its original orientation:
        ecg = ExtendedCactusGraph(self.single_edge_req, self.substrate)
        u_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "u", "layer")
        v_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "v", "layer")
        w_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "w", "layer")
        expected_layer_edges = {(u_ij, v_ij), (v_ij, w_ij), (w_ij, u_ij)}  # u -> v, v -> w, w -> u
        assert expected_layer_edges <= ecg.edges
        #  check that the ecg rooted at j contains the substrate in reversed orientation:
        self.single_edge_req.graph["root"] = "j"
        ecg = ExtendedCactusGraph(self.single_edge_req, self.substrate)
        u_ij = ExtendedCactusGraph._super_node_name(("j", "i"), "u", "layer")
        v_ij = ExtendedCactusGraph._super_node_name(("j", "i"), "v", "layer")
        w_ij = ExtendedCactusGraph._super_node_name(("j", "i"), "w", "layer")
        expected_layer_edges = {(v_ij, u_ij), (w_ij, v_ij), (u_ij, w_ij)}  # v -> u, w -> v, u -> w
        assert expected_layer_edges <= ecg.edges

    def test_correct_topology_for_simple_cycle_request(self):
        ecg = ExtendedCactusGraph(self.simple_cycle_req, self.substrate)

        u_source = ExtendedCactusGraph._super_node_name("i", "u", "source")
        u_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "u", "layer_cycle", "w")
        v_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "v", "layer_cycle", "w")
        w_ij = ExtendedCactusGraph._super_node_name(("i", "j"), "w", "layer_cycle", "w")
        u_ik = ExtendedCactusGraph._super_node_name(("i", "k"), "u", "layer_cycle", "w")
        v_ik = ExtendedCactusGraph._super_node_name(("i", "k"), "v", "layer_cycle", "w")
        w_ik = ExtendedCactusGraph._super_node_name(("i", "k"), "w", "layer_cycle", "w")
        u_jk = ExtendedCactusGraph._super_node_name(("j", "k"), "u", "layer_cycle", "w")
        v_jk = ExtendedCactusGraph._super_node_name(("j", "k"), "v", "layer_cycle", "w")
        w_jk = ExtendedCactusGraph._super_node_name(("j", "k"), "w", "layer_cycle", "w")
        w_k_sink = ExtendedCactusGraph._super_node_name("k", "w", "sink")
        expected_nodes = {
            u_source, w_k_sink,
            u_ij, v_ij, w_ij,
            u_ik, v_ik, w_ik,
            u_jk, v_jk, w_jk
        }
        expected_edges = {
            (u_source, u_ij), (u_source, u_ik),
            (u_ij, v_ij), (v_ij, w_ij), (w_ij, u_ij),  # layer i -> j
            (v_ij, v_jk),  # inter-layer edge
            (u_jk, v_jk), (v_jk, w_jk), (w_jk, u_jk),  # layer j -> k
            (v_ik, u_ik), (w_ik, v_ik), (u_ik, w_ik),  # layer i -> k. Note the reversed edge orientation
            (w_ik, w_k_sink), (w_jk, w_k_sink)
        }
        assert ecg.nodes == expected_nodes
        assert ecg.edges == expected_edges

        ecg_cycle = ecg.ecg_cycles[0]
        assert ecg_cycle.start_node == "i"
        assert len(ecg_cycle.ext_graph_branches) == 2

    def test_can_discover_a_path_above_cycle(self):
        self.simple_cycle_req.add_node("l", 0.0, "universal", ["v"])
        self.simple_cycle_req.add_edge("l", "i", 0.0)
        self.simple_cycle_req.graph["root"] = "l"
        ecg = ExtendedCactusGraph(self.simple_cycle_req, self.substrate)
        expected_path = [("l", "i")]
        assert len(ecg._paths) == 1
        assert expected_path in ecg._paths

    def test_can_discover_a_cycle_next_to_a_cycle(self):
        self.simple_cycle_req.add_node("l", 0.0, "universal", ["v"])
        self.simple_cycle_req.add_node("n", 0.0, "universal", ["u"])
        self.simple_cycle_req.add_edge("n", "i", 0.0)
        self.simple_cycle_req.add_edge("n", "l", 0.0)
        self.simple_cycle_req.add_edge("l", "i", 0.0)
        for root in self.simple_cycle_req.nodes:
            self.simple_cycle_req.graph["root"] = root
            ecg = ExtendedCactusGraph(self.simple_cycle_req, self.substrate)
            assert len(ecg._cycles) == 2

    def test_correct_node_edge_count_for_example_from_paper(self):
        ecg = ExtendedCactusGraph(self.request, self.substrate)
        # print "\n"
        # print util.get_graph_viz_string(ecg)  # to verify the topology
        # print "\n"
        assert len(ecg.nodes) == 49  # 49 = 36 layer nodes + 5 sources + 8 sinks
        assert len(ecg.edges) == 66  # 66 = 36 layer edges + 14 source edges + 10 sink edges + 6 inter layer edges

    def test_bug_request(self):
        req = datamodel.Request("test")
        req.add_node("n1", 0.0, "universal", ["u", "w"])
        req.add_node("n2", 0.0, "universal", ["u", "w"])
        req.add_node("n3", 0.0, "universal", ["u", "w"])
        req.add_node("n4", 0.0, "universal", ["u", "w"])
        req.add_node("n5", 0.0, "universal", ["u", "w"])
        req.add_node("n6", 0.0, "universal", ["u", "w"])
        req.add_node("n7", 0.0, "universal", ["u", "w"])
        req.add_edge("n1", "n2", 0.0)
        req.add_edge("n2", "n3", 0.0)
        req.add_edge("n3", "n7", 0.0)
        req.add_edge("n3", "n6", 0.0)
        req.add_edge("n2", "n4", 0.0)
        req.add_edge("n4", "n5", 0.0)
        req.add_edge("n5", "n6", 0.0)
        req.graph["root"] = "n1"

        eg = ExtendedCactusGraph(req, self.substrate)

    def test_exclude_edge_mappings_with_insufficient_resources(self):
        sub = datamodel.Substrate("paper_example_substrate")
        sub.add_node("u", ["universal"], {"universal": 100}, {"universal": 0.0})
        sub.add_node("v", ["universal"], {"universal": 100}, {"universal": 0.0})
        sub.add_node("w", ["universal"], {"universal": 100}, {"universal": 0.0})
        sub.add_edge("u", "v", capacity=1, bidirected=False)
        sub.add_edge("v", "w", capacity=1000, bidirected=False)
        sub.add_edge("w", "u", capacity=1000, bidirected=False)

        req = datamodel.Request("test")
        req.add_node("n1", 0.0, "universal", ["u"])
        req.add_node("n2", 0.0, "universal", ["v"])
        req.add_node("n3", 0.0, "universal", ["w"])
        req.add_edge("n1", "n2", 10.0)
        req.add_edge("n2", "n3", 0.0)
        req.graph["root"] = "n1"

        insufficient_ext_edge = (
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "u", "layer"),
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "v", "layer")
        )
        ok_ext_edge = (
            ExtendedCactusGraph._super_node_name(("n2", "n3"), "u", "layer"),
            ExtendedCactusGraph._super_node_name(("n2", "n3"), "v", "layer")
        )

        eg = ExtendedCactusGraph(req, sub)

        assert insufficient_ext_edge not in eg.edges, "Extended graph contained edge corresponding to infeasible edge mapping!"
        assert ok_ext_edge in eg.edges, "Extended graph did not contain edge corresponding to feasible edge mapping!"

    def test_exclude_node_mappings_with_insufficient_resources(self):
        sub = datamodel.Substrate("paper_example_substrate")
        sub.add_node("u", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("v", ["t1", "t2"], {"t1": 100, "t2": 0.0}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("w", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_edge("u", "v", capacity=1000, bidirected=False)
        sub.add_edge("v", "w", capacity=1000, bidirected=False)
        sub.add_edge("w", "u", capacity=1000, bidirected=False)

        req = datamodel.Request("test")
        req.add_node("n1", 10.0, "t1", allowed_nodes=["u"])
        req.add_node("n2", 10.0, "t2", allowed_nodes=["v", "w"])
        req.add_node("n3", 10.0, "t1", allowed_nodes=["w"])
        req.add_edge("n1", "n2", 10.0)
        req.add_edge("n2", "n3", 0.0)
        req.graph["root"] = "n1"

        should_not_exist = (
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "v", "layer"),
            ExtendedCactusGraph._super_node_name("n2", "v", "sink")
        )
        should_exist = (
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "w", "layer"),
            ExtendedCactusGraph._super_node_name("n2", "w", "sink")
        )

        eg = ExtendedCactusGraph(req, sub)
        for u in eg.nodes: print u
        for e in eg.edges: print e
        assert should_not_exist not in eg.edges, "Extended graph contained edge corresponding to infeasible node mapping in path"
        assert should_exist in eg.edges, "Extended graph did not contain edge corresponding to feasible node mapping"

    def test_exclude_node_mappings_with_insufficient_resources_cycle(self):
        # check for a cycle
        sub = datamodel.Substrate("test_substrate")
        sub.add_node("u", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("v", ["t1", "t2"], {"t1": 100, "t2": 0.0}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("w", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("x", ["t1"], {"t1": 1.0}, {"t1": 0.0})
        sub.add_edge("u", "v", capacity=1000, bidirected=False)
        sub.add_edge("v", "w", capacity=1000, bidirected=False)
        sub.add_edge("w", "u", capacity=1000, bidirected=False)
        sub.add_edge("w", "x", capacity=1000, bidirected=False)

        req = datamodel.Request("test_request")
        req.graph["root"] = "n1"
        req.add_node("n1", 10.0, "t1", allowed_nodes=["u"])
        req.add_node("n2", 10.0, "t2", allowed_nodes=["v", "w"])
        req.add_node("n3", 10.0, "t1", allowed_nodes=["w"])
        req.add_node("target", 10.0, "t1", allowed_nodes=["w", "x"])
        req.add_edge("n1", "n2", 10.0)
        req.add_edge("n2", "target", 10.0)
        req.add_edge("n1", "n3", 10.0)
        req.add_edge("n3", "target", 10.0)

        eg = ExtendedCactusGraph(req, sub)
        should_not_exist = (
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "v", "layer_cycle", branch_substrate_node="w"),
            ExtendedCactusGraph._super_node_name(("n2", "target"), "v", "layer_cycle", branch_substrate_node="w")
        )
        should_exist = (
            ExtendedCactusGraph._super_node_name(("n1", "n2"), "w", "layer_cycle", branch_substrate_node="w"),
            ExtendedCactusGraph._super_node_name(("n2", "target"), "w", "layer_cycle", branch_substrate_node="w")
        )

        print should_not_exist
        print should_exist
        for u in eg.nodes: print u
        for e in eg.edges: print e
        assert should_exist in eg.edges, "Extended graph did not contain edge corresponding to feasible node mapping"
        assert should_not_exist not in eg.edges, "Extended graph contained edge corresponding to infeasible node mapping in cycle"
        should_not_exist = ExtendedCactusGraph._super_node_name(("n1", "n2"), "v", "layer_cycle", branch_substrate_node="x")
        should_exist = ExtendedCactusGraph._super_node_name(("n1", "n2"), "v", "layer_cycle", branch_substrate_node="w")
        assert should_not_exist not in eg.nodes, "Extended graph contained edge corresponding to infeasible node mapping in cycle"

    def test_exclude_edge_mappings_with_insufficient_resources_cycle(self):
        sub = datamodel.Substrate("test_substrate")
        sub.add_node("u", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("v", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_node("w", ["t1", "t2"], {"t1": 100, "t2": 100}, {"t1": 0.0, "t2": 0.0})
        sub.add_edge("w", "u", capacity=10.0, bidirected=False)
        sub.add_edge("v", "w", capacity=50.0, bidirected=False)
        sub.add_edge("u", "v", capacity=100.0, bidirected=False)

        req = datamodel.Request("test_request")
        req.graph["root"] = "n1"
        req.add_node("n1", 10.0, "t1", allowed_nodes=["u", "v", "w"])
        req.add_node("n2", 10.0, "t2", allowed_nodes=["u", "v", "w"])
        req.add_node("n3", 10.0, "t1", allowed_nodes=["w"])
        req.add_edge("n1", "n2", 1.0)
        req.add_edge("n2", "n3", 50.0)
        req.add_edge("n1", "n3", 100.0)

        eg = ExtendedCactusGraph(req, sub)

        from alib.util import get_graph_viz_string
        print get_graph_viz_string(eg)
        should_not_exist = [
            (
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "w", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "u", "layer_cycle", branch_substrate_node="w")
            ),
            (
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "w", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "u", "layer_cycle", branch_substrate_node="w")
            ),
            (
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "v", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "w", "layer_cycle", branch_substrate_node="w")
            ),
        ]
        should_exist = [
            (
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "u", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "v", "layer_cycle", branch_substrate_node="w")
            ),
            (
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "v", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n2", "n3"), "w", "layer_cycle", branch_substrate_node="w")
            ),
            (
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "u", "layer_cycle", branch_substrate_node="w"),
                ExtendedCactusGraph._super_node_name(("n1", "n3"), "v", "layer_cycle", branch_substrate_node="w")
            ),
        ]

        for e in should_exist:
            assert e in eg.edges
        for e in should_not_exist:
            assert e not in eg.edges

    def test_multiple_compenents_raise_exception(self):
        request = datamodel.Request("foo")
        request.add_node("i1", 1, "universal", {"u"})
        request.add_node("i2", 1, "universal", {"u"})
        request.add_node("i3", 1, "universal", {"u"})
        request.add_node("i4", 1, "universal", {"u"})
        request.add_edge("i1", "i2", 1)
        request.add_edge("i3", "i4", 1)

        with pytest.raises(
                ExtendedCactusGraphError) as excinfo:
            ExtendedCactusGraph(request, self.substrate)
        assert excinfo.match("Request graph may have multiple components:")
