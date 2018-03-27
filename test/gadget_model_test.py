import random

import pytest

from alib import datamodel, scenariogeneration, solutions, util
from vnep_approx import gadget_model, modelcreator_ecg_decomposition


def get_test_triangle_substrate():
    substrate = datamodel.Substrate("substrate")
    substrate.add_node("u", ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substrate.add_node("v", ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substrate.add_node("w", ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substrate.add_edge("u", "v", capacity=10000, cost=1, bidirected=True)
    substrate.add_edge("v", "w", capacity=10000, cost=1, bidirected=True)
    substrate.add_edge("u", "w", capacity=10000, cost=1, bidirected=True)
    return substrate


def connect_gadgets_with_single_edge(parent_gadget_request, child_in_node, child_gadget_request):
    child_gadget_request.add_node(
        child_in_node,
        parent_gadget_request.get_node_demand(child_in_node),
        parent_gadget_request.get_type(child_in_node),
        parent_gadget_request.get_allowed_nodes(child_in_node)
    )
    child_gadget_request.add_edge(child_in_node, child_gadget_request.graph["root"], 1)
    child_gadget_request.graph["root"] = child_in_node


def connect_gadgets_with_decision_gadget(parent, children):
    decision_request = datamodel.Request("decision_{}".format(parent.name))
    in_node = next(iter(parent.out_nodes))
    ntype = parent.request.get_type(in_node)
    demand = parent.request.get_node_demand(in_node)
    allowed = parent.request.get_allowed_nodes(in_node)
    decision_request.add_node(in_node, demand, ntype=ntype, allowed_nodes=allowed)
    out_nodes = []
    for c in children:
        out = c.in_node
        out_nodes.append(out)
        ntype = c.request.get_type(out)
        demand = c.request.get_node_demand(out)
        allowed = c.request.get_allowed_nodes(out)
        decision_request.add_node(out, demand, ntype=ntype, allowed_nodes=allowed)
        decision_request.add_edge(in_node, out, 1.0)
        decision_request.edge[in_node, out]["edge_profit"] = 1000  # * random.random()

    return gadget_model.DecisionGadget("{}_gadget".format(decision_request.name),
                                       decision_request,
                                       in_node,
                                       out_nodes)


def randomize_substrate_costs(substrate):
    for u in substrate.nodes:
        substrate.node[u]["cost"]["t1"] = 0.5 + random.random()
    for uv in substrate.edges:
        substrate.edge[uv]["cost"] = 0.5 + random.random()


def combine_gadget_requests_to_single_request(requests):
    combined_request = datamodel.Request("combined")
    for req in requests:
        for i in req.nodes:
            if i in combined_request.nodes:
                continue
            combined_request.add_node(i, req.get_node_demand(i), req.get_type(i), req.get_allowed_nodes(i))

        for ij in req.edges:
            i, j = ij
            combined_request.add_edge(i, j, req.get_edge_demand(ij))
    return combined_request


def get_test_decision_gadget(
        gadget_name="decision_gadget",
        in_node="int_1",
        inner_nodes=(),
        out_nodes=("int_2", "int_3"),
        get_allowed_nodes=lambda i: {"u", "v", "w"}):
    """
    Make a decision gadget by defining a set of layers, where each node is connected
    to all nodes in the next layer.

    :param gadget_name: name of the returned :class:`gadget_model.DecisionGadget`
    :param in_node: in-node of the returned :class:`gadget_model.DecisionGadget`
    :param inner_nodes: a iterable of integers that specify the number of nodes per layer
    :param out_nodes: out-nodes of the returned :class:`gadget_model.DecisionGadget`
    :param get_allowed_nodes: callable that returns a set of substrate nodes given the request node name
    :return:
    """
    decision_request = datamodel.Request(gadget_name + "_topology")
    decision_request.add_node(in_node, 1, "t1", allowed_nodes=get_allowed_nodes(in_node))
    previous_layer_nodes = [in_node]
    for i, num_nodes in enumerate(inner_nodes):
        this_layer_nodes = []
        for j in range(num_nodes):
            node = "{}_{}_{}".format(gadget_name, i + 1, j + 1)
            decision_request.add_node(
                node,
                demand=1,
                ntype="t1",
                allowed_nodes=get_allowed_nodes(node)
            )
            this_layer_nodes.append(node)
            for prev_node in previous_layer_nodes:
                decision_request.add_edge(prev_node, node, demand=1)
        previous_layer_nodes = this_layer_nodes
    for node in out_nodes:
        decision_request.add_node(node, 1, "t1", allowed_nodes=get_allowed_nodes(node))
        for prev_node in previous_layer_nodes:
            decision_request.add_edge(prev_node, node, demand=1)

    return gadget_model.DecisionGadget(gadget_name, decision_request, in_node, out_nodes)


class TestGadgetContainerRequest:
    def setup(self):
        self.substrate = get_test_triangle_substrate()
        request1 = datamodel.Request("request1")
        request1.add_node("request1_1", 1, "t1")
        request2 = datamodel.Request("request2")
        request2.add_node("request2_1", 1, "t1")
        request3 = datamodel.Request("request3")
        request3.add_node("request3_1", 1, "t1", allowed_nodes=[])
        request4 = datamodel.Request("request4")
        request4.add_node("request4_1", 1, "t1", allowed_nodes=[])
        self.container_request = gadget_model.GadgetContainerRequest("request", 1000.0)
        self.gadget1 = gadget_model.CactusGadget("gadget1", request1, "request1_1", [])
        self.gadget2 = gadget_model.CactusGadget("gadget2", request2, "request2_1", [])
        self.gadget3 = gadget_model.DecisionGadget("gadget3", request3, "request3_1", [])
        self.gadget4 = gadget_model.DecisionGadget("gadget4", request4, "request4_1", [])

    def test_add_gadget(self):
        assert self.container_request.gadgets == {}
        self.container_request.add_gadget(self.gadget1)
        self.container_request.add_gadget(self.gadget3)
        assert self.container_request.gadgets == {
            "gadget1": self.gadget1,
            "gadget3": self.gadget3,
        }
        assert self.gadget1.container_request == self.container_request
        assert self.gadget3.container_request == self.container_request

    def test_add_gadget_duplicate(self):
        # duplicate name is not allowed
        self.gadget2.name = "gadget1"
        self.container_request.add_gadget(self.gadget1)
        with pytest.raises(ValueError):
            self.container_request.add_gadget(self.gadget2)

    def test_check_and_update(self):
        request1 = datamodel.Request("request1")
        request1.add_node("int_1", 1, "t1", allowed_nodes=["u"])
        request1.add_node("int_2", 1, "t1", allowed_nodes=["v", "w"])
        request1.add_edge("int_1", "int_2", 1)
        gadget1 = gadget_model.CactusGadget("gadget1", request1, "int_1", ["int_2"])
        request2 = datamodel.Request("request1")
        request2.add_node("int_2", 1, "t1", allowed_nodes=["v", "w"])
        request2.add_node("int_3", 1, "t1", allowed_nodes=["v", "w"])
        request2.add_edge("int_2", "int_3", 1)
        gadget2 = gadget_model.CactusGadget("gadget2", request2, "int_2", ["int_3"])
        self.container_request.add_gadget(gadget1)
        self.container_request.add_gadget(gadget2)
        self.container_request.check_and_update()  # should work without exceptions

    def test_update_root_single(self):
        self.gadget1.in_node = "int_1"
        self.container_request.add_gadget(self.gadget1)
        self.container_request.update_root()
        assert self.container_request.root_gadget == self.gadget1

    def test_update_root_complex(self):
        self.gadget1.in_node = "int_1"
        self.gadget1.out_nodes = ["int_2", "int_3"]
        self.gadget2.in_node = "int_2"
        self.gadget3.in_node = "int_3"
        self.gadget3.out_nodes = ["int_4"]
        self.gadget4.in_node = "int_4"
        self.container_request.add_gadget(self.gadget1)
        self.container_request.update_root()
        assert self.container_request.root_gadget == self.gadget1

    def test_update_root_multiple_roots(self):
        # multiple root gadgets are not allowed
        self.gadget1.in_node = "int_1"
        self.gadget2.in_node = "int_2"
        self.container_request.add_gadget(self.gadget1)
        self.container_request.add_gadget(self.gadget2)
        with pytest.raises(
                gadget_model.GadgetError) as excinfo:
            self.container_request.update_root()
        assert excinfo.match(r"there must be exactly one root gadget, found 2$")

    def test_check_nodes_out_node_does_not_exist(self):
        request = datamodel.Request("request")
        request.add_node("int_1", 1, "t1", allowed_nodes=["u"])
        gadget = gadget_model.CactusGadget("gadget", request, "int_1", ["int_2"])
        self.container_request.add_gadget(gadget)
        with pytest.raises(
                gadget_model.GadgetError) as excinfo:
            self.container_request.check_nodes()
        assert excinfo.match(r"out-node 'int_2' not in <CactusGadget name=gadget>$")

    def test_check_nodes_interface_nodes_parameters(self):
        request1 = datamodel.Request("request1")
        request1.add_node("int_1", 1, "t1", allowed_nodes=["u"])
        request1.add_node("int_2", 1, "t1", allowed_nodes=["v", "w"])
        request1.add_edge("int_1", "int_2", 1)
        gadget1 = gadget_model.CactusGadget("gadget1", request1, "int_1", ["int_2"])
        request2 = datamodel.Request("request1")
        request2.add_node("int_2", 1, "t1", allowed_nodes=["u"])
        request2.add_node("int_3", 1, "t1", allowed_nodes=["v", "w"])
        request2.add_edge("int_2", "int_3", 1)
        gadget2 = gadget_model.CactusGadget("gadget2", request2, "int_2", ["int_3"])
        self.container_request.add_gadget(gadget1)
        self.container_request.add_gadget(gadget2)
        with pytest.raises(
                gadget_model.GadgetError) as excinfo:
            self.container_request.check_nodes()
        assert excinfo.match(r"interface node 'int_2' has different parameters: ")

    def test_check_nodes_inner_nodes_unique(self):
        request1 = datamodel.Request("request1")
        request1.add_node("int_1", 1, "t1", allowed_nodes=["u"])
        request1.add_node("i", 1, "t1", allowed_nodes=["u"])
        request1.add_node("int_2", 1, "t1", allowed_nodes=["v", "w"])
        request1.add_edge("int_1", "i", 1)
        request1.add_edge("i", "int_2", 1)
        gadget1 = gadget_model.CactusGadget("gadget1", request1, "int_1", ["int_2"])
        request2 = datamodel.Request("request1")
        request2.add_node("int_2", 1, "t1", allowed_nodes=["v", "w"])
        request2.add_node("i", 1, "t1", allowed_nodes=["u"])
        request2.add_node("int_3", 1, "t1", allowed_nodes=["v", "w"])
        request2.add_edge("int_2", "i", 1)
        request2.add_edge("i", "int_3", 1)
        gadget2 = gadget_model.CactusGadget("gadget2", request2, "int_2", ["int_3"])
        self.container_request.add_gadget(gadget1)
        self.container_request.add_gadget(gadget2)
        with pytest.raises(
                gadget_model.GadgetError) as excinfo:
            self.container_request.check_nodes()
        assert excinfo.match(r"inner node 'i' is in multiple gadgets$")

    def test_check_gadget_tree_two_parents(self):
        self.gadget1.in_node = "int_1"
        self.gadget1.out_nodes = ["int_2"]
        self.gadget2.in_node = "int_2"
        self.gadget2.out_nodes = ["int_3"]
        self.gadget3.in_node = "int_2"
        self.gadget3.out_nodes = ["int_3"]
        self.container_request.add_gadget(self.gadget1)
        self.container_request.add_gadget(self.gadget2)
        self.container_request.add_gadget(self.gadget3)
        with pytest.raises(
                gadget_model.GadgetError) as excinfo:
            self.container_request.check_gadget_tree()
        assert excinfo.match(r"out-node 'int_3' is used by multiple gadgets$")

    def test_inconsistent_mapping_causes_exceptions(self):
        cactus_request = datamodel.Request("cactus_gadget_1")
        cactus_request.add_node("int_1", 1, "t1", allowed_nodes={"u"})
        cactus_request.add_node("int_2", 1, "t1", allowed_nodes={"w"})
        cactus_request.add_edge("int_1", "int_2", 1)
        self.container_request.add_gadget(gadget_model.CactusGadget("gadget_1", cactus_request, "int_1", ["int_2"]))

        decision_gadget = get_test_decision_gadget(
            gadget_name="gadget_2",
            in_node="int_2", out_nodes=["d1", "d2"],
            get_allowed_nodes=lambda i: {"int_2": {"w"}, "d1": {"v", "w"}, "d2": {"w"}}[i]
        )
        self.container_request.add_gadget(decision_gadget)
        self.container_request.check_and_update()

        m = solutions.Mapping("test_mapping", self.container_request, self.substrate, True)
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("request, gadget_1 was only partially mapped")

        # try an invalid node mapping, where int_1 is mapped to a bad substrate node
        m.mapping_nodes["int_1"] = "v"  # invalid node mapping
        m.mapping_nodes["int_2"] = "w"  # ok
        m.mapping_edges[("int_1", "int_2")] = [("v", "w")]  # ok
        m.mapping_nodes["d1"] = "v"  # ok
        m.mapping_edges[("int_2", "d1")] = [("w", "u"), ("u", "v")]  # ok
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Improperly mapped nodes:\s*int_1")

        # try an invalid edge mapping, where the substrate edges are there but out of order
        m.mapping_nodes["int_1"] = "u"  # ok
        m.mapping_edges[("int_2", "d1")] = [("u", "v"), ("w", "u")]  # invalid edge mapping: bad order
        m.mapping_edges[("int_1", "int_2")] = [("u", "w")]  # ok
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Edge mapping \('int_2', 'd1'\) .* inconsistent with node mappings int_2 -> w, d1 -> v")

        # try an invalid edge mapping, where the substrate edges are completely messed up
        m.mapping_edges[("int_2", "d1")] = [("w", "u"), ("w", "v")]  # invalid edge mapping
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Edge \('int_2', 'd1'\) has inconsistent mapping: \[\('w', 'u'\), \('w', 'v'\)\]")

        # try an invalid edge mapping where an edge between non-colocated nodes is mapped to empty list
        m.mapping_edges[("int_2", "d1")] = []  # invalid
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Edge \('int_2', 'd1'\) has empty mapping, but node mappings are int_2 -> w, d1 -> v")

        # map both exits of the decision gadget
        m.mapping_edges[("int_2", "d1")] = [("w", "u"), ("u", "v")]  # ok
        m.mapping_nodes["d2"] = "w"  # ok
        m.mapping_edges[("int_2", "d2")] = []  # ok
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Multiple paths of decision gadget request, gadget_2 were mapped: int_2 -> d1 and int_2 -> d2 in mapping")

        # do not map any exit of the decision gadget
        del m.mapping_nodes["d1"]
        del m.mapping_nodes["d2"]
        del m.mapping_edges[("int_2", "d1")]
        del m.mapping_edges[("int_2", "d2")]
        # self.container_request.verify_request_mapping(m)
        with pytest.raises(gadget_model.DecisionModelError) as excinfo:
            self.container_request.verify_request_mapping(m)
        assert excinfo.match("Mapping did not end in out node")


class TestCactusGadget:
    def setup(self):
        self.request_generation_base_parameters = {
            "topology": "Geant2012",
            "node_types": ("t1",),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 0.5,
            "arbitrary_edge_orientations": False,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 6,
            "iterations": 10,
            "max_cycles": 999,
            "layers": 4,
            "fix_root_mapping": True,
            "fix_leaf_mapping": True,
            "branching_distribution": (0.2, 0.4, 0.0, 0.0, 0.4),
            "probability": 1.0,
            "node_resource_factor": 0.05,
            "edge_resource_factor": 20.0,
            "potential_nodes_factor": 0.5
        }

        self.cactus_request = datamodel.Request("cr_1")
        self.cactus_request.add_node("cr1_1", 1, "t1", allowed_nodes={"u"})
        self.cactus_request.add_node("cr1_2", 1, "t1", allowed_nodes={"v", "w"})
        self.cactus_request.add_node("int_1", 1, "t1", allowed_nodes={"u"})
        self.cactus_request.add_edge("cr1_1", "cr1_2", 1)
        self.cactus_request.add_edge("cr1_1", "int_1", 1)

        self.substrate = get_test_triangle_substrate()

        self.gcr = gadget_model.GadgetContainerRequest("foo", 1234567)

    def test_single_gadget_request_generates_some_variables(self):
        cactus_gadget = gadget_model.CactusGadget("1", self.cactus_request, "cr1_1", ["int_1"])
        self.gcr.add_gadget(cactus_gadget)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()

        old_mc = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario(
                "asdf", self.substrate, [self.cactus_request]
            )
        )
        old_mc.init_model_creator()

        sol = mmc.compute_fractional_solution()

        assert len(mmc.model.getVars()) != 0
        assert len(mmc.model.getConstrs()) != 0

    def test_mapping_single_gadget_different_costs(self):
        """
        Test variable mappings of a single cactus gadget with different node
        and edge costs using the ``MIN_COST`` objective.
        """
        cactus_request1 = datamodel.Request("cr1")
        cactus_request1.add_node("cr1_1", 1, "t1", allowed_nodes={"u"})
        cactus_request1.add_node("int_1", 1, "t1", allowed_nodes={"u"})
        cactus_request1.add_node("cr2_1", 1, "t1", allowed_nodes={"v", "w"})
        cactus_request1.add_edge("cr1_1", "int_1", 1)
        cactus_request1.add_edge("int_1", "cr2_1", 1)

        for case in range(6):
            print "case", case
            expected_mapping_nodes, expected_mapping_edges = self._configure_substrate_different_costs(case)

            self.gcr = gadget_model.GadgetContainerRequest("foo", 1234567)
            self.gcr.add_gadget(gadget_model.CactusGadget("gadget1", cactus_request1, "cr1_1", ["int_1"]))
            self.gcr.check_and_update()

            mmc = gadget_model.GadgetModelCreator(
                datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
            )

            mmc.init_model_creator()
            sol = mmc.compute_fractional_solution()

            assert self.gcr in sol.request_mapping
            mappings = sol.request_mapping[self.gcr]
            assert len(mappings) == 1
            mapping = mappings[0]
            assert mapping.mapping_nodes == expected_mapping_nodes
            assert mapping.mapping_edges == expected_mapping_edges

    def test_mapping_two_gadgets_different_costs(self):
        """
        Test variable mappings of two connected cactus gadgets with different
        node and edge costs using the ``MIN_COST`` objective.
        """
        cactus_request1 = datamodel.Request("cr1")
        cactus_request1.add_node("cr1_1", 1, "t1", allowed_nodes={"u"})
        cactus_request1.add_node("int_1", 1, "t1", allowed_nodes={"u"})
        cactus_request1.add_edge("cr1_1", "int_1", 1)

        cactus_request2 = datamodel.Request("cr2")
        cactus_request2.add_node("int_1", 1, "t1", allowed_nodes={"u"})
        cactus_request2.add_node("cr2_1", 1, "t1", allowed_nodes={"v", "w"})
        cactus_request2.add_edge("int_1", "cr2_1", 1)

        for case in range(6):
            print "case", case
            expected_mapping_nodes, expected_mapping_edges = self._configure_substrate_different_costs(case)

            self.gcr = gadget_model.GadgetContainerRequest("foo", 1234567)
            self.gcr.add_gadget(gadget_model.CactusGadget("gadget1", cactus_request1, "cr1_1", ["int_1"]))
            self.gcr.add_gadget(gadget_model.CactusGadget("gadget2", cactus_request2, "int_1", ["cr2_1"]))
            self.gcr.check_and_update()

            mmc = gadget_model.GadgetModelCreator(
                datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
            )

            mmc.init_model_creator()
            sol = mmc.compute_fractional_solution()

            assert self.gcr in sol.request_mapping
            mappings = sol.request_mapping[self.gcr]
            assert len(mappings) == 1
            mapping = mappings[0]
            assert mapping.mapping_nodes == expected_mapping_nodes
            assert mapping.mapping_edges == expected_mapping_edges

    def _configure_substrate_different_costs(self, case):
        self.substrate.node["u"]["cost"]["t1"] = 1
        self.substrate.node["v"]["cost"]["t1"] = 1
        self.substrate.node["w"]["cost"]["t1"] = 1
        self.substrate.edge[("u", "v")]["cost"] = 1
        self.substrate.edge[("v", "u")]["cost"] = 1
        self.substrate.edge[("u", "w")]["cost"] = 1
        self.substrate.edge[("w", "u")]["cost"] = 1
        self.substrate.edge[("v", "w")]["cost"] = 1
        self.substrate.edge[("w", "v")]["cost"] = 1

        if case == 0:
            # case 0: should map cr2_1 -> v due to edge cost
            self.substrate.edge[("u", "w")]["cost"] = 2
            self.substrate.edge[("w", "u")]["cost"] = 2
            self.substrate.edge[("v", "w")]["cost"] = 2
            self.substrate.edge[("w", "v")]["cost"] = 2
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "v",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "v")],
            }
        elif case == 1:
            # case 1: should map cr2_1 -> w due to edge cost
            self.substrate.edge[("u", "v")]["cost"] = 2
            self.substrate.edge[("v", "u")]["cost"] = 2
            self.substrate.edge[("v", "w")]["cost"] = 2
            self.substrate.edge[("w", "v")]["cost"] = 2
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "w",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "w")],
            }
        elif case == 2:
            # case 2: should map cr2_1 -> v due to node cost
            self.substrate.node["w"]["cost"]["t1"] = 2
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "v",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "v")],
            }
        elif case == 3:
            # case 3: should map cr2_1 -> w due to node cost
            self.substrate.node["v"]["cost"]["t1"] = 2
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "w",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "w")],
            }
        elif case == 4:
            # case 4: should map cr2_1 -> v due to node cost
            # and (int_1, cr2_1) to [(u, w), (w, v)] due to edge cost
            self.substrate.node["w"]["cost"]["t1"] = 3
            self.substrate.edge[("u", "v")]["cost"] = 3
            self.substrate.edge[("v", "u")]["cost"] = 3
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "v",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "w"), ("w", "v")],
            }
        elif case == 5:
            # case 5: should map cr2_1 -> w due to node cost
            # and (int_1, cr2_1) to [(u, v), (v, w)] due to edge cost
            self.substrate.node["v"]["cost"]["t1"] = 3
            self.substrate.edge[("u", "w")]["cost"] = 3
            self.substrate.edge[("w", "u")]["cost"] = 3
            expected_mapping_nodes = {
                "cr1_1": "u",
                "int_1": "u",
                "cr2_1": "w",
            }
            expected_mapping_edges = {
                ("cr1_1", "int_1"): [],
                ("int_1", "cr2_1"): [("u", "v"), ("v", "w")],
            }
        else:
            raise Exception("Should not happen!")
        return expected_mapping_nodes, expected_mapping_edges

    def test_two_connected_generated_cactus_requests(self):
        # random.seed(0)
        self.request_generation_base_parameters["number_of_requests"] = 2

        randomize_substrate_costs(self.substrate)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.request_generation_base_parameters, self.substrate
        )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)

        req_a = cactus_requests[0]
        req_b = cactus_requests[1]
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        node_restriction_generator.generate_restrictions_single_request(
            req_a, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_b, self.substrate, self.request_generation_base_parameters
        )

        out_node_a = random.choice(list(req_a.nodes - {req_a.graph["root"]}))
        out_node_b = random.choice(list(req_b.nodes))

        connect_gadgets_with_single_edge(req_a, out_node_a, req_b)

        gadget_a = gadget_model.CactusGadget("cactus_a", req_a, req_a.graph["root"], [out_node_a])
        gadget_b = gadget_model.CactusGadget("cactus_b", req_b, out_node_a, [out_node_b])
        container_request.add_gadget(gadget_a)
        container_request.add_gadget(gadget_b)
        container_request.check_and_update()

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MIN_COST)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        combined_request = combine_gadget_requests_to_single_request(cactus_requests)
        cactus_modelcreator = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario("combined_scenario", self.substrate, [combined_request], datamodel.Objective.MIN_COST)
        )
        cactus_modelcreator.init_model_creator()
        cactus_solution = cactus_modelcreator.compute_fractional_solution()

        assert (cactus_solution.request_mapping[combined_request][0].mapping_nodes
                == gadget_solution.request_mapping[container_request][0].mapping_nodes)
        assert (cactus_solution.request_mapping[combined_request][0].mapping_edges
                == gadget_solution.request_mapping[container_request][0].mapping_edges)

    def test_gadgets_with_shared_innode(self):
        # random.seed(0)
        self.request_generation_base_parameters["number_of_requests"] = 3

        randomize_substrate_costs(self.substrate)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.request_generation_base_parameters, self.substrate
        )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)

        req_a = cactus_requests[0]
        req_b = cactus_requests[1]
        req_c = cactus_requests[2]
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        node_restriction_generator.generate_restrictions_single_request(
            req_a, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_b, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_c, self.substrate, self.request_generation_base_parameters
        )

        out_node_a = random.choice(list(req_a.nodes - {req_a.graph["root"]}))
        out_node_b = random.choice(list(req_b.nodes))
        out_node_c = random.choice(list(req_c.nodes))

        connect_gadgets_with_single_edge(req_a, out_node_a, req_b)
        connect_gadgets_with_single_edge(req_a, out_node_a, req_c)

        gadget_a = gadget_model.CactusGadget("cactus_a", req_a, req_a.graph["root"], [out_node_a])
        gadget_b = gadget_model.CactusGadget("cactus_b", req_b, out_node_a, [out_node_b])
        gadget_c = gadget_model.CactusGadget("cactus_c", req_c, out_node_a, [out_node_c])
        container_request.add_gadget(gadget_a)
        container_request.add_gadget(gadget_b)
        container_request.add_gadget(gadget_c)
        container_request.check_and_update()

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MIN_COST)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        combined_request = combine_gadget_requests_to_single_request(cactus_requests)
        cactus_modelcreator = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario("combined_scenario", self.substrate, [combined_request], datamodel.Objective.MIN_COST)
        )
        cactus_modelcreator.init_model_creator()
        cactus_solution = cactus_modelcreator.compute_fractional_solution()

        assert (cactus_solution.request_mapping[combined_request][0].mapping_nodes
                == gadget_solution.request_mapping[container_request][0].mapping_nodes)
        assert (cactus_solution.request_mapping[combined_request][0].mapping_edges
                == gadget_solution.request_mapping[container_request][0].mapping_edges)

    def test_connected_gadgets_different_innodes(self):
        # random.seed(0)
        self.request_generation_base_parameters["number_of_requests"] = 3

        randomize_substrate_costs(self.substrate)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.request_generation_base_parameters, self.substrate
        )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)

        req_a = cactus_requests[0]
        req_b = cactus_requests[1]
        req_c = cactus_requests[2]
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        node_restriction_generator.generate_restrictions_single_request(
            req_a, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_b, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_c, self.substrate, self.request_generation_base_parameters
        )

        out_node_ab = random.choice(list(req_a.nodes - {req_a.graph["root"]}))
        out_node_ac = random.choice(list(req_a.nodes - {req_a.graph["root"], out_node_ab}))
        out_node_b = random.choice(list(req_b.nodes))
        out_node_c = random.choice(list(req_c.nodes))

        # make the gadgets overlap

        connect_gadgets_with_single_edge(req_a, out_node_ab, req_b)
        connect_gadgets_with_single_edge(req_a, out_node_ac, req_c)

        gadget_a = gadget_model.CactusGadget("cactus_a", req_a, req_a.graph["root"], [out_node_ab, out_node_ac])
        gadget_b = gadget_model.CactusGadget("cactus_b", req_b, out_node_ab, [out_node_b])
        gadget_c = gadget_model.CactusGadget("cactus_c", req_c, out_node_ac, [out_node_c])
        container_request.add_gadget(gadget_a)
        container_request.add_gadget(gadget_b)
        container_request.add_gadget(gadget_c)
        container_request.check_and_update()

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MIN_COST)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        combined_request = combine_gadget_requests_to_single_request(cactus_requests)
        cactus_modelcreator = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario("combined_scenario", self.substrate, [combined_request], datamodel.Objective.MIN_COST)
        )
        cactus_modelcreator.init_model_creator()
        cactus_solution = cactus_modelcreator.compute_fractional_solution()

        # print get_graph_viz_string(
        #     combined_request, get_edge_style=lambda e: "color=" + (
        #         "red" if e in gadget_a.request.edges else "blue" if e in gadget_b.request.edges else "green"
        #     )
        # )

        assert (cactus_solution.request_mapping[combined_request][0].mapping_nodes
                == gadget_solution.request_mapping[container_request][0].mapping_nodes)
        assert (cactus_solution.request_mapping[combined_request][0].mapping_edges
                == gadget_solution.request_mapping[container_request][0].mapping_edges)

    def test_three_gadgets_two_with_shared_innode_starting_on_cycles(self):
        # random.seed(0)
        self.request_generation_base_parameters["number_of_requests"] = 3

        randomize_substrate_costs(self.substrate)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.request_generation_base_parameters, self.substrate
        )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)

        req_a = cactus_requests[0]
        req_b = cactus_requests[1]
        req_c = cactus_requests[2]
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        node_restriction_generator.generate_restrictions_single_request(
            req_a, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_b, self.substrate, self.request_generation_base_parameters
        )
        node_restriction_generator.generate_restrictions_single_request(
            req_c, self.substrate, self.request_generation_base_parameters
        )

        out_node_a = random.choice(list(req_a.nodes - {req_a.graph["root"]}))
        root_b = req_b.graph["root"]
        out_node_b = random.choice(list(req_b.nodes))
        root_c = req_c.graph["root"]
        out_node_c = random.choice(list(req_c.nodes))

        # make the gadgets overlap
        demand = req_a.get_node_demand(out_node_a)
        node_type = req_a.get_type(out_node_a)
        allowed_nodes = req_a.get_allowed_nodes(out_node_a)
        req_b.add_node(out_node_a, demand, node_type, allowed_nodes)
        req_b.add_node("node_b", demand, node_type, {"w"})
        req_b.add_edge(out_node_a, "node_b", 1)
        req_b.add_edge("node_b", root_b, 1)
        req_b.add_edge(out_node_a, root_b, 1)

        req_c.add_node(out_node_a, demand, node_type, allowed_nodes)
        req_c.add_node("node_c1", demand, node_type, {"u"})
        req_c.add_node("node_c2", demand, node_type, {"v"})
        req_c.add_edge(out_node_a, "node_c1", 1)
        req_c.add_edge(out_node_a, "node_c2", 1)
        req_c.add_edge("node_c1", root_c, 1)
        req_c.add_edge("node_c2", root_c, 1)

        gadget_a = gadget_model.CactusGadget("cactus_a", req_a, req_a.graph["root"], [out_node_a])
        gadget_b = gadget_model.CactusGadget("cactus_b", req_b, out_node_a, [out_node_b])
        gadget_c = gadget_model.CactusGadget("cactus_c", req_c, out_node_a, [out_node_c])
        container_request.add_gadget(gadget_a)
        container_request.add_gadget(gadget_b)
        container_request.add_gadget(gadget_c)
        container_request.check_and_update()

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MIN_COST)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        combined_request = combine_gadget_requests_to_single_request(cactus_requests)
        cactus_modelcreator = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario("combined_scenario", self.substrate, [combined_request], datamodel.Objective.MIN_COST)
        )
        cactus_modelcreator.init_model_creator()
        cactus_solution = cactus_modelcreator.compute_fractional_solution()

        assert (cactus_solution.request_mapping[combined_request][0].mapping_nodes
                == gadget_solution.request_mapping[container_request][0].mapping_nodes)
        assert (cactus_solution.request_mapping[combined_request][0].mapping_edges
                == gadget_solution.request_mapping[container_request][0].mapping_edges)

    def test_many_gadgets(self):
        # random.seed(0)
        self.request_generation_base_parameters["number_of_requests"] = 10
        # self.request_generation_base_parameters["min_number_of_nodes"] = 7
        # self.request_generation_base_parameters["max_number_of_nodes"] = 15
        # self.request_generation_base_parameters["layers"] = 7

        randomize_substrate_costs(self.substrate)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.request_generation_base_parameters, self.substrate
        )
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        for req in cactus_requests:
            node_restriction_generator.generate_restrictions_single_request(
                req, self.substrate, self.request_generation_base_parameters
            )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)

        root_req = cactus_requests[0]
        root_out_nodes = random.sample(list(root_req.nodes - {root_req.graph["root"]}), random.randint(1, 2))

        container_request.add_gadget(
            gadget_model.CactusGadget("root_gadget", root_req, root_req.graph["root"], root_out_nodes)
        )
        out_node_map = {
            i: root_req for i in root_out_nodes
        }
        for req in cactus_requests[1:]:
            in_node, parent_req = random.choice(out_node_map.items())
            out_nodes = random.sample(list(req.nodes), random.randint(1, 2))
            out_node_map.update({i: req for i in out_nodes})
            connect_gadgets_with_single_edge(parent_req, in_node, req)
            container_request.add_gadget(
                gadget_model.CactusGadget("{}_gadget".format(req.name), req, in_node, out_nodes)
            )

        container_request.check_and_update()

        combined_request = combine_gadget_requests_to_single_request(cactus_requests)

        # print get_graph_viz_string(
        #     combined_request, get_edge_style=util.graph_viz_edge_color_according_to_request_list(
        #         [req.edges for req in cactus_requests]
        #     )
        # )

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MIN_COST)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        cactus_modelcreator = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            datamodel.Scenario("combined_scenario", self.substrate, [combined_request], datamodel.Objective.MIN_COST)
        )
        cactus_modelcreator.init_model_creator()
        cactus_solution = cactus_modelcreator.compute_fractional_solution()

        assert (cactus_solution.request_mapping[combined_request][0].mapping_nodes
                == gadget_solution.request_mapping[container_request][0].mapping_nodes)
        assert (cactus_solution.request_mapping[combined_request][0].mapping_edges
                == gadget_solution.request_mapping[container_request][0].mapping_edges)


class TestDecisionGadget:
    def setup(self):
        self.substrate = get_test_triangle_substrate()
        self.gcr = gadget_model.GadgetContainerRequest("foo", 1234567)

    def test_simple_chain_request_generates_solution_and_mapping(self):
        decision_gadget_simple_chain = get_test_decision_gadget(
            "decision_simple_chain",
            in_node="int_1",
            out_nodes=["int_2"],
            inner_nodes=[1, 1]
        )
        self.gcr.add_gadget(decision_gadget_simple_chain)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()
        mapping = sol.request_mapping[self.gcr][0]
        assert mapping is not None
        assert sol is not None

    def test_decision_gadget_with_split_maps_exactly_one_out_node(self):
        gadget = get_test_decision_gadget(
            "decision_simple_split",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1]
        )
        gadget.request.add_edge("int_2", "int_3", 1)

        self.gcr.add_gadget(gadget)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()
        mapping = sol.request_mapping[self.gcr][0]
        assert mapping is not None
        assert sol is not None
        out_mapped = 0
        for out_node in gadget.out_nodes:
            if out_node in mapping.mapping_nodes:
                out_mapped += 1
        assert out_mapped == 1

    def test_decision_gadget_with_split_min_cost(self):
        gadget = get_test_decision_gadget(
            "decision",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1],
            get_allowed_nodes=lambda i: {
                "int_1": {"w"},
                "int_2": {"v"},
                "int_3": {"w"},
                "decision_1_1": {"u"},
            }[i]
        )

        self.gcr.add_gadget(gadget)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()
        mapping = sol.request_mapping[self.gcr][0]
        assert mapping is not None
        assert sol is not None

        assert mapping.mapping_nodes in [
            {"int_1": "w", "decision_1_1": "u", "int_2": "v"},
            {"int_1": "w", "decision_1_1": "u", "int_3": "w"},
        ]
        assert mapping.mapping_edges in [
            {("int_1", "decision_1_1"): [("w", "u")],
             ("decision_1_1", "int_2"): [("u", "v")]},
            {("int_1", "decision_1_1"): [("w", "u")],
             ("decision_1_1", "int_3"): [("u", "w")]},
        ]

    def test_new_profit_model(self):
        decision_gadget = get_test_decision_gadget(
            "decision",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1],
            get_allowed_nodes=lambda i: {
                "int_1": {"w"},
                "int_2": {"v"},
                "int_3": {"w"},
                "decision_1_1": {"u"},
            }[i]
        )
        decision_gadget.request.add_edge("int_2", "int_3", 1)

        decision_gadget.request.edge[("int_1", "decision_1_1")]["edge_profit"] = 7
        decision_gadget.request.edge[("decision_1_1", "int_2")]["edge_profit"] = 13
        decision_gadget.request.edge[("decision_1_1", "int_3")]["edge_profit"] = 19
        decision_gadget.request.edge[("int_2", "int_3")]["edge_profit"] = 1

        self.gcr.add_gadget(decision_gadget)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()

        expected_profit = self.gcr.profit
        mapping_list = sol.request_mapping.values()[0]
        for mapping in mapping_list:
            flow = sol.mapping_flows[mapping.name]
            for ij in mapping.mapping_edges:
                expected_profit += flow * decision_gadget.request.edge[ij]["edge_profit"]

        assert mmc.status.objValue == pytest.approx(expected_profit)

    def test_large_gadget_with_many_layers(self):
        layers = [3, 4, 5, 4, 3]
        decision_gadget = get_test_decision_gadget(
            "decision",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=layers
        )

        self.gcr.add_gadget(decision_gadget)
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()

        for m in sol.request_mapping[self.gcr]:
            assert len(m.mapping_nodes) == len(layers) + 2
            prev_layer_node = decision_gadget.in_node
            for layer_index, layer in enumerate(layers):
                mapped_count = 0
                this_layer_node = None
                for i in range(layer):
                    layer_node = "decision_{}_{}".format(layer_index + 1, i + 1)
                    if layer_node in m.mapping_nodes:
                        mapped_count += 1
                        this_layer_node = layer_node
                assert mapped_count == 1
                assert (prev_layer_node, this_layer_node) in m.mapping_edges
                prev_layer_node = this_layer_node

            assert "int_2" in m.mapping_nodes or "int_3" in m.mapping_nodes
            assert "int_2" not in m.mapping_nodes or "int_3" not in m.mapping_nodes
            if "int_2" in m.mapping_nodes:
                assert (prev_layer_node, "int_2") in m.mapping_edges
            if "int_3" in m.mapping_nodes:
                assert (prev_layer_node, "int_3") in m.mapping_edges

    def test_chained_decision_gadgets_no_shared_innodes(self):
        randomize_substrate_costs(self.substrate)

        parent = get_test_decision_gadget(
            "parent",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[],
            get_allowed_nodes=lambda i: {
                "int_1": {"u"},
                "int_2": {"w"},
                "int_3": {"v"},
            }[i]
        )
        child_a = get_test_decision_gadget(
            "child_a",
            in_node="int_2",
            out_nodes=["a_1", "a_2"],
            inner_nodes=[],
            get_allowed_nodes=lambda i: {
                "int_2": {"w"},
                "a_1": {"w"},
                "a_2": {"u"},
            }[i]
        )
        child_b = get_test_decision_gadget(
            "child_b",
            in_node="int_3",
            out_nodes=["b_1", "b_2"],
            inner_nodes=[],
            get_allowed_nodes=lambda i: {
                "int_3": {"v"},
                "b_1": {"v"},
                "b_2": {"w"},
            }[i]
        )
        self.gcr.add_gadget(parent)
        self.gcr.add_gadget(child_a)
        self.gcr.add_gadget(child_b)
        self.gcr.check_and_update()
        gmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
        )
        gmc.init_model_creator()
        sol = gmc.compute_fractional_solution()
        # gmc.model.write("foo/individual_gcr.lp")

        # combine to a single decision gadget
        combined_request = combine_gadget_requests_to_single_request([parent.request, child_a.request, child_b.request])

        combined_gcr = gadget_model.GadgetContainerRequest(
            "foo", 1234567.0
        )
        combined_gadget = gadget_model.DecisionGadget("foo", combined_request, "int_1", ["a_1", "a_2", "b_1", "b_2"])
        combined_gcr.add_gadget(combined_gadget)
        combined_gcr.check_and_update()
        combined_gmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [combined_gcr], objective=datamodel.Objective.MIN_COST)
        )
        combined_gmc.init_model_creator()
        combined_sol = combined_gmc.compute_fractional_solution()
        # combined_gmc.model.write("foo/combined_gcr.lp")
        assert (sol.request_mapping[self.gcr][0].mapping_nodes
                == combined_sol.request_mapping[combined_gcr][0].mapping_nodes)
        assert (sol.request_mapping[self.gcr][0].mapping_edges
                == combined_sol.request_mapping[combined_gcr][0].mapping_edges)

    @pytest.mark.parametrize("repetition", range(50))
    def test_chained_decision_gadgets_same_result_as_combined_gadget(self, repetition):
        # There was a bug where some of the node loads were tracked incorrectly by the DecisionGadget.
        randomize_substrate_costs(self.substrate)

        parent = get_test_decision_gadget(
            "parent",
            in_node="int_1",
            out_nodes=["int_2"],
            inner_nodes=[],
            get_allowed_nodes=lambda i: {
                "int_1": {"u"},
                "int_2": {"w"},
            }[i]
        )
        child_a = get_test_decision_gadget(
            "c_a",
            in_node="int_2",
            out_nodes=["a_1", "a_2"],
            inner_nodes=[],
            get_allowed_nodes=lambda i: {
                "int_2": {"w"},
                "a_1": {"w"},
                "a_2": {"u"},
            }[i]
        )
        self.gcr.add_gadget(parent)
        self.gcr.add_gadget(child_a)
        self.gcr.check_and_update()
        gmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
        )
        gmc.init_model_creator()
        sol = gmc.compute_fractional_solution()

        # combine to a single decision gadget
        combined_request = combine_gadget_requests_to_single_request([parent.request, child_a.request])

        combined_gcr = gadget_model.GadgetContainerRequest(
            "foo", 1234567.0
        )
        combined_gadget = gadget_model.DecisionGadget("foo", combined_request, "int_1", ["a_1", "a_2"])
        combined_gcr.add_gadget(combined_gadget)
        combined_gcr.check_and_update()
        combined_gmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [combined_gcr], objective=datamodel.Objective.MIN_COST)
        )
        combined_gmc.init_model_creator()
        combined_sol = combined_gmc.compute_fractional_solution()
        # combined_gmc.model.write("foo/combined_gcr.lp")
        assert (sol.request_mapping[self.gcr][0].mapping_nodes
                == combined_sol.request_mapping[combined_gcr][0].mapping_nodes)
        assert (sol.request_mapping[self.gcr][0].mapping_edges
                == combined_sol.request_mapping[combined_gcr][0].mapping_edges)

    def test_can_extract_mapping_from_handcrafted_solution(self, import_gurobi_mock):
        MockVar = import_gurobi_mock.MockVar
        gadget = get_test_decision_gadget(
            "foo",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1]
        )

        self.gcr.add_gadget(gadget)
        self.gcr.check_and_update()
        gmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MIN_COST)
        )
        gmc.init_model_creator()

        for i, u_var_dict in gadget.gurobi_vars["node_flow"].iteritems():
            for u in u_var_dict:
                gadget.gurobi_vars["node_flow"][i][u] = MockVar()
        for ext_edge in gadget.gurobi_vars["edge_flow"]:
            gadget.gurobi_vars["edge_flow"][ext_edge] = MockVar()

        mapping_dicts = [
            {
                "flow": 0.25,
                "mapping_nodes": {
                    "int_1": "u",
                    "foo_1_1": "u",
                    "int_2": "u"
                },
                "mapping_edges": {
                    ('int_1', 'foo_1_1'): [],
                    ('foo_1_1', 'int_2'): []
                }
            },
            {
                "flow": 0.25,
                "mapping_nodes": {
                    "int_1": "u",
                    "foo_1_1": "w",
                    "int_3": "w"
                },
                "mapping_edges": {
                    ('int_1', 'foo_1_1'): [("u", "w")],
                    ('foo_1_1', 'int_3'): []
                }
            },
            {
                "flow": 0.5,
                "mapping_nodes": {
                    "int_1": "u",
                    "foo_1_1": "u",
                    "int_3": "w"
                },
                "mapping_edges": {
                    ('int_1', 'foo_1_1'): [],
                    ('foo_1_1', 'int_3'): [("u", "w")]
                }
            },
            {
                "flow": 0.5,
                "mapping_nodes": {
                    "int_1": "u",
                    "foo_1_1": "u",
                    "int_3": "w"
                },
                "mapping_edges": {
                    ('int_1', 'foo_1_1'): [],
                    ('foo_1_1', 'int_3'): [("u", "v"), ("v", "w")]
                }
            }
        ]
        for m_dict in mapping_dicts:
            self._apply_mapping_to_mock_dict(m_dict, gadget)

        expected = set([(
            md["flow"],
            frozenset(md["mapping_nodes"].items()),
            frozenset((k, tuple(v)) for k, v in md["mapping_edges"].items())
        )
            for md in mapping_dicts])

        obtained = set()
        for _ in mapping_dicts:
            m = solutions.Mapping("asdf", self.gcr, self.substrate, False)
            load = {res: 0.0 for res in self.gcr.substrate_resources}
            flow = gadget.extend_mapping_by_own_solution(m, load, set())[0]
            gadget.reduce_flow_on_last_returned_mapping(flow)

            obtained.add(
                (flow, frozenset(m.mapping_nodes.items()), frozenset((k, tuple(v)) for k, v in m.mapping_edges.items()))
            )
        assert expected == obtained

    def _apply_mapping_to_mock_dict(self, mapping_dict, gadget):
        ext_graph = gadget.ext_graph
        flow = mapping_dict["flow"]
        request_source_sink_node_set = set(ext_graph.source_nodes.keys()) | set(ext_graph.sink_nodes.keys())

        for i, u in mapping_dict["mapping_nodes"].items():
            if i in request_source_sink_node_set:
                gadget.gurobi_vars["node_flow"][i][u].x += flow

                if i in ext_graph.sink_nodes:
                    ext_node = ext_graph.sink_nodes[i][u]
                    ext_edge = ext_graph.in_edges[ext_node][0]
                    gadget.gurobi_vars["edge_flow"][ext_edge].x += flow

        for ij, uv_list in mapping_dict["mapping_edges"].iteritems():
            i, j = ij
            u = mapping_dict["mapping_nodes"][i]

            incoming_ext_edge = ext_graph.inter_layer_edges[ij][u]
            gadget.gurobi_vars["edge_flow"][incoming_ext_edge].x += flow

            for uv in uv_list:
                layer_ext_edge = ext_graph.layer_edges[j][uv]
                gadget.gurobi_vars["edge_flow"][layer_ext_edge].x += flow


class TestMixedGadget:
    def setup(self):
        self.gcr = gadget_model.GadgetContainerRequest("test_request", profit=1234567)
        self.substrate = get_test_triangle_substrate()

        self.base_parameters = {
            "topology": "Geant2012",
            "node_types": ("t1",),
            "node_cost_factor": 1.0,
            "node_capacity": 100.0,
            "edge_cost": 1.0,
            "edge_capacity": 100.0,
            "node_type_distribution": 0.5,
            "arbitrary_edge_orientations": False,
            "min_number_of_nodes": 3,
            "max_number_of_nodes": 6,
            "number_of_requests": 3,
            "iterations": 10,
            "max_cycles": 999,
            "layers": 4,
            "fix_root_mapping": True,
            "fix_leaf_mapping": True,
            "branching_distribution": (0.2, 0.4, 0.0, 0.0, 0.4),
            "probability": 1.0,
            "node_resource_factor": 0.05,
            "edge_resource_factor": 20.0,
            "potential_nodes_factor": 1.0
        }

    def test_decision_after_cactus(self):
        cactus_request = datamodel.Request("cr_1")
        cactus_request.add_node("i_c1", 1, "t1", allowed_nodes={"u"})
        cactus_request.add_node("j_c1", 1, "t1", allowed_nodes={"v"})
        cactus_request.add_node("int_1", 1, "t1", allowed_nodes={"w"})
        cactus_request.add_edge("i_c1", "j_c1", 1)
        cactus_request.add_edge("j_c1", "int_1", 1)
        cactus_request.add_edge("i_c1", "int_1", 1)

        self.gcr.add_gadget(gadget_model.CactusGadget("cg_1", cactus_request, "i_c1", ["int_1"]))

        cactus_request = datamodel.Request("cr_1")
        cactus_request.add_node("i_c2", 1, "t1", allowed_nodes={"u"})
        cactus_request.add_node("j_c2", 1, "t1", allowed_nodes={"v"})
        cactus_request.add_node("int_2", 1, "t1", allowed_nodes={"v"})
        cactus_request.add_edge("int_2", "i_c2", 1)
        cactus_request.add_edge("int_2", "j_c2", 1)
        cactus_request.add_edge("j_c2", "i_c2", 1)

        self.gcr.add_gadget(gadget_model.CactusGadget("cg_2", cactus_request, "int_2", ["i_c2"]))

        cactus_request = datamodel.Request("cr_1")
        cactus_request.add_node("i_c3", 1, "t1", allowed_nodes={"u"})
        cactus_request.add_node("j_c3", 1, "t1", allowed_nodes={"v"})
        cactus_request.add_node("int_3", 1, "t1", allowed_nodes={"w"})
        cactus_request.add_edge("int_3", "i_c3", 1)
        cactus_request.add_edge("int_3", "j_c3", 1)
        cactus_request.add_edge("j_c3", "i_c3", 1)

        self.gcr.add_gadget(gadget_model.CactusGadget("cg_3", cactus_request, "int_3", ["i_c3"]))

        decision_gadget = get_test_decision_gadget(
            "decision",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1],
            get_allowed_nodes=lambda i: {
                "int_1": {"w"},
                "int_2": {"v"},
                "int_3": {"w"},
                "decision_1_1": {"u"},
            }[i]
        )
        decision_gadget.request.add_edge("int_2", "int_3", 1)

        self.gcr.add_gadget(decision_gadget)

        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        solution = mmc.compute_fractional_solution()

        assert solution is not None

        # assert mapping is not empty
        mapping = solution.request_mapping[self.gcr][0]
        assert mapping is not None

        # assert that in_node of decision gadget is mapped
        assert mapping.get_mapping_of_node(decision_gadget.in_node) is not None

        # assert that one of the out nodes of decision gadget is mapped
        out_mapped_counter = 0
        for out_node in decision_gadget.out_nodes:
            if out_node in mapping.mapping_nodes:
                out_mapped_counter += 1
        assert out_mapped_counter == 1

    def test_chained_gadget_with_split(self):
        decision_gadget = get_test_decision_gadget(
            "decision",
            in_node="int_1",
            out_nodes=["int_2", "int_3"],
            inner_nodes=[1],
            get_allowed_nodes=lambda i: {
                "int_1": {"w"},
                "int_2": {"v"},
                "int_3": {"w"},
                "decision_1_1": {"u"},
            }[i]
        )
        decision_gadget.request.add_edge("int_2", "int_3", 1)

        self.gcr.add_gadget(decision_gadget)

        # print get_graph_viz_string(decision_gadget.ext_graph)

        cactus_request = datamodel.Request("cr_1")
        cactus_request.add_node("i_c1", 1, "t1", allowed_nodes={"u"})
        cactus_request.add_node("j_c1", 1, "t1", allowed_nodes={"v", "w"})
        cactus_request.add_node("int_2", 1, "t1", allowed_nodes={"v"})
        cactus_request.add_edge("i_c1", "j_c1", 1)
        cactus_request.add_edge("i_c1", "int_2", 1)

        self.gcr.add_gadget(gadget_model.CactusGadget("cg_1", cactus_request, "int_2", []))

        cactus_request_2 = datamodel.Request("cr_2")
        cactus_request_2.add_node("int_3", 1, "t1", allowed_nodes={"w"})
        cactus_request_2.add_node("i_c2", 1, "t1", allowed_nodes={"v", "w"})
        cactus_request_2.add_node("j_c2", 1, "t1", allowed_nodes={"w"})
        cactus_request_2.add_edge("int_3", "i_c2", 1)
        cactus_request_2.add_edge("int_3", "j_c2", 1)

        self.gcr.add_gadget(gadget_model.CactusGadget("cg_2", cactus_request_2, "int_3", []))
        self.gcr.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [self.gcr], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        solution = mmc.compute_fractional_solution()
        assert solution is not None

        # assert mapping is not empty
        mapping = solution.request_mapping[self.gcr][0]
        assert mapping is not None

        # assert that in_node of decision gadget is mapped
        assert mapping.get_mapping_of_node(decision_gadget.in_node) is not None

        # assert that one of the out nodes of decision gadget is mapped
        out_mapped_flag = False
        dec_out_node = None
        for out_node in decision_gadget.out_nodes:
            if out_node in mapping.mapping_nodes:
                out_mapped_flag = True
                dec_out_node = out_node
        assert out_mapped_flag

        # assert that decided cactus gadget is also mapped
        out_mapped_flag = False
        for gagdet in self.gcr.gadgets.values():
            if dec_out_node == gagdet.in_node:
                for out_node in decision_gadget.out_nodes:
                    if out_node in mapping.mapping_nodes:
                        out_mapped_flag = True
        assert out_mapped_flag

    def test_large_request_of_cactus_with_decision_gadget_in_the_middle(self):
        # random.seed(0)

        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.base_parameters, self.substrate
        )
        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)
        interface_nodes = []

        for i, cr in enumerate(cactus_requests):
            in_node = cr.graph["root"]
            out_node = in_node
            while out_node == in_node:
                out_node = random.choice(list(cr.nodes))
            cactus_gadget = gadget_model.CactusGadget("cactus_gadget_{}".format(i + 1), cr, in_node, [out_node])

            container_request.add_gadget(cactus_gadget)
            if i == 0:
                interface_nodes.append(out_node)
            else:
                interface_nodes.append(in_node)

        decision_gadget = get_test_decision_gadget(
            "decision_gadget", in_node=interface_nodes[0], out_nodes=interface_nodes[1:], inner_nodes=[1],
            get_allowed_nodes=lambda _i: {
                node: req.get_allowed_nodes(node) for (node, req) in zip(interface_nodes, cactus_requests)
            }.get(_i, {"u"})
        )
        for int_node, request in zip(interface_nodes, cactus_requests):
            decision_gadget.request.node[int_node]["demand"] = request.node[int_node]["demand"]

        container_request.add_gadget(decision_gadget)
        container_request.check_and_update()
        mmc = gadget_model.GadgetModelCreator(
            datamodel.Scenario("foo", self.substrate, [container_request], objective=datamodel.Objective.MAX_PROFIT)
        )
        mmc.init_model_creator()
        sol = mmc.compute_fractional_solution()

        assert container_request in sol.request_mapping
        assert len(sol.request_mapping[container_request]) != 0

        for m in sol.request_mapping[container_request]:
            for i in cactus_requests[0].nodes:
                assert i in m.mapping_nodes
            for ij in cactus_requests[0].edges:
                assert ij in m.mapping_edges
            assert interface_nodes[1] in m.mapping_nodes or interface_nodes[2] in m.mapping_nodes
            assert interface_nodes[1] not in m.mapping_nodes or interface_nodes[2] not in m.mapping_nodes
            if interface_nodes[1] in m.mapping_nodes:
                for i in cactus_requests[1].nodes:
                    assert i in m.mapping_nodes
                for ij in cactus_requests[1].edges:
                    assert ij in m.mapping_edges
            else:
                for i in cactus_requests[2].nodes:
                    assert i in m.mapping_nodes
                for ij in cactus_requests[2].edges:
                    assert ij in m.mapping_edges

    def test_alternating_layers_of_decision_and_cactus_produces_a_mapping(self):
        # random.seed(0)
        self.base_parameters["number_of_requests"] = 50
        # self.request_generation_base_parameters["min_number_of_nodes"] = 7
        # self.request_generation_base_parameters["max_number_of_nodes"] = 15
        # self.request_generation_base_parameters["layers"] = 7

        randomize_substrate_costs(self.substrate)

        container_request = gadget_model.GadgetContainerRequest("request", 1000.0)
        cactus_requests = scenariogeneration.CactusRequestGenerator().generate_request_list(
            self.base_parameters, self.substrate
        )
        cactus_gadgets = []

        cactus_edges = set()
        node_restriction_generator = scenariogeneration.UniformEmbeddingRestrictionGenerator()
        for req in cactus_requests:
            cactus_edges |= req.edges
            node_restriction_generator.generate_restrictions_single_request(
                req, self.substrate, self.base_parameters
            )
            root = req.graph["root"]
            out_nodes = random.sample([i for i in req.nodes if i != root], 1)
            g = gadget_model.CactusGadget("{}_gadget".format(req.name), req, root, out_nodes)
            cactus_gadgets.append(g)
            container_request.add_gadget(g)

        parent_queue = {cactus_gadgets.pop()}
        children_queue = set(cactus_gadgets)
        decision_edges = set()
        while children_queue:
            parent = parent_queue.pop()
            children = random.sample(list(children_queue), min(random.randint(1, 3), len(children_queue)))
            children_queue -= set(children)
            parent_queue |= set(children)
            decision_gadget = connect_gadgets_with_decision_gadget(parent, children)
            decision_edges |= decision_gadget.request.edges
            container_request.add_gadget(decision_gadget)

        container_request.check_and_update()
        # print get_graph_viz_string(container_request.get_gadget_tree_graph())

        gadget_modelcreator = gadget_model.GadgetModelCreator(
            datamodel.Scenario("gadget_scenario", self.substrate, [container_request], datamodel.Objective.MAX_PROFIT)
        )

        gadget_modelcreator.init_model_creator()
        gadget_solution = gadget_modelcreator.compute_fractional_solution()

        combined = datamodel.Graph("combined")
        combined.edges |= decision_edges
        combined.edges |= cactus_edges

        for m in gadget_solution.request_mapping[container_request]:
            def edge_style(e):
                edge_groups = [decision_edges, cactus_edges]
                colors = ["blue", "darkgreen"]
                edge_group_color_function = util.graph_viz_edge_color_according_to_request_list(edge_groups, colors)
                return edge_group_color_function(e) + ",penwidth={}".format(6.0 if e in m.mapping_edges else 2.0)

            print util.get_graph_viz_string(
                combined,
                get_edge_style=edge_style
            )

        assert len(gadget_solution.request_mapping[container_request]) != 0
