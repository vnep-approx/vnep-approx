import copy

import pytest
from gurobipy import GRB

from alib import datamodel, mip, modelcreator, scenariogeneration, solutions
from vnep_approx import extendedcactusgraph, modelcreator_ecg_decomposition


class TestCactusModelCreator:
    def setup(self):
        scenariogeneration.random.seed(5)
        self.substrate = datamodel.Substrate("paper_example_substrate")
        self.substrate.add_node("u", ["universal"], {"universal": 10}, {"universal": 0.8})
        self.substrate.add_node("v", ["universal"], {"universal": 10}, {"universal": 1.2})
        self.substrate.add_node("w", ["universal"], {"universal": 10}, {"universal": 1.0})
        self.substrate.add_edge("u", "v", capacity=100, bidirected=False)
        self.substrate.add_edge("v", "w", capacity=100, bidirected=False)
        self.substrate.add_edge("w", "u", capacity=100, bidirected=False)

        self.request = datamodel.Request("test")
        self.request.add_node("i", 0.33, "universal", ["w"])
        self.request.add_node("j", 0.33, "universal", ["v", "w"])
        self.request.add_node("k", 0.33, "universal", ["u"])
        self.request.add_node("l", 0.33, "universal", ["u", "w"])
        self.request.add_node("m", 0.33, "universal", ["u", "v"])
        self.request.add_node("n", 0.33, "universal", ["u", "v"])
        self.request.add_node("p", 0.33, "universal", ["v"])
        self.request.add_node("q", 0.33, "universal", ["u", "w"])
        self.request.add_edge("i", "j", 0.25)
        self.request.add_edge("j", "k", 0.25)
        self.request.add_edge("k", "l", 0.25)
        self.request.add_edge("l", "m", 0.25)
        self.request.add_edge("m", "j", 0.25)
        self.request.add_edge("m", "p", 0.25)
        self.request.add_edge("p", "n", 0.25)
        self.request.add_edge("p", "q", 0.25)
        self.request.profit = 1000.0
        self.request.graph["root"] = "j"

        scenariogeneration.random.seed(0)
        self.sg = scenariogeneration.ScenarioGenerator("test")

    def test_variables(self):
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request])
        mc = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc.init_model_creator()

    def test_example_get_load_based_on_integral_solution(self):
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MIN_COST)
        mc_mip = mip.ClassicMCFModel(scenario)
        mc_mip.init_model_creator()
        sol = mc_mip.compute_integral_solution().solution
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(sol.scenario)
        mc_ecg.init_model_creator()
        mc_ecg.fix_mapping_variables_according_to_integral_solution(sol)
        mc_ecg.model.optimize()
        assert mc_mip.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(mc_ecg.model.getAttr(GRB.Attr.ObjVal))

    def test_can_handle_request_whose_node_demand_exceeds_all_substrate_capacities(self):
        self.request.node["p"]["demand"] = 10**10
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MAX_PROFIT)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        mc_ecg.model.optimize()
        assert mc_ecg.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(0.0)

    def test_can_handle_request_whose_edge_demand_exceeds_all_substrate_capacities(self):
        self.request.edge[("m", "p")]["demand"] = 10**10
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MAX_PROFIT)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        sol = mc_ecg.compute_fractional_solution()
        assert mc_ecg.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(1000.0)  # Can be solved by colocation

        # forbid colocation => should become infeasible
        self.request.set_allowed_nodes("m", {"u"})
        self.request.set_allowed_nodes("p", {"v"})
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        sol = mc_ecg.compute_fractional_solution()
        assert mc_ecg.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(0.0)  # Cannot be solved by colocation

    def test_can_handle_request_whose_node_demand_exceeds_some_substrate_capacities(self):
        self.request.node["q"]["demand"] = 10.0  # q is allowed on u, w
        self.substrate.node["w"]["capacity"]["universal"] = 1.0  # q no longer fits on w
        self.substrate.node["u"]["capacity"]["universal"] = 100.0  # q still fits on u
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MAX_PROFIT)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        mc_ecg.model.optimize()
        assert mc_ecg.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(1000.0)

    def test_can_handle_request_whose_edge_demand_exceeds_some_substrate_capacities(self):
        self.request.edge[("m", "p")]["demand"] = 10.0  # m is allowed on u, v and p on v
        self.substrate.edge["w", "u"]["capacity"] = 1.0
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MAX_PROFIT)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        mc_ecg.model.optimize()
        assert mc_ecg.model.getAttr(GRB.Attr.ObjVal) == pytest.approx(1000.0)

    def test_mc_ecg_ignores_unprofitable_requests_when_using_max_profit_objective(self):
        profit_req = copy.deepcopy(self.request)
        no_profit_req = copy.deepcopy(self.request)
        profit_req.name = "profit_req"
        no_profit_req.name = "no_profit_req"
        profit_req.profit = 1000.0
        no_profit_req.profit = 0.0
        scenario = datamodel.Scenario("test_scenario", self.substrate, [profit_req, no_profit_req],
                                      objective=datamodel.Objective.MAX_PROFIT)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        assert no_profit_req not in mc_ecg.requests
        assert no_profit_req in mc_ecg.all_requests
        assert profit_req in mc_ecg.requests

        mc_ecg.init_model_creator()

        # Check that the unprofitable request is not in the Gurobi Model:
        assert no_profit_req not in mc_ecg.extended_graphs
        assert no_profit_req not in mc_ecg.var_node_flow
        assert no_profit_req not in mc_ecg.var_edge_flow
        assert no_profit_req not in mc_ecg.var_request_load
        assert no_profit_req not in mc_ecg.var_embedding_decision
        assert no_profit_req not in mc_ecg.var_request_load

        fs = mc_ecg.compute_fractional_solution()
        # Check that the unprofitable request *is* in the fractional solution
        assert isinstance(fs, solutions.FractionalScenarioSolution)
        assert profit_req in fs.request_mapping
        assert no_profit_req not in fs.request_mapping

        # Ensure that profit_req was mapped
        assert sum([fs.mapping_flows[m.name] for m in fs.request_mapping[profit_req]]) == 1
        for i in profit_req.nodes:
            assert all(i in mapping.mapping_nodes for mapping in fs.request_mapping[profit_req])
        for ij in profit_req.edges:
            assert all(ij in mapping.mapping_edges for mapping in fs.request_mapping[profit_req])

    def test_mc_ecg_does_not_ignore_unprofitable_requests_when_using_min_cost_objective(self):
        profit_req = copy.deepcopy(self.request)
        no_profit_req = copy.deepcopy(self.request)
        profit_req.name = "profit_req"
        no_profit_req.name = "no_profit_req"
        profit_req.profit = 1000.0
        no_profit_req.profit = 0.0
        scenario = datamodel.Scenario("test_scenario", self.substrate, [profit_req, no_profit_req],
                                      objective=datamodel.Objective.MIN_COST)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        assert no_profit_req in mc_ecg.requests
        assert no_profit_req in mc_ecg.all_requests
        assert profit_req in mc_ecg.requests

        mc_ecg.init_model_creator()

        # Check that the unprofitable request is in the Gurobi Model:
        assert no_profit_req in mc_ecg.extended_graphs
        assert no_profit_req in mc_ecg.var_node_flow
        assert no_profit_req in mc_ecg.var_edge_flow
        assert no_profit_req in mc_ecg.var_request_load
        assert no_profit_req in mc_ecg.var_embedding_decision
        assert no_profit_req in mc_ecg.var_request_load

        fs = mc_ecg.compute_fractional_solution()
        # Check that the unprofitable request *is* in the fractional solution
        assert isinstance(fs, solutions.FractionalScenarioSolution)
        assert profit_req in fs.request_mapping
        assert no_profit_req in fs.request_mapping
        assert sum([fs.mapping_flows[m.name] for m in fs.request_mapping[profit_req]]) == 1
        assert sum([fs.mapping_flows[m.name] for m in fs.request_mapping[no_profit_req]]) == 1
        for i in profit_req.nodes:
            assert all(i in mapping.mapping_nodes for mapping in fs.request_mapping[profit_req])
        for ij in profit_req.edges:
            assert all(ij in mapping.mapping_edges for mapping in fs.request_mapping[profit_req])

    def test_can_get_fractional_solution(self):
        scenario = datamodel.Scenario("test_scenario", self.substrate, [self.request],
                                      objective=datamodel.Objective.MIN_COST)
        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(scenario)
        mc_ecg.init_model_creator()
        fs = mc_ecg.compute_fractional_solution()
        assert isinstance(fs, solutions.FractionalScenarioSolution)
        for req in scenario.requests:
            assert req in fs.request_mapping
            # ensure that all nodes & edges are in each mapping
            for i in req.nodes:
                assert all(i in mapping.mapping_nodes for mapping in fs.request_mapping[req])
            for ij in req.edges:
                assert all(ij in mapping.mapping_edges for mapping in fs.request_mapping[req])

    def test_model_fixing_simple_scenario(self):
        sub = datamodel.Substrate("fixing_sub")
        sub.add_node("u", {"t1"}, {"t1": 10}, {"t1": 100})
        sub.add_node("v", {"t1"}, {"t1": 10}, {"t1": 100})
        sub.add_node("w", {"t1"}, {"t1": 10}, {"t1": 100})
        sub.add_node("x", {"t1"}, {"t1": 10}, {"t1": 100})
        sub.add_edge("u", "v")
        sub.add_edge("v", "w")
        sub.add_edge("w", "x")
        sub.add_edge("x", "u")

        req = datamodel.Request("fixing_req")
        req.profit = 1000
        req.add_node("i", 5, "t1", {"u", "v", "w", "x"})
        req.add_node("j", 5, "t1", {"u", "v", "w", "x"})
        req.add_edge("i", "j", 0.5)

        mapping = solutions.Mapping("fixing_mapping", req, sub, True)
        mapping.map_node("i", "u")
        mapping.map_node("j", "x")
        mapping.map_edge(("i", "j"), [("u", "v"), ("v", "w"), ("w", "x")])

        scenario = datamodel.Scenario("fixing_scen", sub, [req], datamodel.Objective.MAX_PROFIT)
        sol = solutions.IntegralScenarioSolution("fixing_sol", scenario)
        sol.add_mapping(req, mapping)

        mc_ecg = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(
            scenario
        )
        mc_ecg.init_model_creator()
        mc_ecg.fix_mapping_variables_according_to_integral_solution(sol)
        ecg_sol = mc_ecg.compute_integral_solution()

        new_mapping = ecg_sol.solution.request_mapping[req]
        assert new_mapping.mapping_nodes == mapping.mapping_nodes
        assert new_mapping.mapping_edges == mapping.mapping_edges

    def test_ecg_and_mcf_should_obtain_same_solution_when_variables_are_fixed(self):
        real_scenario = self.make_real_scenario()
        real_scenario.objective = datamodel.Objective.MAX_PROFIT
        time_limit = 200

        mc_preprocess = mip.ClassicMCFModel(real_scenario)
        mc_preprocess.init_model_creator()
        mc_preprocess.set_gurobi_parameter(modelcreator.Param_TimeLimit, time_limit)
        sol_preprocess = mc_preprocess.compute_integral_solution().solution

        usable_requests = [req for req in real_scenario.requests if sol_preprocess.request_mapping[req].is_embedded]
        assert len(usable_requests) != 0

        real_scenario.requests = usable_requests
        real_scenario.objective = datamodel.Objective.MIN_COST

        # First, compute solution using MCF model and fix variables of Decomposition
        mc_mcf = mip.ClassicMCFModel(real_scenario)
        mc_mcf.init_model_creator()
        mc_mcf.set_gurobi_parameter(modelcreator.Param_TimeLimit, time_limit)
        sol_mcf = mc_mcf.compute_integral_solution().get_solution()
        sol_mcf.validate_solution_fulfills_capacity()

        assert any(m.is_embedded for m in sol_mcf.request_mapping.values())

        mc_ecg_fixed_mapping = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(real_scenario)
        mc_ecg_fixed_mapping.init_model_creator()
        mc_ecg_fixed_mapping.set_gurobi_parameter(modelcreator.Param_TimeLimit, time_limit)
        mc_ecg_fixed_mapping.fix_mapping_variables_according_to_integral_solution(sol_mcf)
        sol_ecg_fixed_mapping = mc_ecg_fixed_mapping.compute_integral_solution()

        assert mc_mcf.model.objVal == pytest.approx(mc_ecg_fixed_mapping.model.objVal)
        for req in usable_requests:
            mcf_mapping = sol_mcf.request_mapping[req]
            ecg_mapping = sol_ecg_fixed_mapping.get_solution().request_mapping[req]
            print mcf_mapping.mapping_nodes
            print ecg_mapping.mapping_nodes
            assert mcf_mapping.mapping_nodes == ecg_mapping.mapping_nodes
            assert mcf_mapping.mapping_edges == ecg_mapping.mapping_edges

        # compute solution using Decomposition and fix variables of MCF model
        mc_ecg_fixed_mapping = modelcreator_ecg_decomposition.ModelCreatorCactusDecomposition(real_scenario)
        mc_ecg_fixed_mapping.init_model_creator()
        mc_ecg_fixed_mapping.set_gurobi_parameter(modelcreator.Param_TimeLimit, time_limit)
        sol_ecg = mc_ecg_fixed_mapping.compute_integral_solution().get_solution()

        assert any(m.is_embedded for m in sol_ecg.request_mapping.values())

        mc_mcf_fixed_mapping = mip.ClassicMCFModel(real_scenario)
        mc_mcf_fixed_mapping.init_model_creator()
        mc_mcf_fixed_mapping.set_gurobi_parameter(modelcreator.Param_TimeLimit, time_limit)
        mc_mcf_fixed_mapping.fix_mapping_variables_according_to_integral_solution(sol_ecg)
        sol_mcf_fixed_mapping = mc_mcf_fixed_mapping.compute_integral_solution().solution
        sol_mcf_fixed_mapping.validate_solution_fulfills_capacity()

        assert mc_mcf_fixed_mapping.model.objVal == pytest.approx(mc_ecg_fixed_mapping.model.objVal)
        for req in usable_requests:
            mcf_mapping = sol_mcf_fixed_mapping.request_mapping[req]
            ecg_mapping = sol_ecg.request_mapping[req]
            assert mcf_mapping.mapping_nodes == ecg_mapping.mapping_nodes
            assert mcf_mapping.mapping_edges == ecg_mapping.mapping_edges

    def make_real_scenario(self):
        scenario_parameters = {
            "maxrepetition": 1,
            "repetition": 1,
            "substrate_generation": {
                "foo": {
                    "TopologyZooReader": {
                        "topology": "Geant2012",
                        "node_types": ("t1", "t2"),
                        "node_cost_factor": 1.0,
                        "node_capacity": 100.0,
                        "edge_cost": 1.0,
                        "node_type_distribution": 0.2,
                        "edge_capacity": 100.0,
                    }
                }
            },
            "profit_calculation": {
                "random": {
                    "RandomEmbeddingProfitCalculator": {
                        "iterations": 3,
                        "profit_factor": 1.0
                    }
                }
            },
            "node_placement_restriction_mapping": {
                "neighbors": {
                    "NeighborhoodSearchRestrictionGenerator": {
                        "potential_nodes_factor": 0.5
                    }
                }
            },
            "request_generation": {
                "cactus": {
                    "CactusRequestGenerator": {
                        "number_of_requests": 5,
                        "min_number_of_nodes": 2,
                        "max_number_of_nodes": 5,
                        "probability": 1,
                        "potential_nodes_factor": 0.1,
                        "node_resource_factor": 0.25,
                        "edge_resource_factor": 5,
                        "normalize": True,
                        "arbitrary_edge_orientations": False,
                        "profit_factor": 1.0,
                        "iterations": 10,
                        "max_cycles": 20,
                        "layers": 3,
                        "fix_root_mapping": True,
                        "fix_leaf_mapping": True,
                        "branching_distribution": (0.0, 0.5, 0.3, 0.15, 0.05)
                    }
                }
            }
        }
        return scenariogeneration.build_scenario((1, scenario_parameters))[1]


class TestDecomposition:
    def test_decomposition_can_handle_splitting_flows(self):
        req = datamodel.Request("test_req")
        req.add_node("root", 1.0, "t1", allowed_nodes=["w"])
        req.add_node("n2", 1.0, "t1", allowed_nodes=["v", "w"])
        req.add_node("n3", 1.0, "t1", allowed_nodes=["u"])

        req.add_edge("root", "n2", 1.0)
        req.add_edge("root", "n3", 1.0)
        req.add_edge("n2", "n3", 1.0)
        req.graph["root"] = "root"

        sub = datamodel.Substrate("test_sub")
        sub.add_node("u", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_node("v", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_node("w", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_edge("u", "v", 1000.0, bidirected=True)
        sub.add_edge("v", "w", 1000.0, bidirected=True)
        sub.add_edge("w", "u", 1000.0, bidirected=True)

        # this is a simple request with a splitting flow that caused 200+ mappings due to a bug
        flow_values = {
            'node': {
                'root': {'w': 0.5652934510902061},
                'n3': {'u': 0.565293451090206}
            },
            'embedding': 0.5652934510902061,
            'edge': {
                ('v_[n2n3]_[u]', 'u_[n2n3]_[u]'): 0.5625924430120608,
                ('w_[n2n3]_[u]', 'v_[n2n3]_[u]'): 0.16087557584604475,
                ('u_[rootn3]_[u]', 'u_[n3]_-'): 0.5652934510902061,
                ('v_[rootn2]_[u]', 'v_[n2n3]_[u]'): 0.40171686716601607,
                ('w_[root]_+', 'w_[rootn2]_[u]'): 0.5652934510902061,
                ('w_[rootn2]_[u]', 'v_[rootn2]_[u]'): 0.40171686716601607,
                ('w_[rootn3]_[u]', 'u_[rootn3]_[u]'): 0.5652934510902061,
                ('w_[root]_+', 'w_[rootn3]_[u]'): 0.5652934510902061,
                ('w_[n2n3]_[u]', 'u_[n2n3]_[u]'): 0.00270100807814525,
                ('w_[rootn2]_[u]', 'w_[n2n3]_[u]'): 0.16357658392419,
                ('u_[n2n3]_[u]', 'u_[n3]_-'): 0.565293451090206
            }
        }

        decomposition = modelcreator_ecg_decomposition.Decomposition(req, sub, flow_values, 0.0001, 0.0001, 1e-10)
        mappings = decomposition.compute_mappings()
        assert len(mappings) == 3

        expected_flow_values = {0.40171686716601607, 0.16087557584604475, 0.00270100807814525}

        for m, flow, load in mappings:
            best_matching_value = min(
                (expected for expected in expected_flow_values),
                key=lambda expected: abs(expected - flow)
            )
            expected_flow_values.remove(best_matching_value)
            assert flow == pytest.approx(best_matching_value, rel=0.001)

    def test_handmade_example_containing_multiple_mappings(self):
        req = datamodel.Request("test_req")
        req.add_node("i", 1.0, "t1", allowed_nodes=["w"])
        req.add_node("j", 1.0, "t1", allowed_nodes=["v", "w"])
        req.add_node("k", 1.0, "t1", allowed_nodes=["u"])
        req.add_node("l", 1.0, "t1", allowed_nodes=["u", "v"])

        req.add_edge("i", "j", 1.0)
        req.add_edge("j", "k", 1.0)
        req.add_edge("i", "k", 1.0)
        req.add_edge("j", "l", 1.0)
        req.graph["root"] = "i"

        sub = datamodel.Substrate("test_sub")
        sub.add_node("u", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_node("v", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_node("w", ["t1"], {"t1": 100}, {"t1": 100})
        sub.add_edge("u", "v", 1000.0, bidirected=True)
        sub.add_edge("v", "w", 1000.0, bidirected=True)
        sub.add_edge("w", "u", 1000.0, bidirected=True)

        ext_graph = extendedcactusgraph.ExtendedCactusGraph(req, sub)

        # print util.get_graph_viz_string(ext_graph)
        # print "\n".join(str(e) for e in ext_graph.nodes)
        # print "\n".join(str(e) for e in ext_graph.edges)

        flow_values = {
            'embedding': 1.0,
            'node': {
                'i': {'w': 1.0},
                'j': {'v': 0.5, 'w': 0.5},
                'k': {'u': 1.0},
                'l': {'v': 0.5, 'u': 0.5}
            },
            'edge': {
                ('w_[i]_+', 'w_[ij]_[u]'): 1.0,
                ('w_[ij]_[u]', 'w_[jk]_[u]'): 0.5,
                ('w_[ij]_[u]', 'v_[ij]_[u]'): 0.5,
                ('v_[ij]_[u]', 'v_[jk]_[u]'): 0.5,
                ('v_[jk]_[u]', 'u_[jk]_[u]'): 0.5,
                ('w_[jk]_[u]', 'u_[jk]_[u]'): 0.5,
                ('u_[jk]_[u]', 'u_[k]_-'): 1.0,
                ('v_[j]_+', 'v_[jl]'): 0.5,
                ('w_[j]_+', 'w_[jl]'): 0.5,
                ('v_[jl]', 'v_[l]_-'): 0.5,
                ('w_[jl]', 'u_[jl]'): 0.5,
                ('u_[jl]', 'u_[l]_-'): 0.5,

                ('w_[i]_+', 'w_[ik]_[u]'): 1.0,
                ('w_[ik]_[u]', 'u_[ik]_[u]'): 1.0,
                ('u_[ik]_[u]', 'u_[k]_-'): 1.0,
            }
        }

        decomposition = modelcreator_ecg_decomposition.Decomposition(req, sub, flow_values, 0.0001, 0.0001,1e-10)
        mappings = decomposition.compute_mappings()
        assert len(mappings) == 2

        expected_flow_values = [0.5, 0.5]

        for m, flow, load in mappings:
            best_matching_value = min(
                (expected for expected in expected_flow_values),
                key=lambda expected: abs(expected - flow)
            )
            expected_flow_values.remove(best_matching_value)
            assert flow == pytest.approx(best_matching_value)
