from alib import scenariogeneration, test_utils
from vnep_approx import randomized_rounding


class TestRandomizedRounding:
    def setup(self):
        self.chain_gen = scenariogeneration.ServiceChainGenerator()

        scenario_parameters = {
            "maxrepetition": 1,
            "repetition": 1,
            "substrate_generation": {
                "foo": {
                    "TopologyZooReader": {
                        "topology": "Aarnet",
                        "node_types": ("t1", "t2"),
                        "node_cost_factor": 1.0,
                        "node_capacity": 100.0,
                        "edge_cost": 1.0,
                        "edge_capacity": 100.0,
                        "node_type_distribution": 0.2
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
                        "number_of_requests": 3,
                        "min_number_of_nodes": 2,
                        "max_number_of_nodes": 5,
                        "probability": 1,
                        "potential_nodes_factor": 0.1,
                        "node_resource_factor": 0.01,
                        "edge_resource_factor": 100,
                        "iterations": 10,
                        "arbitrary_edge_orientations": False,
                        "max_cycles": 20,
                        "layers": 4,
                        "normalize": True,
                        "fix_root_mapping": True,
                        "fix_leaf_mapping": True,
                        "branching_distribution": (0.0, 0.5, 0.3, 0.15, 0.05)
                    }
                }
            }
        }
        # scenario_generator = scenariogeneration.ScenarioGenerator("foo", self.chain_gen, profit_calculator=scenariogeneration.OptimalEmbeddingProfitCalculator(), top_zoo_path="../alib/data/topologyZoo")
        self.easy_scenario = scenariogeneration.build_scenario((1, scenario_parameters))[1]
        self.example_scenario = test_utils.get_example_scenario_from_paper()

    def test_can_solve_example_scenario(self):
        alg = randomized_rounding.RandomizedRounding(self.example_scenario)
        assert hasattr(alg, "mc")
        alg.init_model_creator()
        sol = alg.compute_integral_solution()
        assert alg.solution is not None
        assert sol is not None

    def test_can_solve_easy_real_world_scenario(self):
        alg = randomized_rounding.RandomizedRounding(self.easy_scenario)
        assert hasattr(alg, "mc")
        sol = alg.init_model_creator()
        sol = alg.compute_integral_solution()
        assert sol is not None
