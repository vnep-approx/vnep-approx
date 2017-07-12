__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'

from alib import datamodel, test_utils
from vnep_approx import modelcreator_decomposition as mc


class TestModelCreator:
    def setup(self):
        self.substrate = test_utils.get_test_substrate(3)
        self.request = test_utils.get_test_linear_request(3, "req1")
        self.request2 = test_utils.get_test_linear_request(3, "req2")
        self.scenario = datamodel.Scenario(
            "Sen1", self.substrate,
            [self.request, self.request2]
        )

    def test_init_model(self):
        # REQUEST LATENCY
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert latency == 5
        # mc = mc.ModelCreator(self.scenario)
        # mc.init_model_creator()
        # mc.compute_integral_solution()

    def test_fractual_model(self):
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert self.request.get_latency_requirement(path) == 5

        self.scenario = datamodel.Scenario("Sen1", self.substrate,
                                           [self.request])

        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        solution = modelcreator.compute_fractional_solution()
        assert self.request in solution.request_mapping
        print "profile:  time pre = {}, optimization = {} , post = {}".format(
            modelcreator.time_preprocess, modelcreator.time_optimization,
            modelcreator.time_postprocessing)
        if solution:
            if solution.validate_solution():
                print solution

    def test_minimize_cost(self):
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert self.request.get_latency_requirement(path) == 5
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        solution = modelcreator.compute_integral_solution()
        if solution:
            if solution.validate_solution():
                print solution

    def test_max_profit(self):
        self.request.profit = 5
        self.request2.profit = 6
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert latency == 5
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        solution = modelcreator.compute_integral_solution()
        if solution:
            if solution.validate_solution():
                print solution

    def test_linear_chain(self):
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert self.request.get_latency_requirement(path) == 5
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        modelcreator.compute_integral_solution()

    def test_linear_chain_multipe_edges(self):
        path = list(self.request.get_edges())
        self.request.add_latency_requirement(path, 5)
        latency = self.request.get_latency_requirement(path)
        assert self.request.get_latency_requirement(path) == 5
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        modelcreator.compute_integral_solution()
