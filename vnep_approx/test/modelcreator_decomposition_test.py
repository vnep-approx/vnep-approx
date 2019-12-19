__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'

from alib import datamodel, test_utils
from vnep_approx.deferred import modelcreator_decomposition as mc


class TestModelCreator:
    def setup(self):
        self.substrate = test_utils.get_test_substrate(3)
        self.request = test_utils.get_test_linear_request(3, "req1")
        self.request2 = test_utils.get_test_linear_request(3, "req2")
        self.scenario = datamodel.Scenario(
            "Sen1", self.substrate,
            [self.request, self.request2]
        )

    def test_fractual_model(self):

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
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        solution = modelcreator.compute_integral_solution()
        if solution:
            if solution.validate_solution():
                print solution

    def test_max_profit(self):
        self.request.profit = 5
        self.request2.profit = 6
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        solution = modelcreator.compute_integral_solution()
        if solution:
            if solution.validate_solution():
                print solution

    def test_linear_chain(self):
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        modelcreator.compute_integral_solution()

    def test_linear_chain_multipe_edges(self):
        modelcreator = mc.ModelCreatorDecomp(self.scenario)
        modelcreator.init_model_creator()
        modelcreator.compute_integral_solution()
