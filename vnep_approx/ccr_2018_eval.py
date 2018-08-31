# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import os
import sys

import pickle
import time
import treewidth_model_experiments
from collections import namedtuple

from alib import run_experiment, util, modelcreator, datamodel
from . import modelcreator_ecg_decomposition, randomized_rounding_triumvirate, treewidth_model
import evaluation_ifip_networking_2018

import gurobipy
from gurobipy import GRB, LinExpr

Param_MIPGap = "MIPGap"
Param_IterationLimit = "IterationLimit"
Param_NodeLimit = "NodeLimit"
Param_Heuristics = "Heuristics"
Param_Threads = "Threads"
Param_TimeLimit = "TimeLimit"
Param_MIPFocus = "MIPFocus"
Param_RootCutPasses = "CutPasses"
Param_Cuts = "Cuts"
Param_NodefileStart = "NodefileStart"
Param_NodeMethod = "NodeMethod"
Param_Method = "Method"
Param_BarConvTol = "BarConvTol"
Param_NumericFocus = "NumericFocus"


CCREvaluationInstance = namedtuple("CCREvaluationInstance", ["scenario",
                                                             "original_scenario_id_",
                                                             "randround_solution",
                                                             "LP_runtime",
                                                             "LPobjValue"])



INFINITY = float("inf")

class SeparationLP_DynVMP(object):


    def __init__(self,
                 ccr_eval_instance,
                 gurobi_settings=None,
                 logger=None):
        self.ccr_eval_instance = ccr_eval_instance
        self.scenario = self.ccr_eval_instance.scenario
        self.substrate = self.scenario.substrate
        self.requests = self.scenario.requests
        self.objective = self.scenario.objective

        if self.objective == datamodel.Objective.MAX_PROFIT:
            pass
        elif self.objective == datamodel.Objective.MIN_COST:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")
        else:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")

        self.gurobi_settings = gurobi_settings

        self.model = None  # the model of gurobi
        self.status = None  # GurobiStatus instance
        self.solution = None  # either a integral solution or a fractional one

        self.temporal_log = modelcreator.TemporalLog()

        self.time_preprocess = None
        self.time_optimization = None
        self.time_postprocessing = None
        self._time_postprocess_start = None

        if logger is None:
            self.logger = util.get_logger(__name__, make_file=False, propagate=True)
        else:
            self.logger = logger

    def init_model_creator(self):
        ''' Initializes the modelcreator by generating the model. Afterwards, model.compute() can be called to let
            Gurobi solve the model.

        :return:
        '''

        time_preprocess_start = time.time()

        self.model = gurobipy.Model("column_generation_is_smooth_af")
        self.model._mc = self

        self.model.setParam("LogToConsole", 1)

        if self.gurobi_settings is not None:
            self.apply_gurobi_settings(self.gurobi_settings)

        self.preprocess_input()
        self.create_empty_capacity_constraints()
        self.create_empty_request_embedding_bound_constraints()
        self.create_empty_objective()

        # initialle there are none!
        #self.create_variables()

        # for making the variables accessible
        self.model.update()

        self.time_preprocess = time.time() - time_preprocess_start

    def preprocess_input(self):

        if len(self.substrate.get_types()) > 1:
            raise ValueError("Can only handle a single node type.")

        self.node_type = list(self.substrate.get_types())[0]
        self.snodes = self.substrate.nodes
        self.sedges = self.substrate.edges

        self.node_capacities = {snode : self.substrate.get_node_type_capacity(snode, self.node_type) for snode in self.snodes}
        self.edge_capacities = {sedge : self.substrate.get_edge_capacity(sedge) for sedge in self.sedges}

        self.dual_costs_requests  = {req : 0 for req in self.requests}
        self.dual_costs_node_resources = {snode: 1 for snode in self.snodes}
        self.dual_costs_edge_resources = {sedge: 1 for sedge in self.sedges}

        self.tree_decomp_computation_times = {req : 0 for req in self.requests}
        self.tree_decomps = {req : None for req in self.requests}
        for req in self.requests:
            self.logger.debug("Computing tree decomposition for request {}".format(req))
            td_computation_time = time.time()
            td_comp = treewidth_model.TreeDecompositionComputation(req)
            tree_decomp = td_comp.compute_tree_decomposition()
            sntd = treewidth_model.SmallSemiNiceTDArb(tree_decomp, req)
            self.tree_decomps[req] = sntd
            self.tree_decomp_computation_times[req] = time.time() - td_computation_time
            self.logger.debug("\tdone.".format(req))

        self.dynvmp_instances = {req : None for req in self.requests}
        self.dynvmp_runtimes_initialization = {req: list() for req in self.requests}

        for req in self.requests:
            self.logger.debug("Initializing DynVMP Instance for request {}".format(req))
            dynvmp_init_time = time.time()
            opt_dynvmp = treewidth_model.OptimizedDynVMP(self.substrate,
                                                         req,
                                                         self.tree_decomps[req],
                                                         initial_snode_costs=self.dual_costs_node_resources,
                                                         initial_sedge_costs=self.dual_costs_edge_resources)
            opt_dynvmp.initialize_data_structures()
            self.dynvmp_runtimes_initialization[req].append(time.time()-dynvmp_init_time)
            self.dynvmp_instances[req] = opt_dynvmp
            self.logger.debug("\tdone.".format(req))

        self.mappings_of_requests = {req: list() for req in self.requests}
        self.dynvmp_runtimes_computation = {req: list() for req in self.requests}
        self.gurobi_runtimes = []

        self.mapping_variables = {req: list() for req in self.requests}


    def create_empty_capacity_constraints(self):
        self.capacity_constraints = {}
        for snode in self.snodes:
            self.capacity_constraints[snode] = self.model.addConstr(0, GRB.LESS_EQUAL, self.node_capacities[snode], name="capacity_node_{}".format(snode))
        for sedge in self.sedges:
            self.capacity_constraints[sedge] = self.model.addConstr(0, GRB.LESS_EQUAL, self.edge_capacities[sedge], name="capacity_edge_{}".format(sedge))

    def create_empty_request_embedding_bound_constraints(self):
        self.embedding_bound = {req : None for req in self.requests}
        for req in self.requests:
            self.embedding_bound[req] = self.model.addConstr(0, GRB.LESS_EQUAL, 1)

    def create_empty_objective(self):
        self.model.setObjective(0, GRB.MAXIMIZE)



    def introduce_new_columns(self, req, maximum_number_of_columns_to_introduce=None, cutoff=INFINITY):
        dynvmp_instance = self.dynvmp_instances[req]
        opt_cost = dynvmp_instance.get_optimal_solution_cost()
        current_new_allocations = []
        current_new_variables = []
        self.logger.debug("Objective when introucding new columns was {}".format(opt_cost))
        (costs, indices) = dynvmp_instance.get_ordered_root_solution_costs_and_mapping_indices(maximum_number_of_solutions_to_return=maximum_number_of_columns_to_introduce)
        mapping_list = dynvmp_instance.recover_list_of_mappings(indices)
        self.logger.debug("Will iterate mapping list {}".format(req.name))
        for index, mapping in enumerate(mapping_list):
            if costs[index] > cutoff:
                break
            #store mapping
            varname = "f_req[{}]_k[{}]".format(req.name, index+len(self.mappings_of_requests[req]))
            new_var = self.model.addVar(lb=0.0,
                                        ub=1.0,
                                        obj=req.profit,
                                        vtype=GRB.CONTINUOUS,
                                        name=varname)
            current_new_variables.append(new_var)

            #compute corresponding substrate allocation and store it
            mapping_allocations = self._compute_allocations(req, mapping)
            current_new_allocations.append(mapping_allocations)

        #make variables accessible
        self.model.update()
        for index, new_var in enumerate(current_new_variables):
            #handle allocations
            corresponding_allocation = current_new_allocations[index]
            for sres, alloc in corresponding_allocation.iteritems():
                constr = self.capacity_constraints[sres]
                self.model.chgCoeff(constr, new_var, alloc)

            self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)

        self.mappings_of_requests[req].extend(mapping_list[:len(current_new_variables)])
        self.mapping_variables[req].extend(current_new_variables)
        self.logger.debug("Introduced {} new mappings for {}".format(len(current_new_variables), req.name))


    def _compute_allocations(self, req, mapping):
        allocations = {}
        for reqnode in req.nodes:
            snode = mapping.mapping_nodes[reqnode]
            if snode in allocations:
                allocations[snode] += req.node[reqnode]['demand']
            else:
                allocations[snode] = req.node[reqnode]['demand']
        for reqedge in req.edges:
            path = mapping.mapping_edges[reqedge]
            for sedge in path:
                stail, shead = sedge
                if sedge in allocations:
                    allocations[sedge] += req.edge[reqedge]['demand']
                else:
                    allocations[sedge] = req.edge[reqedge]['demand']
        return allocations

    def perform_separation_and_introduce_new_columns(self):
        new_columns_generated = False

        for req in self.requests:
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None and opt_cost < 0.999*(req.profit - self.dual_costs_requests[req]):
                self.introduce_new_columns(req, maximum_number_of_columns_to_introduce=5, cutoff = req.profit-self.dual_costs_requests[req])
                new_columns_generated = True

        return new_columns_generated

    def update_dual_costs_and_reinit_dynvmps(self):
        #update dual costs
        for snode in self.snodes:
            self.dual_costs_node_resources[snode] = self.capacity_constraints[snode].Pi
        for sedge in self.sedges:
            self.dual_costs_edge_resources[sedge] = self.capacity_constraints[sedge].Pi
        for req in self.requests:
            self.dual_costs_requests[req] = self.embedding_bound[req].Pi

        #reinit dynvmps
        for req in self.requests:
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_reinit_time = time.time()
            dynvmp_instance.reinitialize(new_node_costs=self.dual_costs_node_resources,
                                         new_edge_costs=self.dual_costs_edge_resources)
            self.dynvmp_runtimes_initialization[req].append(time.time() - single_dynvmp_reinit_time)

    def compute_solution(self):
        ''' Abstract function computing an integral solution to the model (generated before).

        :return: Result of the optimization consisting of an instance of the GurobiStatus together with a result
                 detailing the solution computed by Gurobi.
        '''
        self.logger.info("Starting computing solution")
        # do the optimization
        time_optimization_start = time.time()

        #do the magic here

        for req in self.requests:
            self.logger.debug("Getting first mappings for request {}".format(req.name))
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                self.logger.debug("Introducing new columns for {}".format(req.name))
                self.introduce_new_columns(req, maximum_number_of_columns_to_introduce=100)

        self.model.update()

        new_columns_generated = True
        counter = 0
        while new_columns_generated:
            gurobi_runtime = time.time()
            self.model.optimize()
            self.gurobi_runtimes.append(time.time() - gurobi_runtime)
            self.model.write("temp_{}.lp".format(counter))
            self.update_dual_costs_and_reinit_dynvmps()

            new_columns_generated = self.perform_separation_and_introduce_new_columns()
            counter += 1

        self.time_optimization = time.time() - time_optimization_start

        # do the postprocessing
        self._time_postprocess_start = time.time()
        self.status = self.model.getAttr("Status")
        objVal = None
        objBound = GRB.INFINITY
        objGap = GRB.INFINITY
        solutionCount = self.model.getAttr("SolCount")

        if solutionCount > 0:
            objVal = self.model.getAttr("ObjVal")

        self.status = modelcreator.GurobiStatus(status=self.status,
                                                solCount=solutionCount,
                                                objValue=objVal,
                                                objGap=objGap,
                                                objBound=objBound,
                                                integralSolution=False)

        result = None
        if self.status.isFeasible():
            self.solution = self.recover_solution_from_variables()

        self.time_postprocessing = time.clock() - self._time_postprocess_start

        self.logger.debug("Preprocessing time:   {}".format(self.time_preprocess))
        self.logger.debug("Optimization time:    {}".format(self.time_optimization))
        self.logger.debug("Postprocessing time:  {}".format(self.time_postprocessing))

        return result

    def recover_solution_from_variables(self):
        pass


    ###
    ###     GUROBI SETTINGS
    ###

    _listOfUserVariableParameters = [Param_MIPGap, Param_IterationLimit, Param_NodeLimit, Param_Heuristics,
                                     Param_Threads, Param_TimeLimit, Param_Cuts, Param_MIPFocus,
                                     Param_RootCutPasses,
                                     Param_NodefileStart, Param_Method, Param_NodeMethod, Param_BarConvTol,
                                     Param_NumericFocus]

    def apply_gurobi_settings(self, gurobiSettings):
        ''' Apply gurobi settings.

        :param gurobiSettings:
        :return:
        '''


        if gurobiSettings.MIPGap is not None:
            self.set_gurobi_parameter(Param_MIPGap, gurobiSettings.MIPGap)
        else:
            self.reset_gurobi_parameter(Param_MIPGap)

        if gurobiSettings.IterationLimit is not None:
            self.set_gurobi_parameter(Param_IterationLimit, gurobiSettings.IterationLimit)
        else:
            self.reset_gurobi_parameter(Param_IterationLimit)

        if gurobiSettings.NodeLimit is not None:
            self.set_gurobi_parameter(Param_NodeLimit, gurobiSettings.NodeLimit)
        else:
            self.reset_gurobi_parameter(Param_NodeLimit)

        if gurobiSettings.Heuristics is not None:
            self.set_gurobi_parameter(Param_Heuristics, gurobiSettings.Heuristics)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.Threads is not None:
            self.set_gurobi_parameter(Param_Threads, gurobiSettings.Threads)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.TimeLimit is not None:
            self.set_gurobi_parameter(Param_TimeLimit, gurobiSettings.TimeLimit)
        else:
            self.reset_gurobi_parameter(Param_TimeLimit)

        if gurobiSettings.MIPFocus is not None:
            self.set_gurobi_parameter(Param_MIPFocus, gurobiSettings.MIPFocus)
        else:
            self.reset_gurobi_parameter(Param_MIPFocus)

        if gurobiSettings.cuts is not None:
            self.set_gurobi_parameter(Param_Cuts, gurobiSettings.cuts)
        else:
            self.reset_gurobi_parameter(Param_Cuts)

        if gurobiSettings.rootCutPasses is not None:
            self.set_gurobi_parameter(Param_RootCutPasses, gurobiSettings.rootCutPasses)
        else:
            self.reset_gurobi_parameter(Param_RootCutPasses)

        if gurobiSettings.NodefileStart is not None:
            self.set_gurobi_parameter(Param_NodefileStart, gurobiSettings.NodefileStart)
        else:
            self.reset_gurobi_parameter(Param_NodefileStart)

        if gurobiSettings.Method is not None:
            self.set_gurobi_parameter(Param_Method, gurobiSettings.Method)
        else:
            self.reset_gurobi_parameter(Param_Method)

        if gurobiSettings.NodeMethod is not None:
            self.set_gurobi_parameter(Param_NodeMethod, gurobiSettings.NodeMethod)
        else:
            self.reset_gurobi_parameter(Param_NodeMethod)

        if gurobiSettings.BarConvTol is not None:
            self.set_gurobi_parameter(Param_BarConvTol, gurobiSettings.BarConvTol)
        else:
            self.reset_gurobi_parameter(Param_BarConvTol)

        if gurobiSettings.NumericFocus is not None:
            self.set_gurobi_parameter(Param_NumericFocus, gurobiSettings.NumericFocus)
        else:
            self.reset_gurobi_parameter(Param_NumericFocus)

    def reset_all_parameters_to_default(self):
        for param in self._listOfUserVariableParameters:
            (name, type, curr, min, max, default) = self.model.getParamInfo(param)
            self.model.setParam(param, default)

    def reset_gurobi_parameter(self, param):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Parameter {} unchanged".format(param))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        self.model.setParam(param, default)

    def set_gurobi_parameter(self, param, value):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Changed value of parameter {} to {}".format(param, value))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.setParam(param, value)

    def getParam(self, param):
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.getParam(param)