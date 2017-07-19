# MIT License
#
# Copyright (c) 2016-2017 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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

import time

from gurobipy import GRB, LinExpr

from alib import solutions, util, modelcreator
from . import extendedcactusgraph


class CactusDecompositionError(Exception): pass


class DecompositionResult(modelcreator.AlgorithmResult):
    def __init__(self, solution, temporal_log, solution_status):
        super(DecompositionResult, self).__init__()
        self.solution = solution
        self.temporal_log = temporal_log
        self.status = solution_status

    def get_solution(self):
        return self.solution

    def _cleanup_references_raw(self, original_scenario):
        own_scenario = self.solution.scenario
        for i, own_req in enumerate(own_scenario.requests):
            mapping = self.solution.request_mapping[own_req]
            del self.solution.request_mapping[own_req]
            original_request = original_scenario.requests[i]
            mapping.request = original_request
            mapping.substrate = original_scenario.substrate
            self.solution.request_mapping[original_request] = mapping
        self.solution.scenario = original_scenario


class ModelCreatorCactusDecomposition(modelcreator.AbstractEmbeddingModelCreator):
    ALGORITHM_ID = "CactusDecomposition"

    def __init__(self, scenario, gurobi_settings=None, logger=None):
        super(ModelCreatorCactusDecomposition, self).__init__(scenario=scenario, gurobi_settings=gurobi_settings, logger=logger)

        self._originally_allowed_nodes = {}
        self.extended_graphs = {}

        self.ext_graph_edges_node = {}
        self.ext_graph_edges_edge = {}

        self.var_node_flow = {}  # f+(r, i, u)
        self.var_edge_flow = {}  # f(r, e)
        self.var_request_load = {}  # l(r, x, y)

        # Used in solution extraction:
        # self._remaining_flow = None  # dictionary storing the remaining on a given extended graph resource during the decomposition algorithm
        self._used_flow = None
        self.fractional_decomposition_accuracy = 0.0001
        self.fractional_decomposition_abortion_flow = 0.001
        self._start_time_recovering_fractional_solution = None
        self._end_time_recovering_fractional_solution = None
        self.lost_flow_in_the_decomposition = 0.0

        # paper mode means that we gracefully accept failures while logging them!
        self.paper_mode = True

    def preprocess_input(self):
        modelcreator.AbstractEmbeddingModelCreator.preprocess_input(self)

        # filter node placement according to substrate capacities create extended graphs
        for req in self.requests:
            self.extended_graphs[req] = extendedcactusgraph.ExtendedCactusGraph(req, self.scenario.substrate)

    def create_variables_other_than_embedding_decision_and_request_load(self):
        # source/sink node flow induction variables
        for req in self.requests:
            self.var_node_flow[req] = {}
            ext_graph = self.extended_graphs[req]
            if ext_graph is None:
                continue
            for i, ecg_node_dict in ext_graph.source_nodes.iteritems():
                if i not in self.var_node_flow[req]:
                    self.var_node_flow[req][i] = {}
                for u, ecg_node in ecg_node_dict.iteritems():
                    if u in self.var_node_flow[req][i]:
                        continue
                    variable_id = modelcreator.construct_name("flow_induction", req_name=req.name, vnode=i, snode=u)
                    self.var_node_flow[req][i][u] = self.model.addVar(lb=0.0,
                                                                      ub=1.0,
                                                                      obj=0.0,
                                                                      vtype=GRB.BINARY,
                                                                      name=variable_id)
            for i, ecg_node_dict in ext_graph.sink_nodes.iteritems():
                if i not in self.var_node_flow[req]:
                    self.var_node_flow[req][i] = {}
                for u, ecg_node in ecg_node_dict.iteritems():
                    if u in self.var_node_flow[req][i]:
                        continue
                    variable_id = modelcreator.construct_name("flow_induction", req_name=req.name, vnode=i, snode=u)
                    self.var_node_flow[req][i][u] = self.model.addVar(lb=0.0,
                                                                      ub=1.0,
                                                                      obj=0.0,
                                                                      vtype=GRB.BINARY,
                                                                      name=variable_id)
        # edge flow variables
        for req in self.requests:
            self.var_edge_flow[req] = {}
            ext_graph = self.extended_graphs[req]
            if ext_graph is None:
                continue
            for ecg_edge in ext_graph.edges:
                variable_id = modelcreator.construct_name("flow_edge", req_name=req.name, other=ecg_edge)
                self.var_edge_flow[req][ecg_edge] = self.model.addVar(lb=0.0,
                                                                      ub=1.0,
                                                                      obj=0.0,
                                                                      vtype=GRB.BINARY,
                                                                      name=variable_id)

        self.model.update()

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        for req in self.requests:
            if self.extended_graphs[req] is None:
                self.logger.info("Fixing request {} to zero, as it contains unembeddable nodes.".format(req.name))
                expr = LinExpr([(1.0, self.var_embedding_decision[req])])
                constr_name = modelcreator.construct_name("fix_infeasible_req_to_zero", req_name=req.name)
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
                continue

            # flow induction at root - constraint 8
            self._add_constraint_root_flow_induction(req)

            # flow induction for cycles (sources) - constraint 9
            self._add_constraint_cycle_flow_induction_at_source(req)

            # same flow in each cycle branch (commutativity) - constraint 10
            self._add_constraint_cycle_flow_commutativity(req)

            # flow induction for paths (sources) - constraint 11
            self._add_constraint_path_flow_induction_at_source(req)

            # flow preservation - constraint 12
            self._add_constraint_flow_preservation(req)

            # flow induction for cycles (sinks) - constraint 13
            self._add_constraint_cycle_flow_induction_at_sink(req)

            # flow induction for paths (sinks) - constraint 14
            self._add_constraint_path_flow_induction_at_sink(req)

            # flow inductions for subgraphs branching off from a cycle - constraint 15
            self._add_constraint_branching_flow_induction(req)

            # load computation for node resources - constraint 16
            self._add_constraint_track_node_load(req)

            # load computation for edge resources - constraint 17
            self._add_constraint_track_edge_load(req)

            # capacity constraint - constraint 18
            # already included by the AbstractEmbeddingModelCreator

        self.model.update()

    def _add_constraint_root_flow_induction(self, req):
        ext_graph = self.extended_graphs[req]
        root = ext_graph.root
        root_source_nodes = ext_graph.source_nodes[root].keys()  # this list contains all source nodes associated with the request root

        expr = LinExpr([(-1.0, self.var_embedding_decision[req])] +
                       [(1.0, self.var_node_flow[req][root][u]) for u in root_source_nodes])

        constr_name = modelcreator.construct_name("flow_induction_root", req_name=req.name)
        # print "Root flow induction ({}):".format(root).ljust(40), expr, "= 0"
        self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_cycle_flow_induction_at_source(self, req):
        ext_graph = self.extended_graphs[req]
        for cycle in ext_graph.ecg_cycles:
            valid_end_nodes = self.substrate.get_valid_nodes(req.get_type(cycle.end_node), req.get_node_demand(cycle.end_node))
            valid_cycle_targets = set(req.get_allowed_nodes(cycle.end_node)).intersection(valid_end_nodes)

            start_type = req.get_type(cycle.start_node)
            start_demand = req.get_node_demand(cycle.start_node)
            valid_start_nodes = self.substrate.get_valid_nodes(start_type, start_demand)

            for u in req.get_allowed_nodes(cycle.start_node):
                if u not in valid_start_nodes:
                    continue
                cycle_source = ext_graph.source_nodes[cycle.start_node][u]
                ij = cycle.original_branches[0][0]  # select the first edge of the first branch of the cycle
                if ij[0] != cycle.start_node:
                    raise CactusDecompositionError("Sanity check")
                expr = [(-1.0, self.var_node_flow[req][cycle.start_node][u])]
                for w in valid_cycle_targets:
                    ext_edge = (cycle_source, ext_graph.cycle_layer_nodes[ij][u][w])
                    expr.append((1.0, self.var_edge_flow[req][ext_edge]))
                expr = LinExpr(expr)
                constr_name = modelcreator.construct_name("cycle_flow_induction_source", req_name=req.name, other="{}->{}".format(cycle.start_node, cycle.end_node))
                # print "Cycle flow induction ({}):".format(cycle_source).ljust(40), expr, "= 0"
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_cycle_flow_commutativity(self, req):
        ext_graph = self.extended_graphs[req]
        for cycle in ext_graph.ecg_cycles:
            valid_end_nodes = self.substrate.get_valid_nodes(req.get_type(cycle.end_node), req.get_node_demand(cycle.end_node))
            valid_cycle_targets = set(req.get_allowed_nodes(cycle.end_node)).intersection(valid_end_nodes)

            start_type = req.get_type(cycle.start_node)
            start_demand = req.get_node_demand(cycle.start_node)
            valid_start_nodes = self.substrate.get_valid_nodes(start_type, start_demand)

            for u in req.get_allowed_nodes(cycle.start_node):
                if u not in valid_start_nodes:
                    continue
                cycle_source = ext_graph.source_nodes[cycle.start_node][u]
                ij = cycle.original_branches[0][0]  # select the first edge of the first branch of the cycle
                ik = cycle.original_branches[1][0]  # select the first edge of the second branch of the cycle
                if ij[0] != cycle.start_node or ik[0] != cycle.start_node:
                    raise CactusDecompositionError("Sanity check")

                for w in valid_cycle_targets:
                    u_ij = ext_graph.cycle_layer_nodes[ij][u][w]
                    u_ik = ext_graph.cycle_layer_nodes[ik][u][w]
                    expr = LinExpr([
                        (1.0, self.var_edge_flow[req][(cycle_source, u_ij)]),
                        (-1.0, self.var_edge_flow[req][(cycle_source, u_ik)])
                    ])
                    constr_name = modelcreator.construct_name("flow_commutativity",
                                                              req_name=req.name,
                                                              vedge="{}->{}, {}->{}".format(ij[0], ij[1], ik[0], ik[1]),
                                                              snode="s={}, t={}".format(u, w))
                    # print "Cycle flow commutativity ({}, {}):".format(cycle_source, w).ljust(40), expr, "= 0"
                    self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_path_flow_induction_at_source(self, req):
        ext_graph = self.extended_graphs[req]
        for path in ext_graph.ecg_paths:
            start_type = req.get_type(path.start_node)
            start_demand = req.get_node_demand(path.start_node)
            valid_start_nodes = self.substrate.get_valid_nodes(start_type, start_demand)

            for u in req.get_allowed_nodes(path.start_node):
                if u not in valid_start_nodes:
                    continue
                path_source = ext_graph.source_nodes[path.start_node][u]
                ij = path.original_path[0]  # select the first edge of the first branch of the cycle
                ext_ij = (path_source, ext_graph.path_layer_nodes[ij][u])
                expr = LinExpr([
                    (1.0, self.var_edge_flow[req][ext_ij]),
                    (-1.0, self.var_node_flow[req][path.start_node][u])
                ])
                constr_name = modelcreator.construct_name("path_flow_induction_source",
                                                          req_name=req.name,
                                                          vedge="{}->{}".format(ij[0], ij[1]),
                                                          snode="s={}".format(u))
                # print "Path flow induction ({}):".format(path_source).ljust(40), expr, "= 0"
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_flow_preservation(self, req):
        ext_graph = self.extended_graphs[req]
        for node in ext_graph.layer_nodes:
            expr = []
            for neighbor in ext_graph.get_out_neighbors(node):
                edge = node, neighbor
                expr.append((1.0, self.var_edge_flow[req][edge]))
            for neighbor in ext_graph.get_in_neighbors(node):
                edge = neighbor, node
                expr.append((-1.0, self.var_edge_flow[req][edge]))
            expr = LinExpr(expr)
            constr_name = modelcreator.construct_name("flow_preservation",
                                                      other=node)
            self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
            # print "Flow Preservation ({}):".format(node)[:37].ljust(40), expr, "= 0"

    def _add_constraint_cycle_flow_induction_at_sink(self, req):
        ext_graph = self.extended_graphs[req]
        for cycle in ext_graph.ecg_cycles:
            valid_cycle_end_nodes = self.substrate.get_valid_nodes(
                req.get_type(cycle.end_node), req.get_node_demand(cycle.end_node)
            )
            for w in req.get_allowed_nodes(cycle.end_node):
                if w not in valid_cycle_end_nodes:
                    continue
                ij = cycle.original_branches[0][-1]  # select the last edge of the first branch of the cycle
                if ij[1] != cycle.end_node:
                    raise CactusDecompositionError("Sanity check")
                cycle_sink = ext_graph.sink_nodes[cycle.end_node][w]
                ext_ij = (ext_graph.cycle_layer_nodes[ij][w][w], cycle_sink)
                expr = LinExpr([
                    (1.0, self.var_node_flow[req][cycle.end_node][w]),
                    (-1.0, self.var_edge_flow[req][ext_ij])
                ])
                constr_name = modelcreator.construct_name("cycle_flow_induction_source", req_name=req.name, other="{}->{}".format(cycle.start_node, cycle.end_node))
                # print "Cycle flow induction ({}):".format(cycle_sink).ljust(40), expr, "= 0"
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_path_flow_induction_at_sink(self, req):
        ext_graph = self.extended_graphs[req]
        for path in ext_graph.ecg_paths:
            valid_end_nodes = self.substrate.get_valid_nodes(
                req.get_type(path.end_node), req.get_node_demand(path.end_node)
            )

            for w in req.get_allowed_nodes(path.end_node):
                if w not in valid_end_nodes:
                    continue
                path_sink = ext_graph.sink_nodes[path.end_node][w]
                ij = path.original_path[-1]  # select the first edge of the first branch of the cycle
                ext_ij = (ext_graph.path_layer_nodes[ij][w], path_sink)
                expr = LinExpr([
                    (1.0, self.var_edge_flow[req][ext_ij]),
                    (-1.0, self.var_node_flow[req][path.end_node][w])
                ])
                constr_name = modelcreator.construct_name("path_flow_induction_sink",
                                                          req_name=req.name,
                                                          vedge="{}->{}".format(ij[0], ij[1]),
                                                          snode="s={}".format(w))
                # print "Path flow induction ({}):".format(path_sink).ljust(40), expr, "= 0"
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_branching_flow_induction(self, req):
        ext_graph = self.extended_graphs[req]
        for cycle in ext_graph.ecg_cycles:
            cycle_targets = req.get_allowed_nodes(cycle.end_node)

            for branch in cycle.original_branches:
                for pos, ij in enumerate(branch):
                    (i, j) = ij
                    if j not in ext_graph.cycle_branch_nodes:
                        continue
                    # j must be a branching node!
                    jl = branch[pos + 1]
                    if jl[0] != j:
                        raise CactusDecompositionError("Sanity Check!")
                    for u in req.get_allowed_nodes(j):
                        expr = [(-1.0, self.var_node_flow[req][j][u])]
                        for w in cycle_targets:
                            u_ij = ext_graph.cycle_layer_nodes[ij][u][w]
                            u_jl = ext_graph.cycle_layer_nodes[jl][u][w]
                            ext_edge = (u_ij, u_jl)
                            expr.append((1.0, self.var_edge_flow[req][ext_edge]))
                        expr = LinExpr(expr)
                        constr_name = modelcreator.construct_name("branch_flow_induction",
                                                                  other=j)
                        self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
                        # print "Branch flow induction ({}):".format(j)[:37].ljust(40), expr, "= 0"

    def _add_constraint_track_node_load(self, req):
        ext_graph = self.extended_graphs[req]
        constraints = {node_resource: [(-1.0, self.var_request_load[req][node_resource])]
                       for node_resource in self.substrate.substrate_node_resources}
        source_sink_request_nodes = set()
        for path in ext_graph.ecg_paths:
            # Paths are currently only 1 edge long
            if path.start_node not in ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(path.start_node)
            if path.end_node not in ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(path.end_node)

        for cycle in ext_graph.ecg_cycles:
            if cycle.start_node not in ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(cycle.start_node)
            if cycle.end_node not in ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(cycle.end_node)

            valid_cycle_end_nodes = self.substrate.get_valid_nodes(
                req.get_type(cycle.end_node), req.get_node_demand(cycle.end_node)
            )

            for branch in cycle.original_branches:
                for pos, ij in enumerate(branch[:-1]):
                    i, j = ij
                    j_type = req.get_type(j)
                    j_demand = req.get_node_demand(j)
                    valid_substrate_nodes = self.substrate.get_valid_nodes(j_type, j_demand)
                    for u in req.get_allowed_nodes(j):
                        if u not in valid_substrate_nodes:
                            continue
                        for w in req.get_allowed_nodes(cycle.end_node):
                            if w not in valid_cycle_end_nodes:
                                continue
                            jl = branch[pos + 1]
                            u_ij = ext_graph.cycle_layer_nodes[ij][u][w]
                            u_jl = ext_graph.cycle_layer_nodes[jl][u][w]
                            ext_edge = u_ij, u_jl
                            constraints[j_type, u].append((j_demand, self.var_edge_flow[req][ext_edge]))
        for i in source_sink_request_nodes:
            i_type = req.get_type(i)
            i_demand = req.get_node_demand(i)
            valid_substrate_nodes = self.substrate.get_valid_nodes(i_type, i_demand)
            for u in req.get_allowed_nodes(i):
                if u not in valid_substrate_nodes:
                    continue
                constraints[(i_type, u)].append((i_demand, self.var_node_flow[req][i][u]))
        for key, expr in constraints.iteritems():
            expr = LinExpr(expr)
            tau, u = key
            constr_name = modelcreator.construct_name("substrate_track_node_load",
                                                      snode=u,
                                                      type=tau)
            self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
            # print "Node load ({}, {}):".format(u, tau)[:37].ljust(40), expr, "= 0"

    def _add_constraint_track_edge_load(self, req):
        ext_graph = self.extended_graphs[req]
        constraints = {edge_resource: [(-1.0, self.var_request_load[req][edge_resource])]
                       for edge_resource in self.substrate.substrate_edge_resources}
        for path in ext_graph.ecg_paths:
            for ij in path.original_path:
                i, j = ij
                if (j, i) in ext_graph.reversed_request_edges:
                    valid_substrate_edges = self.substrate.get_valid_edges(req.get_edge_demand((j, i)))
                else:
                    valid_substrate_edges = self.substrate.get_valid_edges(req.get_edge_demand(ij))
                for u, v in valid_substrate_edges:  # TODO replace with allowed_edges when they are implemented
                    u_ij = ext_graph.path_layer_nodes[ij][u]
                    v_ij = ext_graph.path_layer_nodes[ij][v]
                    if (j, i) in ext_graph.reversed_request_edges:
                        ext_edge = v_ij, u_ij
                        demand = req.edge[(j, i)]["demand"]
                    else:
                        ext_edge = u_ij, v_ij
                        demand = req.edge[ij]["demand"]
                    constraints[(u, v)].append((demand, self.var_edge_flow[req][ext_edge]))
        for cycle in ext_graph.ecg_cycles:
            valid_cycle_end_nodes = self.substrate.get_valid_nodes(
                req.get_type(cycle.end_node), req.get_node_demand(cycle.end_node)
            )

            for branch in cycle.original_branches:
                for ij in branch:
                    i, j = ij
                    if (j, i) in ext_graph.reversed_request_edges:
                        valid_substrate_edges = self.substrate.get_valid_edges(req.get_edge_demand((j, i)))
                    else:
                        valid_substrate_edges = self.substrate.get_valid_edges(req.get_edge_demand(ij))
                    for u, v in valid_substrate_edges:
                        for w in req.get_allowed_nodes(cycle.end_node):
                            if w not in valid_cycle_end_nodes:
                                continue
                            u_ij = ext_graph.cycle_layer_nodes[ij][u][w]
                            v_ij = ext_graph.cycle_layer_nodes[ij][v][w]
                            if (j, i) in ext_graph.reversed_request_edges:
                                ext_edge = v_ij, u_ij
                                demand = req.get_edge_demand((j, i))
                            else:
                                ext_edge = u_ij, v_ij
                                demand = req.get_edge_demand(ij)
                            constraints[(u, v)].append((demand, self.var_edge_flow[req][ext_edge]))

        for key, expr in constraints.iteritems():
            expr = LinExpr(expr)
            u, v = key
            constr_name = modelcreator.construct_name("substrate_track_edge_load",
                                                      snode=u,
                                                      type=v)
            self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
            # print "Edge load ({}, {}):".format(u, v)[:37].ljust(40), expr, "= 0"

    def recover_fractional_solution_from_variables(self):
        self.logger.info("Recovering fractional solution for {}".format(self.scenario.name))

        flow_dict_start_time = time.clock()
        flow_values = self.make_flow_variable_dict()
        flow_dict_end_time = time.clock()

        # Since the generation of the flow variable dictionary
        # is an implementation detail to separate the modelcreator
        # from the decomposition and not part of the algorithm,
        # we exclude it from the post processing time.

        self._time_postprocess_start += (flow_dict_end_time - flow_dict_start_time)
        self._start_time_recovering_fractional_solution = flow_dict_end_time

        self.solution = solutions.FractionalScenarioSolution(
            "{}_fractional_solution".format(self.scenario.name),
            self.scenario
        )
        for req in self.requests:
            d = Decomposition(req,
                              self.substrate,
                              flow_values[req.name],
                              self.fractional_decomposition_abortion_flow,
                              self.fractional_decomposition_accuracy,
                              extended_graph=self.extended_graphs[req],
                              substrate_resources=self.substrate.substrate_resources,  # TODO: this is redundant, we could rewrite the Decomposition so that it receives a substrateX
                              logger=self.logger)
            mapping_flow_load_list = d.compute_mappings()
            self.lost_flow_in_the_decomposition = d.lost_flow_in_the_decomposition
            for mapping, flow, load in mapping_flow_load_list:
                self.solution.add_mapping(req, mapping, flow, load)

        self._end_time_recovering_fractional_solution = time.clock()
        self.logger.info("Time to extract flow values from model:         {:.5f}".format(flow_dict_end_time - flow_dict_start_time))
        self.logger.info("Time to decompose solution from flow variables: {:.5f}".format(
            self._end_time_recovering_fractional_solution - flow_dict_end_time
        ))
        return self.solution

    def post_process_fractional_computation(self):
        return self.solution

    def recover_integral_solution_from_variables(self):
        name = modelcreator.construct_name("solution_", sol_name=self.scenario.name)
        solution = solutions.IntegralScenarioSolution(name, self.scenario)
        # idea: reuse the code for the fractional solution: There should be at most one mapping with probability ~1
        fractional_solution = self.recover_fractional_solution_from_variables()
        for req in self.scenario.requests:
            mapping_name = modelcreator.construct_name("mapping_", req_name=req.name)
            mapping = solutions.Mapping(mapping_name, req, self.substrate, False)
            if req in fractional_solution.request_mapping:
                mapping_list = fractional_solution.request_mapping[req]
                if mapping_list:
                    most_likely_mapping = max(mapping_list, key=lambda m: fractional_solution.mapping_flows[m.name])
                    embedding_value = fractional_solution.mapping_flows[most_likely_mapping.name]
                    if abs(embedding_value - 1.0) > 0.001:
                        raise ModelCreatorCactusDecomposition("Could not find integral mapping for {}")
                    mapping.mapping_nodes = most_likely_mapping.mapping_nodes
                    mapping.mapping_edges = most_likely_mapping.mapping_edges
                    mapping.is_embedded = True
            solution.add_mapping(req, mapping)
        return solution

    def post_process_integral_computation(self):
        return DecompositionResult(self.solution, self.temporal_log, self.status)

    def make_flow_variable_dict(self):
        result = {}
        for req in self.requests:
            result[req.name] = {
                "embedding": self.var_embedding_decision[req].X,
                "node": {
                    i: {u: var.X for (u, var) in u_var_dict.iteritems() if var.X != 0.0}
                    for (i, u_var_dict) in self.var_node_flow[req].iteritems()
                },
                "edge": {
                    ext_edge: var.X for (ext_edge, var) in self.var_edge_flow[req].iteritems() if var.X != 0.0
                }
            }
        return result

    def fix_mapping_variables_according_to_integral_solution(self, solution):
        if not isinstance(solution, solutions.IntegralScenarioSolution):
            msg = "Expected solutions.IntegralScenarioSolution instance, received {} of type {}".format(
                solution, type(solution)
            )
            raise TypeError(msg)
        if solution.scenario is not self.scenario:
            msg = "This method requires that the solution is based on the same scenario as the Modelcreator."
            raise CactusDecompositionError(msg)

        self.logger.info("Fixing mapping variables according to solution {}".format(solution.name))
        for req in self.requests:
            ext_graph = self.extended_graphs[req]
            mapping = solution.request_mapping[req]
            self._fix_embedding_variable(req, mapping)
            if not mapping.is_embedded:
                continue

            self._fix_mapping_of_source_sink_nodes(req, ext_graph, mapping)

            for path in ext_graph.ecg_paths:
                self._fix_mapping_of_path_according_to_mapping(req, path, mapping)
            # fix mappings of cycle layer nodes
            for cycle in ext_graph.ecg_cycles:
                self._fix_mapping_of_cycle_according_to_mapping(req, cycle, mapping)

    def _fix_embedding_variable(self, req, mapping):
        force_embedding_constraint = LinExpr([(1.0, self.var_embedding_decision[req])])
        name = modelcreator.construct_name("force_embedding", req_name=req.name)
        if mapping.is_embedded:
            self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 1.0, name=name)
        else:
            self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 0.0, name=name)

    def _fix_mapping_of_source_sink_nodes(self, req, ext_graph, mapping):
        # fix mappings of source/sink nodes
        for i, u in mapping.mapping_nodes.iteritems():
            if i in ext_graph.source_nodes:
                force_embedding_constraint = LinExpr([(1.0, self.var_node_flow[req][i][u])])
                name = modelcreator.construct_name("force_mapping", req_name=req.name, snode=u, vnode=i)
                self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 1.0, name=name)
            if i in ext_graph.sink_nodes:
                force_embedding_constraint = LinExpr([(1.0, self.var_node_flow[req][i][u])])
                name = modelcreator.construct_name("force_mapping", req_name=req.name, snode=u, vnode=i)
                self.model.addConstr(force_embedding_constraint, GRB.EQUAL, 1.0, name=name)

    def _fix_mapping_of_path_according_to_mapping(self, req, path, mapping):
        ext_graph = self.extended_graphs[req]
        for ij in path.original_path:
            i, j = ij

            u_i = mapping.mapping_nodes[i]
            v_j = mapping.mapping_nodes[j]

            source = ext_graph.source_nodes[i][u_i]
            u_ext = ext_graph.path_layer_nodes[ij][u_i]
            v_ext = ext_graph.path_layer_nodes[ij][v_j]
            sink = ext_graph.sink_nodes[j][v_j]

            i_node_mapping_constraint = LinExpr([(1.0, self.var_edge_flow[req][(source, u_ext)])])
            name = modelcreator.construct_name("force_mapping", req_name=req.name, snode=u_i, vnode=i)
            self.model.addConstr(i_node_mapping_constraint, GRB.EQUAL, 1.0, name=name)
            j_node_mapping_constraint = LinExpr([(1.0, self.var_edge_flow[req][(v_ext, sink)])])
            name = modelcreator.construct_name("force_mapping", req_name=req.name, snode=v_j, vnode=j)
            self.model.addConstr(j_node_mapping_constraint, GRB.EQUAL, 1.0, name=name)

            # fix the edges
            is_reversed_edge = ij in ext_graph.reversed_request_edges or (j, i) in ext_graph.reversed_request_edges

            if is_reversed_edge:  # need to reverse mapped path
                uv_list = [(v, u) for (u, v) in reversed(mapping.mapping_edges[j, i])]
            else:
                uv_list = mapping.mapping_edges[ij]
            if not uv_list:
                continue
            for (u, v) in self.substrate.edges:
                if (u, v) not in self.substrate.get_valid_edges(req.get_edge_demand(ij)):
                    continue
                if is_reversed_edge:
                    u, v = v, u
                u_ext = ext_graph.path_layer_nodes[ij][u]
                v_ext = ext_graph.path_layer_nodes[ij][v]
                ij_uv_constraint = LinExpr([(1.0, self.var_edge_flow[req][(u_ext, v_ext)])])
                name = modelcreator.construct_name("force_mapping", req_name=req.name, sedge=(u, v), vedge=ij)
                if (u, v) in uv_list:
                    self.model.addConstr(ij_uv_constraint, GRB.EQUAL, 1.0, name=name)
                else:
                    self.model.addConstr(ij_uv_constraint, GRB.EQUAL, 0.0, name=name)

    def _fix_mapping_of_cycle_according_to_mapping(self, req, cycle, mapping):
        ext_graph = self.extended_graphs[req]
        s = cycle.start_node
        t = cycle.end_node
        w = mapping.mapping_nodes[t]
        for branch in cycle.original_branches:
            connection_previous_layer = ext_graph.source_nodes[s][mapping.mapping_nodes[s]]
            for ij in branch:
                i, j = ij
                is_reversed_edge = (j, i) in ext_graph.reversed_request_edges
                if is_reversed_edge:  # need to reverse mapped path
                    ij_original_orientation = (j, i)
                    uv_list = [(v, u) for (u, v) in reversed(mapping.mapping_edges[j, i])]
                else:
                    ij_original_orientation = ij
                    uv_list = mapping.mapping_edges[ij]

                ui, vj = mapping.mapping_nodes[i], mapping.mapping_nodes[j]
                ui_ext = ext_graph.cycle_layer_nodes[ij][ui][w]
                vj_ext = ext_graph.cycle_layer_nodes[ij][vj][w]
                inter_layer_edge = (connection_previous_layer, ui_ext)
                cycle_node_mapping_constraint = LinExpr([(1.0, self.var_edge_flow[req][inter_layer_edge])])
                name = modelcreator.construct_name("force_mapping_cycle", req_name=req.name, snode=ui, vnode=i)
                self.model.addConstr(cycle_node_mapping_constraint, GRB.EQUAL, 1.0, name=name)

                for u, v in self.substrate.edges:
                    if (u, v) not in self.substrate.get_valid_edges(req.get_edge_demand(ij_original_orientation)):
                        continue
                    if is_reversed_edge:
                        u, v = v, u
                    u_ext = ext_graph.cycle_layer_nodes[ij][u][w]
                    v_ext = ext_graph.cycle_layer_nodes[ij][v][w]

                    ij_uv_constraint = LinExpr([(1.0, self.var_edge_flow[req][u_ext, v_ext])])
                    name = modelcreator.construct_name("force_mapping_cycle", req_name=req.name, sedge=(u, v), vedge=ij)
                    if (u, v) in uv_list:
                        self.model.addConstr(ij_uv_constraint, GRB.EQUAL, 1.0, name=name)
                    else:
                        self.model.addConstr(ij_uv_constraint, GRB.EQUAL, 0.0, name=name)
                connection_previous_layer = vj_ext
            cycle_node_mapping_constraint = LinExpr([(1.0, self.var_edge_flow[req][(connection_previous_layer,
                                                                                    ext_graph.sink_nodes[t][w])])])
            name = modelcreator.construct_name("force_mapping_cycle", req_name=req.name, snode=w, vnode=t)
            self.model.addConstr(cycle_node_mapping_constraint, GRB.EQUAL, 1.0, name=name)


class Decomposition(object):
    def __init__(self,
                 req,
                 substrate,
                 flow_values,
                 fractional_decomposition_abortion_flow,
                 fractional_decomposition_accuracy,
                 extended_graph=None,  # optional, can be provided by the caller to save time
                 substrate_resources=None,  # optional, can be provided by the caller to save time
                 logger=None):
        if logger is None:
            logger = util.get_logger("Decomposition", make_file=False, propagate=True)
        self.logger = logger

        # TODO: Replace with new SubstrateX
        self.substrate = substrate
        if substrate_resources is None:
            substrate_resources = list(self.substrate.edges)
            for ntype in self.substrate.get_types():
                for snode in self.substrate.get_nodes_by_type(ntype):
                    substrate_resources.append((ntype, snode))

        self.substrate_resources = substrate_resources

        self.flow_values = flow_values

        self.paper_mode = True

        self.request = req
        if extended_graph is None:
            extended_graph = extendedcactusgraph.ExtendedCactusGraph(req, self.substrate)
        self.ext_graph = extended_graph

        self._mapping_count = None
        self._used_flow = None

        self.fractional_decomposition_abortion_flow = fractional_decomposition_abortion_flow
        self.fractional_decomposition_accuracy = fractional_decomposition_accuracy
        self._used_ext_graph_edge_resources = None  # set of edges in the extended graph that are used in a single iteration of the decomposition algorithm
        self._used_ext_graph_node_resources = None  # same for nodes
        self._abort_decomposition_based_on_numerical_trouble = False
        self.lost_flow_in_the_decomposition = 0.0

    def compute_mappings(self):
        result = []
        self._mapping_count = 0
        self._used_flow = {"embedding": 0.0, "node": {}, "edge": {}}
        while (self.flow_values["embedding"] - self._used_flow["embedding"]) > self.fractional_decomposition_abortion_flow:
            if self._abort_decomposition_based_on_numerical_trouble:
                return result
            self._used_ext_graph_edge_resources = set()  # use the request's original root to store the maximal possible flow according to embedding value
            self._used_ext_graph_node_resources = set()
            mapping = self._decomposition_iteration()

            if not self._abort_decomposition_based_on_numerical_trouble:
                # diminish the flow on the used edges
                node_flow_list = []
                for ext_node in self._used_ext_graph_node_resources:
                    i, u = self.ext_graph.get_associated_original_resources(ext_node)
                    node_flow_list.append(
                        self.flow_values["node"].get(i, {u: 0.0})[u] - self._used_flow["node"].get(ext_node, 0.0)
                    )

                flow = min(
                    min(node_flow_list),
                    self.flow_values["embedding"] - self._used_flow.get("embedding", 0.0),
                    min(self.flow_values["edge"].get(eedge, 0.0) - self._used_flow["edge"].get(eedge, 0.0) for eedge in self._used_ext_graph_edge_resources)
                )
                self._used_flow["embedding"] += flow
                for eedge in self._used_ext_graph_edge_resources:
                    self._used_flow["edge"].setdefault(eedge, 0.0)
                    self._used_flow["edge"][eedge] += flow
                for enode in self._used_ext_graph_node_resources:
                    self._used_flow["node"].setdefault(enode, 0.0)
                    self._used_flow["node"][enode] += flow
                load = self._calculate_substrate_load_for_mapping(mapping)
                result.append((mapping, flow, load))
        remaining_flow = self.flow_values["embedding"] - self._used_flow["embedding"]
        if remaining_flow > self.fractional_decomposition_accuracy:
            self.lost_flow_in_the_decomposition += remaining_flow
            self.logger.error("ERROR: Losing {} amount of flow as the decomposition did not perfectly work.".format(
                remaining_flow
            ))
        return result

    def _decomposition_iteration(self):
        self._mapping_count += 1
        mapping_name = modelcreator.construct_name("mapping_", req_name=self.request.name, other=self._mapping_count)
        mapping = solutions.Mapping(mapping_name, self.request, self.substrate, True)

        ext_root = self._map_root_node_on_node_with_nonzero_flow(mapping)
        self._used_ext_graph_node_resources.add(ext_root)
        queue = {self.ext_graph.root}
        while queue:
            i = queue.pop()
            for cycle in self.ext_graph.ecg_cycles:
                if cycle.start_node != i:
                    continue
                branch_1 = cycle.ext_graph_branches[0]
                branch_2 = cycle.ext_graph_branches[1]

                flow_path_1, ext_path_1, sink = self._choose_flow_path_in_extended_cycle(branch_1, mapping)
                if self.paper_mode and flow_path_1 is None:
                    break

                flow_path_2, ext_path_2, sink_2 = self._choose_flow_path_in_extended_cycle(branch_2, mapping, sink=sink)
                if self.paper_mode and flow_path_2 is None:
                    break

                if sink_2 != sink:
                    msg = "Both branches of cycle need to start & end in the same node: {}, {}".format(sink, sink_2)
                    raise CactusDecompositionError(msg)

                self._process_path(ext_path_1, flow_path_1, mapping, queue)
                self._process_path(ext_path_2, flow_path_2, mapping, queue)
            for ext_path in self.ext_graph.ecg_paths:
                if self.paper_mode and self._abort_decomposition_based_on_numerical_trouble:
                    break
                if ext_path.start_node != i:
                    continue
                flow_path, sink = self._choose_flow_path_in_extended_path(ext_path, mapping)
                self._process_path(ext_path, flow_path, mapping, queue)
        return mapping

    def _map_root_node_on_node_with_nonzero_flow(self, mapping):
        root = self.ext_graph.root
        for (u, ext_node) in self.ext_graph.source_nodes[root].iteritems():
            ext_node_outflow = sum(self.flow_values["edge"].get(eedge, 0.0) - self._used_flow["edge"].get(eedge, 0.0)
                                   for eedge in self.ext_graph.out_edges[ext_node])
            if ext_node_outflow > self.fractional_decomposition_accuracy:
                mapping.map_node(root, u)
                return ext_node
        raise CactusDecompositionError("No valid root mapping found for {}.".format(self.request.name))

    def _choose_flow_path_in_extended_cycle(self, branch, mapping, sink=None):
        if sink is None:  # this is the case for the first processed branch
            ext_path = self._choose_path_for_cycle_branch(branch, self.ext_graph, mapping)
            if self.paper_mode and ext_path is None:
                self._abort_decomposition_based_on_numerical_trouble = True
                return None, None, None
            sink_nodes = [sink_node for (last_layer_node, sink_node) in ext_path.extended_path[ext_path.end_node]]
        else:  # for the second processed branch, the sink mapping is already fixed and this branch must be chosen accordingly
            u_sink = self.ext_graph.node[sink]["substrate_node"]
            ext_path = branch[u_sink]
            sink_nodes = [sink]
        eedge_path, sink = self._choose_flow_path_in_extended_path(ext_path, mapping, sink_nodes=sink_nodes)
        return eedge_path, ext_path, sink

    def _choose_path_for_cycle_branch(self, branch, ext_graph, mapping):
        chosen_path = None
        # Iterate over the branch copies corresponding to different target node mappings
        for end_node, path in branch.iteritems():

            ext_source = ext_graph.source_nodes[path.start_node][mapping.mapping_nodes[path.start_node]]
            # print "shit starting now..."
            for eedge in ext_graph.out_edges[ext_source]:
                # print "ext_source is {}".format(ext_source)
                ee_tail, ee_head = eedge
                # print ee_tail, ee_head

                if ee_head in path.extended_nodes:
                    # print "identified {} as correctly belonging to {} (this doesn't really make sense)".format(eedge, ext_source)
                    if self.flow_values["edge"].get(eedge, 0.0) - self._used_flow["edge"].get(eedge, 0.0) > self.fractional_decomposition_accuracy:
                        # print "choosing path ... because of this edge!"
                        chosen_path = path
                        break

            if chosen_path is not None:
                break

        if self.paper_mode and chosen_path is None:
            self.logger.error("ERROR: Couldn't determine a branch to start the decomposition ...")
            self._abort_decomposition_based_on_numerical_trouble = True
        return chosen_path

    def _choose_flow_path_in_extended_path(self, path, mapping, sink_nodes=None):
        # if a sink node is specified, we search the extended graph until that node is reached. Otherwise, all sink nodes are viable:
        if sink_nodes is None:
            sink_nodes = self.ext_graph.sink_nodes[path.end_node].values()
        ext_source = self.ext_graph.source_nodes[path.start_node][mapping.mapping_nodes[path.start_node]]
        # Depth-First Search until we hit one of the viable sinks:
        predecessor = {enode: None for enode in self.ext_graph.nodes}
        stack = [ext_source]
        sink = None
        while stack:
            current_enode = stack.pop()
            # print "path search, current node:", current_enode
            if current_enode in sink_nodes:
                sink = current_enode
                break
            for eedge in self.ext_graph.out_edges[current_enode]:
                if eedge not in path.extended_edges:
                    continue
                ee_tail, ee_head = eedge
                # ignore flow-less edges:

                flow = self.flow_values["edge"].get(eedge, 0.0) - self._used_flow["edge"].get(eedge, 0.0)
                if flow > self.fractional_decomposition_accuracy and predecessor[ee_head] is None:
                    stack.append(ee_head)
                    predecessor[ee_head] = ee_tail
        if sink is None:
            if self.paper_mode:
                self.logger.error("ERROR: Couldn't find a path in the decomposition process")
                return None, None
            else:
                raise CactusDecompositionError("Sanity Check: Path search did not reach the intended sink node")

        eedge_path = Decomposition._dfs_assemble_path_from_predecessor_dictionary(predecessor, ext_source, sink)
        for edge in eedge_path:
            self._used_ext_graph_edge_resources.add(edge)
            self._used_ext_graph_node_resources.add(sink)
        return eedge_path, sink

    @staticmethod
    def _dfs_assemble_path_from_predecessor_dictionary(predecessor, ext_source_node, ext_sink_node):
        eedge_path = []
        current_enode = ext_sink_node
        while current_enode != ext_source_node:
            previous_hop = predecessor[current_enode]
            ext_edge = (previous_hop, current_enode)
            eedge_path.append(ext_edge)
            current_enode = previous_hop
        # reverse edges such that path leads from super source to super sink
        eedge_path.reverse()
        return eedge_path

    def _process_path(self, extended_path, flow_path, mapping, queue):
        for ij in extended_path.original_path:
            i, j = ij
            for uu_ext in extended_path.extended_path[j]:
                # extended_path.extended_path[j] should contain all inter-layer edges associated with the node mapping of j
                if uu_ext in flow_path:
                    u1_ext, u2_ext = uu_ext
                    u1 = self.ext_graph.node[u1_ext]["substrate_node"]
                    u2 = self.ext_graph.node[u2_ext]["substrate_node"]
                    if u1 != u2:
                        msg = "Inter-layer edge should connect nodes corresponding to the same substrate node! Instead: {} -> {} (= {}, {})".format(u1_ext, u2_ext, u1, u2)
                        raise CactusDecompositionError(msg)
                    if j in mapping.mapping_nodes:
                        u_prev = mapping.mapping_nodes[j]  # previous mapping of
                        if u_prev != u1:
                            msg = "Tried remapping node {}, which was previously mapped to {}, to a different substrate node {}".format(j, u_prev, u1)
                            raise CactusDecompositionError(msg)
                    else:
                        mapping.map_node(j, u1)
            if j not in mapping.mapping_nodes:
                raise CactusDecompositionError("Sanity Check: Mapping of node {} failed!".format(j))

            if j in self.ext_graph.source_nodes:
                queue.add(j)
                self._used_ext_graph_node_resources.add(self.ext_graph.source_nodes[j][mapping.mapping_nodes[j]])

            # extended_path.extended_path[ij] should contain all intra-layer edges associated with the edge mapping of ij:
            ext_graph_edge_mapping = [(self.ext_graph.node[ext_u]["substrate_node"], self.ext_graph.node[ext_v]["substrate_node"])
                                      for (ext_u, ext_v) in flow_path if (ext_u, ext_v) in extended_path.extended_path[ij]]

            ij_original_orientation = ij
            if (j, i) in self.ext_graph.reversed_request_edges:
                ext_graph_edge_mapping = [(v, u) for (u, v) in ext_graph_edge_mapping]
                ext_graph_edge_mapping.reverse()
                ij_original_orientation = (j, i)
            if any(uv not in self.substrate.edges for uv in ext_graph_edge_mapping):
                msg = "Mapped edge {} onto edges {}, which are not substrate edges: {}".format(ij, ext_graph_edge_mapping, self.substrate.edges)
                raise CactusDecompositionError(msg)
            mapping.map_edge(ij_original_orientation, ext_graph_edge_mapping)

    def _calculate_substrate_load_for_mapping(self, mapping):
        load = {(x, y): 0.0 for (x, y) in self.substrate_resources}
        req = self.request
        for i in req.nodes:
            u = mapping.get_mapping_of_node(i)
            t = req.node[i]["type"]
            demand = req.node[i]["demand"]
            load[(t, u)] += demand
        for ij in req.edges:
            demand = req.edge[ij]["demand"]
            for uv in mapping.mapping_edges[ij]:
                load[uv] += demand
        return load
