# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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

from gurobipy import GRB, LinExpr

from alib import solutions, modelcreator
from . import extendedgraph


class DecompositionError(Exception): pass


class ModelCreatorDecomp(modelcreator.AbstractEmbeddingModelCreator):
    """
    Solve models for the embedding of linear requests. Deferred. It serves no real purpose anymore.
    """

    def __init__(self, scenario, gurobi_settings=None):
        super(ModelCreatorDecomp, self).__init__(scenario=scenario, gurobi_settings=gurobi_settings)

        self.extended_graph = {}

        self.ext_graph_edges_node = {}
        self.ext_graph_edges_edge = {}

        self.var_flow = {}

        self.temporal_log = modelcreator.TemporalLog()

    def preprocess_input(self):
        modelcreator.AbstractEmbeddingModelCreator.preprocess_input(self)

        # create extended graphs
        for req in self.requests:
            ext_graph = extendedgraph.ExtendedGraph(req, self.substrate)
            self.extended_graph[req] = ext_graph

            self.ext_graph_edges_node[req] = {}
            self.ext_graph_edges_edge[req] = {}

            for (ntype, snode) in self.substrate.substrate_node_resources:
                self.ext_graph_edges_node[req][(ntype, snode)] = []

            for (stail, shead) in self.substrate.substrate_edge_resources:
                self.ext_graph_edges_edge[req][(stail, shead)] = []

            for ext_edge in ext_graph.edges:
                if "node_origin" in ext_graph.edge[ext_edge]:
                    ntype, snode, vnode = ext_graph.edge[ext_edge]["node_origin"]
                    self.ext_graph_edges_node[req][(ntype, snode)].append((ext_edge, vnode))

                elif "edge_origin" in ext_graph.edge[ext_edge]:
                    sedge, vedge = ext_graph.edge[ext_edge]["edge_origin"]
                    self.ext_graph_edges_edge[req][sedge].append((ext_edge, vedge))

    def create_variables_other_than_embedding_decision_and_request_load(self):
        # flow variables
        for req in self.requests:
            self.var_flow[req] = {}

            for sedge_ext in self.extended_graph[req].edges:
                variable_id = modelcreator.construct_name("flow", req_name=req.name, other=sedge_ext)
                self.var_flow[req][sedge_ext] = self.model.addVar(lb=0.0,
                                                                  ub=1.0,
                                                                  obj=0.0,
                                                                  vtype=GRB.BINARY,
                                                                  name=variable_id)

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        self._flow_induction_constraints()

        self._flow_preservation_constraints()

        self._load_computation_nodes_constraints()

        self._load_computation_edges_constraints()

        self.model.update()

    def _flow_induction_constraints(self):
        for req in self.requests:
            ext_graph = self.extended_graph[req]

            ssource = ext_graph.super_source

            expr = LinExpr([(-1.0, self.var_embedding_decision[req])] +
                           [(1.0, self.var_flow[req][sedge_ext]) for sedge_ext in ext_graph.out_edges[ssource]])

            constr_name = modelcreator.construct_name("flow_induction", req_name=req.name)
            self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _flow_preservation_constraints(self):
        for req in self.requests:
            ext_graph = self.extended_graph[req]

            for ext_node in ext_graph.nodes:

                if ext_node == ext_graph.super_source or ext_node == ext_graph.super_sink:
                    continue

                expr = LinExpr([(+1.0, self.var_flow[req][sedge_ext]) for sedge_ext in ext_graph.out_edges[ext_node]] +
                               [(-1.0, self.var_flow[req][sedge_ext]) for sedge_ext in ext_graph.in_edges[ext_node]])

                constr_name = modelcreator.construct_name("flow_preservation", req_name=req.name, other=ext_node)
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _load_computation_nodes_constraints(self):
        for req in self.requests:
            for (ntype, snode) in self.substrate.substrate_node_resources:
                expr = LinExpr([(req.get_node_demand(i), self.var_flow[req][sedge_ext])
                                for (sedge_ext, i) in self.ext_graph_edges_node[req][(ntype, snode)]] +
                               [(-1.0, self.var_request_load[req][(ntype, snode)])])

                constr_name = modelcreator.construct_name("compute_node_load", req_name=req.name, snode=snode, type=ntype)
                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _load_computation_edges_constraints(self):
        for req in self.requests:

            for (u, v) in self.substrate.substrate_edge_resources:
                expr = LinExpr(
                    [(req.get_edge_demand(vedge), self.var_flow[req][sedge_ext])
                     for (sedge_ext, vedge) in self.ext_graph_edges_edge[req][(u, v)]] +
                    [(-1.0, self.var_request_load[req][(u, v)])]
                )

                constr_name = modelcreator.construct_name("compute_edge_load", req_name=req.name, sedge=(u, v))

                self.model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def recover_integral_solution_from_variables(self):
        solution_name = modelcreator.construct_name("solution_", req_name="",
                                                    sub_name=self.substrate.name)
        self.solution = solutions.IntegralScenarioSolution(solution_name, self.scenario)

        for req in self.requests:

            # create mapping for specific req and substrate
            mapping_name = modelcreator.construct_name("mapping_", req_name=req.name,
                                                       sub_name=self.substrate.name)

            is_embedded = self.var_embedding_decision[req].x > 0.5

            mapping = solutions.Mapping(mapping_name, req, self.substrate, is_embedded=is_embedded)

            if is_embedded:
                self.obtain_integral_mapping_of_request(req, mapping)

            self.solution.add_mapping(req, mapping)

        return self.solution

    def post_process_integral_computation(self):
        return self.solution

    def obtain_integral_mapping_of_request(self, req, mapping):

        ext_graph = self.extended_graph[req]
        flow = self.var_flow[req]

        queue = [ext_graph.super_source]
        predecessor = {}

        for enode in ext_graph.nodes:
            predecessor[enode] = None

        # PERFORM BFS
        while len(queue) > 0:
            current_enode = queue[0]

            if current_enode == ext_graph.super_sink:
                break

            queue = queue[1:]

            for eedge in ext_graph.out_edges[current_enode]:
                ee_tail, ee_head = eedge

                if flow[eedge].X > 0.5 and predecessor[ee_head] is None:
                    queue = queue + [ee_head]
                    predecessor[ee_head] = ee_tail

        if predecessor[ext_graph.super_sink] is None:
            print "Predecessor dictionary:"
            print predecessor
            raise DecompositionError("Decomposition did not reach super sink node!")

        # reconstruct path
        eedge_path = []
        current_enode = ext_graph.super_sink

        while current_enode is not ext_graph.super_source:
            previous_hop = predecessor[current_enode]

            eedge_path.append((previous_hop, current_enode))

            current_enode = previous_hop

        # reverse edges such that path leads from super source to super sink
        eedge_path.reverse()

        # RECONSTRUCT MAPPING FROM PATH

        last_used_edge = 0
        previous_vnode = None

        for index, eedge in enumerate(eedge_path):
            if index == 0:
                # map first node of service chain according to first edge
                ntype, snode, vnode = ext_graph.edge[eedge]["node_origin"]
                mapping.map_node(vnode, snode)
                last_used_edge = 0
                previous_vnode = vnode
            else:
                if "node_origin" in ext_graph.edge[eedge]:
                    # this edge represents a node mapping
                    ntype, snode, vnode = ext_graph.edge[eedge]["node_origin"]
                    mapping.map_node(vnode, snode)
                    # map path between previous_vnode and vnode

                    eedge_path_connecting_previous_vnode_and_vnode = eedge_path[last_used_edge + 1:index]

                    se_path = [ext_graph.edge[eedge]["edge_origin"][0] for eedge in eedge_path_connecting_previous_vnode_and_vnode]

                    mapping.map_edge((previous_vnode, vnode), se_path)

                    last_used_edge = index
                    previous_vnode = vnode

    def recover_fractional_solution_from_variables(self):
        solution_name = modelcreator.construct_name("solution_", req_name="",
                                                    sub_name=self.substrate.name)
        self.solution = solutions.FractionalScenarioSolution(solution_name, self.scenario)
        for req in self.requests:
            # mapping_name = modelcreator.construct_name("fractual_mapping_", req_name=req.name,
            #                              sub_name=self.substrate.name)
            is_embedded = self.var_embedding_decision[req].x > 0.5
            # mapping = mp_pkg.Mapping(mapping_name, req, self.substrate, is_embedded=is_embedded)

            if is_embedded:
                maps = self.obtain_fractional_mappings_of_request(req)
                if not maps:
                    mapping_name = modelcreator.construct_name("empty_mapping_", req_name=req.name,
                                                               sub_name=self.substrate.name)
                    self.solution.add_mapping(req, solutions.Mapping(mapping_name, req, self.substrate, False), {}, {})
                for m, flow, load in maps:
                    self.solution.add_mapping(req, m, flow, load)
                    # TODO: even if the request is not embedded, a mapping object should be created, indicating that the request was not embedded
        return self.solution

    def post_process_fractional_computation(self):
        return self.solution

    def obtain_fractional_mappings_of_request(self, req, eps=0.00001):
        maps = []
        total_flow = self.var_flow[req]
        ext_graph = self.extended_graph[req]
        # dictionary: edge(from extended graph) -> remaining flow
        remaining_flow_dict = {eedge: total_flow[eedge].X for eedge in ext_graph.edges}
        sum_outgoing_flow_super_source = self.var_embedding_decision[req].x
        number_maps = 0
        while sum_outgoing_flow_super_source > eps:
            predecessor_dict = ModelCreatorDecomp._search_path_through_extended_graph(ext_graph, remaining_flow_dict)

            # reconstruct path
            eedge_path = []
            current_enode = ext_graph.super_sink
            min_flow_on_path = 1.0
            while current_enode is not ext_graph.super_source:
                previous_hop = predecessor_dict[current_enode]
                eedge = (previous_hop, current_enode)
                min_flow_on_path = min(min_flow_on_path, remaining_flow_dict[eedge])
                eedge_path.append((previous_hop, current_enode))
                current_enode = previous_hop
            # reverse edges such that path leads from super source to super sink
            eedge_path.reverse()
            for eedge in eedge_path:
                remaining_flow_dict[eedge] -= min_flow_on_path
            sum_outgoing_flow_super_source -= min_flow_on_path
            # lookup minimal used_flow on any of the edges TODO

            mapping_name = modelcreator.construct_name("mapping_" + str(number_maps), req_name=req.name,
                                                       sub_name=self.substrate.name)
            number_maps += 1
            mapping, load = self._get_fractional_mapping_and_load_from_path(req, mapping_name, ext_graph, eedge_path)
            # RECONSTRUCT MAPPING FROM PATH
            maps.append((mapping, min_flow_on_path, load))
        return maps

    @staticmethod
    def _search_path_through_extended_graph(ext_graph, remaining_flow_dict):
        stack = [ext_graph.super_source]
        predecessor = {enode: None for enode in ext_graph.nodes}
        while stack:
            current_enode = stack.pop()
            if current_enode == ext_graph.super_sink:
                del stack
                break

            for eedge in ext_graph.out_edges[current_enode]:
                ee_tail, ee_head = eedge
                remaining_flow = remaining_flow_dict[eedge]

                if remaining_flow > 0.0 and predecessor[ee_head] is None:
                    stack.append(ee_head)
                    predecessor[ee_head] = ee_tail
        if predecessor[ext_graph.super_sink] is None:
            raise DecompositionError("Never possible")  # TODO..
        return predecessor

    def _get_fractional_mapping_and_load_from_path(self, req, mapping_name, ext_graph, eedge_path):
        last_used_edge = 0
        previous_vnode = None
        is_embedded = self.var_embedding_decision[req].x > 0.5
        mapping = solutions.Mapping(mapping_name, req, self.substrate, is_embedded=is_embedded)
        load = {x: 0.0 for x in self.substrate.substrate_resources}
        for index, eedge in enumerate(eedge_path):
            if index == 0:
                # map first node of service chain according to first edge
                ntype, snode, vnode = ext_graph.edge[eedge]["node_origin"]
                mapping.map_node(vnode, snode)
                load[(ntype, snode)] += req.get_node_demand(vnode)
                last_used_edge = 0
                previous_vnode = vnode
            else:
                if "node_origin" in ext_graph.edge[eedge]:
                    # this edge represents a node mapping
                    ntype, snode, vnode = ext_graph.edge[eedge]["node_origin"]
                    mapping.map_node(vnode, snode)
                    load[(ntype, snode)] += req.get_node_demand(vnode)
                    # map path between previous_vnode and vnode

                    eedge_path_connecting_previous_vnode_and_vnode = eedge_path[last_used_edge + 1:index]

                    se_path = [ext_graph.edge[eedge]["edge_origin"][0] for eedge in eedge_path_connecting_previous_vnode_and_vnode]

                    new_edge = (previous_vnode, vnode)
                    mapping.map_edge(new_edge, se_path)
                    for u, v in se_path:
                        load[(u, v)] += req.get_edge_demand(new_edge)

                    last_used_edge = index
                    previous_vnode = vnode
        return mapping, load
