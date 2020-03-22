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

"""
The decision cactus model consists of multiple gadgets.
"""

import itertools

from gurobipy import GRB, LinExpr

from alib import datamodel, modelcreator, solutions
from . import extendedcactusgraph, modelcreator_ecg_decomposition


class MappingCombinationError(Exception):
    pass


class DecisionModelError(Exception):
    pass


class GadgetError(Exception):
    pass


class GadgetModelCreator(modelcreator.AbstractEmbeddingModelCreator):
    ''' Gurobi model creator for our implementation based on 'gadgets'. In particular, the formulation allows extended semantics:
        it allows for decision requests, i.e. two outgoing edges of a virtual node represent choices and only one of the
        outgoing edges (and its following subgraphs) are embedded.

        This model is (in general) based on our paper:
        "Guy Even, Matthias Rost, Stefan Schmid: An Approximation Algorithm for Path Computation and Function Placement in SDNs. SIROCCO 2016: 374-390"

        Note however, that our implementation extends the above presented formulation by allowing to mix decision gadgets
        with non-decision gadgets that may contain cycles.
    '''
    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=None,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(GadgetModelCreator, self).__init__(
            scenario,
            gurobi_settings=gurobi_settings,
            optimization_callback=optimization_callback,
            lp_output_file=lp_output_file,
            potential_iis_filename=potential_iis_filename,
            logger=logger
        )

    def create_variables_other_than_embedding_decision_and_request_load(self):
        for req in self.requests:
            req.set_substrate(self.substrate)
            req.var_embedding_decision = self.var_embedding_decision[req]
            req.var_request_load = self.var_request_load[req]
            req.generate_variables(self.model)

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        for req in self.requests:
            req.generate_constraints(self.model)

    def recover_integral_solution_from_variables(self):
        fractional_solution = self.recover_fractional_solution_from_variables()
        self.solution = solutions.IntegralScenarioSolution(
            "integral_solution_{}".format(self.scenario.name),
            self.scenario
        )

        for req in self.scenario.requests:
            mapping_name = modelcreator.construct_name("mapping_", req_name=req.name)
            mapping = solutions.Mapping(mapping_name, req, self.substrate, False)
            if req in fractional_solution.request_mapping:
                mapping_list = fractional_solution.request_mapping[req]
                if mapping_list:
                    most_likely_mapping = max(mapping_list, key=lambda m: fractional_solution.mapping_flows[m.name])
                    embedding_value = fractional_solution.mapping_flows[most_likely_mapping.name]
                    if abs(embedding_value - 1.0) > 0.001:
                        raise DecisionModelError("Could not find integral mapping for {}")
                    mapping.mapping_nodes = most_likely_mapping.mapping_nodes
                    mapping.mapping_edges = most_likely_mapping.mapping_edges
                    mapping.is_embedded = True
            self.solution.add_mapping(req, mapping)
        return self.solution

    def recover_fractional_solution_from_variables(self):
        self.solution = solutions.FractionalScenarioSolution("fractional_solution_{}".format(self.scenario.name),
                                                             self.scenario)
        for req in self.requests:
            for (mapping, flow, load) in req.extract_mappings():
                self.solution.add_mapping(req, mapping, flow, load)
        return self.solution

    def post_process_fractional_computation(self):
        return self.solution

    def create_objective(self):
        super(GadgetModelCreator, self).create_objective()
        self.model.update()
        if self.scenario.objective == datamodel.Objective.MAX_PROFIT:
            obj_expr = self.model.getObjective()
            for req in self.requests:
                obj_expr = req.adapt_objective(obj_expr)
            self.model.setObjective(obj_expr)

    def post_process_integral_computation(self):
        pass


class GadgetContainerRequest(object):
    """
    A request that consists of several gadgets.

    The gadgets form a tree, there is exactly one root gadget and cycles are
    forbidden. Gadgets are connected by sharing a common node. Nodes that are
    designated to be used by multiple gadgets are called interface nodes.
    """

    def __init__(self, name, profit, rounding_threshold=0.001):
        #: request name
        self.name = name
        #: request profit
        self.profit = profit
        #: rounding threshold
        self.rounding_threshold = rounding_threshold

        #: gadgets by name
        self.gadgets = {}
        #: map in-nodes to a corresponding list of gadgets
        self.gadgets_by_in_nodes = {}
        #: the root gadget
        self.root_gadget = None

        # attributes set from GadgetModelCreator
        self.substrate = None
        self.substrate_resources = []
        self.var_embedding_decision = None
        self.var_request_load = {}

        self.gurobi_vars = {}

    def set_substrate(self, substrate):
        # set it & pass through to gadgets
        self.substrate = substrate
        self.substrate_resources = substrate.substrate_resources
        for g in self.gadgets.values():
            g.set_substrate(substrate)

    def add_gadget(self, gadget):
        """
        Add a gadget to the request.

        :param AbstractGadget gadget: the new gadget
        """
        if gadget.name in self.gadgets:
            raise ValueError("Duplicate gadget name!")
        self.gadgets[gadget.name] = gadget
        gadget.container_request = self
        gadget.rounding_threshold = self.rounding_threshold
        self.gadgets_by_in_nodes.setdefault(gadget.in_node, []).append(gadget)

    def check_and_update(self):
        """
        Check if gadgets form a tree and set the root gadget.

        :raises GadgetError: if a check fails
        """
        self.check_nodes()
        self.update_root()
        self.check_gadget_tree()

    def check_nodes(self):
        """
        Check interface nodes and inner nodes.

        :raises GadgetError: if a check fails
        """
        interface_nodes = {}  # node names to node parameters
        inner_nodes = set()  # node names

        for gadget in self.gadgets.itervalues():
            # check if the in- and out-nodes of gadget exists
            if gadget.in_node not in gadget.request.node:
                raise GadgetError("in-node '{}' not in {}".format(gadget.in_node, gadget))
            for out_node in gadget.out_nodes:
                if out_node not in gadget.request.node:
                    raise GadgetError("out-node '{}' not in {}".format(out_node, gadget))

            for node, params in gadget.request.node.iteritems():
                relevant_parameters = ["type", "allowed_nodes", "demand"]
                if node == gadget.in_node or node in gadget.out_nodes:
                    # check if node parameters are the same for interface nodes
                    params2 = interface_nodes.setdefault(node, params)
                    if any(params[p] != params2[p] for p in relevant_parameters):
                        raise GadgetError("interface node '{}' has different parameters: {} and {}".format(
                            node, params, params2))
                else:
                    # check if node names are unique for inner nodes
                    if node in inner_nodes:
                        raise GadgetError("inner node '{}' is in multiple gadgets".format(node))
                    else:
                        inner_nodes.add(node)

            # check if interface nodes and inner nodes do not overlap
            interface_and_inner = set(interface_nodes) & inner_nodes
            if interface_and_inner:
                raise GadgetError("nodes used as interface and inner node: {}".format(", ".join(interface_and_inner)))

    def check_gadget_tree(self):
        """
        Check if the gadgets are connected as a tree.

        :raises GadgetError: if a check fails
        """
        all_out_nodes = set()
        for gadget in self.gadgets.itervalues():
            for out_node in gadget.out_nodes:
                if out_node in all_out_nodes:
                    raise GadgetError("out-node '{}' is used by multiple gadgets".format(out_node))
                else:
                    all_out_nodes.add(out_node)

    def update_root(self):
        """
        Update root gadget.

        A root gadget is a gadget with in-nodes that are not out-nodes of
        other gadgets.

        :raises GadgetError: if there is not exactly one root gadget
        """
        out_nodes = set()
        for gadget in self.gadgets.itervalues():
            out_nodes.update(gadget.out_nodes)

        # find root gadgets
        root_in_nodes = set(self.gadgets_by_in_nodes) - out_nodes  # in-nodes that are not out-nodes
        root_gadgets = {g for i in root_in_nodes for g in self.gadgets_by_in_nodes[i]}
        if len(root_gadgets) == 1:
            self.root_gadget = root_gadgets.pop()
        else:
            raise GadgetError("there must be exactly one root gadget, found {}".format(len(root_gadgets)))

    def add_node_flow_var(self, model, i, u):
        """
        Return a node flow variable of a model.

        Creates the variable if it does not exist.

        :param gurobipy.Model model: LP model
        :param str i: request node
        :param str u: substrate node
        :return: the node flow variable
        :rtype: gurobipy.Var
        """
        variable_id = modelcreator.construct_name("node_flow", req_name=self.name, vnode=i, snode=u)
        if variable_id not in self.gurobi_vars:
            self.gurobi_vars[variable_id] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=0.0,
                vtype=GRB.BINARY,
                name=variable_id
            )
        return self.gurobi_vars[variable_id]

    def generate_variables(self, model):
        """
        Generate LP variables for each gadget.

        :param gurobipy.Model model: LP model
        """
        if self.root_gadget is None:
            raise GadgetError("Root gadget is undefined!")
        for gadget in self.gadgets.itervalues():
            gadget.generate_variables(model)

    def generate_constraints(self, model):
        """
        Generate LP constraints for each gadget and induce flow at root gadget.

        :param gurobipy.Model model: LP model
        """
        if self.root_gadget is None:
            raise GadgetError("Root gadget is undefined!")
        self.root_gadget.generate_flow_induction_at_root_constraint(model)

        for gadget in self.gadgets.itervalues():
            gadget.generate_constraints(model)
        self._generate_load_constraints(model)

    def _generate_load_constraints(self, model):
        """
        Generate load constraints.

        :param gurobipy.Model model: LP model
        """
        constraints = {sub_resource: [(-1.0, self.var_request_load[sub_resource])]
                       for sub_resource in self.substrate_resources}

        already_handled_request_nodes = set()
        for gadget in self.gadgets.itervalues():
            gadget.extend_load_constraints(constraints, already_handled_request_nodes)

        for key, expr in constraints.iteritems():
            expr = LinExpr(expr)
            if key in self.substrate.edges:
                constr_name = modelcreator.construct_name(
                    "substrate_track_edge_load",
                    sedge=key,
                )
            else:
                nt, u = key
                constr_name = modelcreator.construct_name(
                    "substrate_track_node_load",
                    snode=u,
                    type=nt,
                )
            model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def extract_mappings(self):
        """
        Generator that yields mappings of this :class:`GadgetContainerRequest`.

        :return: yields tuples of (mapping, flow_value, load_dictionary)
        """
        if self.root_gadget is None:
            raise GadgetError("Root gadget is undefined!")
        remaining_flow = self.var_embedding_decision.x
        for mapping_count in itertools.count(1, 1):
            if remaining_flow < self.rounding_threshold:
                break

            name = modelcreator.construct_name("mapping_{}".format(mapping_count), req_name=self.name)
            used_flow = 1.0
            mapping = solutions.Mapping(name, substrate=self.substrate, request=self, is_embedded=False)  # todo: what do we do with is_embedded
            load = {res: 0.0 for res in self.substrate_resources}
            mapped_gadgets = set()
            gadget_queue = {self.root_gadget}
            nodes_handled_by_other_gadgets = set()
            while gadget_queue:
                g = gadget_queue.pop()
                mapped_gadgets.add(g)
                g_used_flow, exit_nodes = g.extend_mapping_by_own_solution(mapping, load, nodes_handled_by_other_gadgets)
                used_flow = min(g_used_flow, used_flow)

                # in a decision mapping, only one outnode is reached by the mapping
                # => depending on the chosen mapping, check what other gadgets need to be mapped:
                for i in exit_nodes:
                    if i in self.gadgets_by_in_nodes and i in mapping.mapping_nodes:
                        connected_gadgets = self.gadgets_by_in_nodes[i]
                        gadget_queue.update(set(connected_gadgets))

            mapping.is_embedded = used_flow > 0.5
            remaining_flow -= used_flow
            for g in mapped_gadgets:
                g.reduce_flow_on_last_returned_mapping(used_flow)

            self.verify_request_mapping(mapping)
            yield mapping, used_flow, load

    def verify_request_mapping(self, mapping):
        """
        Verify a request mapping.

        :param solutions.Mapping mapping: the request mapping
        """
        self._verify_consistent_edge_mapping(mapping)
        gadget_queue = {self.root_gadget}
        while gadget_queue:
            g = gadget_queue.pop()
            exit_nodes = g.verify_mapping(mapping)
            # in a decision mapping, only one outnode is reached by the mapping
            # => depending on the chosen mapping, check what other gadgets need to be mapped:
            for i in exit_nodes:
                if i in self.gadgets_by_in_nodes and i in mapping.mapping_nodes:
                    connected_gadgets = self.gadgets_by_in_nodes[i]
                    gadget_queue.update(set(connected_gadgets))

    def _verify_consistent_edge_mapping(self, mapping):
        """
        Verify the edge mapping of a request mapping.

        :param solutions.Mapping mapping: the request mapping
        :raises DecisionModelError: if the request mapping contains
            inconsistent edge mappings
        """
        error_msg = ""
        for ij, uv_list in mapping.mapping_edges.iteritems():
            i, j = ij
            if not uv_list:
                # this implies colocation of i and j
                u_i = mapping.mapping_nodes.get(i, None)
                u_j = mapping.mapping_nodes.get(j, None)
                if u_i != u_j:
                    error_msg += "\n\tEdge {} has empty mapping, but node mappings are {} -> {}, {} -> {}".format(
                        ij, i, u_i, j, u_j
                    )
                continue
            subfirsttail, subfirsthead = uv_list[0]
            sublasttail, sublasthead = uv_list[-1]
            # check that tail, head of ve are correctly mapped on tail, head of path
            if mapping.mapping_nodes[i] != subfirsttail or mapping.mapping_nodes[j] != sublasthead:
                error_msg += "\n\tEdge mapping {} -> {} inconsistent with node mappings {} -> {}, {} -> {}".format(
                    ij, uv_list, i, mapping.mapping_nodes[i], j, mapping.mapping_nodes[j]
                )
            else:
                if not len(uv_list) > 1:
                    # it's only single edge mapped on single edge
                    pass
                else:
                    # check wether path is a real edge path and connected
                    for idx, uv in enumerate(uv_list):
                        if idx < len(uv_list) - 1:
                            currenttail, currenthead = uv
                            nexttail, nexthead = uv_list[idx + 1]
                            if not currenthead == nexttail:
                                error_msg += "\n\tEdge {} has inconsistent mapping: {}".format(
                                    ij, uv_list
                                )
        if error_msg:
            raise DecisionModelError("Inconsistent edge mappings found:\n" + error_msg)

    def adapt_objective(self, obj_expr):
        """
        Adapt the objective.

        :param gurobipy.LinExpr obj_expr: the objective expression
        :return: the modified objective expression
        :rtype: gurobipy.LinExpr
        """
        for gadget in self.gadgets.itervalues():
            obj_expr = gadget.adapt_model_objective_according_to_local_profits(obj_expr)
        return obj_expr

    def get_gadget_tree_graph(self):
        result = datamodel.Graph("{}_gadget_tree".format(self.name))

        for g in self.gadgets.values():
            result.add_node(g.name)

        gadget_queue = {self.root_gadget}
        while gadget_queue:
            g = gadget_queue.pop()
            exit_nodes = g.out_nodes

            # in a decision mapping, only one outnode is reached by the mapping
            # => depending on the chosen mapping, check what other gadgets need to be mapped:
            for i in exit_nodes:
                if i in self.gadgets_by_in_nodes:
                    connected_gadgets = self.gadgets_by_in_nodes[i]
                    for other_g in connected_gadgets:
                        result.add_edge(g.name, other_g.name)
                    gadget_queue.update(set(connected_gadgets))
        return result


class AbstractGadget(object):
    """
    An abstract gadget used by :class:`GadgetContainerRequest`.

    Gadgets are connected by sharing a common node. Nodes that are designated
    to be used by multiple gadgets are called interface nodes.

    A gadget has exactly one in-node and one or multiple out-nodes.
    """

    def __init__(self, name, request, in_node, out_nodes):
        #: gadget name
        self.name = name
        #: in-node of the gadget
        self.in_node = in_node
        #: out-nodes of the gadget
        self.out_nodes = frozenset(out_nodes)
        #: all interface nodes (in- and out-nodes)
        self.interface_nodes = self.out_nodes | {self.in_node}
        #: the request graph of the gadget
        self.request = request
        #: the substrate
        self.substrate = None

        # attributes set from GadgetContainerRequest
        self.container_request = None
        self.rounding_threshold = None

        self.gurobi_vars = {}
        self._most_recent_mapping = None

    def extend_load_constraints(self, constraint_dict, nodes_handled_by_other_gadgets):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def extend_mapping_by_own_solution(self, mapping, load, nodes_handled_by_other_gadgets):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def generate_constraints(self, model):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def generate_flow_induction_at_root_constraint(self, model):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def generate_variables(self, model):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def reduce_flow_on_last_returned_mapping(self, used_flow):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def adapt_model_objective_according_to_local_profits(self, obj_expr):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def set_substrate(self, substrate):
        self.substrate = substrate
        self._initialize_extended_graph()

    def _initialize_extended_graph(self):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def verify_mapping(self, mapping):
        """
        Perform certain sanity checks usually performed by the :class:`alib.solutions.Mapping`
        class.

        :param solutions.Mapping mapping: the request mapping
        :return: a list of out-nodes that are mapped
        """
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")

    def __repr__(self):
        return "<{} name={}>".format(type(self).__name__, self.name)


class DecisionGadget(AbstractGadget):
    def __init__(self, name, request, in_node, out_nodes):
        super(DecisionGadget, self).__init__(name, request, in_node, out_nodes)
        request.graph["root"] = in_node
        self.ext_graph = None
        self._used_flow = {}
        self._edge_vars_used_in_most_recent_mapping = None
        self._node_vars_used_in_most_recent_mapping = None

    def generate_variables(self, model):
        self.gurobi_vars["node_flow"] = {}
        for i in self.interface_nodes:
            if i not in self.gurobi_vars["node_flow"]:
                self.gurobi_vars["node_flow"][i] = {}
            for u in self.request.get_allowed_nodes(i):
                if u in self.gurobi_vars["node_flow"][i]:
                    continue
                self.gurobi_vars["node_flow"][i][u] = self.container_request.add_node_flow_var(model, i, u)
        self.gurobi_vars["edge_flow"] = {}
        for i, uv_edge_dict in self.ext_graph.layer_edges.iteritems():
            for uv, edge in uv_edge_dict.iteritems():
                variable_id = modelcreator.construct_name(
                    "edge_flow",
                    req_name=self.container_request.name,
                    vnode=i,
                    sedge=uv,
                )
                self.gurobi_vars["edge_flow"][edge] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    obj=0.0,
                    vtype=GRB.BINARY,
                    name=variable_id
                )
        for ij, u_edge_dict in self.ext_graph.inter_layer_edges.iteritems():
            for u, edge in u_edge_dict.iteritems():
                variable_id = modelcreator.construct_name(
                    "interlayer_edge_flow",
                    req_name=self.container_request.name,
                    vedge=ij,
                    snode=u,
                )
                self.gurobi_vars["edge_flow"][edge] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    obj=0.0,
                    vtype=GRB.BINARY,
                    name=variable_id
                )
        for i, u_edge_dict in self.ext_graph.sink_edges.iteritems():
            for u, edge in u_edge_dict.iteritems():
                variable_id = modelcreator.construct_name(
                    "interlayer_sink_edge_flow",
                    req_name=self.container_request.name,
                    vnode=i,
                    snode=u,
                )
                self.gurobi_vars["edge_flow"][edge] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    obj=0.0,
                    vtype=GRB.BINARY,
                    name=variable_id
                )

    def adapt_model_objective_according_to_local_profits(self, obj_expr):
        for ij, u_edge_dict in self.ext_graph.inter_layer_edges.iteritems():
            if "edge_profit" in self.request.edge[ij]:
                profit = self.request.edge[ij]["edge_profit"]
                for ext_edge in u_edge_dict.values():
                    obj_expr.addTerms(profit, self.gurobi_vars["edge_flow"][ext_edge])
        return obj_expr

    def generate_flow_induction_at_root_constraint(self, model):
        in_expr = [(-1.0, self.container_request.var_embedding_decision)]
        for u in self.request.get_allowed_nodes(self.in_node):
            in_expr.append((1.0, self.gurobi_vars["node_flow"][self.in_node][u]))
        in_expr = LinExpr(in_expr)
        constraint_name = "flow_induction_root_gadget_{}".format(self.name)
        model.addConstr(in_expr, GRB.EQUAL, 0.0, name=constraint_name)

    def generate_constraints(self, model):
        for ext_node in self.ext_graph.nodes:
            expr = []
            u = self.ext_graph.node[ext_node]["substrate_node"]
            i = self.ext_graph.node[ext_node]["request_node"]

            # handle in-flow
            if ext_node in self.ext_graph.source_node_set:
                expr.append((1.0, self.gurobi_vars["node_flow"][i][u]))
            for ext_edge in self.ext_graph.get_in_edges(ext_node):
                expr.append((1.0, self.gurobi_vars["edge_flow"][ext_edge]))

            # handle out-flow
            if ext_node in self.ext_graph.sink_node_set:
                expr.append((-1.0, self.gurobi_vars["node_flow"][i][u]))
            for ext_edge in self.ext_graph.get_out_edges(ext_node):
                expr.append((-1.0, self.gurobi_vars["edge_flow"][ext_edge]))

            expr = LinExpr(expr)
            constraint_name = "flow_pres_{}_{}".format(self.name, ext_node)
            model.addConstr(expr, GRB.EQUAL, 0, name=constraint_name)

    def extend_load_constraints(self, constraint_dict, nodes_handled_by_other_gadgets):
        handled_nodes = set()
        for ij, u_edge_dict in self.ext_graph.inter_layer_edges.iteritems():
            i, j = ij
            i_type = self.request.get_type(i)
            for u, edge in u_edge_dict.iteritems():
                u_i_ext, v_j_ext = edge
                # the source nodes may have been handled by another gadget
                if u_i_ext in self.ext_graph.source_node_set:
                    if i in nodes_handled_by_other_gadgets:
                        continue
                    handled_nodes.add(i)
                constraint_dict[(i_type, u)].append((1.0, self.gurobi_vars["edge_flow"][edge]))
        for i, u_edge_dict in self.ext_graph.sink_edges.iteritems():
            i_type = self.request.get_type(i)
            if i in nodes_handled_by_other_gadgets:
                continue
            handled_nodes.add(i)
            for u, edge in u_edge_dict.iteritems():
                constraint_dict[(i_type, u)].append((1.0, self.gurobi_vars["edge_flow"][edge]))
        for i, uv_edge_dict in self.ext_graph.layer_edges.iteritems():
            for uv, edge in uv_edge_dict.iteritems():
                constraint_dict[uv].append((1.0, self.gurobi_vars["edge_flow"][edge]))
        nodes_handled_by_other_gadgets.update(handled_nodes)

    def extend_mapping_by_own_solution(self, mapping, load, nodes_handled_by_other_gadgets):
        self._node_vars_used_in_most_recent_mapping = set()
        self._edge_vars_used_in_most_recent_mapping = set()

        flow, u_in_node = self._pick_consistent_mapping_for_in_node(mapping)
        u_in_node_ext = self.ext_graph.source_nodes[self.in_node][u_in_node]
        self._node_vars_used_in_most_recent_mapping.add((self.in_node, u_in_node))

        if self.in_node not in mapping.mapping_nodes:
            mapping.mapping_nodes[self.in_node] = u_in_node
            if self.in_node not in nodes_handled_by_other_gadgets:
                nodes_handled_by_other_gadgets.add(self.in_node)
                load[(self.request.get_type(self.in_node), u_in_node)] += self.request.get_node_demand(self.in_node)

        # collect the possible exits from the extended graph:
        extgraph_exits = set()
        for i in self.out_nodes:
            extgraph_exits.update(self.ext_graph.sink_nodes[i].values())
        next_ext_node = u_in_node_ext
        i_previous = None
        exit_node = None
        while next_ext_node is not None:
            # in one iteration of this loop, we go from the current extended graph layer to the next;
            # i.e. each iteration corresponds to handling a single request edge
            u_i_ext = next_ext_node

            predecessor, v_j_ext = self._find_path_to_next_layer(u_i_ext)
            if v_j_ext is None:
                # This means that we could not find a connected non-zero flow inter_layer_edge
                raise DecisionModelError("Could not find a connected non-zero flow inter_layer_edge from {} in gadget {}. Current flow is {}".format(u_i_ext, self.name, flow))

            j = self.ext_graph.node[v_j_ext]["request_node"]
            # First, handle the node mapping

            i = self.ext_graph.node[u_i_ext]["request_node"]
            if i not in mapping.mapping_nodes:
                u = self.ext_graph.node[predecessor[v_j_ext]]["substrate_node"]
                # print "Mapping node", i, " -> ", u
                mapping.mapping_nodes[i] = u
                i_type = self.request.get_type(i)
                nodes_handled_by_other_gadgets.add(i)
                load[(i_type, u)] += self.request.get_node_demand(i)
            next_ext_node = v_j_ext

            if i_previous is not None:
                edge_mapping_flow, sub_edges = self._make_edge_mapping_from_predecessor_dict(predecessor, u_i_ext, v_j_ext)
                mapping.mapping_edges[(i_previous, i)] = sub_edges
                # print "Mapping edge", (i_previous, i), " -> ", sub_edges
                flow = min(flow, edge_mapping_flow)
                for uv in sub_edges:
                    load[uv] += self.request.get_edge_demand((i_previous, i))
            i_previous = i

            if v_j_ext in self.ext_graph.sink_node_set:
                exit_node = self.ext_graph.node[v_j_ext]["request_node"]
                mapped_exit_node = self.ext_graph.node[v_j_ext]["substrate_node"]

                exit_flow = self.gurobi_vars["node_flow"][exit_node][mapped_exit_node].x
                if exit_node in self._used_flow:
                    if mapped_exit_node in self._used_flow[exit_node]:
                        exit_flow -= self._used_flow[exit_node][mapped_exit_node]
                flow = min(flow, exit_flow)
                self._edge_vars_used_in_most_recent_mapping.add((u_i_ext, v_j_ext))
                self._node_vars_used_in_most_recent_mapping.add((j, self.ext_graph.node[v_j_ext]["substrate_node"]))
                break

        return flow, [exit_node]

    def _pick_consistent_mapping_for_in_node(self, mapping):
        u_in_node = None
        flow = 1.0
        if self.in_node in mapping.mapping_nodes:
            u_in_node = mapping.mapping_nodes[self.in_node]
            used = 0.0
            if self.in_node in self._used_flow:
                if u_in_node in self._used_flow[self.in_node]:
                    used = self._used_flow[self.in_node][u_in_node]
            flow = self.gurobi_vars["node_flow"][self.in_node][u_in_node].x - used
        else:
            # This only applies if this is the root gadget!
            for u in self.request.get_allowed_nodes(self.in_node):
                used = 0.0
                if self.in_node in self._used_flow:
                    if u in self._used_flow[self.in_node]:
                        used = self._used_flow[self.in_node][u]
                remaining_flow = self.gurobi_vars["node_flow"][self.in_node][u].x - used
                if remaining_flow > self.rounding_threshold:
                    # self._vars_used_in_most_recent_mapping
                    u_in_node = u
                    flow = remaining_flow
                    break
        return flow, u_in_node

    def _find_path_to_next_layer(self, layer_entry_node):
        visited = set()
        layer_queue = [layer_entry_node]  # layer_queue should only contain unvisited nodes within the same layer that are reachable with non-zero flow
        predecessor = {layer_entry_node: None}
        v_j_ext = None  # j is the next request node we go to, and v is the node to which j is mapped
        while layer_queue:  # in this loop, we find an edge that brings us to another layer
            ext_node = layer_queue.pop()
            visited.add(ext_node)
            for u_i_ext_neighbor in self.ext_graph.get_out_neighbors(ext_node):
                if u_i_ext_neighbor in visited:
                    continue
                ext_edge = ext_node, u_i_ext_neighbor
                predecessor[u_i_ext_neighbor] = ext_node
                remaining_flow_ext_edge = self.gurobi_vars["edge_flow"][ext_edge].x - self._used_flow.get(ext_edge, 0.0)
                if remaining_flow_ext_edge > self.rounding_threshold:
                    if ext_edge in self.ext_graph.inter_layer_edge_set or ext_edge in self.ext_graph.sink_edge_set:
                        self._edge_vars_used_in_most_recent_mapping.add(ext_edge)
                        v_j_ext = u_i_ext_neighbor
                        break
                    layer_queue.append(u_i_ext_neighbor)
            if v_j_ext is not None:
                break
        return predecessor, v_j_ext

    def _make_edge_mapping_from_predecessor_dict(self, predecessor, layer_entry, layer_exit):
        flow = 1.0
        reverse_queue = {predecessor[layer_exit]}  # initialize with the predecessor, because we need to move back to the previous layer first
        pred = None
        sub_edges = []
        while pred != layer_entry:
            current = reverse_queue.pop()
            pred = predecessor[current]
            if pred is None:
                break
            u = self.ext_graph.node[pred]["substrate_node"]
            v = self.ext_graph.node[current]["substrate_node"]
            ext_edge = (pred, current)
            remaining_flow_ext_edge = self.gurobi_vars["edge_flow"][ext_edge].x - self._used_flow.get(ext_edge, 0.0)
            flow = min(flow, remaining_flow_ext_edge)
            self._edge_vars_used_in_most_recent_mapping.add(ext_edge)
            sub_edges.append((u, v))
            reverse_queue.add(pred)
        sub_edges.reverse()
        return flow, sub_edges

    def reduce_flow_on_last_returned_mapping(self, used_flow):
        if self._edge_vars_used_in_most_recent_mapping is None:
            raise DecisionModelError("Need to call extend_mapping_by_own_solution before reduce_flow_on_last_returned_mapping!")

        for ext_edge in self._edge_vars_used_in_most_recent_mapping:
            if ext_edge not in self._used_flow:
                self._used_flow[ext_edge] = 0.0
            self._used_flow[ext_edge] += used_flow
        for i, u in self._node_vars_used_in_most_recent_mapping:
            if i not in self._used_flow:
                self._used_flow[i] = {u: 0.0}
            if u not in self._used_flow[i]:
                raise DecisionModelError("Mapped {} onto multiple substrate nodes: {}, {}".format(
                    i, self._node_vars_used_in_most_recent_mapping[i].keys(), u
                ))
            self._used_flow[i][u] += used_flow

        self._node_vars_used_in_most_recent_mapping = None
        self._edge_vars_used_in_most_recent_mapping = None

    def verify_mapping(self, mapping):
        error_msg = ""
        i_queue = {self.in_node}
        exit_node = None
        while i_queue:
            i = i_queue.pop()
            found_neighbor = None
            for j in self.request.get_out_neighbors(i):
                if (i, j) in mapping.mapping_edges:
                    if found_neighbor is not None:
                        error_msg += "Multiple paths of decision gadget {}, {} were mapped: {} -> {} and {} -> {} in mapping".format(
                            self.container_request.name,
                            self.name,
                            i, found_neighbor,
                            i, j
                        )
                    i_queue.add(j)
                    found_neighbor = j
            if not i_queue:
                exit_node = i
        if exit_node not in self.out_nodes:
            error_msg += "\n\tMapping did not end in out node: {} (expected one of {})".format(exit_node, self.out_nodes)
        if error_msg:
            raise DecisionModelError("Invalid decision mapping:" + error_msg)
        return [exit_node]

    def _initialize_extended_graph(self):
        self.ext_graph = ExtendedDecisionGraph(self)


class ExtendedDecisionGraph(datamodel.Graph):
    def __init__(self, parent_gadget):
        super(ExtendedDecisionGraph, self).__init__("{}_ext".format(parent_gadget.name))
        self.parent_gadget = parent_gadget
        self.req = parent_gadget.request
        self.sub = parent_gadget.substrate
        self.root = self.parent_gadget.in_node
        self.source_nodes = {}
        self.source_node_set = set()
        self.sink_nodes = {}
        self.sink_node_set = set()
        self.layer_nodes = {}
        self.layer_node_set = set()
        self.layer_edges = {i: {} for i in self.req.nodes}
        self.layer_edge_set = set()
        self.inter_layer_edges = {ij: {u: None for u in self.req.get_allowed_nodes(ij[0])} for ij in self.req.edges}
        self.inter_layer_edge_set = set()
        self.sink_edges = {i: {} for i in self.parent_gadget.out_nodes}
        self.sink_edge_set = set()
        self._initialize()

    def _initialize(self):
        # generate the layers
        for i in self.req.nodes:
            if i == self.root:
                continue
            self._make_substrate_copy_for_gadget_node(i)

        # make & connect source nodes
        for u in self.req.get_allowed_nodes(self.root):
            for j in self.req.get_out_neighbors(self.root):
                u_i_source = self._add_and_get_super_source_node(self.root, u)
                u_j_layer = self.layer_nodes[j][u]
                self.add_edge(u_i_source, u_j_layer, bidirected=False, request_node=self.root, substrate_node=u)
                ext_edge = (u_i_source, u_j_layer)
                self.inter_layer_edges[(self.root, j)][u] = ext_edge
                self.inter_layer_edge_set.add(ext_edge)

        # connect layers according to request edges
        for ij in self.req.edges:
            i, j = ij
            if i == self.root:
                continue
            for u in self.req.get_allowed_nodes(i):
                u_i = self.layer_nodes[i][u]
                u_j = self.layer_nodes[j][u]
                self.add_edge(u_i, u_j, bidirected=False, request_node=j, substrate_node=u)
                ext_edge = (u_i, u_j)
                self.inter_layer_edges[ij][u] = ext_edge
                self.inter_layer_edge_set.add(ext_edge)

        # make & connect sink nodes
        for i in self.parent_gadget.out_nodes:
            for u in self.req.get_allowed_nodes(i):
                u_i_layer = self.layer_nodes[i][u]
                u_i_sink = self._add_and_get_super_sink_node(i, u)
                self.add_edge(u_i_layer, u_i_sink, bidirected=False, request_node=i, substrate_node=u)
                ext_edge = u_i_layer, u_i_sink
                self.sink_edges[i][u] = ext_edge
                self.sink_edge_set.add(ext_edge)

    def _make_substrate_copy_for_gadget_node(self, i):
        for u in self.sub.nodes:
            u_layer = self._add_and_get_layer_node(i, u)
            self.layer_nodes[i][u] = u_layer

        for uv in self.sub.edges:
            u, v = uv
            u_layer = self._add_and_get_layer_node(i, u)
            v_layer = self._add_and_get_layer_node(i, v)
            self.add_edge(u_layer, v_layer, bidirected=False, substrate_edge=(u, v), request_node=i)
            ext_edge = (u_layer, v_layer)
            self.layer_edges[i][uv] = ext_edge
            self.layer_edge_set.add(ext_edge)

    def _add_and_get_super_source_node(self, i, u):
        if i not in self.source_nodes:
            self.source_nodes[i] = {}
        new_node = "{}_[{}]_+".format(u, i)
        if u not in self.source_nodes[i]:
            self.source_nodes[i][u] = new_node
            self.add_node(new_node, request_node=i, substrate_node=u)
            self.source_node_set.add(new_node)
        return new_node

    def _add_and_get_super_sink_node(self, i, u):
        if i not in self.sink_nodes:
            self.sink_nodes[i] = {}
        new_node = "{}_[{}]_-".format(u, i)
        if u not in self.sink_nodes[i]:
            self.sink_nodes[i][u] = new_node
            self.add_node(new_node, request_node=i, substrate_node=u)
            self.sink_node_set.add(new_node)
        return new_node

    def _add_and_get_layer_node(self, i, u):
        if i not in self.layer_nodes:
            self.layer_nodes[i] = {}
        new_node = "{}_[{}]".format(u, i)
        if u not in self.layer_nodes[i]:
            self.add_node(new_node, request_node=i, substrate_node=u)
            self.layer_nodes[i][u] = new_node
            self.layer_node_set.add(new_node)
        return new_node


class CactusGadget(AbstractGadget):
    def __init__(self, name, request, in_node, out_nodes, parent_name=None):
        super(CactusGadget, self).__init__(name, request, in_node, out_nodes)
        self.parent_name = parent_name
        request.graph["root"] = self.in_node
        self.ext_graph = None
        self._mappings = None
        self._remaining_flow_on_mapping = None
        self._mapping_flows = None
        self._mapping_loads = None

    def generate_variables(self, model):
        self.gurobi_vars["node_flow"] = {}
        for i in set(self.ext_graph.source_nodes) | self.out_nodes:
            if i not in self.gurobi_vars["node_flow"]:
                self.gurobi_vars["node_flow"][i] = {}
            for u in self.request.get_allowed_nodes(i):
                if u not in self.substrate.get_valid_nodes(self.request.get_type(i), self.request.get_node_demand(i)):
                    continue
                if u not in self.gurobi_vars["node_flow"][i]:
                    self.gurobi_vars["node_flow"][i][u] = self.container_request.add_node_flow_var(model, i, u)
        for i, ecg_node_dict in self.ext_graph.sink_nodes.iteritems():
            if i not in self.gurobi_vars["node_flow"]:
                self.gurobi_vars["node_flow"][i] = {}
            for u, ecg_node in ecg_node_dict.iteritems():
                if u not in self.gurobi_vars["node_flow"][i]:
                    self.gurobi_vars["node_flow"][i][u] = self.container_request.add_node_flow_var(model, i, u)

        # edge flow variables
        self.gurobi_vars["edge_flow"] = {}
        for ecg_edge in self.ext_graph.edges:
            variable_id = modelcreator.construct_name("flow_edge", req_name=self.container_request.name, other=ecg_edge)
            self.gurobi_vars["edge_flow"][ecg_edge] = model.addVar(lb=0.0,
                                                                   ub=1.0,
                                                                   obj=0.0,
                                                                   vtype=GRB.BINARY,
                                                                   name=variable_id)

    def generate_constraints(self, model):
        if self.ext_graph is None:
            expr = LinExpr([(1.0, self.container_request.var_embedding_decision)])
            constr_name = modelcreator.construct_name("fix_infeasible_req_to_zero", req_name=self.container_request.name)
            model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
            return
        # flow induction for cycles (sources) - constraint 9
        self._add_constraint_cycle_flow_induction_at_source(model)

        # same flow in each cycle branch (commutativity) - constraint 10
        self._add_constraint_cycle_flow_commutativity(model)

        # flow induction for paths (sources) - constraint 11
        self._add_constraint_path_flow_induction_at_source(model)

        # flow preservation - constraint 12
        self._add_constraint_flow_preservation(model)

        # flow induction for cycles (sinks) - constraint 13
        self._add_constraint_cycle_flow_induction_at_sink(model)

        # flow induction for paths (sinks) - constraint 14
        self._add_constraint_path_flow_induction_at_sink(model)

        # flow inductions for subgraphs branching off from a cycle - constraint 15
        self._add_constraint_branching_flow_induction(model)

    def extend_load_constraints(self, constraint_dict, nodes_handled_by_other_gadgets):
        self._add_constraint_track_node_load(constraint_dict, nodes_handled_by_other_gadgets)
        self._add_constraint_track_edge_load(constraint_dict)

    def generate_flow_induction_at_root_constraint(self, model):
        root = self.ext_graph.root
        root_source_nodes = self.ext_graph.source_nodes[root].keys()  # this list contains all source nodes associated with the request root

        expr = LinExpr([(-1.0, self.container_request.var_embedding_decision)] +
                       [(1.0, self.gurobi_vars["node_flow"][root][u]) for u in root_source_nodes])

        constr_name = modelcreator.construct_name("flow_induction_root_gadget",
                                                  req_name="{}_{}".format(self.container_request.name, self.name),
                                                  )
        # print "Root flow induction ({}):".format(root).ljust(40), expr, "= 0"
        model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_cycle_flow_induction_at_source(self, model):
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            for u in self.request.get_allowed_nodes(cycle.start_node):
                if u not in self.substrate.get_valid_nodes(self.request.get_type(cycle.start_node),
                                                           self.request.get_node_demand(cycle.start_node)):
                    continue
                cycle_source = self.ext_graph.source_nodes[cycle.start_node][u]
                ij = cycle.original_branches[0][0]  # select the first edge of the first branch of the cycle
                if ij[0] != cycle.start_node:
                    raise Exception("Sanity check")
                expr = [(-1.0, self.gurobi_vars["node_flow"][cycle.start_node][u])]
                for w in cycle_targets:
                    ext_edge = (cycle_source, self.ext_graph.cycle_layer_nodes[ij][u][w])
                    expr.append((1.0, self.gurobi_vars["edge_flow"][ext_edge]))
                expr = LinExpr(expr)
                constr_name = modelcreator.construct_name("cycle_flow_induction_source",
                                                          req_name="{}_{}".format(self.container_request.name, self.name),
                                                          other="{}->{}".format(cycle.start_node,
                                                                                cycle.end_node))
                # print "Cycle flow induction ({}):".format(cycle_source).ljust(40), expr, "= 0"
                model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_cycle_flow_commutativity(self, model):
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            for u in self.request.get_allowed_nodes(cycle.start_node):
                if u not in self.substrate.get_valid_nodes(self.request.get_type(cycle.start_node),
                                                           self.request.get_node_demand(cycle.start_node)):
                    continue
                cycle_source = self.ext_graph.source_nodes[cycle.start_node][u]
                ij = cycle.original_branches[0][0]  # select the first edge of the first branch of the cycle
                ik = cycle.original_branches[1][0]  # select the first edge of the second branch of the cycle
                if ij[0] != cycle.start_node or ik[0] != cycle.start_node:
                    raise Exception("Sanity check")

                for w in cycle_targets:
                    u_ij = self.ext_graph.cycle_layer_nodes[ij][u][w]
                    u_ik = self.ext_graph.cycle_layer_nodes[ik][u][w]
                    expr = LinExpr([
                        (1.0, self.gurobi_vars["edge_flow"][(cycle_source, u_ij)]),
                        (-1.0, self.gurobi_vars["edge_flow"][(cycle_source, u_ik)])
                    ])
                    constr_name = modelcreator.construct_name("flow_commutativity",
                                                              req_name="{}_{}".format(self.container_request.name,
                                                                                      self.name),
                                                              vedge="{}->{}, {}->{}".format(ij[0], ij[1], ik[0], ik[1]),
                                                              snode="s={}, t={}".format(u, w))
                    # print "Cycle flow commutativity ({}, {}):".format(cycle_source, w).ljust(40), expr, "= 0"
                    model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_path_flow_induction_at_source(self, model):
        for path in self.ext_graph.ecg_paths:
            for u in self.request.get_allowed_nodes(path.start_node):
                if u not in self.substrate.get_valid_nodes(self.request.get_type(path.start_node),
                                                           self.request.get_node_demand(path.start_node)):
                    continue

                path_source = self.ext_graph.source_nodes[path.start_node][u]
                ij = path.original_path[0]  # select the first edge of the first branch of the cycle
                ext_ij = (path_source, self.ext_graph.path_layer_nodes[ij][u])
                expr = LinExpr([
                    (1.0, self.gurobi_vars["edge_flow"][ext_ij]),
                    (-1.0, self.gurobi_vars["node_flow"][path.start_node][u])
                ])
                constr_name = modelcreator.construct_name("path_flow_induction_source",
                                                          req_name="{}_{}".format(self.container_request.name,
                                                                                  self.name),
                                                          vedge="{}->{}".format(ij[0], ij[1]),
                                                          snode="s={}".format(u))
                # print "Path flow induction ({}):".format(path_source).ljust(40), expr, "= 0"
                model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_flow_preservation(self, model):
        for node in self.ext_graph.layer_nodes:
            expr = []
            for neighbor in self.ext_graph.get_out_neighbors(node):
                edge = node, neighbor
                expr.append((1.0, self.gurobi_vars["edge_flow"][edge]))
            for neighbor in self.ext_graph.get_in_neighbors(node):
                edge = neighbor, node
                expr.append((-1.0, self.gurobi_vars["edge_flow"][edge]))
            expr = LinExpr(expr)
            constr_name = modelcreator.construct_name("flow_preservation",
                                                      other=node)
            model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
            # print "Flow Preservation ({}):".format(node)[:37].ljust(40), expr, "= 0"

    def _add_constraint_cycle_flow_induction_at_sink(self, model):
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            for w in cycle_targets:
                ij = cycle.original_branches[0][-1]  # select the last edge of the first branch of the cycle
                if ij[1] != cycle.end_node:
                    raise Exception("Sanity check")
                cycle_sink = self.ext_graph.sink_nodes[cycle.end_node][w]
                ext_ij = (self.ext_graph.cycle_layer_nodes[ij][w][w], cycle_sink)
                expr = LinExpr([
                    (1.0, self.gurobi_vars["node_flow"][cycle.end_node][w]),
                    (-1.0, self.gurobi_vars["edge_flow"][ext_ij])
                ])
                constr_name = modelcreator.construct_name("cycle_flow_induction_source",
                                                          req_name="{}_{}".format(self.container_request.name,
                                                                                  self.name),
                                                          other="{}->{}".format(cycle.start_node, cycle.end_node))
                # print "Cycle flow induction ({}):".format(cycle_sink).ljust(40), expr, "= 0"
                model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_path_flow_induction_at_sink(self, model):
        for path in self.ext_graph.ecg_paths:
            for w in self.request.get_allowed_nodes(path.end_node):
                if w not in self.substrate.get_valid_nodes(self.request.get_type(path.end_node),
                                                           self.request.get_node_demand(path.end_node)):
                    continue

                path_sink = self.ext_graph.sink_nodes[path.end_node][w]
                ij = path.original_path[-1]  # select the first edge of the first branch of the cycle
                ext_ij = (self.ext_graph.path_layer_nodes[ij][w], path_sink)
                expr = LinExpr([
                    (1.0, self.gurobi_vars["edge_flow"][ext_ij]),
                    (-1.0, self.gurobi_vars["node_flow"][path.end_node][w])
                ])
                constr_name = modelcreator.construct_name("path_flow_induction_sink",
                                                          req_name="{}_{}".format(self.container_request.name,
                                                                                  self.name),
                                                          vedge="{}->{}".format(ij[0], ij[1]),
                                                          snode="s={}".format(w))
                # print "Path flow induction ({}):".format(path_sink).ljust(40), expr, "= 0"
                model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)

    def _add_constraint_branching_flow_induction(self, model):
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            for branch in cycle.original_branches:
                for pos, ij in enumerate(branch):
                    (i, j) = ij
                    if j not in self.out_nodes and j not in self.ext_graph.cycle_branch_nodes or j == cycle.end_node:
                        continue

                    # j must be a branching node!
                    jl = branch[pos + 1]
                    if jl[0] != j:
                        raise Exception("Sanity Check!")
                    for u in self.request.get_allowed_nodes(j):
                        if u not in self.substrate.get_valid_nodes(self.request.get_type(j),
                                                                   self.request.get_node_demand(j)):
                            continue

                        expr = [(-1.0, self.gurobi_vars["node_flow"][j][u])]
                        for w in cycle_targets:
                            u_ij = self.ext_graph.cycle_layer_nodes[ij][u][w]
                            u_jl = self.ext_graph.cycle_layer_nodes[jl][u][w]
                            ext_edge = (u_ij, u_jl)
                            expr.append((1.0, self.gurobi_vars["edge_flow"][ext_edge]))
                        expr = LinExpr(expr)
                        constr_name = modelcreator.construct_name("branch_flow_induction",
                                                                  other=j)
                        model.addConstr(expr, GRB.EQUAL, 0.0, name=constr_name)
                        # print "Branch flow induction ({}):".format(j)[:37].ljust(40), expr, "= 0"

    def _add_constraint_track_node_load(self, constraints, nodes_handled_by_other_gadgets):
        source_sink_request_nodes = set()
        for path in self.ext_graph.ecg_paths:
            if path.start_node not in self.ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(path.start_node)
            if path.end_node not in self.ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(path.end_node)
            for pos, ij in enumerate(path.original_path[:-1]):
                i, j = ij
                if j in nodes_handled_by_other_gadgets:
                    continue
                nodes_handled_by_other_gadgets.add(j)
                j_type = self.request.get_type(j)
                for u in self.request.get_allowed_nodes(j):
                    if u not in self.substrate.get_valid_nodes(self.request.get_type(j),
                                                               self.request.get_node_demand(j)):
                        continue
                    jl = path.original_path[pos + 1]
                    u_ij = self.ext_graph.path_layer_nodes[ij][u]
                    u_jl = self.ext_graph.path_layer_nodes[jl][u]
                    ext_edge = u_ij, u_jl
                    demand = self.request.get_node_demand(j)
                    constraints[j_type, u].append((demand, self.gurobi_vars["edge_flow"][ext_edge]))
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            if cycle.start_node not in self.ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(cycle.start_node)
            if cycle.end_node not in self.ext_graph.cycle_branch_nodes:
                source_sink_request_nodes.add(cycle.end_node)
            for branch in cycle.original_branches:
                for pos, ij in enumerate(branch[:-1]):
                    i, j = ij
                    if j in nodes_handled_by_other_gadgets:
                        continue
                    nodes_handled_by_other_gadgets.add(j)
                    j_type = self.request.get_type(j)
                    for u in self.request.get_allowed_nodes(j):
                        if u not in self.substrate.get_valid_nodes(self.request.get_type(j),
                                                                   self.request.get_node_demand(j)):
                            continue
                        for w in cycle_targets:
                            jl = branch[pos + 1]
                            u_ij = self.ext_graph.cycle_layer_nodes[ij][u][w]
                            u_jl = self.ext_graph.cycle_layer_nodes[jl][u][w]
                            ext_edge = u_ij, u_jl
                            demand = self.request.get_node_demand(j)
                            constraints[j_type, u].append((demand, self.gurobi_vars["edge_flow"][ext_edge]))
        for i in source_sink_request_nodes:
            if i in nodes_handled_by_other_gadgets:
                continue
            nodes_handled_by_other_gadgets.add(i)
            i_type = self.request.get_type(i)
            demand = self.request.get_node_demand(i)
            for u in self.request.get_allowed_nodes(i):
                if u not in self.substrate.get_valid_nodes(self.request.get_type(i),
                                                           self.request.get_node_demand(i)):
                    continue
                if i in self.ext_graph.source_nodes:
                    constraints[(i_type, u)].append((demand, self.gurobi_vars["node_flow"][i][u]))
                elif i in self.ext_graph.sink_nodes:
                    constraints[(i_type, u)].append((demand, self.gurobi_vars["node_flow"][i][u]))

    def _add_constraint_track_edge_load(self, constraints):
        for path in self.ext_graph.ecg_paths:
            for ij in path.original_path:
                i, j = ij
                for u, v in self.substrate.edges:  # TODO replace with allowed_edges when they are implemented
                    u_ij = self.ext_graph.path_layer_nodes[ij][u]
                    v_ij = self.ext_graph.path_layer_nodes[ij][v]
                    if (j, i) in self.ext_graph.reversed_request_edges:
                        ext_edge = v_ij, u_ij
                        demand = self.request.get_edge_demand((j, i))
                    else:
                        ext_edge = u_ij, v_ij
                        demand = self.request.get_edge_demand(ij)
                    constraints[(u, v)].append((demand, self.gurobi_vars["edge_flow"][ext_edge]))
        for cycle in self.ext_graph.ecg_cycles:
            valid_target_substrate_nodes = self.substrate.get_valid_nodes(
                self.request.get_type(cycle.end_node), self.request.get_node_demand(cycle.end_node)
            )
            cycle_targets = valid_target_substrate_nodes.intersection(self.request.get_allowed_nodes(cycle.end_node))

            for branch in cycle.original_branches:
                for ij in branch:
                    i, j = ij
                    for u, v in self.substrate.edges:
                        for w in cycle_targets:
                            u_ij = self.ext_graph.cycle_layer_nodes[ij][u][w]
                            v_ij = self.ext_graph.cycle_layer_nodes[ij][v][w]
                            if (j, i) in self.ext_graph.reversed_request_edges:
                                ext_edge = v_ij, u_ij
                                demand = self.request.get_edge_demand((j, i))
                            else:
                                ext_edge = u_ij, v_ij
                                demand = self.request.get_edge_demand(ij)
                            constraints[(u, v)].append((demand, self.gurobi_vars["edge_flow"][ext_edge]))

                            # print "Edge load ({}, {}):".format(u, v)[:37].ljust(40), expr, "= 0"

    def extend_mapping_by_own_solution(self, mapping, load, nodes_handled_by_other_gadgets):
        if self._mappings is None:
            self._extract_all_mappings()

        chosen_gadget_mapping = None
        if self.in_node in mapping.mapping_nodes:
            # choose consistently with existing mapping
            u_in_node = mapping.mapping_nodes[self.in_node]
            for gadget_mapping in self._mappings[(self.in_node, u_in_node)]:
                if self._remaining_flow_on_mapping[gadget_mapping] > self.rounding_threshold:
                    chosen_gadget_mapping = gadget_mapping
                    break
        else:
            # choose arbitrarily:
            for gadget_mapping, remaining_flow in self._remaining_flow_on_mapping.iteritems():
                if remaining_flow > self.rounding_threshold:
                    chosen_gadget_mapping = gadget_mapping
                    break
        if chosen_gadget_mapping is None:
            raise DecisionModelError("No remaining consistent mapping with non-zero flow!")
        self._most_recent_mapping = chosen_gadget_mapping

        for i, u in chosen_gadget_mapping.mapping_nodes.iteritems():
            nodes_handled_by_other_gadgets.add(i)
            if i in mapping.mapping_nodes:
                if u != mapping.mapping_nodes[i]:
                    raise DecisionModelError("Sanity check!")
            else:
                mapping.mapping_nodes[i] = u  # direct assignment is necessary to avoid the checks in the map_node function (e.g. our "request" has no nodes attribute)

        for ij, uv_list in chosen_gadget_mapping.mapping_edges.iteritems():
            if ij in mapping.mapping_edges:
                raise DecisionModelError("Sanity check! No edge overlap between gadgets!")
            mapping.mapping_edges[ij] = uv_list

        for res, x in self._mapping_loads[chosen_gadget_mapping].iteritems():
            load[res] += x

        # handle potential overlap with other gadgets:
        for i in nodes_handled_by_other_gadgets:
            if i in self.request.nodes:
                ntype = self.request.get_type(i)
                u = chosen_gadget_mapping.mapping_nodes[i]
                load[(ntype, u)] -= self.request.get_node_demand(i)

        return self._remaining_flow_on_mapping[chosen_gadget_mapping], self.out_nodes

    def reduce_flow_on_last_returned_mapping(self, used_flow):
        if self._most_recent_mapping is None:
            raise DecisionModelError("Need to call extend_mapping_by_own_solution before reduce_flow_on_last_returned_mapping!")

        self._remaining_flow_on_mapping[self._most_recent_mapping] -= used_flow

        if self._remaining_flow_on_mapping[self._most_recent_mapping] < -self.rounding_threshold:
            raise DecisionModelError("Sanity check!")
        self._most_recent_mapping = None

    def _extract_all_mappings(self):
        flow_values = self._make_flow_variable_dict()
        decomposition_alg = modelcreator_ecg_decomposition.Decomposition(
            self.request,
            self.substrate,
            flow_values,
            self.container_request.rounding_threshold,
            self.container_request.rounding_threshold,
            1e-10,
            extended_graph=self.ext_graph,
            substrate_resources=self.container_request.substrate_resources,
            # logger=self.logger
        )
        self._mappings = {(self.in_node, u): [] for u in self.request.get_allowed_nodes(self.in_node)}
        self._remaining_flow_on_mapping = {}
        self._mapping_flows = {}
        self._mapping_loads = {}
        for mapping, flow, load in decomposition_alg.compute_mappings():
            u = mapping.mapping_nodes[self.in_node]
            self._mappings[(self.in_node, u)].append(mapping)
            self._mapping_flows[mapping] = flow
            self._mapping_loads[mapping] = load
            self._remaining_flow_on_mapping[mapping] = flow

    def _make_flow_variable_dict(self):
        node_flow = self.gurobi_vars["node_flow"]
        edge_flow = self.gurobi_vars["edge_flow"]
        result = dict(
            embedding=sum(
                node_flow[self.ext_graph.root][u].x
                for u in self.ext_graph.source_nodes[self.ext_graph.root]
            ),
            node={
                i: {u: var.x for (u, var) in u_var_dict.iteritems() if var.x != 0.0}
                for (i, u_var_dict) in node_flow.iteritems()
            },
            edge={
                ext_edge: var.x for (ext_edge, var) in edge_flow.iteritems() if var.x != 0.0
            }
        )
        return result

    def adapt_model_objective_according_to_local_profits(self, obj_expr):
        return obj_expr

    def verify_mapping(self, mapping):
        """
        For a cactus gadget, each node and each edge must be mapped.

        :param mapping:
        :return:
        """
        missing_nodes = set()
        bad_node_mappings = set()
        missing_edges = set()
        for i in self.request.nodes:
            if i not in mapping.mapping_nodes:
                missing_nodes.add(i)
            elif mapping.mapping_nodes[i] not in self.request.get_allowed_nodes(i):
                bad_node_mappings.add(i)

        for ij in self.request.edges:
            if ij not in mapping.mapping_edges:
                missing_edges.add(ij)
                continue

        is_bad_mapping = False
        error_message = "{}, {} was only partially mapped.".format(self.container_request.name, self.name)
        if missing_nodes:
            is_bad_mapping = True
            error_message += "\n\tMissing node mappings:\n\t\t{}\n".format("\n\t\t".join(missing_nodes))
        if bad_node_mappings:
            is_bad_mapping = True
            error_message += "\n\tImproperly mapped nodes:"
            for i in bad_node_mappings:
                error_message += "\n\t\t{}  ->  {}  (expected one of {})".format(
                    i, mapping.mapping_nodes[i], self.request.get_allowed_nodes(i)
                )
        if missing_edges:
            is_bad_mapping = True
            error_message += "\n\tMissing edge mappings:\n\t\t{}\n".format("\n\t\t".join(str(ij) for ij in missing_edges))
        if is_bad_mapping:
            raise DecisionModelError(error_message)
        return self.out_nodes

    def _initialize_extended_graph(self):
        self.ext_graph = extendedcactusgraph.ExtendedCactusGraph(self.request, self.substrate)
