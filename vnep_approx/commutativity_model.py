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

"""
The commutativity model is a generalization of the cactus model.
"""
import itertools
from collections import deque
from random import Random

import gurobipy

from alib import datamodel, modelcreator, solutions

random = Random("extended_cactus_graph")

construct_name = modelcreator.build_construct_name(modelcreator.construct_name.spec + [
    ("comm_index", None, lambda v: "__".join(
        "{}_{}".format(i, u) for i, u in sorted(v, key=lambda iu: iu[0])
    )),
    ("bag", None, lambda v: "_".join(sorted(v))),
])


class CommutativityModelError(Exception):
    pass


class CommutativityModelCreator(modelcreator.AbstractEmbeddingModelCreator):
    """
    Model creator of the generalized commutativity model.
    """

    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=modelcreator.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(CommutativityModelCreator, self).__init__(
            scenario,
            gurobi_settings=gurobi_settings,
            optimization_callback=optimization_callback,
            lp_output_file=lp_output_file,
            potential_iis_filename=potential_iis_filename,
            logger=logger
        )
        self.request_labels = None

        self.var_node_mapping = {}
        self.var_aggregated_node_mapping = {}
        self.edge_sub_lp = {}  #: request -> request edge -> commutativity index -> EdgeSubLP
        self.var_request_load = {}

        # Used in solution extraction:
        # self._remaining_flow = None  # dictionary storing the remaining on a given extended graph resource during the decomposition algorithm
        self._used_flow_node_mapping = None
        self._used_flow_embedding_decision = {req: 0.0 for req in self.requests}

        self.fractional_decomposition_accuracy = 0.0001
        self.fractional_decomposition_abortion_flow = 0.001
        self._start_time_recovering_fractional_solution = None
        self._end_time_recovering_fractional_solution = None
        self.lost_flow_in_the_decomposition = 0.0

        self.dag_requests = {req: self._initialize_dag_request(req) for req in self.requests}
        self.reversed_substrate = self._make_reversed_substrate()

    def preprocess_input(self):
        """
        Create labels and edge sub-LPs for every request.
        """
        self.request_labels = {
            req: CommutativityLabels.create_labels(dag_request)
            for req, dag_request in self.dag_requests.iteritems()
        }
        self._prepare_edge_sub_lp()

    def _prepare_edge_sub_lp(self):
        """
        Create an :class:`EdgeSubLP` for every commutativity index of all
        request edges.
        """
        for req, dag_request in self.dag_requests.iteritems():
            labels = self.request_labels[req]
            self.edge_sub_lp[req] = {}
            for ij in dag_request.edges:
                self.edge_sub_lp[req][ij] = {}

                is_reversed_edge = ij not in req.edges
                oriented_sub = self.reversed_substrate if is_reversed_edge else self.substrate
                ij_labels = labels.get_edge_labels(ij)

                for commutativity_index in self.generate_label_comm_index(req, ij_labels):
                    self.edge_sub_lp[req][ij][commutativity_index] = EdgeSubLP(
                        dag_request, oriented_sub, ij, commutativity_index, is_reversed_edge
                    )

    def create_variables_other_than_embedding_decision_and_request_load(self):
        """
        Create node mapping and sub-LP Gurobi variables.
        """
        self._create_node_mapping_variables()
        self._create_sub_lp_variables()

    def _create_node_mapping_variables(self):
        """
        Create a node mapping variable for every commutativity index of all
        request nodes. Also initialize the used flows.
        """
        self._used_flow_node_mapping = {}
        for req in self.requests:
            labels = self.request_labels[req]

            self.var_aggregated_node_mapping[req] = {}
            self.var_node_mapping[req] = {}
            self._used_flow_node_mapping[req] = {}
            for i in req.get_nodes():
                self.var_aggregated_node_mapping[req][i] = {}
                self.var_node_mapping[req][i] = {}
                self._used_flow_node_mapping[req][i] = {}

                valid_nodes_i = self.substrate.get_valid_nodes(req.get_type(i), req.get_node_demand(i))
                allowed_valid_nodes_i = set(req.get_allowed_nodes(i)) & valid_nodes_i
                for u in allowed_valid_nodes_i:
                    variable_id = construct_name(
                        name="node_mapping_agg",
                        req_name=req.name,
                        snode=u,
                        vnode=i
                    )
                    self.var_aggregated_node_mapping[req][i][u] = self.model.addVar(
                        lb=0.0,
                        ub=1.0,
                        obj=0.0,
                        vtype=gurobipy.GRB.BINARY,
                        name=variable_id
                    )
                    self.var_node_mapping[req][i][u] = {}
                    self._used_flow_node_mapping[req][i][u] = {}

                    bag_list = labels.label_bags[i]
                    for bag_key in bag_list:
                        for commutativity_index in self.generate_label_comm_index(req, bag_key, {i: u}):
                            variable_id = construct_name(
                                name="node_mapping",
                                req_name=req.name,
                                snode=u,
                                vnode=i,
                                comm_index=commutativity_index
                            )
                            self.var_node_mapping[req][i][u][commutativity_index] = self.model.addVar(
                                lb=0.0,
                                ub=1.0,
                                obj=0.0,
                                vtype=gurobipy.GRB.BINARY,
                                name=variable_id
                            )
                            self._used_flow_node_mapping[req][i][u][commutativity_index] = 0.0

    def _create_sub_lp_variables(self):
        """
        Create the Gurobi variables of every sub-LP.
        """
        for req_edge_sub_lp in self.edge_sub_lp.values():
            for edge_sub_lp in req_edge_sub_lp.values():
                for sub_lp in edge_sub_lp.values():
                    sub_lp.extend_model_variables(self.model)

    def _get_valid_substrate_nodes(self, req, labels):
        """
        Returns a dictionary mapping request nodes to sets of allowed
        and valid substrate nodes.

        :param req:
        :param labels: iterable of request nodes (usually labels)
        :return:
        """
        return {
            k: set(req.get_allowed_nodes(k)) & self.substrate.get_valid_nodes(
                req.get_type(k), req.get_node_demand(k)
            )
            for k in labels
        }

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        """
        Create the Gurobi constraints for the sub-LPs, the node mappings, the
        node and edge loads and the embedding constraints.
        """
        self._create_force_embedding_constraint()
        self._create_sub_lp_constraints()
        self._create_node_mapping_constraints()
        self._create_constraints_track_node_loads()
        self._create_constraints_track_edge_loads()

    def _create_force_embedding_constraint(self):
        """
        Create a constraint for every request to force the embedding.
        """
        for req in self.requests:
            var_embedding_decision = self.var_embedding_decision[req]
            root_node = req.graph["root"]
            expr = [(-1.0, var_embedding_decision)]
            for agg_var in self.var_aggregated_node_mapping[req][root_node].values():
                expr.append((1.0, agg_var))

            constr_name = construct_name(
                "force_node_mapping_for_embedded_request",
                req_name=req.name
            )
            self.model.addConstr(gurobipy.LinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name=constr_name)

    def _create_sub_lp_constraints(self):
        """
        Create the Gurobi constraints of every sub-LP.
        """
        for req_edge_sub_lp in self.edge_sub_lp.values():
            for edge_sub_lp in req_edge_sub_lp.values():
                for sub_lp in edge_sub_lp.values():
                    sub_lp.extend_model_constraints(self.model)

    def _create_node_mapping_constraints(self):
        """
        Create the node mapping constraints.
        """
        for req, dag_request in self.dag_requests.iteritems():
            labels = self.request_labels[req]
            for i in req.get_nodes():
                valid_nodes_i = self.substrate.get_valid_nodes(req.get_type(i), req.get_node_demand(i))
                allowed_valid_nodes_i = set(req.get_allowed_nodes(i)) & valid_nodes_i
                for u in allowed_valid_nodes_i:
                    for bag_key in labels.label_bags[i]:
                        self._create_node_mapping_aggregation_constraints(req, i, u, bag_key)
                        self._create_node_mapping_constraints_between_in_edges_and_bags(req, i, u, bag_key)
                        self._create_node_mapping_constraints_within_bag(req, i, u, bag_key)

    def _create_node_mapping_aggregation_constraints(self, req, i, u, bag_key):
        """
        Create the node mapping aggregation constraints.

        The sum of the node mapping variables of a bag should be equal to the
        aggregated node mapping variable.

        :param datamodel.Request req: the request
        :param i: request node
        :param u: substrate node of i
        :param bag_key: labels of the bag
        """
        expr = [(-1.0, self.var_aggregated_node_mapping[req][i][u])]
        for comm_index in self.generate_label_comm_index(req, bag_key):
            expr.append((1.0, self.var_node_mapping[req][i][u][comm_index]))

        constr_name = construct_name(
            "node_mapping_aggregation",
            req_name=req.name,
            vnode=i,
            snode=u,
            bag=bag_key,
        )
        self.model.addConstr(gurobipy.LinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name=constr_name)

    def _create_node_mapping_constraints_between_in_edges_and_bags(self, req, i, u, bag_key):
        """
        Create the node mapping constraints between the in-edges of a node and the bags (which contain its out-edges).

        :param datamodel.Request req: the request
        :param i: request node
        :param u: substrate node of i
        :param bag_key: labels of the bag
        """
        labels = self.request_labels[req]
        # TODO: We might only need to consider one in-edge here
        for ji in self.dag_requests[req].get_in_edges(i):
            ji_labels = labels.get_edge_labels(ji)
            intersection_labels = ji_labels & bag_key

            for intersection_comm_index in self.generate_label_comm_index(req, intersection_labels):
                expr = []
                extra_labels_in_bag = bag_key - ji_labels
                if extra_labels_in_bag:
                    for bag_comm_index in self.generate_label_comm_index(req, extra_labels_in_bag):
                        # complete mapping_combination to match bag_key
                        complete_index = intersection_comm_index | bag_comm_index
                        expr.append((1.0, self.var_node_mapping[req][i][u][complete_index]))
                else:
                    expr.append((1.0, self.var_node_mapping[req][i][u][intersection_comm_index]))

                extra_labels_in_edge = ji_labels - bag_key
                if extra_labels_in_edge:
                    for edge_comm_index in self.generate_label_comm_index(req, extra_labels_in_edge, fixed_mappings={i: u}):
                        # complete mapping_combination to match edge comm_index
                        complete_index = intersection_comm_index | edge_comm_index
                        expr.append((-1.0, self.edge_sub_lp[req][ji][complete_index].var_node_flow_sink[u]))
                else:
                    expr.append((-1.0, self.edge_sub_lp[req][ji][intersection_comm_index].var_node_flow_sink[u]))

                constr_name = construct_name(
                    "node_mapping_in_edge_to_bag",
                    req_name=req.name,
                    vnode=i,
                    snode=u,
                    vedge=ji,
                    comm_index=intersection_comm_index,
                    bag=bag_key
                )
                self.model.addConstr(gurobipy.LinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name=constr_name)

    def _create_node_mapping_constraints_within_bag(self, req, i, u, bag_key):
        """
        Create the node mapping constraints within a bag.

        :param datamodel.Request req: the request
        :param i: request node
        :param u: substrate node of i
        :param bag_key: labels of the bag
        """
        for ij in self.request_labels[req].label_bags[i][bag_key]:
            for comm_index, sub_lp in self.edge_sub_lp[req][ij].iteritems():
                source_var = sub_lp.var_node_flow_source[u]

                expr = [(-1.0, source_var)]
                missing_labels = bag_key - sub_lp.labels
                for completion_comm_index in self.generate_label_comm_index(req, missing_labels, fixed_mappings={i: u}):
                    complete_comm_index = sub_lp.commutativity_index | completion_comm_index
                    expr.append((1.0, self.var_node_mapping[req][i][u][complete_comm_index]))

                constr_name = construct_name(
                    "node_mapping",
                    req_name=req.name,
                    vnode=i,
                    snode=u,
                    vedge=sub_lp.ij,
                    comm_index=comm_index
                )
                self.model.addConstr(gurobipy.LinExpr(expr), gurobipy.GRB.EQUAL, 0.0, name=constr_name)

    def _create_constraints_track_node_loads(self):
        """
        Create constraints to track the node loads.
        """
        for req in self.requests:
            node_load_constr_dict = {
                node_res: [(-1.0, self.var_request_load[req][node_res])]
                for node_res in self.substrate.substrate_node_resources
            }
            for i in req.nodes:
                i_type = req.get_type(i)
                i_demand = req.get_node_demand(i)
                for u, var in self.var_aggregated_node_mapping[req][i].items():
                    node_res = i_type, u
                    node_load_constr_dict[node_res].append((i_demand, var))

            for node_res, expr in node_load_constr_dict.iteritems():
                u_type, u = node_res
                constr_name = construct_name(
                    "track_node_load",
                    req_name=req.name,
                    snode=u,
                    type=u_type
                )
                self.model.addConstr(gurobipy.LinExpr(expr),
                                     gurobipy.GRB.EQUAL,
                                     0.0,
                                     name=constr_name)

    def _create_constraints_track_edge_loads(self):
        """
        Create constraints to track the edge loads.
        """
        for req, edge_comm_index_sub_lp_dict in self.edge_sub_lp.iteritems():
            edge_load_constr_dict = {
                uv: [(-1.0, self.var_request_load[req][uv])]
                for uv in self.substrate.edges
            }

            for comm_index_sub_lp_dict in edge_comm_index_sub_lp_dict.values():
                for sub_lp in comm_index_sub_lp_dict.values():
                    sub_lp.extend_edge_load_constraints(edge_load_constr_dict)

            for uv, expr in edge_load_constr_dict.iteritems():
                constr_name = construct_name(
                    "track_edge_load",
                    req_name=req.name,
                    sedge=uv
                )
                self.model.addConstr(gurobipy.LinExpr(expr),
                                     gurobipy.GRB.EQUAL,
                                     0.0,
                                     name=constr_name)

    def generate_label_comm_index(self, req, labels, fixed_mappings=None):
        """
        For a set of labels, yield each possible commutativity index as a list of tuples.

        :param datamodel.Request req: the request
        :param labels: a label set
        :param fixed_mappings: nodes which should be included in the comm-index with a fixed mapping
        """
        allowed_label_mappings = self._get_valid_substrate_nodes(req, labels)
        if fixed_mappings:
            allowed_label_mappings.update(fixed_mappings)

        sorted_labels = sorted(labels)
        mapping_combinations = itertools.product(*[allowed_label_mappings[k] for k in sorted_labels])
        for label_mapping_combination in mapping_combinations:
            yield frozenset(zip(sorted_labels, label_mapping_combination))

    def recover_integral_solution_from_variables(self):
        frac_sol = self.recover_fractional_solution_from_variables()
        self.solution = solutions.IntegralScenarioSolution(
            "integral_solution_{}".format(self.scenario.name),
            self.scenario
        )
        for req, mapping_list in frac_sol.request_mapping.items():
            for m in mapping_list:
                if abs(1.0 - frac_sol.mapping_flows[m.name]) < self.fractional_decomposition_accuracy:
                    self.solution.add_mapping(req, m)
                    break
        return self.solution

    def post_process_integral_computation(self):
        return self.solution

    def recover_fractional_solution_from_variables(self):
        """
        Recover the fractional solution from the Gurobi variables.

        Extract mappings as long as flow is remaining.
        """
        self.solution = solutions.FractionalScenarioSolution(
            "fractional_solution_{}".format(self.scenario.name),
            self.scenario
        )
        for req in self.requests:
            labels = self.request_labels[req]
            mapping_count = 1

            while self.var_embedding_decision[req].x - self._used_flow_embedding_decision[req] > self.fractional_decomposition_abortion_flow:
                mapping, mapping_flow = self.extract_request_mapping(req, labels, mapping_count)
                mapping_count += 1
                self.reduce_flow(mapping, mapping_flow)
                mapping_load = self._calculate_substrate_load_for_mapping(mapping)
                self.solution.add_mapping(req, mapping, mapping_flow, mapping_load)
        return self.solution

    def post_process_fractional_computation(self):
        return self.solution

    def extract_request_mapping(self, req, labels, mapping_count):
        """
        Extract a request mapping.

        This is called by :meth:`recover_fractional_solution_from_variables`
        as long as there is flow remaining. If there is remaining flow,
        this function should find a mapping.

        :param datamodel.Request req: the request
        :param CommutativityLabels labels: the labels of the request
        :param int mapping_count: the mapping count used for the mapping name
        :return: a mapping of the request and the corresponding flow
        :rtype: (solutions.Mapping, float)
        """
        mapping = solutions.Mapping("mapping_{}_{}".format(req.name, mapping_count), req, self.substrate, True)  # TODO: Ask Matthias about is_embedded
        root = req.graph["root"]
        queue = deque([root])
        added_to_queue = {root}
        min_flow_for_mapping = self.var_embedding_decision[req].x - self._used_flow_embedding_decision[req]

        u_root_candidates = self.var_node_mapping[req][root].keys()
        for u in u_root_candidates:
            for comm_index, var in self.var_node_mapping[req][root][u].iteritems():
                if var.x - self._used_flow_node_mapping[req][root][u][comm_index] > self.fractional_decomposition_accuracy:
                    mapping.map_node(root, u)
                    break
            if root in mapping.mapping_nodes:
                break

        while queue:
            i = queue.popleft()
            u = mapping.mapping_nodes[i]
            for comm_index, var in self.var_node_mapping[req][i][u].iteritems():
                i_mapping_remaining_flow = var.x - self._used_flow_node_mapping[req][i][u][comm_index]
                if i_mapping_remaining_flow <= self.fractional_decomposition_accuracy:
                    continue

                bag_key = {k for k, v in comm_index}
                mapped_nodes_index = self.get_commutativity_index_from_mapping(bag_key, mapping)
                if not (mapped_nodes_index <= comm_index):
                    # index is incompatible with current mapping
                    continue

                # comm_index is a valid choice, extend mapping by it
                for k, w_k in comm_index:
                    if k not in mapping.mapping_nodes:
                        mapping.map_node(k, w_k)
                min_flow_for_mapping = min(min_flow_for_mapping, i_mapping_remaining_flow)

            for j in self.dag_requests[req].get_out_neighbors(i):
                # handle edge ij
                # pick a comm index consistent with mapping
                ij = i, j
                ij_labels = labels.get_edge_labels(ij)

                # labels of ij are already fixed because labels of i are fixed
                comm_index = self.get_commutativity_index_from_mapping(ij_labels, mapping)

                ij_sub_lp = self.edge_sub_lp[req][ij][comm_index]
                uv_list, v_j, flow = ij_sub_lp.extract_edge_mapping(mapping)
                min_flow_for_mapping = min(min_flow_for_mapping, flow)

                if ij_sub_lp.is_reversed_edge:
                    ij_original_orientation = j, i
                else:
                    ij_original_orientation = ij

                if j not in mapping.mapping_nodes:
                    mapping.map_node(j, v_j)
                mapping.map_edge(ij_original_orientation, uv_list)

                if j not in added_to_queue:
                    queue.append(j)
                    added_to_queue.add(j)
        return mapping, min_flow_for_mapping

    def reduce_flow(self, mapping, used_flow):
        """
        Reduce the flow on the mapping by increasing the used flows.

        :param mapping: the full mapping
        :param used_flow: the amount of flow to reduce
        """
        req = mapping.request
        self._used_flow_embedding_decision[req] += used_flow
        labels = self.request_labels[req]

        # reduce flow of node mapping variables
        for i, u in mapping.mapping_nodes.iteritems():
            for bag_key in labels.label_bags[i]:
                comm_index = self.get_commutativity_index_from_mapping(bag_key, mapping)
                self._used_flow_node_mapping[req][i][u][comm_index] += used_flow

        # reduce flow of edge variables
        for ij, uv_list in mapping.mapping_edges.iteritems():
            if ij not in self.dag_requests[req].edges:
                ij_dag_orientation = ij[1], ij[0]
            else:
                ij_dag_orientation = ij
            ij_labels = labels.get_edge_labels(ij_dag_orientation)
            comm_index = self.get_commutativity_index_from_mapping(ij_labels, mapping)
            self.edge_sub_lp[req][ij_dag_orientation][comm_index].reduce_flow(mapping, used_flow)

    def _calculate_substrate_load_for_mapping(self, mapping):
        """
        Calculate the substrate load for the mapping.

        :param mapping:
        :return: a load dict
        """
        req = mapping.request
        load_dict = {(x, y): 0.0 for (x, y) in self.substrate.substrate_resources}
        for i in req.nodes:
            u = mapping.get_mapping_of_node(i)
            t = req.get_type(i)
            demand = req.get_node_demand(i)
            load_dict[(t, u)] += demand
        for ij in req.edges:
            demand = req.get_edge_demand(ij)
            for uv in mapping.mapping_edges[ij]:
                load_dict[uv] += demand
        return load_dict

    @staticmethod
    def get_commutativity_index_from_mapping(labels, mapping):
        """
        Get the mapping of the labels that are already fixed.

        :param labels:
        :param solutions.Mapping mapping:
        :return: the commutativity index
        """
        return frozenset((k, mapping.mapping_nodes[k]) for k in labels if k in mapping.mapping_nodes)

    @classmethod
    def _initialize_dag_request(cls, req):
        """
        Create a DAG request for the request.

        :param req: the request
        :return: a rooted DAG request
        """
        dag_request = datamodel.Request("{}_dag".format(req.name))
        for node in req.nodes:
            demand = req.get_node_demand(node)
            ntype = req.get_type(node)
            allowed = req.get_allowed_nodes(node)
            dag_request.add_node(node, demand=demand, ntype=ntype, allowed_nodes=allowed)
        cls._add_dag_edges_and_initialize_exploration_queue_with_leaf_nodes(req, dag_request)
        return dag_request

    @classmethod
    def _add_dag_edges_and_initialize_exploration_queue_with_leaf_nodes(cls, req, dag_request):
        """
        Explore the original request and create the edges of the DAG.

        :param req: the original request
        :param dag_request: the DAG request
        """
        visited = set()
        root = req.graph.get("root", None)
        if root is None:
            root = random.choice(list(req.nodes))
        dag_request.graph["root"] = root

        queue = deque([root])
        while queue:
            current_node = queue.popleft()
            for out_neighbor in req.get_out_neighbors(current_node):
                if out_neighbor in visited:
                    continue
                cls._add_edge_to_dag_request(req, dag_request, (current_node, out_neighbor), is_reversed=False)
                if out_neighbor not in queue:
                    queue.append(out_neighbor)
            for in_neighbor in req.get_in_neighbors(current_node):
                if in_neighbor in visited:
                    continue
                cls._add_edge_to_dag_request(req, dag_request, (in_neighbor, current_node), is_reversed=True)
                if in_neighbor not in queue:
                    queue.append(in_neighbor)
            visited.add(current_node)
        if len(visited) != len(req.nodes):
            missing_nodes = req.nodes - visited
            msg = "Request graph may have multiple components: Nodes {} were not visited by bfs.\n{}".format(
                missing_nodes, req
            )
            raise CommutativityModelError(msg)

    @staticmethod
    def _add_edge_to_dag_request(req, dag_request, edge, is_reversed):
        """
        Add an edge to the DAG.

        :param datamodel.Request req: the original request
        :param datamodel.Request dag_request: the DAG request
        :param edge:
        :param is_reversed:
        """
        if is_reversed:
            new_head, new_tail = edge
        else:
            new_tail, new_head = edge
        demand = req.get_edge_demand(edge)
        dag_request.add_edge(new_tail, new_head, demand)

    def _make_reversed_substrate(self):
        """
        Generate a copy of the substrate with all edges reversed.

        All default substrate node and edge properties are preserved.

        :return: the reversed substrate
        :rtype: datamodel.SubstrateX
        """
        reversed_substrate = datamodel.Substrate("{}_reversed".format(self.substrate.name))
        for node in self.substrate.nodes:
            reversed_substrate.add_node(node,
                                        self.substrate.node[node]["supported_types"],
                                        self.substrate.node[node]["capacity"],
                                        self.substrate.node[node]["cost"])
            reversed_substrate.node[node] = self.substrate.node[node]
        for tail, head in self.substrate.edges:
            original_edge_properties = self.substrate.edge[(tail, head)]
            reversed_substrate.add_edge(head, tail,
                                        latency=original_edge_properties["latency"],
                                        capacity=original_edge_properties["capacity"],
                                        cost=original_edge_properties["cost"],
                                        bidirected=False)
            # copy any additional entries of the substrate's edge dict:
            for key, value in self.substrate.edge[(tail, head)].iteritems():
                if key not in reversed_substrate.edge[(head, tail)]:
                    reversed_substrate.edge[(head, tail)][key] = value
        return datamodel.SubstrateX(reversed_substrate)


class EdgeSubLP(object):
    """
    An edge sub-LP handles the model variables and constraints and is used to
    extract the mapping.
    """

    def __init__(self, dag_request, substrate_oriented, ij, commutativity_index, is_reversed_edge):
        self.dag_req = dag_request
        self.ij = ij

        self.commutativity_index = commutativity_index
        self.labels = set(i for i, u in commutativity_index)

        self.i, self.j = ij
        self.is_reversed_edge = is_reversed_edge
        self.substrate_oriented = substrate_oriented

        self.var_edge_flow = {}
        self.var_node_flow_source = {}
        self.var_node_flow_sink = {}

        valid_nodes_i = self.substrate_oriented.get_valid_nodes(self.dag_req.get_type(self.i), self.dag_req.get_node_demand(self.i))
        self.allowed_valid_nodes_i = set(self.dag_req.get_allowed_nodes(self.i)) & valid_nodes_i

        valid_nodes_j = self.substrate_oriented.get_valid_nodes(self.dag_req.get_type(self.j), self.dag_req.get_node_demand(self.j))
        allowed_valid_nodes_j = set(self.dag_req.get_allowed_nodes(self.j)) & valid_nodes_j
        self.allowed_valid_nodes_j = {
            v for v in allowed_valid_nodes_j
            if self.j not in self.labels or (self.j, v) in self.commutativity_index
        }

        self.valid_edges = self.substrate_oriented.get_valid_edges(self.dag_req.get_edge_demand(self.ij))

        self._used_flow = {uv: 0.0 for uv in self.valid_edges}
        self._used_flow_source = {u: 0.0 for u in self.allowed_valid_nodes_i}
        self._used_flow_sink = {v: 0.0 for v in self.allowed_valid_nodes_j}

    def extend_model_variables(self, model):
        """
        Extend the Gurobi model with the node and edge flow variables of the
        edge sub-LP.

        :param gurobipy.Model model: the Gurobi model
        """
        req_name = self.dag_req.name.rsplit("_dag", 1)[0]
        for u in self.allowed_valid_nodes_i:
            variable_id = construct_name(
                name="node_flow_source",
                req_name=req_name,
                snode=u,
                vedge=self.ij,
                comm_index=self.commutativity_index
            )
            self.var_node_flow_source[u] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=0.0,
                vtype=gurobipy.GRB.BINARY,
                name=variable_id
            )

        for uv in self.valid_edges:
            variable_id = construct_name(
                name="edge_flow",
                req_name=req_name,
                sedge=uv,
                vedge=self.ij,
                comm_index=self.commutativity_index
            )
            self.var_edge_flow[uv] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=0.0,
                vtype=gurobipy.GRB.BINARY,
                name=variable_id
            )

        for v in self.allowed_valid_nodes_j:
            variable_id = construct_name(
                name="node_flow_sink",
                req_name=req_name,
                snode=v,
                vedge=self.ij,
                comm_index=self.commutativity_index
            )
            self.var_node_flow_sink[v] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=0.0,
                vtype=gurobipy.GRB.BINARY,
                name=variable_id
            )

    def extend_model_constraints(self, model):
        """
        Extend the Gurobi model with the flow preservation constraints of the
        edge sub-LP.

        :param gurobipy.Model model: the Gurobi model
        """
        req_name = self.dag_req.name.rsplit("_dag", 1)[0]
        for u in self.substrate_oriented.nodes:
            expr = []
            # inflow
            for v in self.substrate_oriented.get_in_neighbors(u):

                if (v, u) in self.valid_edges:
                    expr.append((-1.0, self.var_edge_flow[v, u]))
            if u in self.allowed_valid_nodes_i:
                expr.append((-1.0, self.var_node_flow_source[u]))

            # outflow
            for v in self.substrate_oriented.get_out_neighbors(u):
                if (u, v) in self.valid_edges:
                    expr.append((1.0, self.var_edge_flow[u, v]))
            if u in self.allowed_valid_nodes_j:
                expr.append((1.0, self.var_node_flow_sink[u]))

            # create constraint for ij, u, commutativity index
            expr = gurobipy.LinExpr(expr)

            constr_name = construct_name(
                "flow_preservation",
                req_name=req_name,
                vedge=self.ij,
                snode=u,
                comm_index=self.commutativity_index
            )
            model.addConstr(expr, gurobipy.GRB.EQUAL, 0.0, name=constr_name)

    def extend_edge_load_constraints(self, load_constraint_dict):
        """
        Extend the edge load constraints by adding this sub-LP's edge mapping
        variables.

        The constraint is converted to a :class:`gurobipy.LinExpr` and added
        to the model by the :class:`CommutativityModelCreator`.

        Node mapping load is handled in the ModelCreator using the (global)
        aggregation layer variables.

        :param load_constraint_dict:
            Dictionary mapping substrate edges to list of tuples of
            coefficients & gurobi variables
        :return: the extended load_constraint_dict
        """
        ij_demand = self.dag_req.get_edge_demand(self.ij)
        for uv, var in self.var_edge_flow.iteritems():
            uv_original_orientation = uv
            if self.is_reversed_edge:
                uv_original_orientation = uv[1], uv[0]
            load_constraint_dict[uv_original_orientation].append(
                (ij_demand, var)
            )
        return load_constraint_dict

    def extract_edge_mapping(self, partial_mapping):
        """
        Extract the edge mapping of the sub-LP.

        The partial mapping is used to specify the allowed entries (sources)
        and exits (sinks) of the sub-LP.
        The entry node should be fixed in the node mapping in
        :meth:`CommutativityModelCreator.extract_request_mapping` before.
        The exit node can also be fixed if it was contained in the
        commutativity index before. Otherwise all sinks with a remaining flow
        larger than 0 are used.

        The path between the entry and a allowed exit is found by using sub-LP
        edges with a remaining flow larger than 0.

        :param partial_mapping: the partial global mapping
        :return: the path from an entry to an exit, the exit,
            the minimum flow on the path
        """
        if self.j in partial_mapping.mapping_nodes:
            allowed_exits = {partial_mapping.mapping_nodes[self.j]}
        else:
            allowed_exits = {
                u for u in self.allowed_valid_nodes_j
                if self.var_node_flow_sink[u].x - self._used_flow_sink[u] > 0.0
            }
        u_i = partial_mapping.mapping_nodes[self.i]
        q = deque([u_i])
        predecessor = {u_i: None}
        v_j = None
        while q:
            u = q.popleft()
            if u in allowed_exits:
                v_j = u
                break
            for uv in self.substrate_oriented.get_out_edges(u):
                v = uv[1]
                if v in predecessor:
                    continue
                if self.var_edge_flow[uv].x - self._used_flow[uv] > 0.0:
                    predecessor[v] = u
                    if v in allowed_exits:
                        q.appendleft(v)  # append it left so that we can quickly exit the outer loop
                    else:
                        q.append(v)
        if v_j is None:
            raise ValueError("Did not find valid edge mapping for {}".format(self.ij))

        uv_list = []
        current_node = v_j
        min_flow_on_path = self.var_node_flow_source[u_i].x - self._used_flow_source[u_i]
        while current_node != u_i:
            parent = predecessor[current_node]
            uv = (parent, current_node)
            uv_list.append(uv)
            min_flow_on_path = min(min_flow_on_path, self.var_edge_flow[uv].x - self._used_flow[uv])
            current_node = parent
        min_flow_on_path = min(min_flow_on_path, self.var_node_flow_sink[v_j].x - self._used_flow_sink[v_j])
        if self.is_reversed_edge:
            uv_list = [(v, u) for (u, v) in uv_list]
        else:
            uv_list.reverse()
        return uv_list, v_j, min_flow_on_path

    def reduce_flow(self, mapping, used_flow):
        """
        Reduce the flow on the edge mapping.

        The flow reduction is done by increasing the used flow of the source,
        the sink and every substrate edge of the path.

        :param mapping: the full mapping
        :param used_flow: the amount of flow to reduce
        """
        if self.is_reversed_edge:
            ij_original_orientation = self.j, self.i
        else:
            ij_original_orientation = self.ij

        uv_list = mapping.mapping_edges[ij_original_orientation]
        if self.is_reversed_edge:
            uv_list = ((v, u) for (u, v) in uv_list)

        for uv in uv_list:
            self._used_flow[uv] += used_flow

        u_i = mapping.mapping_nodes[self.i]
        self._used_flow_source[u_i] += used_flow

        v_j = mapping.mapping_nodes[self.j]
        self._used_flow_sink[v_j] += used_flow

    def __repr__(self):
        return construct_name(
            "EdgeSubLP",
            req_name=self.dag_req.name.rsplit("_dag", 1)[0],
            vedge=self.ij,
            comm_index=self.commutativity_index
        )


class CommutativityLabels(object):
    """
    A :class:`CommutativityLabels` object contains labels for all nodes of a
    DAG request.

    A node is labeled with the end nodes of all possible cycles containing the
    node.
    """

    def __init__(self, node_labels, dag_request):
        self.node_labels = node_labels
        self.dag_request = dag_request
        self.label_bags = {i: self.calculate_edge_label_bags(i) for i in dag_request.nodes}

    @classmethod
    def create_labels(cls, dag_request):
        """
        Calculate labels of a DAG request.

        :param dag_request: the DAG request
        :return: the calculated labels
        :rtype: CommutativityLabels
        """
        root = dag_request.graph["root"]

        # 1. Determine the reachable end nodes of every request node. An end
        # node of a cycle is a node with at least two in-neighbors. Start from
        # leaf nodes going up.

        visited = set()
        node_queue = {i for i in dag_request.nodes
                      if not dag_request.get_out_neighbors(i)}
        # request node i -> all "end nodes" that are reachable from i
        reachable_end_nodes = {}
        while node_queue:
            i = node_queue.pop()
            visited.add(i)
            i_parents = dag_request.get_in_neighbors(i)

            reachable_end_nodes[i] = {j for child in dag_request.get_out_neighbors(i)
                                      for j in reachable_end_nodes[child]}
            if i != root and not i_parents:
                raise CommutativityModelError(("Node {} has no parents, but is not root (root is {}). "
                                               "May have multiple components or not be rooted").format(i, root))
            elif i == root and i_parents:
                raise CommutativityModelError("Root node {} has in-neighbors {}".format(root, i_parents))
            if len(i_parents) >= 2:
                reachable_end_nodes[i].add(i)
            for p in i_parents:
                if all(child in visited for child in dag_request.get_out_neighbors(p)):
                    node_queue.add(p)

        # 2. Check pairwise overlaps of the children of every request node that has
        # multiple children. Identify end nodes that are in at least one overlap of
        # the children.

        # request node i -> set of end nodes that are reachable from more than one of i's children
        overlaps = {i: set() for i in dag_request.nodes}
        for i in dag_request.nodes:
            children = dag_request.get_out_neighbors(i)
            # note that combinations are empty if there are less than two children
            for left_child, right_child in itertools.combinations(children, 2):
                left_reachable = reachable_end_nodes[left_child]
                right_reachable = reachable_end_nodes[right_child]
                overlap = left_reachable & right_reachable
                overlaps[i] |= overlap

        # 3. Reduce the overlapping sets by removing dominated nodes so that only the
        # actual cycle end nodes of simple cycles remain.
        # If it is not possible to find a path from a possible start node to an end
        # node from the overlapping set without visiting one avoided node, the
        # possible start node is not a start node of the end node (dominated). Try
        # other nodes from the overlapping set as avoid node.

        # request node i -> set of end nodes of cycles starting at i
        end_nodes = {i: set(overlaps[i]) for i in dag_request.nodes}
        for i in dag_request.nodes:
            for j in overlaps[i]:
                j_is_dominated = False
                for avoid_node in overlaps[i] - {j}:
                    # find path from i to j without visiting avoid_node
                    path_found = False
                    stack = [i]
                    while stack and not path_found:
                        node = stack.pop()
                        for child in dag_request.get_out_neighbors(node):
                            if child == j:
                                path_found = True
                                break
                            if child != avoid_node:
                                stack.append(child)
                    if not path_found:
                        j_is_dominated = True
                        break

                if j_is_dominated:
                    end_nodes[i].remove(j)

        # 4. Propagate the labels to all nodes on paths between the start and end nodes.

        # request node i -> set of end nodes of all cycles to which i belongs
        node_labels = {i: set() for i in dag_request.nodes}
        node_queue = {root}
        reachable_start_node_labels = {}  # labels of reachable start nodes
        while node_queue:
            i = node_queue.pop()

            try:
                reachable_start_node_labels[i] = {j for parent in dag_request.get_in_neighbors(i)
                                                  for j in reachable_start_node_labels[parent]}
            except KeyError as e:
                k = e.args[0]
                raise CommutativityModelError("Request is no DAG: cycle contains {}, {}".format(i, k))
            reachable_start_node_labels[i] |= end_nodes[i]
            node_labels[i] = reachable_end_nodes[i] & reachable_start_node_labels[i]

            for child in dag_request.get_out_neighbors(i):
                if all(parent in reachable_start_node_labels for parent in dag_request.get_in_neighbors(child)):
                    node_queue.add(child)

        return cls(node_labels, dag_request)

    def calculate_edge_label_bags(self, i):
        """
        Given a request node, group its outgoing edges into bags according
        to overlapping edge labels.

        :param i: request node
        :return: dictionary (node label set -> set of edges in that bag)
        """

        label_groups = {}  # label -> frozenset of labels
        bags = {}  # frozenset of labels -> set of edges in this bag
        tree_edges = set()
        for ij in self.dag_request.get_out_edges(i):
            ij_labels = frozenset(self.get_edge_labels(ij))
            if not ij_labels:
                tree_edges.add(ij)
                continue
            merge_labels = ij_labels
            merge_edges = {ij}
            for label in ij_labels:
                if label in label_groups:
                    merge_labels |= label_groups[label]
                    if label_groups[label] in bags:
                        merge_edges |= bags.pop(label_groups[label])
            for label in merge_labels:
                label_groups[label] = merge_labels
            bags[merge_labels] = merge_edges

        if tree_edges:
            bags[frozenset()] = tree_edges

        if not bags:
            # if the node has no out-edges, there are no bags, but we still want to generate
            # a node mapping variable & constraints to make the decomposition easier
            bags[frozenset()] = set()

        return bags

    def get_edge_labels(self, ij):
        """
        Calculate the labels of the request edge ij.

        The labels of ij are the intersection of the labels of i and j.

        :param ij: the edge
        :return: the labels of ij
        """
        i, j = ij
        labels_i = self.node_labels[i]
        labels_j = self.node_labels[j]
        return labels_i & labels_j
