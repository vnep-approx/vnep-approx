import random
from copy import deepcopy

from gurobipy import GRB, LinExpr

from alib import datamodel as dm
from alib import mip as mip
from alib import modelcreator as mc
from alib import solutions
from alib import util


class NodeMappingMethod(object):
    DETERMINISTIC = "deterministic"
    RANDOMIZED = "randomized"

    AVAILABLE_METHODS = [DETERMINISTIC, RANDOMIZED]


class EdgeMappingMethod(object):
    SPLITTABLE = "splittable"
    SHORTEST_PATH = "shortest_path"

    AVAILABLE_METHODS = [SPLITTABLE, SHORTEST_PATH]


class WiNESingleWindow(object):
    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None,
                 node_mapping_method=NodeMappingMethod.DETERMINISTIC,
                 edge_mapping_method=EdgeMappingMethod.SHORTEST_PATH,
                 use_load_balancing_objective=False):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename

        if logger is None:
            logger = util.get_logger(__name__, make_file=False, propagate=True)

        self.logger = logger
        self.scenario = scenario
        self.node_mapping_method = node_mapping_method
        self.edge_mapping_method = edge_mapping_method
        self.use_load_balancing_objective = use_load_balancing_objective

    def init_modelcreator(self):
        pass

    def compute_integral_solution(self):
        # TODO: Measure runtimes
        vine_instance = ViNESingleScenario(
            scenario=self.scenario,
            node_mapping_method=self.node_mapping_method,
            edge_mapping_method=self.edge_mapping_method,
            use_load_balancing_objective=self.use_load_balancing_objective,
            gurobi_settings=self.gurobi_settings,
            optimization_callback=self.optimization_callback,
            lp_output_file=self.lp_output_file,
            potential_iis_filename=self.potential_iis_filename,
            logger=self.logger,
        )
        solution_name = mc.construct_name("solution_", sub_name=self.scenario.name)
        solution = solutions.IntegralScenarioSolution(solution_name, self.scenario)
        for req in sorted(self.scenario.requests, key=lambda r: r.profit, reverse=True):
            mapping = vine_instance.vine_procedure_single_request(req)
            solution.add_mapping(req, mapping)
        return solution


class SplittableMapping(solutions.Mapping):
    def map_edge(self, ij, edge_vars):
        self.mapping_edges[ij] = edge_vars


class FractionalClassicMCFModel(mip.ClassicMCFModel):
    """ This Modelcreator is used to access the raw LP values. """

    def __init__(self, scenario,
                 use_load_balancing_objective,
                 gurobi_settings=None,
                 logger=None):
        super(FractionalClassicMCFModel, self).__init__(
            scenario=scenario, gurobi_settings=gurobi_settings, logger=logger
        )
        self.use_load_balancing_objective = use_load_balancing_objective

    def recover_fractional_solution_from_variables(self):
        """
        As the ClassicMCFModel does not implement any fractional solution methods, we define this placeholder, which
        is required by AbstractModelCreator.compute_fractional_solution.
        """
        pass

    def post_process_fractional_computation(self):
        """ Collect all LP variable assignments in a single dictionary """
        variable_assignment = {}
        for req in self.scenario.requests:
            variable_assignment[req] = dict(
                node_vars={},
                edge_vars={},
            )
            for i in req.nodes:
                variable_assignment[req]["node_vars"][i] = {}
                allowed_nodes = req.get_allowed_nodes(i)
                for u in allowed_nodes:
                    variable_assignment[req]["node_vars"][i][u] = self.var_y[req][i][u].x
            for ij in req.edges:
                variable_assignment[req]["edge_vars"][ij] = {}
                sub_edges = req.get_allowed_edges(ij)
                if sub_edges is None:
                    sub_edges = self.substrate.edges
                for uv in sub_edges:
                    variable_assignment[req]["edge_vars"][ij][uv] = self.var_z[req][ij][uv].x
        return variable_assignment

    def create_constraints_fix_node_mappings(self, req, node_mapping_dict):
        """ Add constraints which enforce the node mappings specified in node_mapping_dict. """
        for i, u_mapped in node_mapping_dict.iteritems():
            for u in req.get_allowed_nodes(i):
                fix_i_u_mapping_constraint = LinExpr([(1.0, self.var_y[req][i][u])])
                name = "{req}_fix_{i}_{u}".format(req=req.name, i=i, u=u)

                if u == u_mapped:
                    self.model.addConstr(fix_i_u_mapping_constraint, GRB.EQUAL, 1.0, name=name)
                else:
                    self.model.addConstr(fix_i_u_mapping_constraint, GRB.EQUAL, 0.0, name=name)
        self.model.update()

    def create_objective(self):
        if not self.use_load_balancing_objective:
            super(FractionalClassicMCFModel, self).create_objective()
        else:
            self.plugin_constraint_embed_all_requests()
            self.plugin_objective_load_balancing()

    def plugin_objective_load_balancing(self):
        """
        Adaptation of AbstractEmbeddingModelcreator.plugin_objective_minimize_cost to include the additional
        coefficients used for load balancing.
        """
        delta = 10 ** -6  # small positive constant to avoid division by zero
        obj_expr = LinExpr()
        for req in self.requests:
            for u, v in self.substrate.substrate_edge_resources:
                lb_coefficient = 1.0 / (self.substrate.get_edge_capacity((u, v)) + delta)
                obj_expr.addTerms(
                    lb_coefficient * self.substrate.get_edge_cost((u, v)),
                    self.var_request_load[req][(u, v)]
                )

            for ntype, snode in self.substrate.substrate_node_resources:
                lb_coefficient = 1.0 / (self.substrate.get_node_type_capacity(snode, ntype) + delta)
                obj_expr.addTerms(
                    lb_coefficient * self.substrate.get_node_type_cost(snode, ntype),
                    self.var_request_load[req][(ntype, snode)]
                )

        self.model.setObjective(obj_expr, GRB.MINIMIZE)


class ViNESingleScenario(object):
    """
    Implementation of the ViNE-SP procedure for a single request, in which edge mappings are determined by a shortest
    path computation.

    A new ViNESingleRequest should be instantiated for each scenario, as the residual capacities are tracked for
    repeated calls to vine_procedure_single_request, updating them whenever a request is embedded.

    By providing an appropriate node_mapper, which should implement the AbstractViNENodeMapper defined below,
    either R-ViNE (RandomizedNodeMapper) or D-ViNE (DeterministicNodeMapper) can be used.
    """

    def __init__(self,
                 scenario,
                 node_mapping_method,
                 edge_mapping_method,
                 use_load_balancing_objective,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename

        if logger is None:
            logger = util.get_logger(__name__, make_file=False, propagate=True)

        self.logger = logger
        self.scenario = scenario

        self.node_mapping_method = node_mapping_method
        self.edge_mapping_method = edge_mapping_method
        self.use_load_balancing_objective = use_load_balancing_objective

        self.residual_capacity_substrate = deepcopy(scenario.substrate)
        if node_mapping_method == NodeMappingMethod.DETERMINISTIC:
            self.node_mapper = DeterministicNodeMapper()
        elif node_mapping_method == NodeMappingMethod.RANDOMIZED:
            self.node_mapper = RandomizedNodeMapper()
        else:
            raise ValueError("Invalid node mapping method: {}".format(node_mapping_method))

        if self.edge_mapping_method not in EdgeMappingMethod.AVAILABLE_METHODS:
            raise ValueError("Invalid edge mapping method: {}".format(edge_mapping_method))

        self._current_request = None
        self._provisional_node_allocations = None
        self._provisional_edge_allocations = None

    def vine_procedure_single_request(self, request):
        """ Perform the ViNE procedure for a single request. """
        self._current_request = request
        self._initialize_provisional_allocations()

        lp_variables = self.solve_vne_lp_relax()
        if lp_variables is None:
            self.logger.debug("Reject {}: No initial LP solution.".format(request.name))
            return None  # REJECT: no LP solution
        node_variables = lp_variables[self._current_request]["node_vars"]

        mapping = self._get_empty_mapping_of_correct_type()
        for i in self._current_request.nodes:
            u = self.node_mapper.get_single_node_mapping(i, node_variables)
            if u is None:
                self.logger.debug("Reject {}: Node mapping failed for {}.".format(request.name, u))
                return None  # REJECT: Failed node mapping
            t = request.get_type(i)
            self._provisional_node_allocations[(u, t)] += request.get_node_demand(i)
            mapping.map_node(i, u)

        if self.edge_mapping_method == EdgeMappingMethod.SHORTEST_PATH:
            mapping = self._map_edges_shortest_path(mapping)
        elif self.edge_mapping_method == EdgeMappingMethod.SPLITTABLE:
            mapping = self._map_edges_splittable(mapping)
        else:
            raise ValueError("Invalid edge mapping method: {}".format(self.edge_mapping_method))

        if mapping is None:
            self.logger.debug("Reject {}: Edge mapping failed.".format(request.name))
            return None  # REJECT: Failed edge mapping

        self._apply_provisional_allocations_to_residual_capacity_substrate()
        return mapping

    def _map_edges_shortest_path(self, mapping):
        for ij in self._current_request.edges:
            i, j = ij
            ij_demand = self._current_request.get_edge_demand(ij)
            ij_allowed_edges = self._current_request.get_allowed_edges(ij)
            if ij_allowed_edges is None:
                ij_allowed_edges = self.residual_capacity_substrate.get_edges()
            u = mapping.get_mapping_of_node(i)
            v = mapping.get_mapping_of_node(j)
            uv_path = self._shortest_substrate_path_respecting_capacities(
                u, v, ij_demand, ij_allowed_edges,
            )
            if uv_path is None:
                return None
            mapping.map_edge(ij, uv_path)
            for uv in uv_path:
                self._provisional_edge_allocations[uv] += ij_demand
        return mapping

    def _shortest_substrate_path_respecting_capacities(self, start, target, min_capacity, allowed_edges):
        distance = {node: float("inf") for node in self.residual_capacity_substrate.nodes}
        prev = {u: None for u in self.residual_capacity_substrate.nodes}
        distance[start] = 0
        q = set(self.residual_capacity_substrate.nodes)
        while q:
            u = min(q, key=distance.get)
            if u == target:
                break
            q.remove(u)
            for uv in self.residual_capacity_substrate.get_out_edges(u):
                if uv not in allowed_edges:
                    continue
                residual_cap = self.residual_capacity_substrate.get_edge_capacity(uv) - self._provisional_edge_allocations[uv]
                if residual_cap < min_capacity:
                    continue  # avoid using edges that are too small
                v = next(v for v in uv if v != u)
                new_dist = distance[u] + self.residual_capacity_substrate.get_edge_cost(uv)
                if new_dist < distance[v]:
                    distance[v] = new_dist
                    prev[v] = u
        if distance[target] == float('inf'):
            return None
        path = []
        u = target
        while u is not start:
            path.append((prev[u], u))
            u = prev[u]
        return list(reversed(path))

    def _map_edges_splittable(self, mapping):
        lp_vars = self.solve_vne_lp_relax(mapping.mapping_nodes)
        if lp_vars is None:
            return None

        lp_vars = lp_vars[self._current_request]["edge_vars"]

        for ij in self._current_request.edges:
            mapping.map_edge(ij, lp_vars[ij])
            ij_demand = self._current_request.get_edge_demand(ij)
            for uv, alloc in lp_vars[ij].iteritems():
                self._provisional_edge_allocations[uv] += alloc * ij_demand
        return mapping

    def solve_vne_lp_relax(self, fixed_node_mappings_dict=None):
        single_req_scenario = dm.Scenario(
            name="{}_{}".format(self.scenario.name, self._current_request.name),
            substrate=self.residual_capacity_substrate,
            requests=[self._current_request],
            objective=dm.Objective.MIN_COST,
        )

        sub_mc = FractionalClassicMCFModel(
            single_req_scenario,
            use_load_balancing_objective=self.use_load_balancing_objective,
            gurobi_settings=self.gurobi_settings,
            logger=self.logger,
        )
        sub_mc.init_model_creator()
        if fixed_node_mappings_dict is not None:
            sub_mc.create_constraints_fix_node_mappings(self._current_request, fixed_node_mappings_dict)
        lp_variables = sub_mc.compute_fractional_solution()
        return lp_variables

    def _initialize_provisional_allocations(self):
        self._provisional_node_allocations = {
            (u, t): 0.0
            for u in self.residual_capacity_substrate.nodes
            for t in self.residual_capacity_substrate.get_supported_node_types(u)
        }
        self._provisional_edge_allocations = {
            uv: 0.0 for uv in self.residual_capacity_substrate.get_edges()
        }

    def _apply_provisional_allocations_to_residual_capacity_substrate(self):
        """
        Apply the current request's provisional allocations.

        This should only be called once a request is certain to be accepted, as it modifies the
        capacities available in the residual_capacity_substrate, which impact embeddings of future
        requests.

        :param m:
        :return:
        """
        for node_resource, alloc in self._provisional_node_allocations.iteritems():
            u, node_type = node_resource
            self.residual_capacity_substrate.node[u]["capacity"][node_type] -= alloc
        for uv, alloc in self._provisional_edge_allocations.iteritems():
            self.residual_capacity_substrate.edge[uv]["capacity"] -= alloc

    def _get_empty_mapping_of_correct_type(self):
        if self.edge_mapping_method == EdgeMappingMethod.SHORTEST_PATH:
            return solutions.Mapping(
                "{}_{}".format(self.scenario.name, self._current_request.name),
                self._current_request, self.scenario.substrate, is_embedded=True,
            )
        elif self.edge_mapping_method == EdgeMappingMethod.SPLITTABLE:
            return SplittableMapping(
                "{}_{}".format(self.scenario.name, self._current_request.name),
                self._current_request, self.scenario.substrate, is_embedded=True,
            )
        else:
            raise ValueError("Invalid edge mapping method: {}".format(self.edge_mapping_method))


class AbstractViNENodeMapper(object):
    def get_single_node_mapping(self, i, node_variables):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")


class DeterministicNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables):
        """ Deterministic node mapping: Node mapping is selected according to the maximal variable in the LP solution. """
        u_max = None
        p_max = float('-inf')
        for u, p_u in node_variables[i].iteritems():
            if p_max < p_u:
                p_max = p_u
                u_max = u
        return u_max


class RandomizedNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables):
        """ Randomized node mapping: Node mapping is selected randomly, interpreting the LP variables as probabilities. """
        u_max = None
        draw = random.random()
        # Node mapping LP variables should already be normalized to 1
        for u, p_u in node_variables[i].iteritems():
            if draw < p_u:
                u_max = u
                break
            draw -= p_u
        return u_max
