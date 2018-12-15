import random
import time
from copy import deepcopy
import enum
from gurobipy import GRB, LinExpr

from alib import datamodel as dm
from alib import mip as mip
from alib import modelcreator as mc
from alib import solutions
from alib import util


@enum.unique
class ViNERequestMappingStatus(enum.Enum):
    is_embedded = "is_embedded"
    initial_lp_failed = "initial_lp_failed"
    node_mapping_failed = "node_mapping_failed"
    edge_mapping_failed = "edge_mapping_failed"


@enum.unique
class ViNELPObjective(enum.Enum):
    LB_IGNORE_COSTS = "LB_IGNORE_COSTS"
    LB_USE_COSTS = "LB_USE_COSTS"
    NO_LB_IGNORE_COSTS = "NO_LB_IGNORE_COSTS"
    NO_LB_USE_COSTS = "NO_LB_USE_COSTS"

    @staticmethod
    def create_from_name(obj_string):
        for objective in ViNELPObjective:
            if objective.name == obj_string:
                return objective

        raise ValueError("Invalid LP Objective: {} (expected one of {})".format(
            obj_string, ", ".join(sorted(o.name for o in ViNELPObjective))
        ))


class NodeMappingMethod(object):
    DETERMINISTIC = "deterministic"
    RANDOMIZED = "randomized"

    AVAILABLE_METHODS = [DETERMINISTIC, RANDOMIZED]


class EdgeMappingMethod(object):
    SPLITTABLE = "splittable"
    SHORTEST_PATH = "shortest_path"

    AVAILABLE_METHODS = [SPLITTABLE, SHORTEST_PATH]


class SplittableMapping(solutions.Mapping):
    EPSILON = 10 ** -5

    def map_edge(self, ij, edge_vars):
        self.mapping_edges[ij] = {
            uv: val for (uv, val) in edge_vars.iteritems()
            if abs(val) >= SplittableMapping.EPSILON
        }


class ViNEResult(mc.AlgorithmResult):
    def __init__(self, solution, vine_parameters, runtime, runtime_per_request, mapping_status_per_request):
        self.solution = solution
        self.runtime = runtime
        self.runtime_per_request = runtime_per_request
        self.mapping_status_per_request = mapping_status_per_request

    def get_solution(self):
        return self.solution

    def _cleanup_references_raw(self, original_scenario):
        own_scenario = self.solution.scenario
        self.solution.scenario = original_scenario

        for own_req, original_request in zip(own_scenario.requests, original_scenario.requests):
            assert own_req.nodes == original_request.nodes
            assert own_req.edges == original_request.edges

            mapping = self.solution.request_mapping[own_req]
            del self.solution.request_mapping[own_req]
            if mapping is not None:
                mapping.request = original_request
                mapping.substrate = original_scenario.substrate
            self.solution.request_mapping[original_request] = mapping

            runtime = self.runtime_per_request[own_req]
            del self.runtime_per_request[own_req]
            self.runtime_per_request[original_request] = runtime

            status = self.mapping_status_per_request[own_req]
            del self.mapping_status_per_request[own_req]
            self.mapping_status_per_request[original_request] = status


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
                 lp_objective=ViNELPObjective.NO_LB_IGNORE_COSTS):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename

        if logger is None:
            logger = util.get_logger(__name__, make_file=False, propagate=True)

        if isinstance(lp_objective, str):
            lp_objective = ViNELPObjective.create_from_name(lp_objective)
        self.lp_objective = lp_objective

        self.logger = logger
        self.scenario = scenario
        self.node_mapping_method = node_mapping_method
        self.edge_mapping_method = edge_mapping_method

    def init_modelcreator(self):
        pass

    def compute_integral_solution(self):
        vine_instance = ViNESingleScenario(
            substrate=self.scenario.substrate,
            node_mapping_method=self.node_mapping_method,
            edge_mapping_method=self.edge_mapping_method,
            lp_objective=self.lp_objective,
            gurobi_settings=self.gurobi_settings,
            optimization_callback=self.optimization_callback,
            lp_output_file=self.lp_output_file,
            potential_iis_filename=self.potential_iis_filename,
            logger=self.logger,
        )
        vine_parameters = dict(
            node_mapping_method=self.node_mapping_method,
            edge_mapping_method=self.edge_mapping_method,
            lp_objective=self.lp_objective.name,
        )

        solution_name = mc.construct_name("solution_", sub_name=self.scenario.name)
        solution = solutions.IntegralScenarioSolution(solution_name, self.scenario)

        overall_runtime_start = time.time()
        runtime_per_request = {}
        mapping_status_per_request = {}
        for req in sorted(self.scenario.requests, key=lambda r: r.profit, reverse=True):
            t_start = time.time()
            mapping, status = vine_instance.vine_procedure_single_request(req)

            runtime_per_request[req] = time.time() - t_start
            mapping_status_per_request[req] = status

            solution.add_mapping(req, mapping)

        overall_runtime = time.time() - overall_runtime_start
        result = ViNEResult(
            solution=solution,
            vine_parameters=vine_parameters,
            runtime=overall_runtime,
            runtime_per_request=runtime_per_request,
            mapping_status_per_request=mapping_status_per_request,
        )
        return result


class FractionalClassicMCFModel(mip.ClassicMCFModel):
    """ This Modelcreator is used to access the raw LP values. """

    def __init__(self, scenario, lp_objective,
                 gurobi_settings=None,
                 logger=None):
        super(FractionalClassicMCFModel, self).__init__(
            scenario=scenario, gurobi_settings=gurobi_settings, logger=logger
        )
        self.lp_objective = lp_objective

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
        if isinstance(self.lp_objective, ViNELPObjective):
            self.plugin_constraint_embed_all_requests()
            self.plugin_objective_load_balancing()
        else:
            msg = "Invalid LP objective: {}. Expected instance of LPComputationObjective defined above!".format(
                self.lp_objective
            )
            raise ValueError(msg)

    def plugin_objective_load_balancing(self):
        """
        Adaptation of AbstractEmbeddingModelcreator.plugin_objective_minimize_cost to include the additional
        coefficients used for load balancing.
        """
        delta = 10 ** -6  # small positive constant to avoid division by zero
        obj_expr = LinExpr()
        for req in self.requests:
            for u, v in self.substrate.substrate_edge_resources:
                cost = self.substrate.get_edge_cost((u, v))
                capacity = self.substrate.get_edge_capacity((u, v)) + delta

                lb_coefficient = self._get_objective_coefficient(capacity, cost)

                obj_expr.addTerms(
                    lb_coefficient,
                    self.var_request_load[req][(u, v)]
                )

            for ntype, snode in self.substrate.substrate_node_resources:
                cost = self.substrate.get_node_type_cost(snode, ntype)
                capacity = self.substrate.get_node_type_capacity(snode, ntype) + delta

                lb_coefficient = self._get_objective_coefficient(capacity, cost)

                obj_expr.addTerms(
                    lb_coefficient,
                    self.var_request_load[req][(ntype, snode)]
                )

        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _get_objective_coefficient(self, capacity, cost):
        if self.lp_objective == ViNELPObjective.NO_LB_IGNORE_COSTS:
            # alpha = beta = residual_capacity, and the coefficient is 1 (ignoring the tiny delta value)
            lb_coefficient = 1.0
        elif self.lp_objective == ViNELPObjective.LB_IGNORE_COSTS:
            # alpha = beta = 1, and the coefficient is the reciprocal of the remaining capacity
            lb_coefficient = 1.0 / capacity
        elif self.lp_objective == ViNELPObjective.NO_LB_USE_COSTS:
            # corresponds to ClassicMCFModel's default MIN_COST objective
            lb_coefficient = cost
        elif self.lp_objective == ViNELPObjective.LB_USE_COSTS:
            # combines the MIN_COST objective with the load balancing approach
            lb_coefficient = cost / capacity
        else:
            msg = "Invalid LP objective: {}. Expected instance of LPComputationObjective defined above!".format(
                self.lp_objective
            )
            raise ValueError(msg)
        return lb_coefficient


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
                 substrate,
                 node_mapping_method,
                 edge_mapping_method,
                 lp_objective,
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
        self.original_substrate = substrate
        self.residual_capacity_substrate = deepcopy(substrate)

        self.node_mapping_method = node_mapping_method
        self.edge_mapping_method = edge_mapping_method
        self.lp_objective = lp_objective

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
            self.logger.debug("Rejected {}: No initial LP solution.".format(request.name))
            return None, ViNERequestMappingStatus.initial_lp_failed  # REJECT: no LP solution
        node_variables = lp_variables[self._current_request]["node_vars"]

        mapping = self._get_empty_mapping_of_correct_type()
        for i in self._current_request.nodes:
            u = self.node_mapper.get_single_node_mapping(i, node_variables)
            if u is None:
                self.logger.debug("Rejected {}: Node mapping failed for {}.".format(request.name, u))
                return None, ViNERequestMappingStatus.node_mapping_failed  # REJECT: Failed node mapping
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
            self.logger.debug("Rejected {}: Edge mapping failed.".format(request.name))
            return None, ViNERequestMappingStatus.edge_mapping_failed  # REJECT: Failed edge mapping

        self.logger.debug("Embedding of {} succeeded: Applying provisional allocations".format(request.name))
        self._apply_provisional_allocations_to_residual_capacity_substrate()
        return mapping, ViNERequestMappingStatus.is_embedded

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
            name="vine_{}".format(self._current_request.name),
            substrate=self.residual_capacity_substrate,
            requests=[self._current_request],
            objective=dm.Objective.MIN_COST,
        )

        sub_mc = FractionalClassicMCFModel(
            single_req_scenario,
            lp_objective=self.lp_objective,
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
            name = mc.construct_name(
                "shortest_path_mapping_", req_name=self._current_request.name, sub_name=self.original_substrate.name
            )
            return solutions.Mapping(
                name,
                self._current_request, self.original_substrate, is_embedded=True,
            )
        elif self.edge_mapping_method == EdgeMappingMethod.SPLITTABLE:
            name = mc.construct_name(
                "splittable_mapping_", req_name=self._current_request.name, sub_name=self.original_substrate.name
            )
            return SplittableMapping(
                name,
                self._current_request, self.original_substrate, is_embedded=True,
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
