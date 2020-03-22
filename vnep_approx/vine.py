import random
import time
from copy import deepcopy
import enum
from gurobipy import GRB, LinExpr
import itertools
from collections import namedtuple

from alib import datamodel as dm
from alib import mip as mip
from alib import modelcreator as mc
from alib import solutions
from alib import util


@enum.unique
class ViNEMappingStatus(enum.Enum):
    is_embedded = "is_embedded"
    initial_lp_failed = "initial_lp_failed"
    node_mapping_failed = "node_mapping_failed"
    edge_mapping_failed = "edge_mapping_failed"


@enum.unique
class ViNELPObjective(enum.Enum):
    ViNE_LB_DEF = "ViNE_LB_DEF"
    ViNE_LB_INCL_SCENARIO_COSTS = "ViNE_LB_INCL_SCENARIO_COSTS"
    ViNE_COSTS_DEF = "ViNE_COSTS_DEF"
    ViNE_COSTS_INCL_SCENARIO_COSTS = "ViNE_COSTS_INCL_SCENARIO_COSTS"

@enum.unique
class ViNERoundingProcedure(enum.Enum):
    DETERMINISTIC = "DET"
    RANDOMIZED = "RAND"

@enum.unique
class ViNEEdgeEmbeddingModel(enum.Enum):
    UNSPLITTABLE = "SP"  #single path
    SPLITTABLE   = "MCF" #multi-commodity flow


ViNESettings = namedtuple("ViNESettings", "edge_embedding_model lp_objective rounding_procedure")


class ViNESettingsFactory(object):

    _known_vine_settings = {}

    @staticmethod
    def get_vine_settings(edge_embedding_model, lp_objective, rounding_procedure):
        if isinstance(edge_embedding_model, str):
            edge_embedding_model = ViNEEdgeEmbeddingModel(edge_embedding_model)
        elif not isinstance(edge_embedding_model, ViNEEdgeEmbeddingModel):
            raise ValueError("model must be of ViNEEdgeEmbeddingModel type.")
        if isinstance(lp_objective, str):
            lp_objective = ViNELPObjective(lp_objective)
        elif not isinstance(lp_objective, ViNELPObjective):
            raise ValueError("objective must be of ViNELPObjective type.")
        if isinstance(rounding_procedure, str):
            rounding_procedure = ViNERoundingProcedure(rounding_procedure)
        elif not isinstance(rounding_procedure, ViNERoundingProcedure):
            raise ValueError("rounding must be of ViNERoundingProcedure type.")

        temp_result = ViNESettings(edge_embedding_model=edge_embedding_model,
                                   lp_objective=lp_objective,
                                   rounding_procedure=rounding_procedure)

        if temp_result in ViNESettingsFactory._known_vine_settings.keys():
            return ViNESettingsFactory._known_vine_settings[temp_result]
        else:
            ViNESettingsFactory._known_vine_settings[temp_result] = temp_result
            return temp_result

    @staticmethod
    def get_vine_settings_from_settings(vine_settings):
        ViNESettingsFactory.check_vine_settings(vine_settings)

        if vine_settings in ViNESettingsFactory._known_vine_settings.keys():
            return ViNESettingsFactory._known_vine_settings[vine_settings]
        else:
            ViNESettingsFactory._known_vine_settings[vine_settings] = vine_settings
            return vine_settings

    @staticmethod
    def check_vine_settings(vine_settings):
        if not isinstance(vine_settings, ViNESettings):
            raise ValueError("vine_settings must be of type ViNESettings.")
        if not isinstance(vine_settings.edge_embedding_model, ViNEEdgeEmbeddingModel):
            raise ValueError("model must be of ViNEEdgeEmbeddingModel type.")
        if not isinstance(vine_settings.lp_objective, ViNELPObjective):
            raise ValueError("objective must be of ViNELPObjective type.")
        if not isinstance(vine_settings.rounding_procedure, ViNERoundingProcedure):
            raise ValueError("rounding must be of ViNERoundingProcedure type.")




class OfflineViNEResult(mc.AlgorithmResult):
    def __init__(self, solution, vine_settings, runtime, runtime_per_request, mapping_status_per_request):
        self.solution = solution
        self.vine_parameters = vine_settings
        self.total_runtime = runtime
        self.profit = self.compute_profit()
        self.runtime_per_request = runtime_per_request
        self.mapping_status_per_request = mapping_status_per_request

    def compute_profit(self):
        profit = 0.0
        for req, mapping in self.solution.request_mapping.iteritems():
            if mapping is not None:
                profit += req.profit
        return profit

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



class SplittableMapping(solutions.Mapping):
    EPSILON = 10 ** -5

    def map_edge(self, ij, edge_vars):
        self.mapping_edges[ij] = {
            uv: val for (uv, val) in edge_vars.iteritems()
            if abs(val) >= SplittableMapping.EPSILON
        }


class OfflineViNEAlgorithm(object):
    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None,
                 vine_settings=None,
                 edge_embedding_model=None,
                 lp_objective=None,
                 rounding_procedure=None
                 ):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename

        if logger is None:
            logger = util.get_logger(__name__, make_file=False, propagate=True)

        if vine_settings is None:

            if rounding_procedure is None or edge_embedding_model is None or lp_objective is None:
                raise ValueError("Either vine_settings or all of the following must be specified: edge_embedding_model, objective, rounding_procedure")

            if isinstance(edge_embedding_model, str):
                edge_embedding_model= ViNEEdgeEmbeddingModel(edge_embedding_model)

            if isinstance(lp_objective, str):
                lp_objective = ViNELPObjective(lp_objective)

            if isinstance(rounding_procedure, str):
                rounding_procedure = ViNERoundingProcedure(rounding_procedure)

            self.vine_settings = ViNESettingsFactory.get_vine_settings(edge_embedding_model, lp_objective, rounding_procedure)
        else:
            ViNESettingsFactory.check_vine_settings(vine_settings)
            self.vine_settings = vine_settings


        self.logger = logger
        self.scenario = scenario

    def init_model_creator(self):
        pass

    def compute_integral_solution(self):
        vine_instance = ViNESingleScenario(
            substrate=self.scenario.substrate,
            vine_settings=self.vine_settings,
            gurobi_settings=self.gurobi_settings,
            optimization_callback=self.optimization_callback,
            lp_output_file=self.lp_output_file,
            potential_iis_filename=self.potential_iis_filename,
            logger=self.logger
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

        # assert solution.validate_solution()                       test is limited but worked, no need to keep it in the evaluation
        # assert solution.validate_solution_fulfills_capacity()     test is limited but worked, no need to keep it in the evaluation

        overall_runtime = time.time() - overall_runtime_start
        result = OfflineViNEResult(
            solution=solution,
            vine_settings=self.vine_settings,
            runtime=overall_runtime,
            runtime_per_request=runtime_per_request,
            mapping_status_per_request=mapping_status_per_request,
        )
        return result


class FractionalClassicMCFModel(mip.ClassicMCFModel):
    """ This Modelcreator is used to access the raw LP values. """

    def __init__(self, scenario, lp_objective,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 logger=None):
        super(FractionalClassicMCFModel, self).__init__(
            scenario=scenario, gurobi_settings=gurobi_settings, logger=logger, optimization_callback=optimization_callback
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

                obj_expr.addTerms(
                    self._get_objective_coefficient(capacity, cost),
                    self.var_request_load[req][(u, v)]
                )

            for ntype, snode in self.substrate.substrate_node_resources:
                cost = self.substrate.get_node_type_cost(snode, ntype)
                capacity = self.substrate.get_node_type_capacity(snode, ntype) + delta

                obj_expr.addTerms(
                    self._get_objective_coefficient(capacity, cost),
                    self.var_request_load[req][(ntype, snode)]
                )

        self.model.setObjective(obj_expr, GRB.MINIMIZE)

    def _get_objective_coefficient(self, capacity, cost):
        if self.lp_objective == ViNELPObjective.ViNE_COSTS_DEF:
            # alpha = beta = residual_capacity, and the coefficient is 1 (ignoring the tiny delta value)
            lb_coefficient = 1.0
        elif self.lp_objective == ViNELPObjective.ViNE_LB_DEF:
            # alpha = beta = 1, and the coefficient is the reciprocal of the remaining capacity
            lb_coefficient = 1.0 / capacity
        elif self.lp_objective == ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS:
            # corresponds to ClassicMCFModel's default MIN_COST objective
            lb_coefficient = cost
        elif self.lp_objective == ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS:
            # combines the MIN_COST objective with the load balancing approach
            lb_coefficient = cost / capacity
        else:
            msg = "Invalid LP objective: {}. Expected instance of LPComputationObjective defined above!".format(
                self.lp_objective
            )
            raise ValueError(msg)
        return lb_coefficient

    def reset_gurobi_parameter(self, param):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.model.setParam(param, default)

    def set_gurobi_parameter(self, param, value):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        if not param in self._listOfUserVariableParameters:
            raise mc.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.setParam(param, value)


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
                 vine_settings,
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

        ViNESettingsFactory.check_vine_settings(vine_settings)
        self.vine_settings = vine_settings

        self.edge_embedding_model = vine_settings.edge_embedding_model
        self.lp_objective = vine_settings.lp_objective
        self.rounding_procedure = vine_settings.rounding_procedure

        self._extended_logging = False

        if self.rounding_procedure == ViNERoundingProcedure.DETERMINISTIC:
            self.node_mapper = DeterministicNodeMapper()
        elif self.rounding_procedure == ViNERoundingProcedure.RANDOMIZED:
            self.node_mapper = RandomizedNodeMapper()
        else:
            raise ValueError("Invalid node mapping method: {}".format(self.rounding_procedure))

        if self.edge_embedding_model not in ViNEEdgeEmbeddingModel:
            raise ValueError("Invalid edge mapping method: {}".format(self.edge_embedding_model))

        self._current_request = None
        self._provisional_node_allocations = None
        self._provisional_edge_allocations = None

    def vine_procedure_single_request(self, request):
        """ Perform the ViNE procedure for a single request. """
        self._current_request = request
        self._initialize_provisional_allocations()

        if self._extended_logging:
            self.logger.debug("Handling request {}..".format(self._current_request.name))
            substrate_state_string = "\n\nCurrent substrate is:\n"
            for ntype in self.residual_capacity_substrate.get_types():
                for node in self.residual_capacity_substrate.get_nodes_by_type(ntype):
                    substrate_state_string += "\t" + "node {:>3} of cap {:7.3f} and costs {:7.3f} \n".format(node, self.residual_capacity_substrate.get_node_type_capacity(node, ntype), self.residual_capacity_substrate.get_node_type_cost(node, ntype))
            for edge in self.residual_capacity_substrate.get_edges():
                substrate_state_string += "\t" + "edge {:>12} of cap {:7.3f}\n".format(edge,
                                                                              self.residual_capacity_substrate.get_edge_capacity(edge))

            self.logger.debug(substrate_state_string + "\n\n")

        lp_variables = self.solve_vne_lp_relax()
        if lp_variables is None:
            self.logger.debug("Rejected {}: No initial LP solution.".format(request.name))
            return None, ViNEMappingStatus.initial_lp_failed  # REJECT: no LP solution

        node_variables = lp_variables[self._current_request]["node_vars"]

        mapping = self._get_empty_mapping_of_correct_type()

        for i in self._current_request.nodes:
            allowed_nodes_of_i = []
            type_of_i = self._current_request.get_type(i)
            if self._extended_logging:
                node_mapping_log_string = "mapping opportunitites for node {} of type {} and demand...\n".format(i, type_of_i, self._current_request.get_node_demand(i))
            for allowed_node in self._current_request.get_allowed_nodes(i):
                actual_residual_cap = self.residual_capacity_substrate.node[allowed_node]['capacity'][type_of_i] - \
                                      self._provisional_node_allocations[(allowed_node, type_of_i)]

                include_node = (actual_residual_cap - self._current_request.get_node_demand(i)) >= 0.0

                if include_node:
                    allowed_nodes_of_i.append(allowed_node)
                    if self._extended_logging:
                        node_mapping_log_string += "\tsnode: {:>3} of cap. {:>7.3f} included ({})\n".format(allowed_node, actual_residual_cap, include_node)

            u = self.node_mapper.get_single_node_mapping(i, node_variables, allowed_nodes_of_i)
            if u is None:
                self.logger.debug("Rejected {}: Node mapping failed for {}.".format(request.name, u))
                return None, ViNEMappingStatus.node_mapping_failed  # REJECT: Failed node mapping
            elif self._extended_logging:
                self.logger.debug("Node {} mapped on {}.\npossibilities were: {}\n".format(i, u, node_mapping_log_string))

            t = request.get_type(i)
            self._provisional_node_allocations[(u, t)] += request.get_node_demand(i)
            mapping.map_node(i, u)

        if self.edge_embedding_model == ViNEEdgeEmbeddingModel.UNSPLITTABLE:
            mapping = self._map_edges_shortest_path(mapping)
        elif self.edge_embedding_model == ViNEEdgeEmbeddingModel.SPLITTABLE:
            mapping = self._map_edges_splittable(mapping)
        else:
            raise ValueError("Invalid edge mapping method: {}".format(self.edge_embedding_model))

        if mapping is None:
            self.logger.debug("Rejected {}: Edge mapping failed.".format(request.name))
            return None, ViNEMappingStatus.edge_mapping_failed  # REJECT: Failed edge mapping

        self.logger.debug("Embedding of {} succeeded: Applying provisional allocations.".format(request.name))
        self._apply_provisional_allocations_to_residual_capacity_substrate()
        return mapping, ViNEMappingStatus.is_embedded

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
        while u != start:
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
            name="vine_scenario_{}".format(self._current_request.name),
            substrate=self.residual_capacity_substrate,
            requests=[self._current_request],
            objective=dm.Objective.MIN_COST,
        )

        sub_mc = FractionalClassicMCFModel(
            single_req_scenario,
            lp_objective=self.lp_objective,
            gurobi_settings=self.gurobi_settings,
            logger=self.logger,
            optimization_callback=None
        )
        sub_mc._disable_temporal_information_output = True
        sub_mc.init_model_creator()
        sub_mc.model.setParam("LogFile", "")
        if fixed_node_mappings_dict is not None:
            sub_mc.create_constraints_fix_node_mappings(self._current_request, fixed_node_mappings_dict)
        lp_variable_assignment = sub_mc.compute_fractional_solution()
        #necessary as otherwise too many gurobi models are created
        del sub_mc.model
        del sub_mc
        return lp_variable_assignment

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
        if self.edge_embedding_model == ViNEEdgeEmbeddingModel.UNSPLITTABLE:
            name = mc.construct_name(
                "shortest_path_mapping_", req_name=self._current_request.name, sub_name=self.original_substrate.name
            )
            return solutions.Mapping(
                name, self._current_request, self.original_substrate, is_embedded=True,
            )
        elif self.edge_embedding_model == ViNEEdgeEmbeddingModel.SPLITTABLE:
            name = mc.construct_name(
                "splittable_mapping_", req_name=self._current_request.name, sub_name=self.original_substrate.name
            )
            return SplittableMapping(
                name, self._current_request, self.original_substrate, is_embedded=True,
            )
        else:
            raise ValueError("Invalid edge mapping method: {}".format(self.edge_embedding_model))




class AbstractViNENodeMapper(object):
    def get_single_node_mapping(self, i, node_variables, allowed_nodes):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")


class DeterministicNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables, allowed_nodes):
        """ Deterministic node mapping: Node mapping is selected according to the maximal variable in the LP solution. """
        u_max = None
        p_max = float('-inf')
        for u, p_u in node_variables[i].iteritems():
            if u not in allowed_nodes:
                continue
            if p_max < p_u:
                p_max = p_u
                u_max = u
        return u_max


class RandomizedNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables, allowed_nodes):
        """ Randomized node mapping: Node mapping is selected randomly, interpreting the LP variables as probabilities. """
        u_max = None

        #to normalize the node variables we iterate over the set of allowed_nodes

        assignment_sum = 0.0
        for u in allowed_nodes:
            assignment_sum += node_variables[i][u]

        if assignment_sum == 0.0:
            return u_max

        draw = random.random() / assignment_sum

        if draw >= 1.0: #in case that numerical difficulties arise, simply select the last one
            draw = 0.999

        for u, p_u in node_variables[i].iteritems():
            if u not in allowed_nodes:
                continue
            if draw < p_u:
                u_max = u
                break
            draw -= p_u
        return u_max


class OfflineViNEResultCollection(mc.AlgorithmResult):

    def __init__(self, vine_settings_list, scenario):
        self.vine_settings_list = vine_settings_list
        for vine_settings in self.vine_settings_list:
            ViNESettingsFactory.check_vine_settings(vine_settings)
        self.scenario = scenario
        self.solutions = {}

    def add_solution(self, vine_settings, offline_vine_result):
        if vine_settings not in self.vine_settings_list:
            raise ValueError("VineSettings diverge from given vine_settings")
        if vine_settings not in self.solutions.keys():
            self.solutions[vine_settings] = []
        if self.scenario != offline_vine_result.solution.scenario:
            raise ValueError("Seems to be another scenario!")

        number_current_solutions = len(self.solutions[vine_settings])
        self.solutions[vine_settings].append((number_current_solutions, offline_vine_result))

    def get_solution(self):
        return self.solutions

    def _check_scenarios_are_equal(self, original_scenario):
        #check can only work if a single solution is returned; we incorporate this in the following function: _cleanup_references_raw
        pass


    def _cleanup_references_raw(self, original_scenario):
        for vine_settings in self.solutions.keys():
            for (result_index, result) in self.solutions[vine_settings]:
                for own_req, original_request in zip(self.scenario.requests, original_scenario.requests):
                    assert own_req.nodes == original_request.nodes
                    assert own_req.edges == original_request.edges

                    mapping = result.solution.request_mapping[own_req]
                    del result.solution.request_mapping[own_req]
                    if mapping is not None:
                        mapping.request = original_request
                        mapping.substrate = original_scenario.substrate
                    result.solution.request_mapping[original_request] = mapping

                    runtime = result.runtime_per_request[own_req]
                    del result.runtime_per_request[own_req]
                    result.runtime_per_request[original_request] = runtime

                    status = result.mapping_status_per_request[own_req]
                    del result.mapping_status_per_request[own_req]
                    result.mapping_status_per_request[original_request] = status
                result.solution.scenario = original_scenario

        #lastly: adapt the collection's scenario
        self.scenario = original_scenario

    def _get_solution_overview(self):
        result = "\n\t{:^10s} | {:^5s} {:^20s} {:^5s} | {:^8s}\n".format("PROFIT", "MODEL", "LP-OBJECTIVE", "PROC", "INDEX")
        for vine_settings in self.vine_settings_list:
            if vine_settings in self.solutions.keys():
                for solution_index, solution in self.solutions[vine_settings]:
                    result += "\t" + "{:^10.5f} | {:^5s} {:^20s} {:^5s} | {:<8d}\n".format(solution.profit,
                                                                                      vine_settings.edge_embedding_model.value,
                                                                                      vine_settings.lp_objective.value,
                                                                                      vine_settings.rounding_procedure.value,
                                                                                      solution_index)
        return result





class OfflineViNEAlgorithmCollection(object):

    ALGORITHM_ID = "OfflineViNEAlgorithmCollection"

    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None,
                 vine_settings_list=None,
                 edge_embedding_model_list=None,
                 lp_objective_list=None,
                 rounding_procedure_list=None,
                 repetitions_for_randomized_experiments=1
                 ):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename

        if logger is None:
            logger = util.get_logger(__name__, make_file=False, propagate=True)

        if vine_settings_list is None:

            if rounding_procedure_list is None or edge_embedding_model_list is None or lp_objective_list is None:
                raise ValueError("Either vine_settings or all of the following must be specified: edge_embedding_model, objective, rounding_procedure")

            self.vine_settings_list = self._construct_vine_settings_combinations(edge_embedding_model_list,
                                                                                 lp_objective_list,
                                                                                 rounding_procedure_list)
        else:
            self.vine_settings_list = []
            for vine_settings in vine_settings_list:
                ViNESettingsFactory.check_vine_settings(vine_settings)
                self.vine_settings_list.append(ViNESettingsFactory.get_vine_settings_from_settings(vine_settings))


        self.logger = logger
        self.scenario = scenario
        if repetitions_for_randomized_experiments <= 0:
            raise ValueError("Repetitions must be larger than or equal to 1.")
        self.repetitions_for_randomized_experiments = repetitions_for_randomized_experiments

        self.result = None


    def _construct_vine_settings_combinations(self,
                                              edge_embedding_model_list,
                                              lp_objective_list,
                                              rounding_procedure_list):

        result = []

        for edge_embedding_model, lp_objective, rounding_procedure in itertools.product(edge_embedding_model_list,
                                                                                        lp_objective_list,
                                                                                        rounding_procedure_list):
            if isinstance(edge_embedding_model, str):
                edge_embedding_model = ViNEEdgeEmbeddingModel(edge_embedding_model)

            if isinstance(lp_objective, str):
                lp_objective = ViNELPObjective(lp_objective)

            if isinstance(rounding_procedure, str):
                rounding_procedure = ViNERoundingProcedure(rounding_procedure)

            result.append(ViNESettingsFactory.get_vine_settings(edge_embedding_model, lp_objective, rounding_procedure))

        return result

    def init_model_creator(self):
        pass

    def compute_integral_solution(self):

        if self.result is  None:
            self.result = OfflineViNEResultCollection(self.vine_settings_list, self.scenario)

            for vine_settings in self.vine_settings_list:

                iterations_to_execute = 1
                if vine_settings.rounding_procedure == ViNERoundingProcedure.RANDOMIZED:
                    iterations_to_execute = self.repetitions_for_randomized_experiments

                self.logger.info("Going to execute {} times the ViNE algorithm with settings {}.".format(iterations_to_execute, vine_settings))

                for iteration in range(iterations_to_execute):
                    vine_algorithm = OfflineViNEAlgorithm(scenario=self.scenario,
                                                          gurobi_settings=self.gurobi_settings,
                                                          optimization_callback=self.optimization_callback,
                                                          lp_output_file=self.lp_output_file,
                                                          potential_iis_filename=self.potential_iis_filename,
                                                          logger=self.logger,
                                                          vine_settings=vine_settings)

                    offline_vine_result = vine_algorithm.compute_integral_solution()
                    self.result.add_solution(vine_settings, offline_vine_result)

                    self.logger.debug(self.result._get_solution_overview())

                    del vine_algorithm

        return self.result




