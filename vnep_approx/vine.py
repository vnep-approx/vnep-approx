import random
from copy import deepcopy

from alib import datamodel as dm
from alib import mip as mip
from alib import modelcreator as mc
from alib import solutions


class WiNESingleWindow(object):
    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename
        self.logger = logger
        self.scenario = scenario

    def init_modelcreator(self):
        pass

    def compute_integral_solution(self):
        # TODO: Measure runtimes
        vine_instance = ViNESingleScenario(
            scenario=self.scenario,
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


class FractionalClassicMCFModel(mip.ClassicMCFModel):
    """ This Modelcreator is used to access the raw LP values. """

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


class ViNESingleScenario(object):
    """
    Implementation of the ViNE-SP procedure for a single request, in which edge mappings are determined by a shortest
    path computation.

    A new ViNESingleRequest should be instantiated for each scenario, as the residual capacities are tracked for
    repeated calls to vine_procedure_single_request, updating them whenever a request is embedded.

    By providing an appropriate node_mapper, which must implement the AbstractViNENodeMapper defined below,
    either R-ViNE (RandomizedNodeMapper) or D-ViNE (DeterministicNodeMapper) can be used.
    """

    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 optimization_callback=mc.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None,
                 node_mapper=None):
        self.gurobi_settings = gurobi_settings
        self.optimization_callback = optimization_callback
        self.lp_output_file = lp_output_file
        self.potential_iis_filename = potential_iis_filename
        self.logger = logger
        self.scenario = scenario

        self.residual_capacity_substrate = deepcopy(scenario.substrate)
        if node_mapper is None:
            node_mapper = DeterministicNodeMapper()
        self.node_mapper = node_mapper

        self.current_request = None

    def vine_procedure_single_request(self, request):
        """ Perform the ViNE procedure for a single request. """
        self.current_request = request
        single_req_scenario = dm.Scenario(
            name="{}_{}".format(self.scenario.name, self.current_request.name),
            substrate=self.residual_capacity_substrate,
            requests=[self.current_request],
            objective=self.scenario.objective,
        )

        sub_mc = FractionalClassicMCFModel(
            single_req_scenario, gurobi_settings=self.gurobi_settings, logger=self.logger
        )
        sub_mc.init_model_creator()
        lp_variables = sub_mc.compute_fractional_solution()
        node_variables = lp_variables[self.current_request]["node_vars"]

        mapping = solutions.Mapping(
            "{}_{}".format(self.scenario.name, self.current_request.name),
            self.current_request, self.scenario.substrate, is_embedded=True,
        )
        for i in self.current_request.nodes:
            u = self.node_mapper.get_single_node_mapping(i, node_variables)
            if u is None:
                return  # REJECT
            mapping.map_node(i, u)

        for ij in self.current_request.edges:
            i, j = ij
            u = mapping.get_mapping_of_node(i)
            v = mapping.get_mapping_of_node(j)

            uv_path = self._shortest_substrate_path_respecting_capacities(
                u, v, self.current_request.get_edge_demand(ij)
            )
            if uv_path is None:
                return  # REJECT
            mapping.map_edge(ij, uv_path)

        self._adjust_residual_capacities(mapping)
        return mapping

    def _shortest_substrate_path_respecting_capacities(self, start, target, min_capacity):
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
                if self.residual_capacity_substrate.get_edge_capacity(uv) < min_capacity:
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

    def _adjust_residual_capacities(self, m):
        for i, u in m.mapping_nodes.iteritems():
            i_type = self.current_request.get_type(i)
            i_demand = self.current_request.get_node_demand(i)
            self.residual_capacity_substrate.node[u]["capacity"][i_type] -= i_demand
        for ij, uv_path in m.mapping_edges.iteritems():
            ij_demand = self.current_request.get_edge_demand(ij)
            for uv in uv_path:
                self.residual_capacity_substrate.edge[uv]["capacity"] -= ij_demand


class AbstractViNENodeMapper(object):
    def get_single_node_mapping(self, i, node_variables):
        raise NotImplementedError("This is an abstract method! Use one of the implementations.")


class DeterministicNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables):
        u_max = None
        p_max = float('-inf')
        for u, p_u in node_variables[i].iteritems():
            if p_max < p_u:
                p_max = p_u
                u_max = u
        return u_max


class RandomizedNodeMapper(AbstractViNENodeMapper):
    def get_single_node_mapping(self, i, node_variables):
        u_max = None
        draw = random.random()
        # Node mapping LP variables should already be normalized to 1
        for u, p_u in node_variables[i].iteritems():
            if draw < p_u:
                u_max = u
                break
            draw -= p_u
        return u_max
