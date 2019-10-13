import numpy as np
import math
from heapq import heappush, heappop


class ShortestValidPathsComputer_NoLatencies(object):

    ''' This class is optimized to compute all shortest paths in the substrate quickly for varying valid edge sets.
    '''
    def __init__(self, substrate, request, valid_mapping_restriction_computer, edge_costs):
        self.substrate = substrate
        self.request = request
        self.valid_mapping_restriction_computer = valid_mapping_restriction_computer
        self.edge_costs = edge_costs
        self.edge_mapping_invalidities = False

    def compute(self):
        self.number_of_nodes = len(self.substrate.nodes)
        self.distance = np.full(self.number_of_nodes, np.inf, dtype=np.float64)
        self.predecessor = np.full(self.number_of_nodes, -1, dtype=np.int32)

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_pred = {id: {} for id in range(self.number_of_valid_edge_sets)}


        self._prepare_numeric_graph()
        self._compute_valid_edge_mapping_costs()

    def recompute_with_new_costs(self, new_edge_costs):
        for sedge, cost in new_edge_costs.iteritems():
            self.edge_costs[sedge] = abs(cost)

        self.compute()

    def _prepare_numeric_graph(self):
        #prepare and store information on "numeric graph"
        self.number_of_nodes = 0
        self.num_id_to_snode_id = []
        self.snode_id_to_num_id = {}

        for snode in self.substrate.nodes:
            self.num_id_to_snode_id.append(snode)
            self.snode_id_to_num_id[snode] = self.number_of_nodes
            self.number_of_nodes += 1

    def _compute_substrate_edge_abstraction_with_integer_nodes(self, valid_edge_set):
        out_neighbors_with_cost = []

        for num_node in range(self.number_of_nodes):

            snode = self.num_id_to_snode_id[num_node]
            neighbor_and_cost_list = []

            for sedge in self.substrate.out_edges[snode]:
                if sedge in valid_edge_set:
                    _, out_neighbor = sedge
                    num_id_neighbor = self.snode_id_to_num_id[out_neighbor]
                    neighbor_and_cost_list.append((num_id_neighbor, self.edge_costs[sedge]))

            out_neighbors_with_cost.append(neighbor_and_cost_list)

        return out_neighbors_with_cost

    def _compute_valid_edge_mapping_costs(self):

        for edge_set_index in range(self.number_of_valid_edge_sets):

            out_neighbors_with_cost = self._compute_substrate_edge_abstraction_with_integer_nodes(self.edge_set_id_to_edge_set[edge_set_index])

            for num_source_node in range(self.number_of_nodes):

                #reinitialize it
                self.distance.fill(np.inf)
                self.predecessor.fill(-1)

                queue = [(0.0, (num_source_node, num_source_node))]
                while queue:
                    dist, (num_node, num_predecessor) = heappop(queue)
                    if self.predecessor[num_node] == -1:
                        self.distance[num_node] = dist
                        self.predecessor[num_node] = num_predecessor

                        for num_neighbor, edge_cost in out_neighbors_with_cost[num_node]:
                            if self.predecessor[num_neighbor] == -1:
                                heappush(queue, (dist + edge_cost, (num_neighbor, num_node)))

                for num_target_node in range(self.number_of_nodes):
                    if self.distance[num_target_node] != np.inf:
                        self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = self.distance[num_target_node]
                    else:
                        self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = np.nan
                        self.edge_mapping_invalidities = True

                converted_pred_dict = {self.num_id_to_snode_id[num_node] : self.num_id_to_snode_id[self.predecessor[num_node]] if self.predecessor[num_node] != -1 else None for num_node in range(self.number_of_nodes)  }
                self.valid_sedge_pred[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_pred_dict

        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()

        for request_edge, edge_set_id_to_edge_set in request_edge_to_edge_set_id.iteritems():
            self.valid_sedge_costs[request_edge] = self.valid_sedge_costs[edge_set_id_to_edge_set]
            self.valid_sedge_pred[request_edge] = self.valid_sedge_pred[edge_set_id_to_edge_set]


    """ Getter functions for retrieving calculation results """
    def get_valid_sedge_costs_for_reqedge(self, request_edge, mapping_edge):
        return self.valid_sedge_costs[request_edge][mapping_edge]

    def get_valid_sedge_costs_from_edgesetindex(self, edge_set_index, reqedge):
        return self.valid_sedge_costs[edge_set_index][reqedge]

    def get_valid_sedge_path(self, request_edge, source_mapping, target_mapping):
        reqedge_predecessors = self.valid_sedge_pred[request_edge][source_mapping]
        u = target_mapping
        path = []
        while u != source_mapping:
            pred = reqedge_predecessors[u]
            path.append((pred, u))
            u = pred
        path = list(reversed(path))
        return path


class ShortestValidPathsComputer(object):

    Approx_NoLatencies = 0
    Approx_Flex = 1
    Approx_Strict = 2
    Approx_Exact = 3

    @staticmethod
    def createSVPC(approx_type, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies=None, epsilon=None, limit=None):
        if approx_type == ShortestValidPathsComputer.Approx_Flex:
            return ShortestValidPathsComputer_Flex		(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        if approx_type == ShortestValidPathsComputer.Approx_Strict:
            return ShortestValidPathsComputer_Strict	(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        if approx_type == ShortestValidPathsComputer.Approx_Exact:
            return ShortestValidPathsComputer_Exact     (substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)

        return ShortestValidPathsComputer_NoLatencies	(substrate, None, valid_mapping_restriction_computer, edge_costs)


    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        self.substrate = substrate
        self.valid_mapping_restriction_computer = valid_mapping_restriction_computer
        self.edge_costs = edge_costs
        self.edge_latencies = edge_latencies
        self.epsilon = epsilon
        self.limit = limit
        self.latency_limit_overstepped = False
        self.edge_mapping_invalidities = False

        self._prepare_numeric_graph()
        self._init_datastructures()

    def _prepare_numeric_graph(self):
        # prepare and store information on "numeric graph"
        self.number_of_nodes = 0
        self.num_id_to_snode_id = []
        self.snode_id_to_num_id = {}

        for snode in self.substrate.nodes:
            self.num_id_to_snode_id.append(snode)
            self.snode_id_to_num_id[snode] = self.number_of_nodes
            self.number_of_nodes += 1

    def _init_datastructures(self):
        """ this method exists to separate the allocation of datastructures from their re-initialization """
        self.temp_latencies = np.full(self.number_of_nodes, -1, dtype=np.float64)
        self.node_infeasible = np.full(self.number_of_nodes, False, dtype=np.bool)
        self.distances = np.full((self.number_of_nodes, 32 + 1), np.inf, dtype=np.float64)

        self.edge_set_id_to_edge_set = 0
        self.number_of_valid_edge_sets = 0
        self.valid_sedge_costs = dict()
        self.valid_sedge_paths = dict()

    def compute(self):
        # reset values
        self.temp_latencies.fill(-1)
        self.node_infeasible.fill(False)

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()


        for edgesetindex in range(self.number_of_valid_edge_sets):
            self.valid_sedge_costs[edgesetindex] = {}
            self.valid_sedge_paths[edgesetindex] = {}

        if self.number_of_valid_edge_sets == 0:
            print "ERROR: no valid edges"
            return

        self._compute_all_pairs()

    def recompute_with_new_costs(self, new_edge_costs):
        for sedge, cost in new_edge_costs.iteritems():
            self.edge_costs[sedge] = abs(cost)

        self.compute()

    def _prepare_valid_edges(self, edge_set_index):
        self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]
        self.sedge_valid = {}
        for e in self.current_valid_edge_set:
            self.sedge_valid[e] = True

    # one run of dijkstra using delay metric to root out infeasible nodes
    def _preprocess(self, num_source_node):
        queue = [(0, num_source_node)]
        self.temp_latencies.fill(np.inf)
        self.temp_latencies[num_source_node] = 0
        self.node_infeasible.fill(False)

        while queue:
            total_latencies, num_current_node = heappop(queue)

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if self.sedge_valid.get(sedge, False):
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    lat = self.edge_latencies[sedge]
                    if total_latencies + lat < self.temp_latencies[num_endpoint]:
                        self.temp_latencies[num_endpoint] = total_latencies + lat
                        heappush(queue, (total_latencies + lat, num_endpoint))

        self.node_nums = []
        self.num_feasible_nodes = 0
        self.edge_mapping_invalidities = False
        for num_node in range(self.number_of_nodes):
            if self.temp_latencies[num_node] > self.limit:
                self.node_infeasible[num_node] = True
                self.edge_mapping_invalidities = True
            else:
                self.node_nums.append(num_node)
                self.num_feasible_nodes += 1

    def _compute_all_pairs(self):
        raise RuntimeError("Abstract Method")


    """ Getter functions for retrieving calculation results """
    def get_valid_sedge_costs_for_reqedge(self, request_edge, mapping_edge):
        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()
        edge_set_id_to_edge_set = request_edge_to_edge_set_id[request_edge]
        return self.get_valid_sedge_costs_from_edgesetindex(edge_set_id_to_edge_set, mapping_edge)

    def get_valid_sedge_costs_from_edgesetindex(self, edge_set_index, mapping):
        return self.valid_sedge_costs[edge_set_index].get(mapping, np.nan)

    def get_valid_sedge_path(self, request_edge, source_mapping, target_mapping):
        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()
        edge_set_id = request_edge_to_edge_set_id[request_edge]
        return self.valid_sedge_paths[edge_set_id][source_mapping].get(target_mapping, [])


class ShortestValidPathsComputer_Flex(ShortestValidPathsComputer):
    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        super(ShortestValidPathsComputer_Flex, self).__init__(substrate, valid_mapping_restriction_computer,
                                                              edge_costs, edge_latencies, epsilon, limit)

        self.preds = np.full((self.number_of_nodes, 32 + 1), -1, dtype=np.int32)
        self.distances = np.full((self.number_of_nodes, 32 + 1), np.inf, dtype=np.float64)
        self.tau_modified_latencies = dict(zip(self.substrate.edges, [0] * len(self.substrate.edges)))

    def _handle_zero_delay_links(self, num_source_node, t):
        queue = [num_source_node]
        steps = 0

        while queue and steps < len(self.substrate.edges):  # len(nodes) !?
            num_current_node = queue.pop()
            steps += 1

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if self.sedge_valid.get(sedge, False) \
                        and self.tau_modified_latencies[sedge] == 0:
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    if not self.node_infeasible[num_endpoint]:
                        val = self.distances[num_current_node][t] + self.edge_costs[sedge]
                        if self.distances[num_endpoint][t] > val:
                            self.distances[num_endpoint][t] = val
                            self.preds[num_endpoint][t] = num_current_node
                            queue.append(num_endpoint)

    def _DAD(self, num_source_node, tau):
        if tau + 1 > np.shape(self.distances)[1]:
            self.distances = np.full((self.number_of_nodes, tau + 1), np.inf, dtype=np.float64)
            self.preds = np.full((self.number_of_nodes, tau + 1), -1, dtype=np.int32)
        else:
            self.distances.fill(np.inf)
            self.preds.fill(-1)

        self.distances[num_source_node][0] = 0
        self.preds[num_source_node][0] = num_source_node

        self._handle_zero_delay_links(num_source_node, 0)

        for t in range(1, tau + 1):
            for num_current_node in self.node_nums:

                self.distances[num_current_node][t] = self.distances[num_current_node][t - 1]
                self.preds[num_current_node][t] = self.preds[num_current_node][t - 1]

                for sedge in self.substrate.in_edges[self.num_id_to_snode_id[num_current_node]]:
                    if self.sedge_valid.get(sedge, False):
                        latency = self.tau_modified_latencies[sedge]
                        if latency <= t:
                            num_in_neighbor = self.snode_id_to_num_id[sedge[0]]
                            val = self.distances[num_in_neighbor][t - latency] + self.edge_costs[sedge]
                            if val < self.distances[num_current_node][t]:
                                self.distances[num_current_node][t] = val
                                self.preds[num_current_node][t] = num_in_neighbor

            for num_current_node in self.node_nums:
                self._handle_zero_delay_links(num_current_node, t)

    def _approx_latencies(self, num_source_node):
        self._preprocess(num_source_node)

        tau = 3

        self.paths = {i: None for i in self.node_nums}
        self.paths[num_source_node] = []

        approx_holds = False

        closed_nodes = np.copy(self.node_infeasible)
        closed_nodes[num_source_node] = True

        while not approx_holds:

            for key, value in self.edge_latencies.iteritems():
                val = int((value * tau) / self.limit)
                self.tau_modified_latencies[key] = val

            self._DAD(num_source_node, tau)

            approx_holds = True

            for num_target_node in self.node_nums:

                if closed_nodes[num_target_node]:
                    continue

                if self.preds[num_target_node][tau] == -1:
                    approx_holds = False
                    tau *= 2
                    break

                path = []
                added_latencies = 0
                n = num_target_node
                t = tau
                while n != num_source_node:
                    pred = self.preds[n][t]
                    sedge = self.num_id_to_snode_id[pred], self.num_id_to_snode_id[n]
                    path.append(sedge)
                    added_latencies += self.edge_latencies[sedge]
                    t -= self.tau_modified_latencies[sedge]
                    n = pred
                path = list(reversed(path))

                if not approx_holds or added_latencies > (1 + self.epsilon) * self.limit:
                    approx_holds = False
                    tau *= 2
                    break

                # approximation good enough, save result
                self.temp_latencies[num_target_node] = added_latencies
                self.paths[num_target_node] = path
                closed_nodes[num_target_node] = True

        return tau - 1

    def _compute_all_pairs(self):
        for edge_set_index in range(self.number_of_valid_edge_sets):

            self._prepare_valid_edges(edge_set_index)

            for num_source_node in range(self.number_of_nodes):

                converted_path_dict = {}
                source_snode = self.num_id_to_snode_id[num_source_node]

                final_tau = self._approx_latencies(num_source_node)

                for num_target_node in self.node_nums:

                    target_snode = self.num_id_to_snode_id[num_target_node]
                    costs = self.distances[num_target_node][final_tau]
                    self.valid_sedge_costs[edge_set_index][(source_snode, target_snode)] = costs

                    if not np.isnan(costs) and self.temp_latencies[num_target_node] > self.limit:
                        self.latency_limit_overstepped = True

                    converted_path_dict[target_snode] = self.paths[num_target_node]

                self.valid_sedge_paths[edge_set_index][source_snode] = converted_path_dict


class ShortestValidPathsComputer_Strict(ShortestValidPathsComputer):
    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        super(ShortestValidPathsComputer_Strict, self).__init__(substrate, valid_mapping_restriction_computer,
                                                                edge_costs, edge_latencies, epsilon, limit)

        self.FAIL = (None, np.nan)
        self.predecessors = np.full(self.number_of_nodes, -1, dtype=np.int32)
        self.distances = np.full((self.number_of_nodes, 1000), np.inf)

    def _SPPP(self, lower, upper, eps, num_source_node, num_target_node):
        S = float(lower * eps) / (self.num_feasible_nodes + 1)

        rescaled_costs = dict(zip(self.current_valid_edge_set, map(lambda x: int(float(self.edge_costs[x]) / S) + 1, self.current_valid_edge_set)))

        U_tilde = int(upper / S) + self.num_feasible_nodes + 1

        if U_tilde + 1 >= np.shape(self.distances)[1]:
            self.distances = np.full((self.number_of_nodes, U_tilde + 1), np.inf, dtype=np.float64)
        else:
            self.distances.fill(np.inf)

        self.distances[num_source_node][0] = 0
        self.predecessors[num_source_node] = num_source_node

        for i in range(1, U_tilde+1):
            for n in self.node_nums:
                self.distances[n][i] = self.distances[n][i-1]

                for e in self.substrate.in_edges[self.num_id_to_snode_id[n]]:
                    if self.sedge_valid.get(e, False) and rescaled_costs[e] <= i:
                        u, _ = e
                        comp_val = self.edge_latencies[e] + self.distances[self.snode_id_to_num_id[u]][i - rescaled_costs[e]]

                        if comp_val < self.distances[n][i]:
                            self.distances[n][i] = comp_val
                            self.predecessors[n] = self.snode_id_to_num_id[u]

            if self.distances[num_target_node][i] <= self.limit:
                # retrace path from t to s
                n = num_target_node
                path = []
                total_costs = 0
                while n != num_source_node:
                    pred = self.predecessors[n]
                    sedge = self.num_id_to_snode_id[pred], self.num_id_to_snode_id[n]
                    path.append(sedge)
                    total_costs += self.edge_costs[sedge]
                    n = pred
                path = list(reversed(path))

                return path, total_costs

        return self.FAIL

    def _Hassin(self, lower, upper, num_source_node, num_target_node):
        b_low = lower
        b_up = math.ceil(upper / 2)

        while b_low == 0 or b_up / b_low > 2:
            b = math.sqrt(b_low * b_up)
            if self._SPPP(b, b, 1, num_source_node, num_target_node) == self.FAIL:
                b_low = b
            else:
                b_up = b

        return self._SPPP(b_low, 2 * b_up, self.epsilon, num_source_node, num_target_node)


    def _sort_edges_distinct(self):
        return list(set([self.edge_costs[x] for x in
                    sorted(self.current_valid_edge_set, key=lambda i: self.edge_costs[i])]))


    def _shortest_path_latencies_limited(self, limit, num_source_node, num_target_node, retPath=False):
        queue = [(0, num_source_node)]
        self.temp_latencies.fill(np.inf)
        self.temp_latencies[num_source_node] = 0
        if retPath:
            preds = [-1] * self.number_of_nodes

        while queue:
            total_latencies, num_current_node = heappop(queue)

            if num_current_node == num_target_node:
                break

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if self.sedge_valid.get(sedge, False): # sedge in self.current_valid_edge_set:
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    cost = self.edge_costs[sedge]
                    lat = self.edge_latencies[sedge]
                    if cost <= limit:  # this is where the induced subgraph G_j comes in
                        if total_latencies + lat < self.temp_latencies[num_endpoint]:
                            self.temp_latencies[num_endpoint] = total_latencies + lat
                            heappush(queue, (total_latencies + lat, num_endpoint))
                            if retPath:
                                preds[num_endpoint] = num_current_node

        if retPath:
            n = num_target_node
            path = []
            total_costs = 0
            while n != num_source_node:
                pred = preds[n]
                sedge = self.num_id_to_snode_id[pred], self.num_id_to_snode_id[n]
                path.append(sedge)
                total_costs += self.edge_costs[sedge]
                n = pred
            path = list(reversed(path))
            return path, total_costs
        else:
            return self.temp_latencies[num_target_node]


    def _approx_latencies(self, num_source_node, num_target_node):

        if num_source_node == num_target_node:
            return [], 0

        low, high = -1, len(self.edge_levels_sorted)-1

        while low < high - 1:
            j = int((low + high) / 2)
            if self._shortest_path_latencies_limited(self.edge_levels_sorted[j], num_source_node,
                                                     num_target_node) < self.limit:
                high = j
            else:
                low = j

        lower = self.edge_levels_sorted[high]

        if lower == 0:
            return self._shortest_path_latencies_limited(0, num_source_node, num_target_node, True)

        upper = lower * self.num_feasible_nodes
        return self._Hassin(lower, upper, num_source_node, num_target_node)


    def _compute_all_pairs(self):

        for edge_set_index in range(self.number_of_valid_edge_sets):

            self._prepare_valid_edges(edge_set_index)
            self.edge_levels_sorted = self._sort_edges_distinct()

            for num_source_node in range(self.number_of_nodes):

                self._preprocess(num_source_node)
                converted_path_dict = {}

                for num_target_node in self.node_nums:

                    path, costs = self._approx_latencies(num_source_node, num_target_node)

                    self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = costs
                    converted_path_dict[self.num_id_to_snode_id[num_target_node]] = path

                self.valid_sedge_paths[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_path_dict


class ShortestValidPathsComputer_Exact(ShortestValidPathsComputer):
    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        super(ShortestValidPathsComputer_Exact, self).__init__(substrate, valid_mapping_restriction_computer,
                                                              edge_costs, edge_latencies, epsilon, limit)


    def _recursive_find_paths(self, current_node, target_node, visited, localPathList):
        """ Has exponential worst-case running time """

        visited[current_node] = True

        if current_node == target_node:
            yield localPathList

        for sedge in self.substrate.out_edges[self.num_id_to_snode_id[current_node]]:

            if sedge not in self.current_valid_edge_set:
                continue

            neighbor = self.snode_id_to_num_id[sedge[1]]
            if not visited[neighbor]:
                localPathList.append(sedge)
                for path in self._recursive_find_paths(neighbor, target_node, visited, localPathList):
                    yield path
                localPathList.remove(sedge)

        visited[current_node] = False


    def _approx_latencies(self, num_source_node, num_target_node):
        visited = np.full(self.number_of_nodes, False, np.bool)
        pathList = []
        lowestCosts = np.nan
        cheapestPath = []

        for path in self._recursive_find_paths(num_source_node, num_target_node, visited, pathList):

            if not path:
                continue

            total_latencies, total_costs = 0, 0

            for edge in path:
                total_latencies += self.edge_latencies[edge]
                total_costs += self.edge_costs[edge]

            if total_latencies <= self.limit:
                # path is feasible

                if not cheapestPath or total_costs < lowestCosts:
                    cheapestPath = list(path)
                    lowestCosts = total_costs

        if np.isnan(lowestCosts):
            return None, np.nan
        else:
            return cheapestPath, lowestCosts


    def _compute_all_pairs(self):
        for edge_set_index in range(self.number_of_valid_edge_sets):

            self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]

            for num_source_node in range(self.number_of_nodes):

                converted_path_dict = {}

                for num_target_node in range(self.number_of_nodes):

                    if num_source_node == num_target_node:
                        converted_path_dict[self.num_id_to_snode_id[num_target_node]] = []
                        self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = 0

                    path, costs = self._approx_latencies(num_source_node, num_target_node)

                    if not np.isnan(costs):

                        self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = costs

                        converted_path_dict[self.num_id_to_snode_id[num_target_node]] = path

                self.valid_sedge_paths[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_path_dict
