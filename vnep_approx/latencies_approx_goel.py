import math
import numpy as np
from heapq import heappush, heappop

class ShortestValidPathsComputerWithLatencies(object):
    ''' This class is optimized to compute all shortest paths in the substrate quickly for varying valid edge sets while
        approximately keeping the per-edge-latencies below a given limit.
    '''

    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        self.substrate = substrate
        self.valid_mapping_restriction_computer = valid_mapping_restriction_computer
        self.edge_costs = edge_costs
        self.edge_latencies = edge_latencies
        self.epsilon = epsilon
        self.limit = limit
        self.edge_mapping_invalidities = False
        self.latency_limit_overstepped = False
        self.FAIL = (None, np.nan, np.nan)

    def compute(self):
        self.number_of_nodes = len(self.substrate.nodes)
        self.predecessor = np.full(self.number_of_nodes, -1, dtype=np.int32)
        self.temp_latencies = np.full(self.number_of_nodes, -1, dtype=np.float64)
        self.node_infeasible = np.full(self.number_of_nodes, False, dtype=np.bool)
        # self.distances = np.full(self.number_of_nodes, np.inf, dtype=np.float64)

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_latencies = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_paths = {id: {} for id in range(self.number_of_valid_edge_sets)}

        self._prepare_numeric_graph()
        self._compute_all_pairs()

    def recompute_with_new_costs(self, new_edge_costs):
        self.edge_costs = new_edge_costs
        self.compute()

    def recompute_with_new_latencies(self, new_edge_latencies):
        self.edge_latencies = new_edge_latencies
        self.compute()

    def _prepare_numeric_graph(self):
        # prepare and store information on "numeric graph"
        self.number_of_nodes = 0
        self.num_id_to_snode_id = []
        self.snode_id_to_num_id = {}

        for snode in self.substrate.nodes:
            self.num_id_to_snode_id.append(snode)
            self.snode_id_to_num_id[snode] = self.number_of_nodes
            self.number_of_nodes += 1


    def _handle_zero_delay_links(self, num_source_node, tau_modified_latencies, t):
        queue = [num_source_node]

        while queue:
            num_current_node = queue.pop()

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if sedge in self.current_valid_edge_set:
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    if self.distances[num_endpoint][t] > self.distances[num_source_node][t]\
                            and tau_modified_latencies[sedge] == 0:
                        self.distances[num_endpoint][t] = self.distances[num_source_node][t]
                        self.preds[num_endpoint][t] = num_current_node
                        queue.append(num_endpoint)


    def _DAD(self, num_source_node, tau, tau_modified_latencies):

        self.distances = np.full((self.number_of_nodes, tau+1), np.inf, dtype=np.float64)
        self.distances[num_source_node][0] = 0

        self.preds = np.full((self.number_of_nodes, tau+1), -1, dtype=np.int32)
        self.preds[num_source_node][0] = num_source_node


        for t in range(1, tau+1):
            for num_current_node in range(self.number_of_nodes):

                if self.node_infeasible[num_current_node]:
                    continue

                self._handle_zero_delay_links(num_current_node, tau_modified_latencies, t)

                val = self.distances[num_current_node][t-1]
                if self.distances[num_current_node][t] > val:
                    self.distances[num_current_node][t] = val
                    self.preds[num_current_node][t] = self.preds[num_current_node][t - 1]

                for sedge in self.current_valid_edge_set:
                    if sedge in self.substrate.in_edges[self.num_id_to_snode_id[num_current_node]]:
                        latency = tau_modified_latencies[sedge]
                        if latency <= t:
                            num_in_neighbor = self.snode_id_to_num_id[sedge[0]]
                            val = self.distances[num_in_neighbor][t - latency] + self.edge_costs[sedge]
                            if val < self.distances[num_current_node][t]:
                                self.distances[num_current_node][t] = val
                                self.preds[num_current_node][t] = num_in_neighbor


    # one run of dijkstra using delay metric to root out infeasible nodes
    def _preprocess(self, num_source_node):

        queue = [(0, num_source_node)]
        self.temp_latencies.fill(np.inf)
        self.temp_latencies[num_source_node] = 0
        self.node_infeasible.fill(False)
        preds = np.full(self.number_of_nodes, -1, dtype=np.int32)

        while queue:
            total_latencies, num_current_node = heappop(queue)

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if sedge in self.current_valid_edge_set:
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    lat = self.edge_latencies[sedge]
                    if total_latencies + lat < self.temp_latencies[num_endpoint]:
                        self.temp_latencies[num_endpoint] = total_latencies + lat
                        heappush(queue, (total_latencies + lat, num_endpoint))
                        preds[num_endpoint] = num_current_node

        # self.num_infeasible_nodes = 0
        for num_node in range(self.number_of_nodes):
            if self.temp_latencies[num_node] > self.limit:
                self.node_infeasible[num_node] = True
                # self.num_infeasible_nodes += 1



    def _approx_latencies(self, num_source_node):

        self._preprocess(num_source_node)

        tau = int(self.limit) / 100 + 1 # max(int(math.log(self.limit) / 2), 1)

        self.paths = {self.snode_id_to_num_id[snode]: None for snode in self.substrate.nodes}

        approx_holds = False

        while not approx_holds:

            tau_modified_latencies = {}
            for key, value in self.edge_latencies.items():
                tau_modified_latencies[key] = int(value * tau / self.limit)

            self._DAD(num_source_node, tau, tau_modified_latencies)

            approx_holds = True

            for num_target_node in range(self.number_of_nodes):

                if self.node_infeasible[num_target_node] or num_source_node == num_target_node:
                    continue

                added_latencies = 0
                n = num_target_node
                t = tau

                if self.preds[num_target_node][t] == -1:
                    approx_holds = False
                    tau *= 2
                    break

                path = [self.num_id_to_snode_id[num_target_node]]
                while n != num_source_node:
                    end = self.num_id_to_snode_id[n]
                    n = self.preds[n][t]

                    if n == -1:
                        # no path could be found!!
                        added_latencies = 0
                        approx_holds = False
                        # print "invalid edge:  " , end , " has no predecessor"
                        break


                    sedge = (self.num_id_to_snode_id[n], end)
                    added_latencies += self.edge_latencies[sedge]
                    path = [self.num_id_to_snode_id[n]] + path
                    t -= tau_modified_latencies[self.num_id_to_snode_id[n], end]

                if added_latencies > (1 + self.epsilon) * self.limit:
                    approx_holds = False

                if not approx_holds:
                    tau *= 2
                    break

                # approximation good enough
                self.temp_latencies[num_target_node] = added_latencies
                self.paths[num_target_node] = path

        return tau-1


    def _compute_all_pairs(self):

        for edge_set_index in range(self.number_of_valid_edge_sets):

            self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]

            for num_source_node in range(self.number_of_nodes):

                converted_path_dict = {}

                final_tau = self._approx_latencies(num_source_node)

                for num_target_node in range(self.number_of_nodes):

                    if self.node_infeasible[num_target_node]:
                        self.valid_sedge_costs[edge_set_index][
                            (self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = np.nan
                        self.valid_sedge_latencies[edge_set_index][
                            (self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = np.nan

                        converted_path_dict[self.num_id_to_snode_id[num_target_node]] = None
                        continue

                    costs = self.distances[num_target_node][final_tau]

                    self.valid_sedge_costs[edge_set_index][
                        (self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = costs
                    self.valid_sedge_latencies[edge_set_index][
                        (self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = self.temp_latencies[num_target_node]

                    if costs == np.nan:
                        self.edge_mapping_invalidities = True
                    elif self.temp_latencies[num_target_node] > self.limit:
                        self.latency_limit_overstepped = True
                        print "WARNING: Latency limit overstepped by {} from {} to {}".format(self.temp_latencies[num_target_node] - self.limit,
                                                                                              self.num_id_to_snode_id[
                                                                                                  num_source_node],
                                                                                              self.num_id_to_snode_id[
                                                                                                  num_target_node])

                    converted_path_dict[self.num_id_to_snode_id[num_target_node]] = self.paths[num_target_node]\
                        if num_target_node != num_source_node else [self.num_id_to_snode_id[num_source_node]]

                self.valid_sedge_paths[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_path_dict

        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()

        for request_edge, edge_set_id_to_edge_set in request_edge_to_edge_set_id.iteritems():
            self.valid_sedge_costs[request_edge] = self.valid_sedge_costs[edge_set_id_to_edge_set]
            self.valid_sedge_paths[request_edge] = self.valid_sedge_paths[edge_set_id_to_edge_set]

