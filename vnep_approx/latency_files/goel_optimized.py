import numpy as np
from heapq import heappush, heappop

class ShortestValidPathsComputer(object):
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

    def compute(self):
        self.number_of_nodes = len(self.substrate.nodes)
        self.temp_latencies = np.full(self.number_of_nodes, -1, dtype=np.float64)
        self.node_infeasible = np.full(self.number_of_nodes, False, dtype=np.bool)
        self.distances = np.full((self.number_of_nodes, 32 + 1), np.inf, dtype=np.float64)
        self.preds = np.full((self.number_of_nodes, 32 + 1), -1, dtype=np.int32)

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_paths = {id: {} for id in range(self.number_of_valid_edge_sets)}

        self._prepare_numeric_graph()
        self._compute_all_pairs()

    def recompute_with_new_costs(self, new_edge_costs):
        self.edge_costs = new_edge_costs
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


    def _prepare_valid_edges(self, edge_set_index):
        self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]
        self.sedge_valid = {}
        for e in self.current_valid_edge_set:
            self.sedge_valid[e] = True

    def _handle_zero_delay_links(self, num_source_node, tau_modified_latencies, t):
        queue = [num_source_node]
        steps = 0

        while queue:
            num_current_node = queue.pop()
            steps += 1

            for sedge in self.substrate.out_edges[self.num_id_to_snode_id[num_current_node]]:
                if self.sedge_valid.get(sedge, False) \
                        and tau_modified_latencies[sedge] == 0:
                    num_endpoint = self.snode_id_to_num_id[sedge[1]]
                    if not self.node_infeasible[num_endpoint]:
                        val = self.distances[num_current_node][t] + self.edge_costs[sedge]
                        if self.distances[num_endpoint][t] > val:
                            self.distances[num_endpoint][t] = val
                            self.preds[num_endpoint][t] = num_current_node
                            queue.append(num_endpoint)

    def _DAD(self, num_source_node, tau, tau_modified_latencies):
        if tau + 1 > np.shape(self.distances)[1]:
            self.distances = np.full((self.number_of_nodes, tau + 1), np.inf, dtype=np.float64)
            self.preds = np.full((self.number_of_nodes, tau + 1), -1, dtype=np.int32)
        else:
            self.distances.fill(np.inf)
            self.preds.fill(-1)

        self.distances[num_source_node][0] = 0
        self.preds[num_source_node][0] = num_source_node

        self._handle_zero_delay_links(num_source_node, tau_modified_latencies, 0)

        for t in range(1, tau + 1):
            if t > 1:
                for num_start_node in self.node_nums:
                    self._handle_zero_delay_links(num_start_node, tau_modified_latencies, t - 1)

            for num_current_node in self.node_nums:

                self.distances[num_current_node][t] = self.distances[num_current_node][t - 1]
                self.preds[num_current_node][t] = self.preds[num_current_node][t - 1]

                for sedge in self.substrate.in_edges[self.num_id_to_snode_id[num_current_node]]:
                    if self.sedge_valid.get(sedge, False):
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
        for num_node in range(self.number_of_nodes):
            if self.temp_latencies[num_node] > self.limit:
                self.node_infeasible[num_node] = True
                self.edge_mapping_invalidities = True
            else:
                self.node_nums.append(num_node)
                self.num_feasible_nodes += 1


    def _approx_latencies(self, num_source_node):

        self._preprocess(num_source_node)

        tau = 3  # int(self.limit) / 100 + 1 # max(int(math.log(self.limit) / 2), 1)

        self.paths = {i: None for i in range(self.number_of_nodes)}
        self.paths[num_source_node] = []

        approx_holds = False

        closed_nodes = np.copy(self.node_infeasible)
        closed_nodes[num_source_node] = True

        while not approx_holds:  # and tau < 200:

            print " ---------------- tau:  ", tau, "\t\t -----------------"

            tau_modified_latencies = {}
            for key, value in self.edge_latencies.items():
                val = int((value * tau) / self.limit)
                tau_modified_latencies[key] = val

            self._DAD(num_source_node, tau, tau_modified_latencies)

            approx_holds = True

            for num_target_node in self.node_nums:

                if closed_nodes[num_target_node]:
                    continue

                if tau / 2 > self.number_of_nodes / self.epsilon:
                    print "ERROR: too many iterations"

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
                    t -= tau_modified_latencies[sedge]
                    n = pred
                path = list(reversed(path))

                if added_latencies > (1 + self.epsilon) * self.limit:
                    approx_holds = False
                    # print "end"

                if not approx_holds:
                    tau *= 2
                    break

                # approximation good enough, save result
                self.temp_latencies[num_target_node] = added_latencies
                self.paths[num_target_node] = path
                closed_nodes[num_target_node] = True

        return  tau - 1 #if approx_holds else tau/2 - 1

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

                    if np.isnan(costs):
                        print "shouldnt happen"
                    elif self.temp_latencies[num_target_node] > self.limit:
                        self.latency_limit_overstepped = True

                    converted_path_dict[target_snode] = self.paths[num_target_node]

                self.valid_sedge_paths[edge_set_index][source_snode] = converted_path_dict

    def get_valid_sedge_costs_for_reqedge(self, request_edge, mapping_edge):
        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()
        edge_set_id_to_edge_set = request_edge_to_edge_set_id[request_edge]
        return self.get_valid_sedge_costs_from_edgesetindex(edge_set_id_to_edge_set, mapping_edge)

    def get_valid_sedge_costs_from_edgesetindex(self, edge_set_index, reqedge):
        return self.valid_sedge_costs[edge_set_index].get(reqedge, np.nan)

    def get_valid_sedge_path(self, request_edge, source_mapping, target_mapping):
        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()
        edge_set_id_to_edge_set = request_edge_to_edge_set_id[request_edge]
        return self.valid_sedge_paths[edge_set_id_to_edge_set][source_mapping].get(target_mapping, None)

