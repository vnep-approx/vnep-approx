from heapq import heappop, heappush
import numpy as np
import math

class ShortestValidPathsComputerLORENZ(object):

    ''' This class is optimized used to check correctness of the optimized lorenz class;
        they should produce the same output
    '''

    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        self.substrate = substrate
        self.valid_mapping_restriction_computer = valid_mapping_restriction_computer
        self.edge_costs = edge_costs
        self.edge_latencies = edge_latencies
        # self.edge_latencies = {sedge: max(lat, -lat) for sedge, lat in edge_latencies.iteritems()}
        self.epsilon = epsilon
        self.limit = limit
        self.edge_mapping_invalidities = False
        self.latency_limit_overstepped = False
        self.FAIL = (None, np.nan, np.nan)

        for sedge, lat in edge_latencies:
            if lat < 0:
                print "ERROR!! lat < 0:  ", lat
                self.edge_latencies[sedge] = -lat

        for sedge, lat in edge_costs:
            if lat < 0:
                print "ERROR!! cost < 0:  ", lat
                self.edge_costs[sedge] = -lat


    def compute(self):
        self.number_of_nodes = len(self.substrate.nodes)
        self.predecessor = np.full(self.number_of_nodes, -1, dtype=np.int32)
        self.temp_latencies = np.full(self.number_of_nodes, np.inf, dtype=np.float64)
        self.node_infeasible = np.full(self.number_of_nodes, False, dtype=np.bool)
        self.node_nums = []

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_latencies = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_paths = {id: {} for id in range(self.number_of_valid_edge_sets)}


        self._prepare_numeric_graph()
        self._compute_all_pairs()

    def recompute_with_new_costs(self, new_edge_costs):
        # self.edge_costs = new_edge_costs

        for sedge, cost in new_edge_costs.iteritems():

            if cost < 0:
                print "WARNING: cost < 0 in recompute.  ", cost

            self.edge_costs[sedge] = max(cost, -cost)

        self.compute()

    def recompute_with_new_latencies(self, new_edge_latencies):
        self.edge_latencies = new_edge_latencies
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


    def _SPPP(self, lower, upper, eps, num_source_node, num_target_node):
        S = float(lower * eps) / (self.number_of_nodes + 1)

        c_tilde = {}
        min_c_tilde = np.inf
        for e in self.current_valid_edge_set:
            val = int(float(self.edge_costs[e]) / S) + 1
            c_tilde[e] = val
            if val < min_c_tilde:
                min_c_tilde = val

        U_tilde = int(upper / S) + self.number_of_nodes + 1

        distances = np.full((self.number_of_nodes, U_tilde+1), np.inf)
        distances[num_source_node][0] = 0
        self.predecessor[num_source_node] = num_source_node

        for i in range(1, U_tilde+1):
            for snode in self.substrate.nodes:
                n = self.snode_id_to_num_id[snode]
                distances[n][i] = distances[n][i-1]

                for e in self.substrate.in_edges[snode]:
                    if e in self.current_valid_edge_set and c_tilde[e] <= i:
                        u, _ = e
                        comp_val = self.edge_latencies[e] + distances[self.snode_id_to_num_id[u]][i - c_tilde[e]]

                        if comp_val < distances[n][i]:
                            distances[n][i] = comp_val
                            self.predecessor[n] = self.snode_id_to_num_id[u]

            if distances[num_target_node][i] <= self.limit:
                # retrace path from t to s
                n = num_target_node
                path = []
                total_costs = 0
                while n != num_source_node:
                    pred = self.predecessor[n]
                    sedge = self.num_id_to_snode_id[pred], self.num_id_to_snode_id[n]
                    path.append(sedge)
                    total_costs += self.edge_costs[sedge]
                    n = pred
                path = list(reversed(path))

                return path, distances[num_target_node][i], total_costs

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
                if sedge in self.current_valid_edge_set:
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
            return path, self.temp_latencies[num_target_node], total_costs
        else:
            return self.temp_latencies[num_target_node]


    def _approx_latencies(self, num_source_node, num_target_node):

        if num_source_node == num_target_node:
            return [], 0, 0

        low, high = -1, len(self.edge_levels_sorted)-1

        while low < high - 1:
            j = int((low + high) / 2)
            if self._shortest_path_latencies_limited(self.edge_levels_sorted[j], num_source_node, num_target_node) < self.limit:
                high = j
            else:
                low = j

        lower = self.edge_levels_sorted[high]

        if lower == 0:
            return self._shortest_path_latencies_limited(0, num_source_node, num_target_node, True)

        upper = lower * self.number_of_nodes
        return self._Hassin(lower, upper, num_source_node, num_target_node)


    def _compute_all_pairs(self):

        for edge_set_index in range(self.number_of_valid_edge_sets):

            #TODO: move to preprocessing
            self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]
            # start_time = time.time()
            self.edge_levels_sorted = self._sort_edges_distinct()
            # print "time for sorting edges:  ", (time.time() - start_time)

            for num_source_node in range(self.number_of_nodes):

                print "starting from ", num_source_node

                converted_path_dict = {}

                for num_target_node in range(self.number_of_nodes):

                    if len(self.current_valid_edge_set) == 0:
                        path, lat, costs = self.FAIL
                    else:
                        self.predecessor.fill(-1)
                        path, lat, costs = self._approx_latencies(num_source_node, num_target_node)

                    self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = costs
                    self.valid_sedge_latencies[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = lat

                    if costs == np.nan:
                        self.edge_mapping_invalidities = True
                    elif lat > self.limit:
                        self.latency_limit_overstepped = True
                        print "WARNING: Latency limit overstepped by {} from {} to {}".format(lat - self.limit, self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])

                    converted_path_dict[self.num_id_to_snode_id[num_target_node]] = path

                self.valid_sedge_paths[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_path_dict

        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()

        for request_edge, edge_set_id_to_edge_set in request_edge_to_edge_set_id.iteritems():
            self.valid_sedge_costs[request_edge] = self.valid_sedge_costs[edge_set_id_to_edge_set]
            self.valid_sedge_paths[request_edge] = self.valid_sedge_paths[edge_set_id_to_edge_set]
