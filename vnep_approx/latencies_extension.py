import math
import numpy as np
from heapq import heappush, heappop
import gurobipy
from alib import datamodel


#       path, costs, latencies
FAIL = (None, np.nan, np.nan)


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

    def compute(self):
        self.number_of_nodes = len(self.substrate.nodes)
        self.predecessor = np.full(self.number_of_nodes, -1, dtype=np.int32)
        self.temp_distances = np.full(self.number_of_nodes, np.inf, dtype=np.float64)

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_pred = {id: {} for id in range(self.number_of_valid_edge_sets)}
        self.valid_sedge_latencies = {id: {} for id in range(self.number_of_valid_edge_sets)}


        self._prepare_numeric_graph()
        self._compute_all_pairs()

    def recompute_with_new_costs(self, new_edge_costs):
        self.edge_costs = new_edge_costs
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
        S = (lower * eps) / (self.number_of_nodes + 1)

        # TODO: save max value of c_tilde -> use to shrink distances matrix (axis 1)
        c_tilde = {}
        max_c_tilde = -1
        for e in self.substrate.edges:
            val = int(self.edge_costs[e] / S) + 1
            c_tilde[e] = val
            if val > max_c_tilde:
                max_c_tilde = val

        U_tilde = int(upper / S) + self.number_of_nodes + 1


        print "U_tilde:", U_tilde, max_c_tilde, num_source_node, num_target_node, lower, upper
        # return FAIL

        # unfortunately has different size for each call to SPPP - (maybe use submatrix?)
        distances = np.full((self.number_of_nodes, U_tilde+1), np.inf)
        distances[num_source_node][0] = 0
        self.predecessor[num_source_node] = num_source_node

        # TODO: build path & costs on the fly
        # DONE: Save only necessary items (head and latencies should suffice)
        # TODO: get rid of predecessor_info
        predecessor_info = self.number_of_nodes * [(-1, 0, 0)]

        for i in range(1, U_tilde+1):
            for snode in self.substrate.nodes:
                n = self.snode_id_to_num_id[snode]
                distances[n][i] = distances[n][i-1]
                for e in self.substrate.in_edges[snode]:
                    if c_tilde[e] <= i:
                        u, _ = e
                        comp_val = self.edge_latencies[e] + distances[self.snode_id_to_num_id[u]][i - c_tilde[e]]

                        if comp_val < distances[n][i]:
                            distances[n][i] = comp_val
                            predecessor_info[n] = (self.snode_id_to_num_id[u], self.edge_latencies[e], self.edge_costs[e])
                            self.predecessor[n] = self.snode_id_to_num_id[u]

            if distances[num_target_node][i] <= self.limit:
                # retrace path from t to s
                n = num_target_node
                path = [self.num_id_to_snode_id[num_target_node]]
                total_latencies = 0
                total_costs = 0
                while n != num_source_node:
                    n, lat, cost = predecessor_info[n]
                    total_latencies += lat
                    total_costs += cost
                    path = [self.num_id_to_snode_id[n]] + path

                if distances[num_target_node][i] != total_latencies:
                    print ("Equality check failed: dist[t][i] = {}  !=  {} = total_lat".format(distances[num_target_node][i], total_latencies))

                return path, total_latencies, total_costs

        return FAIL

    def _Hassin(self, lower, upper, num_source_node, num_target_node):
        b_low = lower
        b_up = math.ceil(upper / 2)

        while b_up / b_low > 2:
            b = math.sqrt(b_low * b_up)
            if self._SPPP(b, b, 1, num_source_node, num_target_node) == FAIL:
                b_low = b
            else:
                b_up = b

        print "\t\t\t\t", b_low, b_up

        return self._SPPP(b_low, 2 * b_up, self.epsilon, num_source_node, num_target_node)


    def _sort_edges_distinct(self):
        return list(set([self.edge_costs[x] for x in
                    sorted(self.substrate.edges, key=lambda i: self.edge_costs[i])]))


    def _shortest_path_latencies_limited(self, limit, num_source_node, num_target_node):
        # DONE: distances is LARGE, consider reusing
        # TODO: reuse shortest paths!?
        # TODO: try to reuse this for later nodes!

        queue = [(0, num_source_node)]
        self.temp_distances[num_source_node] = 0

        while queue:
            costs, node = heappop(queue)

            if node == num_target_node:
                break

            for u, v in self.substrate.out_edges[node]:
                lat = self.edge_costs[(u, v)]
                if lat <= limit:  # this is where the induced subgraph G_j comes in
                    if costs + lat < self.temp_distances[v]:
                        self.temp_distances[v] = costs + lat
                        heappush(queue, (costs + lat, v))

        return self.temp_distances[num_target_node]


    def _approx_latencies(self, num_source_node, num_target_node):
        low, high = 0, len(self.edge_levels_sorted)

        while low < high-1:
            j = int((low + high) / 2)
            if self._shortest_path_latencies_limited(self.edge_levels_sorted[j], num_source_node, num_target_node) < self.limit:
                high = j
            else:
                low = j

        lower = self.edge_levels_sorted[low]
        upper = lower * self.number_of_nodes
        return self._Hassin(lower, upper, num_source_node, num_target_node)


    def _compute_all_pairs(self):

        self.edge_levels_sorted = self._sort_edges_distinct()

        for edge_set_index in range(self.number_of_valid_edge_sets):

            for num_source_node in range(self.number_of_nodes):

                for num_target_node in range(self.number_of_nodes):
                    path, lat, costs = self._approx_latencies(num_source_node, num_target_node)

                    print "from ", self.num_id_to_snode_id[num_source_node], " to", \
                        self.num_id_to_snode_id[num_target_node], ":\n\t", path, lat, costs

                    print "predecessors:\n", self.predecessor, "\n"

                    # TODO: what if approx_latency returns FAIL

                    self.valid_sedge_costs[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = costs
                    self.valid_sedge_latencies[edge_set_index][(self.num_id_to_snode_id[num_source_node], self.num_id_to_snode_id[num_target_node])] = lat

                    if costs == np.nan:
                        self.edge_mapping_invalidities = True
                    elif lat > self.limit:
                        self.latency_limit_overstepped = True

                converted_pred_dict = {self.num_id_to_snode_id[num_node]: self.num_id_to_snode_id[self.predecessor[num_node]]
                    if self.predecessor[num_node] != -1 else None for num_node in range(self.number_of_nodes)}

                self.valid_sedge_pred[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_pred_dict

        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()

        for request_edge, edge_set_id_to_edge_set in request_edge_to_edge_set_id.iteritems():
            self.valid_sedge_costs[request_edge] = self.valid_sedge_costs[edge_set_id_to_edge_set]
            self.valid_sedge_pred[request_edge] = self.valid_sedge_pred[edge_set_id_to_edge_set]
