import math
import numpy as np


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
        self.temp_latencies = np.full(self.number_of_nodes, -1, dtype=np.int32)
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


    def _DAD(self, num_source_node, tau):

        print("DAD? is that you?\nstarting from {}".format(self.num_id_to_snode_id[num_source_node]))

        self.distances = np.full((self.number_of_nodes, tau+1), np.inf, dtype=np.float64)
        self.distances[num_source_node][0] = 0

        self.paths = {self.snode_id_to_num_id[snode]: [snode] for snode in self.substrate.nodes}

        tau_modified_latencies = {}
        for key, value in self.edge_latencies.items():
            tau_modified_latencies[key] = value * tau / self.limit

        for t in range(1, tau+1):
            for num_current_node in range(self.number_of_nodes):
                self.distances[num_current_node][t] = self.distances[num_current_node][t - 1]
                for sedge in self.current_valid_edge_set:
                    if sedge in self.substrate.in_edges[self.num_id_to_snode_id[num_current_node]]:
                        latency = tau_modified_latencies[sedge]
                        if latency < t:
                            num_in_neighbor = self.snode_id_to_num_id[sedge[0]]
                            val = self.distances[num_in_neighbor][t - latency] + self.edge_costs[sedge]
                            if val < self.distances[num_current_node][t]:
                                self.distances[num_current_node][t] = val

                                last_node_on_path = self.paths[num_current_node][-1]

                                if last_node_on_path != sedge[1]:
                                    self.paths[num_current_node][-1] = sedge[0]
                                    print sedge
                                else:
                                    self.paths[num_current_node] += self.num_id_to_snode_id[num_in_neighbor]


    def _approx_latencies(self, num_source_node):

        # One run of dijkstra with latencies as metric

        tau = 100# int(math.log(self.limit) / 2)

        self.predecessor.fill(-1)

        approx_holds = False

        while not approx_holds:

            self._DAD(num_source_node, tau)

            approx_holds = True

            for snode in self.substrate.nodes:
                added_latencies = 0

                print "to ", snode, ":  " , self.paths[self.snode_id_to_num_id[snode]]

                for next_snode in self.paths[self.snode_id_to_num_id[snode]][1:-1]:
                    sedge = (snode, next_snode)
                    snode = next_snode
                    added_latencies += self.edge_latencies[sedge]

                self.temp_latencies[self.snode_id_to_num_id[snode]] = added_latencies

                if added_latencies > (1 + self.epsilon) * self.limit:
                    tau *= 2
                    approx_holds = False
                    break

        return tau-1


    def _compute_all_pairs(self):

        for edge_set_index in range(self.number_of_valid_edge_sets):

            self.current_valid_edge_set = self.edge_set_id_to_edge_set[edge_set_index]

            for num_source_node in range(self.number_of_nodes):

                converted_path_dict = {}

                final_tau = self._approx_latencies(num_source_node)

                for num_target_node in range(self.number_of_nodes):

                    costs = self.distances[num_target_node][final_tau]

                    # n = num_target_node
                    # path = [self.num_id_to_snode_id[num_target_node]]
                    # total_costs = 0
                    # while n != num_source_node:
                    #     end = self.num_id_to_snode_id[n]
                    #     n = self.predecessor[n]
                    #     sedge = (self.num_id_to_snode_id[n], end)
                    #     total_costs += self.edge_costs[sedge]
                    #     path = [self.num_id_to_snode_id[n]] + path


                    # print "Equality check:  {} == {} \t \t -> {}".format(costs, total_costs, total_costs == costs)


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

                    converted_path_dict[self.num_id_to_snode_id[num_target_node]] = \
                        self.paths[num_target_node] if len(self.paths[num_target_node]) > 1 \
                            or num_source_node == num_target_node else None

                self.valid_sedge_paths[edge_set_index][self.num_id_to_snode_id[num_source_node]] = converted_path_dict

        request_edge_to_edge_set_id = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()

        for request_edge, edge_set_id_to_edge_set in request_edge_to_edge_set_id.iteritems():
            self.valid_sedge_costs[request_edge] = self.valid_sedge_costs[edge_set_id_to_edge_set]
            self.valid_sedge_paths[request_edge] = self.valid_sedge_paths[edge_set_id_to_edge_set]
