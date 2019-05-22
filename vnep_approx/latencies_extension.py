import math
import numpy as np
from heapq import heappush, heappop
import gurobipy
from alib import datamodel

FAIL = (None, None)
_latency_identifier = "latency"
_cost_identifier    = "cost"





class ShortestValidPathsComputerWithLatencies(object):

    ''' This class is optimized to compute all shortest paths in the substrate quickly for varying valid edge sets while
        approximately keeping the per-edge-larencies below a given limit.
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
        self.edge_costs = new_edge_costs
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


def SPPP(substrate, limit, lower, upper, eps, s, t):
    num_nodes = len(substrate.nodes)
    S = (lower * eps) / (num_nodes + 1)

    # TODO: save max value of c_tilde -> use to shrink distances (axis 1)
    c_tilde = {}
    for i, e in enumerate(substrate.edges):
        c_tilde[e] = int(substrate.edge[e][_cost_identifier] / S) + 1

    U_tilde = int(upper / S) + num_nodes + 1

    print "U_tilde", U_tilde

    distances = np.full((num_nodes, U_tilde+1), float('inf'))
    distances[s][0] = 0

    # TODO: build path & costs on the fly
    # DONE: Save only necessary items (head and latencies should suffice)
    predecessor_info = [(-1, 0)] * len(substrate.nodes)

    for i in range(1, U_tilde+1):
        for n in substrate.nodes:
            distances[n][i] = distances[n][i-1]
            for e in substrate.in_edges[n]:
                if c_tilde[e] <= i:
                    u, _ = e
                    comp_val = substrate.edge[e][_latency_identifier] + distances[u][i - c_tilde[e]]

                    if comp_val < distances[n][i]:
                        distances[n][i] = comp_val
                        predecessor_info[n] = (u, substrate.edge[e][_latency_identifier],
                                               substrate.edge[e][_cost_identifier])

        if distances[t][i] <= limit:
            # retrace path from t to s
            n = t
            path = [t]
            total_latencies = 0
            total_costs = 0
            while n != s:
                (n, lat, cost) = predecessor_info[n]
                total_latencies += lat
                total_costs += cost
                path = [n] + path

            print(total_latencies, distances[t][i])

            return path, total_latencies, total_costs

    return FAIL


def T(b, graph, limit, s, t):
    return SPPP(graph, limit, b, b, 1, s, t) is FAIL


def Hassin(graph, limit, lower, upper, eps, s, t):
    b_low = lower
    b_up = math.ceil(upper / 2)

    while b_up / b_low > 2:
        b = math.sqrt(b_low * b_up)
        if T(b, graph, limit, s, t):
            b_low = b
        else:
            b_up = b

    return SPPP(graph, limit, b_low, 2 * b_up, eps, s, t)


def _sort_edges_distinct(graph):
    return list(set([graph.edge[x][_cost_identifier] for x in
                sorted(graph.edges, key=lambda x: graph.edge[x][_cost_identifier])]))


def _shortest_path_latencies(graph, limit, s, t):
    # TODO: distances is LARGE, consider reusing
    # TODO: reuse shortest paths!?

    queue = [(0, s)]
    distances = [float('inf')] * len(graph.nodes)
    distances[s] = 0

    while queue:
        costs, node = heappop(queue)

        if node == t:
            break

        for u, v in graph.out_edges[str(node)]:
            lat = graph.edge[(u, v)][_latency_identifier]
            if lat <= limit:  # this is where the induced subgraph G_j comes in
                if costs + lat < distances[v]:
                    distances[v] = costs + lat
                    heappush(queue, (costs + lat, v))

    return distances[t]


def approx_latencies(graph, limit, eps, s, t):
    edges_sorted = _sort_edges_distinct(graph)

    l = len(edges_sorted)

    low, high = 1, l

    while low < high-1:
        j = int((low + high) / 2)
        # only pass j and edges_sorted, in shortest path: only use edges whose values doesnt exceed c_j
        if _shortest_path_latencies(graph, edges_sorted[j], s, t) < limit:
            high = j
        else:
            low = j

    lower = edges_sorted[low]
    upper = lower * len(graph.nodes)
    return Hassin(graph, limit, lower, upper, eps, s, t)


class Graph2:
    def __init__(self, num_nodes):
        self.nodes = list(range(num_nodes))
        self.edges = []
        self.in_edges = {}
        self.out_edges = {}
        self.edge = {}

        for n in self.nodes:
            self._init_node(n)

    def _init_node(self, node):
        self.in_edges[node] = []
        self.out_edges[node] = []

    def add_edge(self, head, tail, **kwargs):
        new_edge = (head, tail)
        self.edges.append(new_edge)
        self.in_edges[tail].append(new_edge)
        self.out_edges[head].append(new_edge)
        self.edge[new_edge] = {}
        for key, value in kwargs.items():
            self.edge[new_edge][key] = value


def _build_example_graph():
    g = datamodel.Graph("Substrate")
    for i in xrange(6):
        g.add_node(i)
    g.add_edge(0, 1, latency=5, cost=10)
    g.add_edge(0, 2, latency=2, cost=7)
    g.add_edge(2, 1, latency=2, cost=13)
    g.add_edge(1, 3, latency=6, cost=6)
    g.add_edge(3, 4, latency=3, cost=2)
    g.add_edge(4, 2, latency=3, cost=30)
    g.add_edge(5, 2, latency=1, cost=6)
    g.add_edge(4, 5, latency=4, cost=9)
    g.add_edge(3, 5, latency=7, cost=12)
    g.add_edge(2, 4, latency=3, cost=30)
    return g


def __test():
    g = _build_example_graph()
    print(approx_latencies(g, 17, 0.2, 0, 4))
    print(approx_latencies(g, 17, 0.10, 0, 5))


if __name__ == '__main__':
    __test()


