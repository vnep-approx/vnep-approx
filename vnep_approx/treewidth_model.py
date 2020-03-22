# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math
import itertools
import os
import subprocess32 as subprocess
from collections import deque
import numpy as np
import random
import time
from heapq import heappush, heappop
import enum
from collections import namedtuple

import time

try:
    import cPickle as pickle
except ImportError:
    import pickle


import gurobipy
from gurobipy import GRB, LinExpr
from alib import datamodel, modelcreator, solutions, util

from . import randomized_rounding_triumvirate as rrt

""" This module contains data structures and algorithms related to treewidth based approximation approaches """

class TreeDecomposition(datamodel.UndirectedGraph):
    ''' Representation of a tree decomposition.'''

    def __init__(self, name):
        super(TreeDecomposition, self).__init__(name)

        self.node_bag_dict = {}  # map TD nodes to their bags
        self.representative_map = {}  # map request nodes to representative TD nodes
        self.complete_request_node_to_tree_node_map = {}

    def add_node(self, node, node_bag=None):
        ''' adds a node to the tree decomposition and stores the bag information; edges must be created externally.

        :param node:
        :param node_bag:
        :return: None
        '''
        if not node_bag:
            raise ValueError("Empty or unspecified node bag: {}".format(node_bag))
        if not isinstance(node_bag, frozenset):
            raise ValueError("Expected node bag as frozenset: {}".format(node_bag))
        super(TreeDecomposition, self).add_node(node)
        self.node_bag_dict[node] = node_bag
        # edges_to_create = set()
        for req_node in node_bag:
            if req_node not in self.representative_map:
                self.representative_map[req_node] = node
            if req_node not in self.complete_request_node_to_tree_node_map:
                self.complete_request_node_to_tree_node_map[req_node] = [node]
            else:
                self.complete_request_node_to_tree_node_map[req_node].append(node)


    def remove_node(self, node):
        del self.node_bag_dict[node]


        for req_node in self.complete_request_node_to_tree_node_map.keys():
            if node in self.complete_request_node_to_tree_node_map[req_node]:
                self.complete_request_node_to_tree_node_map[req_node].remove(node)
            if self.representative_map[req_node] == node:
                self.representative_map[req_node] = self.complete_request_node_to_tree_node_map[req_node][0]

        super(TreeDecomposition, self).remove_node(node)

    def convert_to_arborescence(self, root=None):
        if not self._verify_intersection_property() or not self._is_tree():
            raise ValueError("Cannot derive decomposition arborescence from invalid tree decomposition!")
        if root is None:
            root = next(iter(self.nodes))
        arborescence = TDArborescence(self.name, root)
        q = {root}
        visited = set()
        while q:
            node = q.pop()
            arborescence.add_node(node)
            visited.add(node)
            for other in self.get_neighbors(node):
                if other not in visited:
                    arborescence.add_node(other)
                    arborescence.add_edge(node, other)
                    q.add(other)
        return arborescence

    @property
    def width(self):
        return max(len(bag) for bag in self.node_bag_dict.values()) - 1

    def get_bag_intersection(self, t1, t2):
        return self.node_bag_dict[t1] & self.node_bag_dict[t2]

    def get_representative(self, req_node):
        if req_node not in self.representative_map:
            raise ValueError("Cannot find representative for unknown node {}!".format(req_node))
        return self.representative_map[req_node]

    def get_any_covering_td_node(self, *args):
        covering_nodes = set(self.node_bag_dict.iterkeys())
        for i in args:
            if i not in self.representative_map:
                raise ValueError("Cannot find representative for unknown node {}!".format(i))
            covering_nodes &= {t for t in covering_nodes if i in self.node_bag_dict[t]}
            if not covering_nodes:
                print "Nodes {} have no covering node!".format(", ".join(str(i) for i in args))
                return None
        return next(iter(covering_nodes))

    def is_tree_decomposition(self, request):
        if not self._is_tree():
            print("Not a tree!")
            return False
        if not self._verify_all_nodes_covered(request):
            print("Not all nodes are covered!")
            return False
        if not self._verify_all_edges_covered(request):
            print("Not all edges are covered!")
            return False
        if not self._verify_intersection_property():
            print("Intersection Property does not hold!")
            return False
        return True

    def _is_tree(self):
        if not self.nodes:
            return True
        start_node = next(iter(self.nodes))
        visited = set()
        q = {start_node}
        while q:
            t = q.pop()
            visited.add(t)
            if len(self.get_neighbors(t) & visited) > 1:
                return False
            q |= self.get_neighbors(t) - visited

        return visited == set(self.nodes)

    def _verify_all_nodes_covered(self, req):
        return set(self.representative_map.keys()) == set(req.nodes)

    def _verify_all_edges_covered(self, req):
        for (i, j) in req.edges:
            # Check that there is some overlap in the sets of representative nodes:
            found_covering_bag = False
            for node_bag in self.node_bag_dict.values():
                if i in node_bag and j in node_bag:
                    found_covering_bag = True
                    break
            if found_covering_bag:
                break
        return True

    def _verify_intersection_property(self):
        # Check that subtrees induced by each request node are connected
        for req_node in self.representative_map:
            subtree_nodes = {t for (t, bag) in self.node_bag_dict.items() if req_node in bag}
            start_node = self.get_representative(req_node)
            visited = set()
            q = {start_node}
            while q:
                t = q.pop()
                visited.add(t)
                subtree_neighbors = self.get_neighbors(t) & subtree_nodes
                unvisited_neighbors = (subtree_neighbors - visited)
                q |= unvisited_neighbors
            if not visited == subtree_nodes:
                return False
        return True


class TDArborescence(datamodel.Graph):

    ''' Representation of a directed and rooted tree decomposition, i.e. an arborescence.'''

    def __init__(self, name, root):
        super(TDArborescence, self).__init__(name)
        self.root = root

    def add_node(self, node, **kwargs):
        super(TDArborescence, self).add_node(
            node, **kwargs
        )

    def add_edge(self, tail, head, **kwargs):
        if self.get_in_neighbors(head):
            raise ValueError("Error: Arborescence must not contain confluences!")
        super(TDArborescence, self).add_edge(
            tail, head, bidirected=False
        )

    def post_order_traversal(self):
        visited = set()
        q = [self.root]
        while q:
            t = q.pop()
            unvisited_children = set(self.out_neighbors[t]) - visited
            if not unvisited_children:  # leaf or children have all been handled
                yield t
                visited.add(t)
            else:
                q.append(t)  # put t back on stack, to revisit after its children
                for c in sorted(unvisited_children, reverse=True):  # TODO: sorted only for testing
                    q.append(c)
        assert visited == self.nodes  # sanity check

class NodeType(object):
    ''' Used within the DynVMP algorithm to classify nodes.'''
    Leaf = 0
    Introduction = 1
    Forget = 2
    Join = 3
    Root = 4

class SmallSemiNiceTDArb(TreeDecomposition):

    ''' A specific Tree Decomposition Arborescence (TDArb):
        - small means that no node bag is a subset of any other node bag (see function _make_small)
        - semi-niceness is a term we have come up with, to denote the following relaxation of niceness (of a tree decomposition)
          A nice tree decomposition is one in which the following holds:
          . the tree is binary
          . bags of leaves only contain a single node
          . each node has one of the following 3 types:
            _ introduce: a new element is contained in the bag,
            _ forget: a previous element is removed,
            _ join: the node has exactly 2 children having the same node bag as the current node.
          For us, a semi-nice tree decomposition is simply a small tree decomposition (hence no node is a subset of any other node)
          in which for each edge one additional node is introduced representing the intersection of the respective node bags.
          Hence, "u - w" becomes "u - v - w" with v corresponding to the intersection of the node bags of u and w.
    '''

    def __init__(self, original_td, request):
        super(SmallSemiNiceTDArb, self).__init__("small_nice({})".format(original_td.name))
        self.original_td = original_td
        self.req = request
        self._initialize_from_original_td()
        self._make_small()
        self._make_semi_nice()

        self.out_neighbors = {}
        self.in_neighbor = {}
        self.root = None
        self.pre_order_traversal = None

        self.node_classification = {}

        self._create_arborescence()


    #essentially just make a copy
    def _initialize_from_original_td(self):
        for node, node_bag in self.original_td.node_bag_dict.iteritems():
            self.add_node(node, node_bag)
        for edge in self.original_td.edges:
            edge_as_list = list(edge)
            self.add_edge(edge_as_list[0], edge_as_list[1])
        # if not self.is_tree_decomposition():
        #     raise ValueError("Something got lost!")

    def _merge_nodes(self, node_to_remove, node_to_keep):

            neighbors_of_removed_node = set(self.get_neighbors(node_to_remove))

            if not node_to_keep in neighbors_of_removed_node:
                raise ValueError("Seems as if the node_to_keep is not connected to the node_to_remove.")
            neighbors_of_removed_node.remove(node_to_keep)

            self.remove_node(node_to_remove)
            for neighbor in neighbors_of_removed_node:
                self.add_edge(neighbor, node_to_keep)

    def _make_small(self):
        progress = True
        while progress:
            progress = False
            edges_to_check = list(self.edges)
            for edge in edges_to_check:
                edge_list = list(edge)
                tail = edge_list[0]
                head = edge_list[1]

                tail_bag = self.node_bag_dict[tail]
                head_bag = self.node_bag_dict[head]

                if tail_bag.issubset(head_bag):
                    self._merge_nodes(tail, head)
                    progress = True
                elif head_bag.issubset(tail_bag):
                    self._merge_nodes(head, tail)
                    progress = True
                if progress:
                    self.is_tree_decomposition(self.req)
                    break

    def _make_semi_nice(self):
        ''' Connected node bags are connected by a novel node representing the intersection of the node bags.

        :return:
        '''

        edges = list(self.edges)
        for edge in edges:
            edge_as_list = list(edge)
            tail = edge_as_list[0]
            head = edge_as_list[1]

            tail_bag = self.node_bag_dict[tail]
            head_bag = self.node_bag_dict[head]

            intersection = tail_bag.intersection(head_bag)

            if tail_bag == intersection or head_bag == intersection:
                raise ValueError("Tree decomposition was not nice in the first place!")

            internode = "bag_intersection_{}_{}".format(tail, head)

            self.add_node(internode, intersection)

            self.remove_edge(tail, head)
            self.add_edge(tail, internode)
            self.add_edge(internode, head)

    def _create_arborescence(self):
        #select root as bag having maximal bag size and maximal degree
        best_root, max_size, max_degree = None, -1, -1
        for node in self.nodes:
            node_bag_size = len(self.node_bag_dict[node])
            node_degree = len(self.incident_edges[node])

            if node_bag_size > max_size or (node_bag_size == max_size and node_degree > max_degree):
                max_size = node_bag_size
                max_degree = node_degree
                best_root = node

        self.root = best_root
        self.pre_order_traversal = []
        #root is set
        visited = set()
        queue = [self.root]
        while len(queue) > 0:
            current_node = queue.pop(0)
            visited.add(current_node)
            self.out_neighbors[current_node] = []
            for edge in self.incident_edges[current_node]:
                edge_as_list = list(edge)
                other = None
                if edge_as_list[0] == current_node:
                    other = edge_as_list[1]
                else:
                    other = edge_as_list[0]

                if other not in visited:
                    self.in_neighbor[other] = current_node
                    self.out_neighbors[current_node].append(other)
                    queue.append(other)

            self.pre_order_traversal.append(current_node)

        self.pre_order_traversal = list(self.pre_order_traversal)

        self.post_order_traversal = list(reversed(self.pre_order_traversal))

        for node in self.post_order_traversal:
            if node not in self.in_neighbor:
                self.in_neighbor[node] = None
                self.node_classification[node] = NodeType.Root
            elif len(self.out_neighbors[node]) == 0:
                self.node_classification[node] = NodeType.Leaf
            elif len(self.out_neighbors[node]) == 1:
                out_neighbor = self.out_neighbors[node][0]
                if self.node_bag_dict[out_neighbor].issubset(self.node_bag_dict[node]):
                    self.node_classification[node] = NodeType.Introduction
                elif self.node_bag_dict[node].issubset(self.node_bag_dict[out_neighbor]):
                    self.node_classification[node] = NodeType.Forget
                else:
                    raise ValueError("Don't know what's happening here!")
            else:
                self.node_classification[node] = NodeType.Join
                for out_neighbor in self.out_neighbors[node]:
                    if not self.node_bag_dict[out_neighbor].issubset(self.node_bag_dict[node]) or not self.node_classification[out_neighbor] == NodeType.Forget:
                        raise ValueError("Children of Join nodes must be forget nodes!")



class ValidMappingRestrictionComputer(object):

    ''' This class facilitates the computation of valid edge sets for the respective request edges. '''

    def __init__(self, substrate, request):
        self.substrate = substrate
        self.request = request

        self.allowed_nodes = {}
        self.allowed_edges = {}
        self.sorted_substrate_nodes = sorted(list(substrate.nodes))

        self.edge_set_to_edge_set_id = {}
        self.request_edge_to_edge_set_id = {}
        self.edge_set_id_to_request_edges = {}
        self.edge_set_id_to_edge_set = {}
        self.number_of_different_edge_sets = 0

    def compute(self):
        self._process_request_node_mapping_restrictions()
        self._process_request_edge_mapping_restrictions()
        self._process_edge_sets()

    def _process_request_node_mapping_restrictions(self):

        for reqnode in self.request.nodes:
            node_demand = self.request.get_node_demand(reqnode)

            allowed_nodes = self.request.get_allowed_nodes(reqnode)
            if allowed_nodes is None:
                allowed_nodes = list(self.substrate.get_nodes_by_type(self.request.get_type(reqnode)))
            allowed_nodes_copy = list(allowed_nodes)

            for allowed_node in allowed_nodes_copy:
                if self.substrate.get_node_capacity(allowed_node) < node_demand:
                    allowed_nodes.remove(allowed_node)
            self.allowed_nodes[reqnode] = sorted(allowed_nodes)

    def _process_request_edge_mapping_restrictions(self):
        for reqedge in self.request.edges:
            edge_demand = self.request.get_edge_demand(reqedge)
            allowed_edges = self.request.get_allowed_edges(reqedge)
            if allowed_edges is None:
                allowed_edges = set(self.substrate.get_edges())
            else:
                allowed_edges = set(allowed_edges)
            copy_of_edges = set(allowed_edges)

            for sedge in copy_of_edges:
                if self.substrate.get_edge_capacity(sedge) < edge_demand:
                    allowed_edges.remove(sedge)

            self.allowed_edges[reqedge] = allowed_edges

    def _process_edge_sets(self):

        for reqedge in self.request.edges:
            frozen_edge_set = frozenset(self.allowed_edges[reqedge])
            if frozen_edge_set in self.edge_set_to_edge_set_id:
                self.request_edge_to_edge_set_id[reqedge] = self.edge_set_to_edge_set_id[frozen_edge_set]
                self.edge_set_id_to_request_edges[self.request_edge_to_edge_set_id[reqedge]].append(reqedge)
            else:
                self.edge_set_id_to_edge_set[self.number_of_different_edge_sets] = frozen_edge_set
                self.edge_set_to_edge_set_id[frozen_edge_set] = self.number_of_different_edge_sets
                self.request_edge_to_edge_set_id[reqedge] = self.number_of_different_edge_sets
                self.edge_set_id_to_request_edges[self.number_of_different_edge_sets] = [reqedge]
                self.number_of_different_edge_sets += 1

    def get_allowed_snode_list(self, reqnode):
        return self.allowed_nodes[reqnode]

    def get_allowed_sedge_set(self, reqedge):
        return self.allowed_edges[reqedge]

    def get_edge_set_mapping(self):
        return self.edge_set_id_to_edge_set

    def get_reqedge_to_edgeset_id_mapping(self):
        return self.request_edge_to_edge_set_id

    def get_edgeset_id_to_reqedge_mapping(self):
        return self.edge_set_id_to_request_edges

    def get_number_of_different_edge_sets(self):
        return self.number_of_different_edge_sets


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

    #TODO make enum
    Approx_NoLatencies = 0
    Approx_Flex = 1
    Approx_Strict = 2
    Approx_Exact = 3
    Approx_Exact_MIP = 4

    @staticmethod
    def createSVPC(approx_type, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies=None, epsilon=None, limit=None):
        if approx_type == ShortestValidPathsComputer.Approx_Flex:
            return ShortestValidPathsComputer_Flex(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        elif approx_type == ShortestValidPathsComputer.Approx_Strict:
            return ShortestValidPathsComputer_Strict(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        elif approx_type == ShortestValidPathsComputer.Approx_Exact:
            return ShortestValidPathsComputer_Exact(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        elif approx_type == ShortestValidPathsComputer.Approx_Exact_MIP:
            return ShortestValidPathsComputer_Exact_MIP(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        elif approx_type == ShortestValidPathsComputer.Approx_NoLatencies:
            return ShortestValidPathsComputer_NoLatencies(substrate, None, valid_mapping_restriction_computer, edge_costs)
        else:
            raise RuntimeError("Latency approximation type not known!")

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
        #TODO: here a new copy logic was introduced,
        for sedge, cost in new_edge_costs.iteritems():
            self.edge_costs[sedge] = abs(cost)  #TODO: abs seems absolutely unncessary

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


# end



def relative_termination_callback(model, where):
    if where == GRB.Callback.MIP:
        best = model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if best < GRB.INFINITY:
            # We have a feasible solution
            if best <= bound * (1.0 + 1e-6 + model._epsilon):
                model.terminate()
                model._logger.debug("\nTermination MIP execution based on the following values:\n\tbest: {:12.4f}\n\t{:12.4f}\n\t{:12.4f}\n\n".format(best, bound, model._epsilon))

class LatencyConstrainedShortestPathsMIPComputer(modelcreator.AbstractModelCreator):

    ''' Gurobi model to construct and solve the multi-commodity flow formulation for the VNEP.

        Important: inheriting from the AbstractEmbeddingModelCreator, only the core functionality is enabled in this class.
    '''

    ALGORITHM_ID = "LatencyConstrainedShortestPathsMIPComputer"

    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit, gurobi_settings=None, logger=None, optimization_callback=relative_termination_callback):

        if epsilon is None:
            epsilon = 0.0
        if gurobi_settings is None:
            gurobi_settings = modelcreator.GurobiSettings(numericfocus=2)
        else:
            gurobi_settings.NumericFocus = 2

        super(LatencyConstrainedShortestPathsMIPComputer, self).__init__(gurobi_settings=gurobi_settings, logger=logger, optimization_callback=optimization_callback)

        self.temporal_log = modelcreator.TemporalLog_Disabled()
        self._disable_temporal_information_output = True
        self._disable_temporal_log_output = True
        self.substrate = substrate
        self.valid_mapping_restriction_computer = valid_mapping_restriction_computer
        self.edge_costs = edge_costs
        self.edge_latencies = edge_latencies
        self.epsilon = epsilon
        self.limit = limit

        self.edge_set_id_to_edge_set = self.valid_mapping_restriction_computer.get_edge_set_mapping()
        self.number_of_valid_edge_sets = self.valid_mapping_restriction_computer.get_number_of_different_edge_sets()
        self.valid_sedge_costs = {edge_set_id: {} for edge_set_id in xrange(self.number_of_valid_edge_sets)}
        self.valid_sedge_latencies = {edge_set_id: {} for edge_set_id in xrange(self.number_of_valid_edge_sets)}
        self.valid_sedge_paths = {edge_set_id: {} for edge_set_id in xrange(self.number_of_valid_edge_sets)}

        self.var_latency_bound = None
        self.var_y = {}
        self.var_z = {}
        self.time_lp = None

    def create_variables(self):
        # node mapping variable
        for snode in self.substrate.nodes:
            variable_name = modelcreator.construct_name("node_mapping",
                                                        snode=snode)

            self.var_y[snode] = self.model.addVar(lb=-1.0,
                                                  ub=1.0,
                                                  obj=0.0,
                                                  vtype=GRB.INTEGER,
                                                  name=variable_name)

        # flow variable
        for (u, v) in self.substrate.edges:
            variable_name = modelcreator.construct_name("flow",
                                                        sedge=(u, v))
            self.var_z[(u, v)] = self.model.addVar(lb=0.0,
                                                   ub=1.0,
                                                   obj=0.0,
                                                   vtype=GRB.BINARY,
                                                   name=variable_name)

        self.var_latency_bound = self.model.addVar(lb=self.limit, ub=self.limit, obj=0.0, vtype=GRB.CONTINUOUS, name="latency_bound")

        self.model.update()

    def create_constraints(self):
        self.create_constraints_flow_preservation_and_induction()
        self.create_constraint_only_one_outgoing_edge_and_one_incoming_edge()
        self.create_latency_bound_constraint()

    def create_constraints_flow_preservation_and_induction(self):
        for u in self.substrate.nodes:
            right_expr = LinExpr(self.var_y[u])

            left_outgoing = LinExpr([(1.0, self.var_z[sedge]) for sedge in self.substrate.out_edges[u]])
            left_incoming = LinExpr([(1.0, self.var_z[sedge]) for sedge in self.substrate.in_edges[u]])
            left_expr = LinExpr(left_outgoing - left_incoming)
            constr_name = modelcreator.construct_name("flow_preservation_and_induction",
                                                      snode=u)  # Matthias: changed to conform to standard naming
            self.model.addConstr(left_expr, GRB.EQUAL, right_expr, name=constr_name)

    def create_constraint_only_one_outgoing_edge_and_one_incoming_edge(self):
        for u in self.substrate.nodes:
            outgoing_flows = LinExpr([(1.0, self.var_z[sedge]) for sedge in self.substrate.out_edges[u]])
            constr_name = modelcreator.construct_name("maximal_one_outgoing_edge",
                                                      snode=u)  # Matthias: changed to conform to standard naming
            self.model.addConstr(outgoing_flows, GRB.LESS_EQUAL, 1.0, name=constr_name)

            outgoing_flows = LinExpr([(1.0, self.var_z[sedge]) for sedge in self.substrate.in_edges[u]])
            constr_name = modelcreator.construct_name("maximal_one_incoming_edge",
                                                      snode=u)  # Matthias: changed to conform to standard naming
            self.model.addConstr(outgoing_flows, GRB.LESS_EQUAL, 1.0, name=constr_name)


    def create_latency_bound_constraint(self):
        latency_of_edges = LinExpr([(self.edge_latencies[sedge], self.var_z[sedge]) for sedge in self.substrate.edges])
        constr_name = "latency_bound_constraint"
        self.model.addConstr(latency_of_edges, GRB.LESS_EQUAL, self.var_latency_bound, name=constr_name)

    def constrain_paths_to_allowed_edges(self, allowed_edges):
        for sedge in self.substrate.edges:
            if sedge in allowed_edges:
                self.var_z[sedge].ub = 1.0
            else:
                self.var_z[sedge].ub = 0.0
        self.model.update()

    def set_source_and_target_node(self, source, target):
        for snode in self.substrate.nodes:
            if snode == source:
                if snode == target:
                    self.var_y[snode].ub = 0.0
                    self.var_y[snode].lb = 0.0
                else:
                    self.var_y[snode].ub = 1.0
                    self.var_y[snode].lb = 1.0
            elif snode == target:
                self.var_y[snode].ub = -1.0
                self.var_y[snode].lb = -1.0
            else:
                self.var_y[snode].ub = 0.0
                self.var_y[snode].lb = 0.0
        self.model.update()

    def preprocess_input(self):
        pass

    def create_objective(self):
        objective_expr = LinExpr([(self.edge_costs[sedge], self.var_z[sedge]) for sedge in self.substrate.edges])
        self.model.setObjective(objective_expr, GRB.MINIMIZE)

    def compute_fractional_solution(self):
        raise NotImplementedError("Computing fraction paths makes no sense here.")

    def compute_all_pairs_shortest_paths(self):
        self.model._epsilon = self.epsilon
        self.model._logger = self.logger
        for edge_set_index in xrange(self.number_of_valid_edge_sets):
            self.constrain_paths_to_allowed_edges(self.edge_set_id_to_edge_set[edge_set_index])

            for snode in self.substrate.nodes:
                for o_snode in self.substrate.nodes:
                    self._snode = snode
                    self._o_snode = o_snode


                    self.set_source_and_target_node(snode, o_snode)
                    self.compute_integral_solution()

                    if self.status.isFeasible():
                        edge_path, cost_value, latency = self.solution
                    else:
                        edge_path = None
                        cost_value = np.nan
                        latency = np.nan

                    self.valid_sedge_costs[edge_set_index][(snode, o_snode)] = cost_value
                    self.valid_sedge_paths[edge_set_index][(snode, o_snode)] = edge_path
                    self.valid_sedge_latencies[edge_set_index][(snode, o_snode)] = latency


    def recover_integral_solution_from_variables(self):
        node_path = []
        edge_path = []
        cost_value = 0.0
        latency = 0.0

        start_snode = self._snode
        end_snode = self._o_snode
        if start_snode != end_snode:
            current_snode = end_snode
            node_path.append(current_snode)
            while current_snode != start_snode:
                found_predecessor = False
                for (pre_snode, snode) in self.substrate.in_edges[current_snode]:
                    if self.var_z[(pre_snode, snode)].X > 0.5:
                        node_path.append(pre_snode)
                        cost_value += self.edge_costs[(pre_snode, snode)]
                        latency += self.edge_latencies[(pre_snode, snode)]
                        current_snode = pre_snode
                        found_predecessor = True
                        break
                if not found_predecessor:
                    raise RuntimeError("Couldn't backtrack path!")
            pre_snode = node_path.pop()
            while len(node_path) > 0:
               post_snode = node_path.pop()
               edge_path.append((pre_snode,post_snode))
               pre_snode = post_snode

        return (edge_path, cost_value, latency)

    def post_process_integral_computation(self):
        return self.solution




class ShortestValidPathsComputer_Exact_MIP(ShortestValidPathsComputer):
    def __init__(self, substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit):
        super(ShortestValidPathsComputer_Exact_MIP, self).__init__(substrate, valid_mapping_restriction_computer,
                                                              edge_costs, edge_latencies, epsilon, limit)

        self.lcspmc = LatencyConstrainedShortestPathsMIPComputer(substrate, valid_mapping_restriction_computer, edge_costs, edge_latencies, epsilon, limit)
        self.lcspmc.init_model_creator()

    def compute(self):
        self.lcspmc.compute_all_pairs_shortest_paths()

    def recompute_with_new_costs(self, new_edge_costs):
        self.lcspmc.edge_costs = new_edge_costs
        self.lcspmc.create_objective()
        self.lcspmc.model.update()
        self.compute()

    """ Getter functions for retrieving calculation results """
    def get_valid_sedge_costs_for_reqedge(self, request_edge, mapping_edge):
        #TODO: refactor: mapping_edge is a little bit misleading here
        edge_set_index = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()[request_edge]
        return self.lcspmc.valid_sedge_costs[edge_set_index][mapping_edge]

    def get_valid_sedge_costs_from_edgesetindex(self, edge_set_index, reqedge):
        return self.valid_sedge_costs[edge_set_index][reqedge]

    def get_valid_sedge_path(self, request_edge, source_mapping, target_mapping):
        edge_set_index = self.valid_mapping_restriction_computer.get_reqedge_to_edgeset_id_mapping()[request_edge]
        return self.lcspmc.valid_sedge_paths[edge_set_index][(source_mapping, target_mapping)]
# end


class OptimizedDynVMPNode(object):

    ''' Represents a node in our Dyn-VNP algorithm. The prefix Optimized here stands for the fact that our
        implementation is significantly more complex (hence optimized) than that presented in the paper.
    '''

    def __init__(self, optdynvmp_parent, treenode):
        self.optdynvmp_parent = optdynvmp_parent
        self.substrate = optdynvmp_parent.substrate
        self.request = optdynvmp_parent.request
        self.ssntda = optdynvmp_parent.ssntda
        self.vmrc = self.optdynvmp_parent.vmrc
        self.svpc = self.optdynvmp_parent.svpc
        self.treenode = treenode
        self.nodetype = self.optdynvmp_parent.ssntda.node_classification[treenode]

    '''
    Helper functions for accessing indices of mappings effectively
    '''

    def get_indices_of_mappings_under_restrictions(self, mapping_restrictions):
        meta_information, construction_rule = self.get_construction_rule_for_indexing_mappings_under_restrictions(mapping_restrictions)

        return self.non_recursive_index_generation(meta_information,
                                                   construction_rule)




    def get_construction_rule_for_indexing_mappings_under_restrictions(self, mapping_restrictions):
        construction_rule = []
        meta_information = []

        for reqnode in self.contained_request_nodes:
            meta_information.append((reqnode, self.number_of_allowed_nodes[reqnode]))
            if reqnode in mapping_restrictions:
                construction_rule.append([self.index_of_allowed_node[reqnode][mapping_restrictions[reqnode]]])
            else:
                construction_rule.append(self.allowed_node_indexes[reqnode])

        return meta_information, construction_rule


    def non_recursive_index_generation(self,
                                       meta_information,
                                       construction_rule):

        for index, (reqnode, number_of_allowed_nodes) in enumerate(meta_information):
            self.maximum_node_indices[index] = len(construction_rule[index])
            self.number_allowed_nodes[index] = number_of_allowed_nodes
            if index == 0:
                self.current_parent_values[index] = construction_rule[index][0]
            else:
                self.current_parent_values[index] = self.current_parent_values[index-1] * self.number_allowed_nodes[index] + construction_rule[index][0]
            self.current_node_indices[index] = 0


        generation_depth = len(construction_rule)-1
        current_depth = generation_depth
        self.current_node_indices[current_depth] -= 1

        item_counter = 0

        while current_depth >= 0:

            self.current_node_indices[current_depth] += 1

            current_node_index = self.current_node_indices[current_depth]
            current_construction_rule = construction_rule[current_depth]
            if current_node_index == self.maximum_node_indices[current_depth]:
                self.current_node_indices[current_depth] = -1
                current_depth -= 1
            else:
                if current_depth == generation_depth:
                    if current_depth > 0:
                        self.list_for_indices[item_counter] = (self.current_parent_values[current_depth - 1] * \
                                                               self.number_allowed_nodes[current_depth] + \
                                                               current_construction_rule[
                                                                   current_node_index])
                        item_counter += 1
                    else:
                        self.list_for_indices[item_counter] =(current_construction_rule[current_node_index])
                        item_counter += 1
                else:
                    if current_depth > 0:
                        self.current_parent_values[current_depth] = self.current_parent_values[current_depth - 1] * \
                                                                    self.number_allowed_nodes[current_depth] + \
                                                               current_construction_rule[current_node_index]
                    else:
                        self.current_parent_values[current_depth] = current_construction_rule[current_node_index]
                    current_depth += 1
        return self.list_for_indices[:item_counter]

    def recursive_index_generation(self,
                                   meta_information,
                                   construction_rule,
                                   result_list,
                                   parent_value = 0,
                                   current_reqnode_index = 0):
        if current_reqnode_index > 0:
            parent_value *= meta_information[current_reqnode_index][1] #multiple number of allowed nodes
        for value in construction_rule[current_reqnode_index]:
            current_value = parent_value + value
            if current_reqnode_index < self.number_of_request_nodes-1:
                self.recursive_index_generation(meta_information,
                                                construction_rule,
                                                result_list,
                                                current_value,
                                                current_reqnode_index+1
                                                )
            else:
                result_list.append(current_value)

    def fill_matrix_with_mapping_indices(self,
                                         mapping_restrictions,
                                         result_array,
                                         result_array_row=0):

        meta_information, construction_rule = self.get_construction_rule_for_indexing_mappings_under_restrictions(mapping_restrictions)
        current_node_indices = np.full(len(construction_rule), 0, dtype=np.int32)
        maximum_node_indices = np.full(len(construction_rule), 0, dtype=np.int32)
        number_allowed_nodes = np.full(len(construction_rule), 0, dtype=np.int32)
        current_parent_values = np.full(len(construction_rule), 0, dtype=np.int32)
        for index, (reqnode, number_of_allowed_nodes) in enumerate(meta_information):
            maximum_node_indices[index] = len(construction_rule[index])
            number_allowed_nodes[index] = number_of_allowed_nodes
            if index == 0:
                current_parent_values[index] = construction_rule[index][0]
            else:
                current_parent_values[index] = current_parent_values[index-1] * number_allowed_nodes[index] + construction_rule[index][0]


        generation_depth = len(construction_rule)-1
        current_depth = generation_depth
        current_node_indices[current_depth] -= 1

        result_array_column = 0

        array_slice = result_array[result_array_row,:]

        while current_depth >= 0:

            current_node_indices[current_depth] += 1

            current_node_index = current_node_indices[current_depth]
            current_construction_rule = construction_rule[current_depth]
            if current_node_index == maximum_node_indices[current_depth]:
                current_node_indices[current_depth] = -1
                current_depth -= 1
            else:
                if current_depth == generation_depth:
                    if current_depth > 0:
                        array_slice[result_array_column] = current_parent_values[current_depth - 1] * \
                                                                              number_allowed_nodes[current_depth] + \
                                                                              current_construction_rule[
                                                                                  current_node_index]
                        result_array_column +=1
                    else:
                        array_slice[result_array_column] = current_construction_rule[current_node_index]
                        result_array_column += 1
                else:
                    if current_depth > 0:
                        current_parent_values[current_depth] = current_parent_values[current_depth - 1] * \
                                                               number_allowed_nodes[current_depth] + \
                                                               current_construction_rule[current_node_index]
                    else:
                        current_parent_values[current_depth] = current_construction_rule[current_node_index]
                    current_depth += 1


    def get_node_mapping_based_on_index(self, mapping_index):
        if mapping_index < 0 or mapping_index >= self.number_of_potential_node_mappings:
            raise ValueError("Given index {} is out of bounds [0,...,{}].".format(mapping_index,
                                                                                  self.number_of_potential_node_mappings))


        result_node_mapping = {}
        for reversed_reqnode_index in range(self.number_of_request_nodes):
            request_node = self._reversed_contained_request_nodes[reversed_reqnode_index]
            number_of_allowed_nodes_for_request_node =  self.number_of_allowed_nodes[request_node]
            snode_index = mapping_index % number_of_allowed_nodes_for_request_node
            snode = self.allowed_nodes[request_node][snode_index]
            result_node_mapping[request_node] = snode
            mapping_index /= number_of_allowed_nodes_for_request_node



        return result_node_mapping


    def initialize(self):
        self.contained_request_nodes = sorted(list(self.ssntda.node_bag_dict[self.treenode]))
        self.number_of_request_nodes = len(self.contained_request_nodes)

        self._reversed_contained_request_nodes = list(reversed(self.contained_request_nodes))
        self.index_of_req_node = {reqnode : self.contained_request_nodes.index(reqnode) for reqnode in self.contained_request_nodes}

        self.contained_request_edges = []
        for reqnode_source in self.contained_request_nodes:
            for reqnode_target in self.request.out_neighbors[reqnode_source]:
                if reqnode_target in self.contained_request_nodes:
                    self.contained_request_edges.append((reqnode_source, reqnode_target))

        self.allowed_nodes = {reqnode : self.vmrc.get_allowed_snode_list(reqnode) for reqnode in self.contained_request_nodes}



        self.list_of_ordered_allowed_nodes = [self.allowed_nodes[reqnode] for reqnode in self.contained_request_nodes]

        self.index_of_allowed_node = {reqnode :
                                          {snode : self.allowed_nodes[reqnode].index(snode)
                                           for snode in self.allowed_nodes[reqnode]}
                                      for reqnode in self.contained_request_nodes}

        self.number_of_allowed_nodes = {reqnode : len(self.allowed_nodes[reqnode]) for reqnode in self.contained_request_nodes}
        self.allowed_node_indexes = {reqnode: [x for x in range(self.number_of_allowed_nodes[reqnode])] for reqnode in self.contained_request_nodes}


        self.number_of_potential_node_mappings =  int(np.prod(self.number_of_allowed_nodes.values()))
        if self.number_of_potential_node_mappings < 1:
            raise ValueError("There exist no valid node mappings")

        minimal_number_different_node_mappings = min([self.number_of_allowed_nodes[reqnode] for reqnode in self.contained_request_nodes])
        maximal_size_of_index_array = self.number_of_potential_node_mappings / minimal_number_different_node_mappings

        self.list_for_indices = [0]*maximal_size_of_index_array

        self.current_node_indices = [0] * self.number_of_request_nodes
        self.maximum_node_indices = [0] * self.number_of_request_nodes
        self.number_allowed_nodes = [0] * self.number_of_request_nodes
        self.current_parent_values = [0] * self.number_of_request_nodes


        self.validity_array = np.full(self.number_of_potential_node_mappings, True, dtype=np.bool)
        self._initialize_validity_array()
        self.mapping_costs = np.full(self.number_of_potential_node_mappings, 0.0, dtype=np.float32)
        self.mapping_costs[~self.validity_array] = np.nan


    def _initialize_validity_array(self):
        if self.svpc.edge_mapping_invalidities:

            for (reqedge_source, reqedge_target) in self.contained_request_edges:

                for mapping_of_source, mapping_of_target in itertools.product(self.allowed_nodes[reqedge_source],
                                                                    self.allowed_nodes[reqedge_target]):
                    if np.isnan(self.svpc.get_valid_sedge_costs_for_reqedge((reqedge_source, reqedge_target), (mapping_of_source, mapping_of_target))):

                        list_of_indices = self.get_indices_of_mappings_under_restrictions(
                            {reqedge_source: mapping_of_source,
                             reqedge_target: mapping_of_target})

                        self.validity_array[list_of_indices] = False


    def initialize_corresponding_to_neighbors(self):
        #reversed arc orientation: the in-neighbor becomes the out-neighbor and the out-neighbors become the in-neighbors
        self.out_neighbor = None
        self.in_neighbors = []
        if self.ssntda.in_neighbor[self.treenode] is not None:
            self.out_neighbor = self.optdynvmp_parent.dynvmp_tree_nodes[self.ssntda.in_neighbor[self.treenode]]
        for neighbor in self.ssntda.out_neighbors[self.treenode]:
            self.in_neighbors.append(self.optdynvmp_parent.dynvmp_tree_nodes[neighbor])


        self.forgotten_virtual_elements_out_neigbor = [[],[]]

        if self.nodetype != NodeType.Root:
            for reqnode in self.contained_request_nodes:
                if reqnode not in self.out_neighbor.contained_request_nodes:
                    if self.out_neighbor.nodetype != NodeType.Forget:
                        raise ValueError("I can only forget something, if the out neighbor is a forget node!")
                    self.forgotten_virtual_elements_out_neigbor[0].append(reqnode)
            for reqedge in self.contained_request_edges:
                if reqedge not in self.out_neighbor.contained_request_edges:
                    if self.out_neighbor.nodetype != NodeType.Forget:
                        raise ValueError("I can only forget something, if the out neighbor is a forget node!")
                    self.forgotten_virtual_elements_out_neigbor[1].append(reqedge)
        else:
            self.forgotten_virtual_elements_out_neigbor[0] = self.contained_request_nodes
            self.forgotten_virtual_elements_out_neigbor[1] = self.contained_request_edges



    def initialize_local_cost_updates_and_selection_matrices(self):
        number_of_nodes = len(self.substrate.nodes)

        if self.nodetype != NodeType.Forget:

            #NODE COSTS
            self.local_node_cost_weights = np.full((number_of_nodes, self.number_of_potential_node_mappings), 0.0, dtype=np.float32)

            for reqnode in self.forgotten_virtual_elements_out_neigbor[0]:
                for snode in self.allowed_nodes[reqnode]:
                    list_of_indices = self.get_indices_of_mappings_under_restrictions({reqnode: snode})
                    self.local_node_cost_weights[
                        self.optdynvmp_parent.sorted_snode_index[snode],list_of_indices] += self.request.get_node_demand(reqnode)

            #EDGE COSTS

            edgeset_id_to_reqedges_mapping = self.vmrc.get_edgeset_id_to_reqedge_mapping()
            edgeset_id_to_forgotten_reqedges_mapping = {edge_set_id : [reqedge for reqedge in edgeset_id_to_reqedges_mapping[edge_set_id] if reqedge in self.forgotten_virtual_elements_out_neigbor[1]] for edge_set_id in edgeset_id_to_reqedges_mapping.keys()}

            self.local_edge_cost_length = {edge_set_id: int(len(edgeset_id_to_forgotten_reqedges_mapping[edge_set_id])) for edge_set_id
                                          in edgeset_id_to_reqedges_mapping.keys()}

            self.local_edge_cost_update_list = []

            for edge_set_id in edgeset_id_to_reqedges_mapping.keys():
                if self.local_edge_cost_length[edge_set_id] > 0:

                    local_edge_cost_indices_array = np.full(
                        (self.number_of_potential_node_mappings, self.local_edge_cost_length[edge_set_id]), 0,
                        dtype=np.int32)
                    local_edge_cost_weights_array = np.full(
                        ( self.number_of_potential_node_mappings, self.local_edge_cost_length[edge_set_id]), 0.0,
                        dtype=np.float32)

                    for reqedge_index, reqedge in enumerate(edgeset_id_to_forgotten_reqedges_mapping[edge_set_id]):
                        req_source, req_target = reqedge

                        for source_mapping, target_mapping in itertools.product(self.allowed_nodes[req_source], self.allowed_nodes[req_target]):
                            sedge_pair_index = self.optdynvmp_parent.sedge_pair_index[(source_mapping, target_mapping)]

                            list_of_indices = self.get_indices_of_mappings_under_restrictions({req_source: source_mapping,
                                                                                       req_target: target_mapping})

                            local_edge_cost_indices_array[list_of_indices, reqedge_index] = sedge_pair_index

                            local_edge_cost_weights_array[list_of_indices, reqedge_index] += self.request.get_edge_demand(
                                reqedge)

                    self.local_edge_cost_update_list.append((edge_set_id, local_edge_cost_indices_array, local_edge_cost_weights_array))

        if self.nodetype == NodeType.Forget:
            if len(self.in_neighbors) != 1:
                raise ValueError("Wasn't expecting more than a single neighbor!")
            in_neighbor = self.in_neighbors[0]
            out_neighbor = self.out_neighbor

            forgotten_nodes_from_in_neighbor = in_neighbor.forgotten_virtual_elements_out_neigbor[0]
            number_of_matching_in_neighbor_mappings = 1
            for forgotten_reqnode in forgotten_nodes_from_in_neighbor:
                number_of_matching_in_neighbor_mappings *= len(in_neighbor.allowed_nodes[forgotten_reqnode])

            #we need to handle this a bit differently as we do not represent what gets added from a forget node towards a introduction/join node
            number_of_matching_out_neighbor_mappings = 1
            for reqnode in out_neighbor.contained_request_nodes:
                if reqnode not in self.contained_request_nodes:
                    number_of_matching_out_neighbor_mappings *= len(out_neighbor.allowed_nodes[reqnode])

            self.minimum_selection_pull = np.full((self.number_of_potential_node_mappings, number_of_matching_in_neighbor_mappings), -1, dtype=np.int32)
            self.minimum_selection_push = np.full((self.number_of_potential_node_mappings, number_of_matching_out_neighbor_mappings), -1, dtype=np.int32)

            partial_fixed_mapping = {reqnode: None for reqnode in self.contained_request_nodes}

            for local_node_mapping_index, node_mapping in enumerate(itertools.product(*self.list_of_ordered_allowed_nodes)):
                for node_mapping_index, reqnode in enumerate(self.contained_request_nodes):
                    partial_fixed_mapping[reqnode] = node_mapping[node_mapping_index]

                # in_neighbor.fill_matrix_with_mapping_indices(partial_fixed_mapping,
                #                                              self.minimum_selection_pull,
                #                                              result_array_row=local_node_mapping_index)
                #
                # out_neighbor.fill_matrix_with_mapping_indices(partial_fixed_mapping,
                #                                               self.minimum_selection_push,
                #                                               result_array_row=local_node_mapping_index)

                list_of_indices = in_neighbor.get_indices_of_mappings_under_restrictions(partial_fixed_mapping)
                self.minimum_selection_pull[local_node_mapping_index, :] = list_of_indices

                list_of_indices = out_neighbor.get_indices_of_mappings_under_restrictions(partial_fixed_mapping)
                self.minimum_selection_push[local_node_mapping_index,:] = list_of_indices

    def _apply_local_costs_updates(self):
        if self.nodetype == NodeType.Forget:
            raise ValueError("This function should never be called for a Forget Node!")

        self.mapping_costs += np.dot(self.optdynvmp_parent.node_costs_array, self.local_node_cost_weights)
        for (edge_set_id, local_edge_cost_update_indices_array, local_edge_cost_update_weights_array) in self.local_edge_cost_update_list:
            self.mapping_costs += np.einsum('ij,ij->i',self.optdynvmp_parent.edge_costs_arrays[edge_set_id][local_edge_cost_update_indices_array], local_edge_cost_update_weights_array)

    def _apply_cost_propagation_forget_node(self):
        if self.nodetype != NodeType.Forget:
            raise ValueError("This function can only be called for forget nodes!")
        if len(self.in_neighbors) != 1:
            raise ValueError("Wasn't expecting more than a single neighbor!")

        in_neighbor = self.in_neighbors[0]
        out_neighbor = self.out_neighbor

        self.mapping_costs = np.nanmin(in_neighbor.mapping_costs[self.minimum_selection_pull], axis=1)
        #out_neighbor.mapping_costs[self.minimum_selection_push] += self.mapping_costs
        for local_node_mapping_index in xrange(int(self.number_of_potential_node_mappings)):
        #     self.mapping_costs[local_node_mapping_index] = np.nanmin(in_neighbor.mapping_costs[self.minimum_selection_pull[local_node_mapping_index,:]])
             out_neighbor.mapping_costs[self.minimum_selection_push[local_node_mapping_index,:]] += self.mapping_costs[local_node_mapping_index]

    def compute_costs_based_on_children(self):
        if self.nodetype == NodeType.Leaf:
            #given that all costs were initialized to 0 / nan, we just need to add costs that will be forgotten when stepping up
            self._apply_local_costs_updates()
        elif self.nodetype == NodeType.Introduction or self.nodetype == NodeType.Join or self.nodetype == NodeType.Root:
            self._apply_local_costs_updates()
        elif self.nodetype == NodeType.Forget:
            self._apply_cost_propagation_forget_node()
        else:
            raise ValueError("Don't know whats going on here!")

    def reinitialize(self):
        self.mapping_costs[self.validity_array] = 0.0


approx_str_to_type = {
    'no latencies': ShortestValidPathsComputer.Approx_NoLatencies,
    'flex': ShortestValidPathsComputer.Approx_Flex,
    'strict': ShortestValidPathsComputer.Approx_Strict,
    'exact': ShortestValidPathsComputer.Approx_Exact
}

class OptimizedDynVMP(object):

    ''' The actual algorithm to compute optimal valid mappings using dynamic programming. Nearly all algorithmic challenging
        tasks are to be found in the implementation of the OptimizedDynVMPNode class.
    '''

    def __init__(self, substrate, request, ssntda, initial_snode_costs=None, initial_sedge_costs=None, epsilon=1, limit=1, lat_approx_type='no latencies'):
        self.substrate = substrate
        self.request = request
        if not isinstance(ssntda, SmallSemiNiceTDArb):
            raise ValueError("The Optimized DYNVMP Algorithm expects a small-semi-nice-tree-decomposition-arborescence")
        self.ssntda = ssntda

        self._initialize_costs(initial_snode_costs, initial_sedge_costs)

        self.vmrc = ValidMappingRestrictionComputer(substrate=substrate, request=request)

        try:
            # rescale limit value to adapt to different substrate sizes
            limit *= substrate.get_average_node_distance()
        except:
            pass

        self.svpc = ShortestValidPathsComputer.createSVPC(approx_str_to_type[lat_approx_type],
                                                          substrate,
                                                          self.vmrc,
                                                          self.sedge_costs,
                                                          self.sedge_latencies,
                                                          epsilon,
                                                          limit)

        self.lat_approx_type = lat_approx_type


    def _initialize_costs(self, snode_costs, sedge_costs):
        if snode_costs is None:
            self.snode_costs = {snode: 1.0 for snode in self.substrate.nodes}
        else:
            self.snode_costs = {snode: snode_costs[snode] for snode in snode_costs.keys()}

        if sedge_costs is None:
            self.sedge_costs = {sedge: 1.0 for sedge in self.substrate.edges}
        else:
            self.sedge_costs = {sedge: sedge_costs[sedge] for sedge in sedge_costs.keys()}

        self.sedge_latencies = {sedge: self.substrate.edge[sedge].get("latency", 1) for sedge in self.substrate.edges}

        self._max_demand = max([self.request.get_node_demand(reqnode) for reqnode in self.request.nodes] +
                               [self.request.get_edge_demand(reqedge) for reqedge in self.request.edges])

        self._max_cost = max(
            [cost for cost in self.snode_costs.values()] + [cost for cost in self.sedge_costs.values()])

        self._mapping_cost_bound = self._max_demand * self._max_cost * 2.0 * len(self.request.edges) * len(self.substrate.edges)



    def initialize_data_structures(self):
        self.vmrc.compute()

        print ("finding shortest paths for {} using {}".format(self.request.name, self.lat_approx_type))
        self.svpc.compute()


        self.sorted_snodes = sorted(list(self.substrate.nodes))
        self.sorted_snode_index = {snode : self.sorted_snodes.index(snode) for snode in self.sorted_snodes}

        self.number_of_nodes = len(self.sorted_snodes)
        self.snode_index = {snode : self.sorted_snodes.index(snode) for snode in self.substrate.nodes}
        self.sedge_pair_index = {(snode_1, snode_2) : self.snode_index[snode_1] * self.number_of_nodes + self.snode_index[snode_2]   #the plus one is important as the 0-th entry always holds 0
                                 for (snode_1,snode_2) in itertools.product(self.sorted_snodes, self.sorted_snodes)}

        self.node_costs_array = np.empty(len(self.sorted_snodes), dtype=np.float32)
        self.edge_costs_arrays = [np.empty(self.number_of_nodes * self.number_of_nodes, dtype=np.float32) for x in range(self.vmrc.get_number_of_different_edge_sets())]
        self._initialize_cost_arrays()

        self.dynvmp_tree_nodes = {}
        for t in self.ssntda.post_order_traversal:
            self.dynvmp_tree_nodes[t] = OptimizedDynVMPNode(self, t)
            self.dynvmp_tree_nodes[t].initialize()

        for t in self.ssntda.post_order_traversal:
            self.dynvmp_tree_nodes[t].initialize_corresponding_to_neighbors()
            self.dynvmp_tree_nodes[t].initialize_local_cost_updates_and_selection_matrices()

    def _initialize_cost_arrays(self):
        self.node_costs_array.fill(0.0)
        for node_index, snode in enumerate(self.sorted_snodes):
            self.node_costs_array[node_index] = self.snode_costs[snode]

        for edge_set_index in range(self.vmrc.get_number_of_different_edge_sets()):
            current_edge_costs_array = self.edge_costs_arrays[edge_set_index]
            current_edge_costs_array.fill(0.0)
            for node_pair_index, (snode_1, snode_2) in enumerate(
                    itertools.product(self.sorted_snodes, self.sorted_snodes)):
                current_edge_costs_array[node_pair_index] = self.svpc.\
                    get_valid_sedge_costs_from_edgesetindex(edge_set_index, (snode_1, snode_2))

    def compute_solution(self):
        for t in self.ssntda.post_order_traversal:
            self.dynvmp_tree_nodes[t].compute_costs_based_on_children()

    def get_optimal_solution_cost(self):
        minimal_cost = None
        dynvmp_root_node = self.dynvmp_tree_nodes[self.ssntda.root]
        try:
            minimal_cost = np.nanmin(dynvmp_root_node.mapping_costs)
        except:
            #if there was no non-nan value..
            pass
        if np.isnan(minimal_cost):
            minimal_cost = None

        return minimal_cost

    def get_ordered_root_solution_costs_and_mapping_indices(self, maximum_number_of_solutions_to_return=None):
        dynvmp_root_node = self.dynvmp_tree_nodes[self.ssntda.root]
        non_nan_indices = np.where(~np.isnan(dynvmp_root_node.mapping_costs))[0]
        if np.shape(non_nan_indices)[0] > 0:
            if maximum_number_of_solutions_to_return is None:
                maximum_number_of_solutions_to_return = np.shape(non_nan_indices)[0]
            non_nan_slice = dynvmp_root_node.mapping_costs[non_nan_indices]
            order_of_indices = np.argsort(non_nan_slice)

            aranged_costs = non_nan_slice[order_of_indices]
            aranged_original_indices = non_nan_indices[order_of_indices]

            if maximum_number_of_solutions_to_return is None or np.shape(non_nan_indices)[0] < maximum_number_of_solutions_to_return:
                maximum_number_of_solutions_to_return = np.shape(non_nan_indices)[0]

            return (aranged_costs[:maximum_number_of_solutions_to_return], aranged_original_indices[:maximum_number_of_solutions_to_return])
        else:
            return (None, None)

    def recover_mapping(self, root_mapping_index=None):
        root_cost, reqnode_mappings = self._recover_node_mapping(root_mapping_index=root_mapping_index)
        if root_cost is None:
            return None, None

        reqedge_mappings = {}
        for reqedge in self.request.edges:
            reqedge_mappings[reqedge] = self._reconstruct_edge_mapping(reqedge, reqnode_mappings[reqedge[0]], reqnode_mappings[reqedge[1]])

        result_mapping = solutions.Mapping("dynvmp_mapping_{}".format(self.request.name),
                                    self.request,
                                    self.substrate,
                                    True)

        for reqnode, mapping_thereof in reqnode_mappings.iteritems():
            result_mapping.map_node(reqnode, mapping_thereof)
        for reqedge, mapping_thereof in reqedge_mappings.iteritems():
            result_mapping.map_edge(reqedge, mapping_thereof)

        return root_cost, result_mapping

    def recover_list_of_mappings(self, list_of_root_mapping_indices):
        result = []
        for mapping_index in list_of_root_mapping_indices:
            result.append(self.recover_mapping(root_mapping_index=mapping_index)[1])
        return result


    def _reconstruct_edge_mapping(self, reqedge, source_mapping, target_mapping):
        return self.svpc.get_valid_sedge_path(reqedge, source_mapping, target_mapping)


    def _recover_node_mapping(self, root_mapping_index = None):
        fixed_node_mappings = {}
        root_cost = None

        for treenode in self.ssntda.pre_order_traversal:

            # find mapping of least cost
            dynvmp_treenode = self.dynvmp_tree_nodes[treenode]
            best_mapping_index = None
            try:
                if not fixed_node_mappings:
                    if root_mapping_index is None:
                        best_mapping_index = np.nanargmin(dynvmp_treenode.mapping_costs)
                        root_cost = dynvmp_treenode.mapping_costs[best_mapping_index]
                    else:
                        best_mapping_index = root_mapping_index
                        root_cost = dynvmp_treenode.mapping_costs[best_mapping_index]

                else:
                    number_of_matching_mappings = dynvmp_treenode.number_of_potential_node_mappings
                    for reqnode in dynvmp_treenode.contained_request_nodes:
                        if reqnode in fixed_node_mappings:
                            number_of_matching_mappings /= dynvmp_treenode.number_of_allowed_nodes[reqnode]

                    list_of_indices = dynvmp_treenode.get_indices_of_mappings_under_restrictions(fixed_node_mappings)



                    best_mapping_index_relative_to_slice = np.nanargmin(dynvmp_treenode.mapping_costs[list_of_indices])
                    best_mapping_index = dynvmp_treenode.list_for_indices[best_mapping_index_relative_to_slice]

            except ValueError:
                # no mapping could be found
                return None, None

            corresponding_node_mapping = dynvmp_treenode.get_node_mapping_based_on_index(best_mapping_index)

            for request_node, substrate_node in corresponding_node_mapping.iteritems():
                if request_node in fixed_node_mappings:
                    if fixed_node_mappings[request_node] != substrate_node:
                        raise ValueError(
                            "Extracted local (optimal) mapping does not aggree with the previously set node mappings:\n\n"
                            "\tfixed_mappings: {}\n"
                            "\tcurrent mapping: {}".format(fixed_node_mappings,
                                                           corresponding_node_mapping))
                else:
                    fixed_node_mappings[request_node] = substrate_node

        return root_cost, fixed_node_mappings

    def reinitialize(self, new_node_costs, new_edge_costs):

        self._initialize_costs(new_node_costs, new_edge_costs)
        self.svpc.recompute_with_new_costs(new_edge_costs)
        self._initialize_cost_arrays()

        for t in self.ssntda.post_order_traversal:
            self.dynvmp_tree_nodes[t].reinitialize()


""" Computing tree decompositions """


class TreeDecompositionComputation(object):
    """
    Use the exact tree decomposition algorithm implementation by Hisao Tamaki and Hiromu Ohtsuka, obtained
    from https://github.com/TCS-Meiji/PACE2017-TrackA, to compute tree decompositions.

    It assumes that the path to the tree decomposition algorithm is stored in the environment variable PACE_TD_ALGORITHM_PATH
    """

    def __init__(self, graph, logger=None, timeout=None):
        self.graph = graph
        self.timeout = timeout
        if logger is None:
            self.logger = util.get_logger(__name__, make_file=False, propagate=True)
        else:
            self.logger = logger

        self.map_nodes_to_numeric_id = None
        self.map_numeric_id_to_nodes = None
        self.DEBUG_MODE = False




    def compute_tree_decomposition(self):
        td_alg_input = self._convert_graph_to_td_input_format()
        result = None
        # There is probably a better way...
        curr_dir = os.getcwd()
        PACE_TD_ALGORITHM_PATH = os.getenv("PACE_TD_ALGORITHM_PATH")
        if PACE_TD_ALGORITHM_PATH is None:
            raise ValueError("PACE_TD_ALGORITHM_PATH environment variable is not set!")
        os.chdir(PACE_TD_ALGORITHM_PATH)
        try:
            p = subprocess.Popen("./tw-exact", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdoutdata, stderrdata = None, None
            try:
                stdoutdata, stderrdata = p.communicate(input=td_alg_input, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.logger.info("Timeout expired when trying to compute tree decomposition. Killing process and discarding potential result.")
                p.kill()
                p.communicate()
            if not stderrdata:
                result = self._convert_result_format_to_tree_decomposition(stdoutdata)
        except subprocess.CalledProcessError as e:
            print "Subprocess Error:"
            print e
            print "Return code {}".format(e.returncode)
        finally:
            os.chdir(curr_dir)
        return result

    def _convert_graph_to_td_input_format(self):
        self.map_nodes_to_numeric_id = {node: str(idx) for (idx, node) in enumerate(sorted(self.graph.nodes), 1)}
        self.map_numeric_id_to_nodes = {v: k for (k, v) in self.map_nodes_to_numeric_id.items()}
        lines = ["p tw {} {}".format(len(self.graph.nodes), len(self.graph.edges))]
        for edge in sorted(self.graph.edges):
            i, j = edge
            lines.append("{} {}".format(
                self.map_nodes_to_numeric_id[i],
                self.map_nodes_to_numeric_id[j],
            ))
        if self.DEBUG_MODE: #TODO make output file unique
            with open("pace_request.txt", "w") as f:
                f.write("\n".join(lines))
                f.write("{}".format(self.map_nodes_to_numeric_id))
                f.write("{}".format(self.map_numeric_id_to_nodes))
        return "\n".join(lines)

    def _convert_result_format_to_tree_decomposition(self, computation_stdout):
        lines = computation_stdout.split("\n")
        td = TreeDecomposition("{}_TD".format(self.graph.name))
        # TODO: Do we need to use the header line for something?
        for line in lines[1:]:
            line = [w.strip() for w in line.split() if w]
            if not line or line[0] == "c":  # ignore empty and comment lines
                continue
            elif line[0] == "b":
                bag_id = self._get_bagid(line[1])
                bag = frozenset([
                    self.map_numeric_id_to_nodes[i] for i in line[2:]
                ])
                td.add_node(bag_id, node_bag=bag)
            else:
                assert len(line) == 2
                i, j = line
                td.add_edge(
                    self._get_bagid(i),
                    self._get_bagid(j),
                )
        return td

    def _get_bagid(self, numeric_id):
        return "bag_{}".format(numeric_id)


def compute_tree_decomposition(request, logger=None, timeout=None):
    return TreeDecompositionComputation(request, logger=logger, timeout=timeout).compute_tree_decomposition()

### Now comes the Separation LP ###

# Copy pasted...
Param_MIPGap = "MIPGap"
Param_IterationLimit = "IterationLimit"
Param_NodeLimit = "NodeLimit"
Param_Heuristics = "Heuristics"
Param_Threads = "Threads"
Param_TimeLimit = "TimeLimit"
Param_MIPFocus = "MIPFocus"
Param_RootCutPasses = "CutPasses"
Param_Cuts = "Cuts"
Param_NodefileStart = "NodefileStart"
Param_NodeMethod = "NodeMethod"
Param_Method = "Method"
Param_BarConvTol = "BarConvTol"
Param_NumericFocus = "NumericFocus"


INFINITY = float("inf")


class SeparationLPSolution(modelcreator.AlgorithmResult):
    def __init__(self,
                 time_preprocessing,
                 time_optimization,
                 time_postprocessing,
                 tree_decomp_runtimes,
                 dynvmp_init_runtimes,
                 dynvmp_computation_runtimes,
                 gurobi_runtimes,
                 status,
                 profit,
                 number_of_generated_mappings
                 ):
        super(SeparationLPSolution, self).__init__()
        self.time_preprocessing = time_preprocessing
        self.time_optimization = time_optimization
        self.time_postprocessing = time_postprocessing
        self.tree_decomp_runtimes = tree_decomp_runtimes
        self.dynvmp_init_runtimes = dynvmp_init_runtimes
        self.dynvmp_computation_runtimes = dynvmp_computation_runtimes
        self.gurobi_runtimes = gurobi_runtimes
        self.status = status
        self.profit = profit
        self.number_of_generated_mappings = number_of_generated_mappings

    def get_solution(self):
        return self

    def cleanup_references(self, original_scenario):
        pass

    def __str__(self):
        output_string = ""

        output_string += "         time_preprocessing: {}\n".format(self.time_preprocessing)
        output_string += "          time_optimization: {}\n".format(self.time_optimization)
        output_string += "        time_postprocessing: {}\n".format(self.time_postprocessing)
        output_string += "       tree_decomp_runtimes: {}\n".format(self.tree_decomp_runtimes)
        output_string += "       dynvmp_init_runtimes: {}\n".format(self.dynvmp_init_runtimes)
        output_string += "dynvmp_computation_runtimes: {}\n".format(self.dynvmp_computation_runtimes)
        output_string += "            gurobi_runtimes: {}\n".format(self.gurobi_runtimes)
        output_string += "                     status: {}\n".format(self.status)
        output_string += "                     profit: {}\n".format(self.profit)

        return output_string


class SeparationLP_OptDynVMP(object):
    ALGORITHM_ID = "SeparationLPDynVMP"

    '''
        Allows the computation of LP solutions using OptDynVMP as a separation oracle.
        Currently, this is only implemented for the Max Profit Objective, but the min cost objective should be easy to
        handle as well.
        
    '''

    def __init__(self,
                 scenario,
                 gurobi_settings=None,
                 logger=None):
        self.scenario = scenario
        self.substrate = self.scenario.substrate
        self.requests = self.scenario.requests
        self.objective = self.scenario.objective

        if self.objective == datamodel.Objective.MAX_PROFIT:
            pass
        elif self.objective == datamodel.Objective.MIN_COST:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")
        else:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")

        self.gurobi_settings = gurobi_settings

        self.model = None  # the model of gurobi
        self.status = None  # GurobiStatus instance
        self.solution = None  # either a integral solution or a fractional one

        self.temporal_log = modelcreator.TemporalLog()

        self.time_preprocess = None
        self.time_optimization = None
        self.time_postprocessing = None
        self._time_postprocess_start = None

        if logger is None:
            self.logger = util.get_logger(__name__, make_file=False, propagate=True)
        else:
            self.logger = logger

    def init_model_creator(self):
        ''' Initializes the modelcreator by generating the model. Afterwards, model.compute() can be called to let
            Gurobi solve the model.

        :return:
        '''

        time_preprocess_start = time.time()

        self.model = gurobipy.Model("column_generation_is_smooth")  #name doesn't matter...
        self.model._mc = self

        self.model.setParam("LogToConsole", 1)

        if self.gurobi_settings is not None:
            self.apply_gurobi_settings(self.gurobi_settings)

        self.preprocess_input()
        self.create_empty_capacity_constraints()
        self.create_empty_request_embedding_bound_constraints()
        self.create_empty_objective()

        self.model.update()

        self.time_preprocess = time.time() - time_preprocess_start

    def preprocess_input(self):

        if len(self.substrate.get_types()) > 1:
            raise ValueError("Can only handle a single node type.")

        self.node_type = list(self.substrate.get_types())[0]
        self.snodes = self.substrate.nodes
        self.sedges = self.substrate.edges

        self.node_capacities = {snode : self.substrate.get_node_type_capacity(snode, self.node_type) for snode in self.snodes}
        self.edge_capacities = {sedge : self.substrate.get_edge_capacity(sedge) for sedge in self.sedges}

        self.dual_costs_requests  = {req : 0 for req in self.requests}
        self.dual_costs_node_resources = {snode: 1 for snode in self.snodes}
        self.dual_costs_edge_resources = {sedge: 1 for sedge in self.sedges}

        self.tree_decomp_computation_times = {req : 0 for req in self.requests}
        self.tree_decomps = {req : None for req in self.requests}
        for req in self.requests:
            self.logger.debug("Computing tree decomposition for request {}".format(req))
            td_computation_time = time.time()
            td_comp = TreeDecompositionComputation(req)
            tree_decomp = td_comp.compute_tree_decomposition()
            sntd = SmallSemiNiceTDArb(tree_decomp, req)
            self.tree_decomps[req] = sntd
            self.tree_decomp_computation_times[req] = time.time() - td_computation_time
            self.logger.debug("\tdone.".format(req))

        self.dynvmp_instances = {req : None for req in self.requests}
        self.dynvmp_runtimes_initialization = {req: list() for req in self.requests}

        for req in self.requests:
            self.logger.debug("Initializing DynVMP Instance for request {}".format(req))
            dynvmp_init_time = time.time()
            opt_dynvmp = OptimizedDynVMP(self.substrate,
                                         req,
                                         self.tree_decomps[req],
                                         initial_snode_costs=self.dual_costs_node_resources,
                                         initial_sedge_costs=self.dual_costs_edge_resources)
            opt_dynvmp.initialize_data_structures()
            self.dynvmp_runtimes_initialization[req].append(time.time()-dynvmp_init_time)
            self.dynvmp_instances[req] = opt_dynvmp
            self.logger.debug("\tdone.".format(req))

        self.mappings_of_requests = {req: list() for req in self.requests}
        self.dynvmp_runtimes_computation = {req: list() for req in self.requests}
        self.gurobi_runtimes = []

        self.mapping_variables = {req: list() for req in self.requests}


    def create_empty_capacity_constraints(self):
        self.capacity_constraints = {}
        for snode in self.snodes:
            self.capacity_constraints[snode] = self.model.addConstr(0, GRB.LESS_EQUAL, self.node_capacities[snode], name="capacity_node_{}".format(snode))
        for sedge in self.sedges:
            self.capacity_constraints[sedge] = self.model.addConstr(0, GRB.LESS_EQUAL, self.edge_capacities[sedge], name="capacity_edge_{}".format(sedge))

    def create_empty_request_embedding_bound_constraints(self):
        self.embedding_bound = {req : None for req in self.requests}
        for req in self.requests:
            self.embedding_bound[req] = self.model.addConstr(0, GRB.LESS_EQUAL, 1)

    def create_empty_objective(self):
        self.model.setObjective(0, GRB.MAXIMIZE)



    def introduce_new_columns(self, req, maximum_number_of_columns_to_introduce=None, cutoff=INFINITY):
        dynvmp_instance = self.dynvmp_instances[req]
        opt_cost = dynvmp_instance.get_optimal_solution_cost()
        current_new_allocations = []
        current_new_variables = []
        self.logger.debug("Objective when introducing new columns was {}".format(opt_cost))
        (costs, indices) = dynvmp_instance.get_ordered_root_solution_costs_and_mapping_indices(maximum_number_of_solutions_to_return=maximum_number_of_columns_to_introduce)
        mapping_list = dynvmp_instance.recover_list_of_mappings(indices)
        self.logger.debug("Will iterate mapping list {}".format(req.name))
        for index, mapping in enumerate(mapping_list):
            if costs[index] > cutoff:
                break
            #store mapping
            varname = "f_req[{}]_k[{}]".format(req.name, index+len(self.mappings_of_requests[req]))
            new_var = self.model.addVar(lb=0.0,
                                        ub=1.0,
                                        obj=req.profit,
                                        vtype=GRB.CONTINUOUS,
                                        name=varname)
            current_new_variables.append(new_var)

            #compute corresponding substrate allocation and store it
            mapping_allocations = self._compute_allocations(req, mapping)
            current_new_allocations.append(mapping_allocations)

        #make variables accessible
        self.model.update()
        for index, new_var in enumerate(current_new_variables):
            #handle allocations
            corresponding_allocation = current_new_allocations[index]
            for sres, alloc in corresponding_allocation.iteritems():
                constr = self.capacity_constraints[sres]
                self.model.chgCoeff(constr, new_var, alloc)

            self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)

        self.mappings_of_requests[req].extend(mapping_list[:len(current_new_variables)])
        self.mapping_variables[req].extend(current_new_variables)
        self.logger.debug("Introduced {} new mappings for {}".format(len(current_new_variables), req.name))


    def _compute_allocations(self, req, mapping):
        allocations = {}
        for reqnode in req.nodes:
            snode = mapping.mapping_nodes[reqnode]
            if snode in allocations:
                allocations[snode] += req.node[reqnode]['demand']
            else:
                allocations[snode] = req.node[reqnode]['demand']
        for reqedge in req.edges:
            path = mapping.mapping_edges[reqedge]
            for sedge in path:
                stail, shead = sedge
                if sedge in allocations:
                    allocations[sedge] += req.edge[reqedge]['demand']
                else:
                    allocations[sedge] = req.edge[reqedge]['demand']
        return allocations

    def perform_separation_and_introduce_new_columns(self):
        new_columns_generated = False

        for req in self.requests:
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None and opt_cost < 0.995*(req.profit - self.dual_costs_requests[req]):
                self.introduce_new_columns(req, maximum_number_of_columns_to_introduce=5, cutoff = req.profit-self.dual_costs_requests[req])
                new_columns_generated = True

        return new_columns_generated

    def update_dual_costs_and_reinit_dynvmps(self):
        #update dual costs
        for snode in self.snodes:
            self.dual_costs_node_resources[snode] = self.capacity_constraints[snode].Pi
        for sedge in self.sedges:
            self.dual_costs_edge_resources[sedge] = self.capacity_constraints[sedge].Pi
        for req in self.requests:
            self.dual_costs_requests[req] = self.embedding_bound[req].Pi

        #reinit dynvmps
        for req in self.requests:
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_reinit_time = time.time()
            dynvmp_instance.reinitialize(new_node_costs=self.dual_costs_node_resources,
                                         new_edge_costs=self.dual_costs_edge_resources)
            self.dynvmp_runtimes_initialization[req].append(time.time() - single_dynvmp_reinit_time)

    def compute_integral_solution(self):
        #the name sucks, but we sadly need it for the framework
        return self.compute_solution()


    def compute_solution(self):
        ''' Abstract function computing an integral solution to the model (generated before).

        :return: Result of the optimization consisting of an instance of the GurobiStatus together with a result
                 detailing the solution computed by Gurobi.
        '''
        self.logger.info("Starting computing solution")
        # do the optimization
        time_optimization_start = time.time()

        #do the magic here

        for req in self.requests:
            self.logger.debug("Getting first mappings for request {}".format(req.name))
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                self.logger.debug("Introducing new columns for {}".format(req.name))
                self.introduce_new_columns(req, maximum_number_of_columns_to_introduce=100)

        self.model.update()

        new_columns_generated = True
        counter = 0
        last_obj = -1
        current_obj = 0
        #the abortion criterion here is not perfect and should probably depend on the relative error instead of the
        #absolute one.
        while new_columns_generated and abs(current_obj-last_obj) > 0.1:
            gurobi_runtime = time.time()
            self.model.optimize()
            last_obj = current_obj
            current_obj = self.model.getAttr("ObjVal")
            self.gurobi_runtimes.append(time.time() - gurobi_runtime)
            self.update_dual_costs_and_reinit_dynvmps()

            new_columns_generated = self.perform_separation_and_introduce_new_columns()

            counter += 1

        self.time_optimization = time.time() - time_optimization_start

        # do the postprocessing
        self._time_postprocess_start = time.time()
        self.status = self.model.getAttr("Status")
        objVal = None
        objBound = GRB.INFINITY
        objGap = GRB.INFINITY
        solutionCount = self.model.getAttr("SolCount")

        if solutionCount > 0:
            objVal = self.model.getAttr("ObjVal")

        self.status = modelcreator.GurobiStatus(status=self.status,
                                                solCount=solutionCount,
                                                objValue=objVal,
                                                objGap=objGap,
                                                objBound=objBound,
                                                integralSolution=False)

        anonymous_decomp_runtimes = []
        anonymous_init_runtimes = []
        anonymous_computation_runtimes = []
        for req in (self.requests):
            anonymous_decomp_runtimes.append(self.tree_decomp_computation_times[req])
            anonymous_init_runtimes.append(self.dynvmp_runtimes_initialization[req])
            anonymous_computation_runtimes.append(self.dynvmp_runtimes_computation[req])

        number_of_generated_mappings = sum([len(self.mapping_variables[req]) for req in self.requests])

        self.result = SeparationLPSolution(self.time_preprocess,
                                           self.time_optimization,
                                           0,
                                           anonymous_decomp_runtimes,
                                           anonymous_init_runtimes,
                                           anonymous_computation_runtimes,
                                           self.gurobi_runtimes,
                                           self.status,
                                           objVal,
                                           number_of_generated_mappings)

        return self.result

    def recover_solution_from_variables(self):
        pass


    ###
    ###     GUROBI SETTINGS
    ###     The following is copy-pasted from the basic modelcreator in the alib, as the separation approach
    ###     breaks the structure of computing a simple LP or IP.
    ###

    _listOfUserVariableParameters = [Param_MIPGap, Param_IterationLimit, Param_NodeLimit, Param_Heuristics,
                                     Param_Threads, Param_TimeLimit, Param_Cuts, Param_MIPFocus,
                                     Param_RootCutPasses,
                                     Param_NodefileStart, Param_Method, Param_NodeMethod, Param_BarConvTol,
                                     Param_NumericFocus]

    def apply_gurobi_settings(self, gurobiSettings):
        ''' Apply gurobi settings.

        :param gurobiSettings:
        :return:
        '''


        if gurobiSettings.MIPGap is not None:
            self.set_gurobi_parameter(Param_MIPGap, gurobiSettings.MIPGap)
        else:
            self.reset_gurobi_parameter(Param_MIPGap)

        if gurobiSettings.IterationLimit is not None:
            self.set_gurobi_parameter(Param_IterationLimit, gurobiSettings.IterationLimit)
        else:
            self.reset_gurobi_parameter(Param_IterationLimit)

        if gurobiSettings.NodeLimit is not None:
            self.set_gurobi_parameter(Param_NodeLimit, gurobiSettings.NodeLimit)
        else:
            self.reset_gurobi_parameter(Param_NodeLimit)

        if gurobiSettings.Heuristics is not None:
            self.set_gurobi_parameter(Param_Heuristics, gurobiSettings.Heuristics)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.Threads is not None:
            self.set_gurobi_parameter(Param_Threads, gurobiSettings.Threads)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.TimeLimit is not None:
            self.set_gurobi_parameter(Param_TimeLimit, gurobiSettings.TimeLimit)
        else:
            self.reset_gurobi_parameter(Param_TimeLimit)

        if gurobiSettings.MIPFocus is not None:
            self.set_gurobi_parameter(Param_MIPFocus, gurobiSettings.MIPFocus)
        else:
            self.reset_gurobi_parameter(Param_MIPFocus)

        if gurobiSettings.cuts is not None:
            self.set_gurobi_parameter(Param_Cuts, gurobiSettings.cuts)
        else:
            self.reset_gurobi_parameter(Param_Cuts)

        if gurobiSettings.rootCutPasses is not None:
            self.set_gurobi_parameter(Param_RootCutPasses, gurobiSettings.rootCutPasses)
        else:
            self.reset_gurobi_parameter(Param_RootCutPasses)

        if gurobiSettings.NodefileStart is not None:
            self.set_gurobi_parameter(Param_NodefileStart, gurobiSettings.NodefileStart)
        else:
            self.reset_gurobi_parameter(Param_NodefileStart)

        if gurobiSettings.Method is not None:
            self.set_gurobi_parameter(Param_Method, gurobiSettings.Method)
        else:
            self.reset_gurobi_parameter(Param_Method)

        if gurobiSettings.NodeMethod is not None:
            self.set_gurobi_parameter(Param_NodeMethod, gurobiSettings.NodeMethod)
        else:
            self.reset_gurobi_parameter(Param_NodeMethod)

        if gurobiSettings.BarConvTol is not None:
            self.set_gurobi_parameter(Param_BarConvTol, gurobiSettings.BarConvTol)
        else:
            self.reset_gurobi_parameter(Param_BarConvTol)

        if gurobiSettings.NumericFocus is not None:
            self.set_gurobi_parameter(Param_NumericFocus, gurobiSettings.NumericFocus)
        else:
            self.reset_gurobi_parameter(Param_NumericFocus)

    def reset_all_parameters_to_default(self):
        for param in self._listOfUserVariableParameters:
            (name, type, curr, min, max, default) = self.model.getParamInfo(param)
            self.model.setParam(param, default)

    def reset_gurobi_parameter(self, param):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Parameter {} unchanged".format(param))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        self.model.setParam(param, default)

    def set_gurobi_parameter(self, param, value):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Changed value of parameter {} to {}".format(param, value))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.setParam(param, value)

    def getParam(self, param):
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.getParam(param)


class RoundingOrder(enum.Enum):
    RANDOM = "RAND"
    STATIC_REQ_PROFIT = "STATIC_REQ_PROFIT"
    ACHIEVED_REQ_PROFIT = "ACHIEVED_REQ_PROFIT"


class LPRecomputationMode(enum.Enum):
    NONE = "NONE"
    RECOMPUTATION_WITHOUT_SEPARATION =     "RECOMPUTATION_WITHOUT_SEPARATION"
    RECOMPUTATION_WITH_SINGLE_SEPARATION = "RECOMPUTATION_WITH_SINGLE_SEPARATION"


RandomizedRoundingSolution = namedtuple("RandomizedRoundingSolution", ["solution",
                                                                       "profit",
                                                                       "max_node_load",
                                                                       "max_edge_load",
                                                                       "time_to_round_solution"])




class RandRoundSepLPOptDynVMPCollectionResult(modelcreator.AlgorithmResult):

    def __init__(self, scenario, lp_computation_information):
        self.scenario = scenario
        self.lp_computation_information = lp_computation_information
        self.solutions = {}

    def add_solution(self, rounding_order, lp_recomputation_mode, solution):
        identifier = (lp_recomputation_mode, rounding_order)
        if identifier not in self.solutions.keys():
            self.solutions[identifier] = []
        solution_index = len(self.solutions[identifier])
        self.solutions[identifier].append(solution)


    def get_solution(self):
        return self.solutions

    def _check_scenarios_are_equal(self, original_scenario):
        #check can only work if a single solution is returned; we incorporate this in the following function: _cleanup_references_raw
        pass


    def _cleanup_references_raw(self, original_scenario):
        for identifier in self.solutions.keys():
            #search for best solution and remove mapping information of all other solutions
            list_of_solutions = self.solutions[identifier]
            best_solution = max(list_of_solutions, key= lambda x: x.profit)
            new_list_of_solutions = []

            for solution in list_of_solutions:
                if solution == best_solution:
                    new_list_of_solutions.append(self._actual_cleanup_of_references_raw(original_scenario, solution))
                else:
                    new_list_of_solutions.append(RandomizedRoundingSolution(solution=None,
                                                                            profit=solution.profit,
                                                                            max_node_load=solution.max_node_load,
                                                                            max_edge_load=solution.max_edge_load,
                                                                            time_to_round_solution=solution.time_to_round_solution))

            self.solutions[identifier] = new_list_of_solutions


        #lastly: adapt the collection's scenario
        self.scenario = original_scenario

    def _actual_cleanup_of_references_raw(self, original_scenario, rr_solution):
        for own_req, original_request in zip(self.scenario.requests, original_scenario.requests):
            assert own_req.nodes == original_request.nodes
            assert own_req.edges == original_request.edges
            assert own_req.name == original_request.name

            if own_req in rr_solution.solution.request_mapping:
                mapping = rr_solution.solution.request_mapping[own_req]

                del rr_solution.solution.request_mapping[own_req]
                if mapping is not None:
                    mapping.request = original_request
                    mapping.substrate = original_scenario.substrate
                rr_solution.solution.request_mapping[original_request] = mapping

        rr_solution.solution.scenario = original_scenario
        return rr_solution



    def _get_solution_overview(self):
        result = "\n\t{:^10s} | {:^40s} {:^20s} | {:^8s}\n".format("PROFIT", "LP Recomputation Mode", "Rounding Order", "INDEX")
        for identifier in self.solutions.keys():
            for solution_index, solution in enumerate(self.solutions[identifier]):
                result += "\t" + "{:^10.5f} | {:^40s} {:^20s} | {:<8d}\n".format(solution.profit,
                                                                                 identifier[0].value,
                                                                                 identifier[1].value,
                                                                                 solution_index)
        return result


class RandRoundSepLPOptDynVMPCollection(object):
    ALGORITHM_ID = "RandRoundSepLPOptDynVMPCollection"

    '''
        Allows the computation of LP solutions using OptDynVMP as a separation oracle.
        Currently, this is only implemented for the Max Profit Objective, but the min cost objective should be easy to
        handle as well.
    '''

    def __init__(self,
                 scenario,
                 rounding_order_list,
                 lp_recomputation_mode_list,
                 lp_relative_quality,
                 rounding_samples_per_lp_recomputation_mode,
                 number_initial_mappings_to_compute,
                 number_further_mappings_to_add,
                 latency_approximation_factor=1,
                 latency_approximation_limit=1,
                 latency_approximation_type=None,
                 gurobi_settings=None,
                 logger=None,
                 ):
        self.scenario = scenario
        self.substrate = self.scenario.substrate
        self.requests = self.scenario.requests
        self.objective = self.scenario.objective

        self.latency_approximation_factor = latency_approximation_factor
        self.latency_approximation_limit = latency_approximation_limit # * total_latencies
        self.latency_approximation_type = latency_approximation_type if latency_approximation_type is not None else 'no latencies'

        self.rounding_order_list = []
        self.lp_recomputation_list = []
        self.lp_relative_quality = lp_relative_quality
        self.rounding_samples_per_lp_recomputation_mode = dict()
        for lp_recomputation_str, number_of_samples in rounding_samples_per_lp_recomputation_mode:
            self.rounding_samples_per_lp_recomputation_mode[LPRecomputationMode(lp_recomputation_str)] = number_of_samples
        self.number_initial_mappings_to_compute = number_initial_mappings_to_compute
        self.number_further_mappings_to_add = number_further_mappings_to_add

        for rounding_order in rounding_order_list:
            if isinstance(rounding_order, str):
                self.rounding_order_list.append(RoundingOrder(rounding_order))
            elif isinstance(rounding_order, RoundingOrder):
                self.rounding_order_list.append(rounding_order)
            else:
                raise ValueError("Cannot handle this rounding order.")

        for lp_recomputation_mode in lp_recomputation_mode_list:
            if isinstance(lp_recomputation_mode, str):
                self.lp_recomputation_list.append(LPRecomputationMode(lp_recomputation_mode))
            elif isinstance(lp_recomputation_mode, LPRecomputationMode):
                self.lp_recomputation_list.append(lp_recomputation_mode)
            else:
                raise ValueError("Cannot handle this LP recomputation mode.")

        if self.objective == datamodel.Objective.MAX_PROFIT:
            pass
        elif self.objective == datamodel.Objective.MIN_COST:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")
        else:
            raise ValueError("The separation LP algorithm can at the moment just handle max-profit instances.")

        self.gurobi_settings = gurobi_settings

        self.model = None  # the model of gurobi
        self.status = None  # GurobiStatus instance
        self.solution = None  # either a integral solution or a fractional one

        self.temporal_log = modelcreator.TemporalLog()

        self.time_preprocess = None
        self.time_optimization = None
        self.time_postprocessing = None
        self._time_postprocess_start = None

        if logger is None:
            self.logger = util.get_logger(__name__, make_file=False, propagate=True)
        else:
            self.logger = logger

    def init_model_creator(self):
        ''' Initializes the modelcreator by generating the model. Afterwards, model.compute() can be called to let
            Gurobi solve the model.

        :return:
        '''

        time_preprocess_start = time.time()

        self.model = gurobipy.Model("column_generation_is_smooth")  #name doesn't matter...
        self.model._mc = self

        self.model.setParam("LogToConsole", 1)

        if self.gurobi_settings is not None:
            self.apply_gurobi_settings(self.gurobi_settings)

        self.preprocess_input()
        self.create_empty_capacity_constraints()
        self.create_empty_request_embedding_bound_constraints()
        self.create_empty_objective()

        self.model.update()

        self.warmstart_lp_basis = {"V": list(), "C": list()}

        self.time_preprocess = time.time() - time_preprocess_start

    def preprocess_input(self):

        if len(self.substrate.get_types()) > 1:
            raise ValueError("Can only handle a single node type.")

        self.node_type = list(self.substrate.get_types())[0]
        self.snodes = self.substrate.nodes
        self.sedges = self.substrate.edges

        self.node_capacities = {snode : self.substrate.get_node_type_capacity(snode, self.node_type) for snode in self.snodes}
        self.edge_capacities = {sedge : self.substrate.get_edge_capacity(sedge) for sedge in self.sedges}

        self.dual_costs_requests  = {req : 0 for req in self.requests}
        self.dual_costs_node_resources = {snode: 1 for snode in self.snodes}
        self.dual_costs_edge_resources = {sedge: 1 for sedge in self.sedges}

        self.tree_decomp_computation_times = {req : 0 for req in self.requests}
        self.tree_decomps = {req : None for req in self.requests}
        for req in self.requests:
            self.logger.debug("Computing tree decomposition for request {}".format(req))
            td_computation_time = time.time()
            td_comp = TreeDecompositionComputation(req)
            tree_decomp = td_comp.compute_tree_decomposition()
            sntd = SmallSemiNiceTDArb(tree_decomp, req)
            self.tree_decomps[req] = sntd
            self.tree_decomp_computation_times[req] = time.time() - td_computation_time
            self.logger.debug("\tdone.".format(req))

        self.dynvmp_instances = {req : None for req in self.requests}
        self.dynvmp_runtimes_initialization = {req: list() for req in self.requests}


        # TODO calculate shortest paths for all reqests at once
        for req in self.requests:
            self.logger.debug("Initializing DynVMP Instance for request {}".format(req))
            dynvmp_init_time = time.time()
            opt_dynvmp = OptimizedDynVMP(self.substrate,
                                         req,
                                         self.tree_decomps[req],
                                         initial_snode_costs=self.dual_costs_node_resources,
                                         initial_sedge_costs=self.dual_costs_edge_resources,
                                         # edge_latencies=self.edge_latencies, handled in DynVP
                                         epsilon=self.latency_approximation_factor,
                                         limit=self.latency_approximation_limit,
                                         lat_approx_type=self.latency_approximation_type
                                         )
            opt_dynvmp.initialize_data_structures()
            self.dynvmp_runtimes_initialization[req].append(time.time()-dynvmp_init_time)
            self.dynvmp_instances[req] = opt_dynvmp
            self.logger.debug("\tdone.".format(req))

        self.mappings_of_requests = {req: list() for req in self.requests}
        self.allocations_of_mappings = {req: list() for req in self.requests}
        self.dynvmp_runtimes_computation = {req: list() for req in self.requests}
        self.gurobi_runtimes = []

        self.mapping_variables = {req: list() for req in self.requests}

        self.substrate_resources = list(self.scenario.substrate.edges)
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
                self.substrate_resources.append((ntype, snode))
        self.adapted_variables = list()


    def create_empty_capacity_constraints(self):
        self.capacity_constraints = {}
        for snode in self.snodes:
            self.capacity_constraints[snode] = self.model.addConstr(0, GRB.LESS_EQUAL, self.node_capacities[snode], name="capacity_node_{}".format(snode))
        for sedge in self.sedges:
            self.capacity_constraints[sedge] = self.model.addConstr(0, GRB.LESS_EQUAL, self.edge_capacities[sedge], name="capacity_edge_{}".format(sedge))

    def create_empty_request_embedding_bound_constraints(self):
        self.embedding_bound = {req : None for req in self.requests}
        for req in self.requests:
            self.embedding_bound[req] = self.model.addConstr(0, GRB.LESS_EQUAL, 1)

    def create_empty_objective(self):
        self.model.setObjective(0, GRB.MAXIMIZE)



    def introduce_new_columns(self, req, maximum_number_of_columns_to_introduce=None, cutoff=INFINITY):
        dynvmp_instance = self.dynvmp_instances[req]
        opt_cost = dynvmp_instance.get_optimal_solution_cost()
        current_new_allocations = []
        current_new_variables = []
        self.logger.debug("Objective when introucding new columns was {}".format(opt_cost))
        (costs, indices) = dynvmp_instance.get_ordered_root_solution_costs_and_mapping_indices(maximum_number_of_solutions_to_return=maximum_number_of_columns_to_introduce)
        mapping_list = dynvmp_instance.recover_list_of_mappings(indices)
        self.logger.debug("Will iterate mapping list {}".format(req.name))
        for index, mapping in enumerate(mapping_list):
            if costs[index] > cutoff:
                break
            #store mapping
            varname = "f_req[{}]_k[{}]".format(req.name, index+len(self.mappings_of_requests[req]))
            new_var = self.model.addVar(lb=0.0,
                                        ub=1.0,
                                        obj=req.profit,
                                        vtype=GRB.CONTINUOUS,
                                        name=varname)
            current_new_variables.append(new_var)

            #compute corresponding substrate allocation and store it
            mapping_allocations = self._compute_allocations(req, mapping)
            current_new_allocations.append(mapping_allocations)

        #make variables accessible
        self.model.update()
        for index, new_var in enumerate(current_new_variables):
            #handle allocations
            corresponding_allocation = current_new_allocations[index]
            for sres, alloc in corresponding_allocation.iteritems():
                if sres not in self.capacity_constraints.keys():
                    continue
                constr = self.capacity_constraints[sres]
                self.model.chgCoeff(constr, new_var, alloc)

            self.model.chgCoeff(self.embedding_bound[req], new_var, 1.0)

        self.mappings_of_requests[req].extend(mapping_list[:len(current_new_variables)])
        self.allocations_of_mappings[req].extend(current_new_allocations[:len(current_new_variables)])
        self.mapping_variables[req].extend(current_new_variables)
        self.logger.debug("Introduced {} new mappings for {}".format(len(current_new_variables), req.name))


    def _compute_allocations(self, req, mapping):
        allocations = {}
        for reqnode in req.nodes:
            snode = mapping.mapping_nodes[reqnode]
            if snode in allocations:
                allocations[snode] += req.node[reqnode]['demand']
            else:
                allocations[snode] = req.node[reqnode]['demand']
            reqnode_type = req.get_type(reqnode)
            res_id = (reqnode_type, snode)
            if res_id in allocations:
                allocations[res_id] += req.node[reqnode]['demand']
            else:
                allocations[res_id] = req.node[reqnode]['demand']
        for reqedge in req.edges:
            path = mapping.mapping_edges[reqedge]
            for sedge in path:
                stail, shead = sedge
                if sedge in allocations:
                    allocations[sedge] += req.edge[reqedge]['demand']
                else:
                    allocations[sedge] = req.edge[reqedge]['demand']
        return allocations

    def perform_separation_and_introduce_new_columns(self, current_objective, ignore_requests=[]):
        new_columns_generated = False

        total_dual_violations = 0

        for req in self.requests:
            if req in ignore_requests:
                continue
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                dual_violation_of_req = (opt_cost - req.profit + self.dual_costs_requests[req])
                if dual_violation_of_req < 0:
                    total_dual_violations += (-dual_violation_of_req)

        self._current_obj_bound = current_objective + total_dual_violations

        self._current_solution_quality = None
        if abs(current_objective) > 0.00001:
            self._current_solution_quality = total_dual_violations / current_objective
        else:
            self.logger.info("WARNING: Current objective is very close to zero; treating it as such.")
            self._current_solution_quality = total_dual_violations

        self.logger.info("\nCurrent LP solution value is {:10.5f}\n"
                           "Current dual upper bound is  {:10.5f}\n"
                         "Accordingly, current solution is at least {}-optimal".format(current_objective, current_objective+total_dual_violations, 1+self._current_solution_quality))

        if self._current_solution_quality < self.lp_relative_quality:
            self.logger.info("Ending LP computation as found solution is {}-near optimal, which lies below threshold of {}".format(self._current_solution_quality, self.lp_relative_quality))
            return False
        else:

            for req in self.requests:
                if req in ignore_requests:
                    continue
                dynvmp_instance = self.dynvmp_instances[req]
                opt_cost = dynvmp_instance.get_optimal_solution_cost()
                if opt_cost is not None and opt_cost < 0.999*(req.profit - self.dual_costs_requests[req]):
                    self.introduce_new_columns(req,
                                               maximum_number_of_columns_to_introduce=self.number_further_mappings_to_add,
                                               cutoff = req.profit-self.dual_costs_requests[req])
                    new_columns_generated = True

            return new_columns_generated

    def update_dual_costs_and_reinit_dynvmps(self):
        #update dual costs
        for snode in self.snodes:
            self.dual_costs_node_resources[snode] = self.capacity_constraints[snode].Pi
        for sedge in self.sedges:
            self.dual_costs_edge_resources[sedge] = self.capacity_constraints[sedge].Pi
        for req in self.requests:
            self.dual_costs_requests[req] = self.embedding_bound[req].Pi

        #reinit dynvmps
        for req in self.requests:
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_reinit_time = time.time()
            dynvmp_instance.reinitialize(new_node_costs=self.dual_costs_node_resources,
                                         new_edge_costs=self.dual_costs_edge_resources)
            self.dynvmp_runtimes_initialization[req].append(time.time() - single_dynvmp_reinit_time)

    def compute_integral_solution(self):
        #the name sucks, but we sadly need it for the framework
        return self.compute_solution()



    def _save_warmstart_lp_basis(self):
        if len(self.warmstart_lp_basis["V"]) > 0 or len(self.warmstart_lp_basis["C"]) > 0:
            self.logger.info("ERROR: Actually, the warmstart basis should only be set once!")
            self.warmstart_lp_basis["V"] = list
            self.warmstart_lp_basis["C"] = list
        for var in self.model.getVars():
            self.warmstart_lp_basis["V"].append((var, var.VBasis))
        for constr in self.model.getConstrs():
            self.warmstart_lp_basis["C"].append((constr, constr.CBasis))
        self.logger.info(
            "Successfully stored warmstart LP Basis.")

    def _load_warmstart_lp_basis(self):
        for var, value in self.warmstart_lp_basis["V"]:
            var.VBasis = value
        for constr, value in self.warmstart_lp_basis["C"]:
            constr.CBasis = value
        self.logger.info("Successfully loaded warmstart LP Basis (whether this basis is used in the following is indicated by the log of gurobi).")


    def compute_solution(self):
        ''' Abstract function computing an integral solution to the model (generated before).

        :return: Result of the optimization consisting of an instance of the GurobiStatus together with a result
                 detailing the solution computed by Gurobi.
        '''
        self.logger.info("Starting computing solution")
        # do the optimization
        time_optimization_start = time.time()

        #do the magic here

        for req in self.requests:
            self.logger.debug("Getting first mappings for request {}".format(req.name))
            # execute algorithm
            dynvmp_instance = self.dynvmp_instances[req]
            single_dynvmp_runtime = time.time()
            dynvmp_instance.compute_solution()
            self.dynvmp_runtimes_computation[req].append(time.time() - single_dynvmp_runtime)
            opt_cost = dynvmp_instance.get_optimal_solution_cost()
            if opt_cost is not None:
                self.logger.debug("Introducing new columns for {}".format(req.name))
                self.introduce_new_columns(req,
                                           maximum_number_of_columns_to_introduce=self.number_initial_mappings_to_compute)

        self.model.update()

        new_columns_generated = True
        counter = 0
        last_obj = -1
        current_obj = 0
        #the abortion criterion here is not perfect and should probably depend on the relative error instead of the
        #absolute one.
        while new_columns_generated:
            gurobi_runtime = time.time()
            self.model.optimize()
            last_obj = current_obj
            current_obj = self.model.getAttr("ObjVal")
            self.gurobi_runtimes.append(time.time() - gurobi_runtime)
            self.update_dual_costs_and_reinit_dynvmps()

            new_columns_generated = self.perform_separation_and_introduce_new_columns(current_objective=current_obj)

            counter += 1


        self.time_optimization = time.time() - time_optimization_start

        # do the postprocessing
        self._time_postprocess_start = time.time()
        self.status = self.model.getAttr("Status")
        objVal = None
        objBound = self._current_obj_bound
        objGap = self._current_solution_quality
        solutionCount = self.model.getAttr("SolCount")

        if solutionCount > 0:
            objVal = self.model.getAttr("ObjVal")

        self.status = modelcreator.GurobiStatus(status=self.status,
                                                solCount=solutionCount,
                                                objValue=objVal,
                                                objGap=objGap,
                                                objBound=objBound,
                                                integralSolution=False)



        self._save_warmstart_lp_basis()

        anonymous_decomp_runtimes = []
        anonymous_init_runtimes = []
        anonymous_computation_runtimes = []
        for req in (self.requests):
            anonymous_decomp_runtimes.append(self.tree_decomp_computation_times[req])
            anonymous_init_runtimes.append(self.dynvmp_runtimes_initialization[req])
            anonymous_computation_runtimes.append(self.dynvmp_runtimes_computation[req])

        number_of_generated_mappings = sum([len(self.mapping_variables[req]) for req in self.requests])


        sep_lp_solution = SeparationLPSolution(self.time_preprocess,
                                               self.time_optimization,
                                               0,
                                               anonymous_decomp_runtimes,
                                               anonymous_init_runtimes,
                                               anonymous_computation_runtimes,
                                               self.gurobi_runtimes,
                                               self.status,
                                               objVal,
                                               number_of_generated_mappings)

        self.result = RandRoundSepLPOptDynVMPCollectionResult(scenario=self.scenario,
                                                              lp_computation_information=sep_lp_solution)

        self.perform_rounding()
        #self.logger.info("Best solution by rounding is {}".format(best_solution))

        return self.result




    def remove_impossible_mappings_and_reoptimize(self, currently_fixed_allocations, fixed_requests, lp_computation_mode):
        #self.logger.debug("Removing mappings that would violate capacities given the current state..")
        for req in self.requests:
            if req in fixed_requests:
                continue
            for mapping_index, mapping in enumerate(self.mappings_of_requests[req]):
                if not self.check_whether_mapping_would_obey_resource_violations(currently_fixed_allocations, self.allocations_of_mappings[req][mapping_index]):
                    self.mapping_variables[req][mapping_index].ub = 0.0
                    self.adapted_variables.append(self.mapping_variables[req][mapping_index])


        #self.logger.debug("Re-compute LP after removal of impossible columns")

        self.model.update()
        self.model.optimize()
        current_obj = self.model.getAttr("ObjVal")

        if lp_computation_mode == LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION:
            new_columns_generated = True
            while new_columns_generated:
                self.update_dual_costs_and_reinit_dynvmps()
                new_columns_generated = self.perform_separation_and_introduce_new_columns(current_objective=current_obj,
                                                                                          ignore_requests=fixed_requests)
                if new_columns_generated:
                    #self.logger.debug("Separation yielded new mappings. Reoptimizing!")
                    self.model.update()
                    self.model.optimize()
                    current_obj = self.model.getAttr("ObjVal")
                break

            #self.logger.debug("Removing columns which will not be feasible")
            for req in self.requests:
                if req in fixed_requests:
                    continue
                for mapping_index, mapping in enumerate(self.mappings_of_requests[req]):
                    if not self.check_whether_mapping_would_obey_resource_violations(currently_fixed_allocations, self.allocations_of_mappings[req][mapping_index]):
                        self.mapping_variables[req][mapping_index].ub = 0.0
                        self.adapted_variables.append(self.mapping_variables[req][mapping_index])


            #self.logger.debug("Re-compute after removal of stupid mappings...")

            self.model.update()
            self.model.optimize()



    def perform_rounding(self):

        actual_lp_computation_modes = []
        if LPRecomputationMode.NONE in self.lp_recomputation_list:
            actual_lp_computation_modes.append(LPRecomputationMode.NONE)
        if LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION in self.lp_recomputation_list:
            actual_lp_computation_modes.append(LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION)
        if LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION in self.lp_recomputation_list:
            actual_lp_computation_modes.append(LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION)

        for lp_computation_mode in actual_lp_computation_modes:
            for rounding_order in self.rounding_order_list:
                result_list = self.round_solution_without_violations(lp_computation_mode, rounding_order)
                for solution in result_list:
                    self.result.add_solution(rounding_order, lp_computation_mode, solution=solution)
        self.logger.debug(self.result._get_solution_overview())

    def round_solution_without_violations(self, lp_computation_mode, rounding_order):

        A = {}

        result_list = []

        self.logger.debug("Executing rounding according to settings: {} {}".format(lp_computation_mode.value, rounding_order.value))

        for q in xrange(self.rounding_samples_per_lp_recomputation_mode[lp_computation_mode]):
            #recompute optimal solution
            self.model.optimize()

            time_rr0 = time.time()

            solution, profit, max_node_load, max_edge_load = self.rounding_iteration_violations_without_violations(A, lp_computation_mode, rounding_order)

            if lp_computation_mode != LPRecomputationMode.NONE:
                self.model.reset()
                self._load_warmstart_lp_basis()
                self.model.optimize()

            time_rr = time.time() - time_rr0

            if not solution.validate_solution():
                self.logger.info("ERROR: scenario solution did not validate!")
            if not solution.validate_solution_fulfills_capacity():
                self.logger.info("ERROR: scenario solution did not satisfy capacity constraints!")

            solution_tuple = RandomizedRoundingSolution(solution=solution,
                                                        profit=profit,
                                                        max_node_load=max_node_load,
                                                        max_edge_load=max_edge_load,
                                                        time_to_round_solution=time_rr)

            result_list.append(solution_tuple)

        return result_list

    def _initialize_allocations_dict(self, A):
        sub = self.scenario.substrate
        for snode in sub.nodes:
            for ntype in sub.node[snode]["capacity"]:
                A[(ntype, snode)] = 0.0
        for u, v in sub.edges:
            A[(u, v)] = 0.0

    def rounding_iteration_violations_without_violations(self, A, lp_computation_mode, rounding_order):
        processed_reqs = set()
        unprocessed_reqs = set(self.scenario.requests)
        B = 0.0
        self._initialize_allocations_dict(A)

        req_list = list(self.scenario.requests)

        if rounding_order == RoundingOrder.RANDOM:
            random.shuffle(req_list)
        elif rounding_order == RoundingOrder.STATIC_REQ_PROFIT:
            req_list = list(sorted(self.scenario.requests, key=lambda r: r.profit, reverse=True))
        elif rounding_order == RoundingOrder.ACHIEVED_REQ_PROFIT:
            cum_embedding_value_of_req = {req : sum(var.X for var in self.mapping_variables[req]) for req in self.scenario.requests}
            req_list = list(sorted(self.scenario.requests, key=lambda r: r.profit * cum_embedding_value_of_req[r], reverse=True))

        scenario_solution = solutions.IntegralScenarioSolution(name="", scenario=self.scenario)

        for i in range(len(req_list)):
            if rounding_order == RoundingOrder.ACHIEVED_REQ_PROFIT and lp_computation_mode != LPRecomputationMode.NONE:
                cum_embedding_value_of_req = {req: sum(var.X for var in self.mapping_variables[req]) for req in
                                              unprocessed_reqs}
                req = max(unprocessed_reqs, key=lambda r: r.profit * cum_embedding_value_of_req[r])
            else:
                req = req_list[i]

            if lp_computation_mode != LPRecomputationMode.NONE:
                self.remove_impossible_mappings_and_reoptimize(A, fixed_requests=processed_reqs, lp_computation_mode=lp_computation_mode)

            p = random.random()
            total_flow = 0.0

            chosen_mapping = None

            for mapping_index, mapping in enumerate(self.mappings_of_requests[req]):
                if req.profit < 0.0001:
                    continue
                total_flow += self.mapping_variables[req][mapping_index].X
                if p < total_flow:
                    chosen_mapping = (mapping_index, mapping)
                    break

            if chosen_mapping is not None:

                if self.check_whether_mapping_would_obey_resource_violations(A, self.allocations_of_mappings[req][chosen_mapping[0]]):
                    B += req.profit
                    for res in self.substrate_resources:
                        if res in self.allocations_of_mappings[req][chosen_mapping[0]].keys():
                            A[res] += self.allocations_of_mappings[req][chosen_mapping[0]][res]

                    if lp_computation_mode != LPRecomputationMode.NONE:
                        self.mapping_variables[req][chosen_mapping[0]].lb = 1.0
                        self.adapted_variables.append(self.mapping_variables[req][chosen_mapping[0]])

                    scenario_solution.add_mapping(req, chosen_mapping[1])
            else:
                if lp_computation_mode != LPRecomputationMode.NONE:
                    for var in self.mapping_variables[req]:
                        var.ub = 0
                        self.adapted_variables.append(var)

            processed_reqs.add(req)
            unprocessed_reqs.remove(req)

        if len(self.adapted_variables) > 0:
            for var in self.adapted_variables:
                var.lb = 0.0
                var.ub = 1.0
            #TODO: handle the case with separation such that introduced columns are removed or at least newly generated
            #TODO: variables get a warmstart basis
            #reset adapted variables
            self.adapted_variables = []

        max_node_load, max_edge_load = self.calc_max_loads(A)
        return scenario_solution, B, max_node_load, max_edge_load


    def calc_max_loads(self, L):
        max_node_load = 0
        max_edge_load = 0
        for (ntype, snode) in self.substrate_node_resources:
            ratio = L[(ntype, snode)] / float(self.scenario.substrate.node[snode]["capacity"][ntype])
            if ratio > max_node_load:
                max_node_load = ratio
        for (u, v) in self.substrate_edge_resources:
            ratio = L[(u, v)] / float(self.scenario.substrate.edge[(u, v)]["capacity"])
            if ratio > max_edge_load:
                max_edge_load = ratio
        return (max_node_load, max_edge_load)

    def check_whether_mapping_would_obey_resource_violations(self, L, mapping_loads):
        result = True

        for (ntype, snode) in self.substrate_node_resources:
            if (ntype, snode) not in mapping_loads:
                continue
            if L[(ntype, snode)] + mapping_loads[(ntype, snode)] > self.scenario.substrate.node[snode]["capacity"][ntype]:
                result = False
                break

        for (u, v) in self.substrate_edge_resources:
            if (u,v) not in mapping_loads:
                continue
            if L[(u, v)] + mapping_loads[(u, v)] > self.scenario.substrate.edge[(u, v)]["capacity"]:
                result = False
                break

        return result

    def recover_solution_from_variables(self):
        pass


    ###
    ###     GUROBI SETTINGS
    ###     The following is copy-pasted from the basic modelcreator in the alib, as the separation approach
    ###     breaks the structure of computing a simple LP or IP.
    ###

    _listOfUserVariableParameters = [Param_MIPGap, Param_IterationLimit, Param_NodeLimit, Param_Heuristics,
                                     Param_Threads, Param_TimeLimit, Param_Cuts, Param_MIPFocus,
                                     Param_RootCutPasses,
                                     Param_NodefileStart, Param_Method, Param_NodeMethod, Param_BarConvTol,
                                     Param_NumericFocus]

    def apply_gurobi_settings(self, gurobiSettings):
        ''' Apply gurobi settings.

        :param gurobiSettings:
        :return:
        '''


        if gurobiSettings.MIPGap is not None:
            self.set_gurobi_parameter(Param_MIPGap, gurobiSettings.MIPGap)
        else:
            self.reset_gurobi_parameter(Param_MIPGap)

        if gurobiSettings.IterationLimit is not None:
            self.set_gurobi_parameter(Param_IterationLimit, gurobiSettings.IterationLimit)
        else:
            self.reset_gurobi_parameter(Param_IterationLimit)

        if gurobiSettings.NodeLimit is not None:
            self.set_gurobi_parameter(Param_NodeLimit, gurobiSettings.NodeLimit)
        else:
            self.reset_gurobi_parameter(Param_NodeLimit)

        if gurobiSettings.Heuristics is not None:
            self.set_gurobi_parameter(Param_Heuristics, gurobiSettings.Heuristics)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.Threads is not None:
            self.set_gurobi_parameter(Param_Threads, gurobiSettings.Threads)
        else:
            self.reset_gurobi_parameter(Param_Heuristics)

        if gurobiSettings.TimeLimit is not None:
            self.set_gurobi_parameter(Param_TimeLimit, gurobiSettings.TimeLimit)
        else:
            self.reset_gurobi_parameter(Param_TimeLimit)

        if gurobiSettings.MIPFocus is not None:
            self.set_gurobi_parameter(Param_MIPFocus, gurobiSettings.MIPFocus)
        else:
            self.reset_gurobi_parameter(Param_MIPFocus)

        if gurobiSettings.cuts is not None:
            self.set_gurobi_parameter(Param_Cuts, gurobiSettings.cuts)
        else:
            self.reset_gurobi_parameter(Param_Cuts)

        if gurobiSettings.rootCutPasses is not None:
            self.set_gurobi_parameter(Param_RootCutPasses, gurobiSettings.rootCutPasses)
        else:
            self.reset_gurobi_parameter(Param_RootCutPasses)

        if gurobiSettings.NodefileStart is not None:
            self.set_gurobi_parameter(Param_NodefileStart, gurobiSettings.NodefileStart)
        else:
            self.reset_gurobi_parameter(Param_NodefileStart)

        if gurobiSettings.Method is not None:
            self.set_gurobi_parameter(Param_Method, gurobiSettings.Method)
        else:
            self.reset_gurobi_parameter(Param_Method)

        if gurobiSettings.NodeMethod is not None:
            self.set_gurobi_parameter(Param_NodeMethod, gurobiSettings.NodeMethod)
        else:
            self.reset_gurobi_parameter(Param_NodeMethod)

        if gurobiSettings.BarConvTol is not None:
            self.set_gurobi_parameter(Param_BarConvTol, gurobiSettings.BarConvTol)
        else:
            self.reset_gurobi_parameter(Param_BarConvTol)

        if gurobiSettings.NumericFocus is not None:
            self.set_gurobi_parameter(Param_NumericFocus, gurobiSettings.NumericFocus)
        else:
            self.reset_gurobi_parameter(Param_NumericFocus)

    def reset_all_parameters_to_default(self):
        for param in self._listOfUserVariableParameters:
            (name, type, curr, min, max, default) = self.model.getParamInfo(param)
            self.model.setParam(param, default)

    def reset_gurobi_parameter(self, param):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Parameter {} unchanged".format(param))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        self.model.setParam(param, default)

    def set_gurobi_parameter(self, param, value):
        (name, type, curr, min_val, max_val, default) = self.model.getParamInfo(param)
        self.logger.debug("Changed value of parameter {} to {}".format(param, value))
        self.logger.debug("    Prev: {}   Min: {}   Max: {}   Default: {}".format(
            curr, min_val, max_val, default
        ))
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.setParam(param, value)

    def getParam(self, param):
        if not param in self._listOfUserVariableParameters:
            raise modelcreator.ModelcreatorError("You cannot access the parameter <" + param + ">!")
        else:
            self.model.getParam(param)

""" Dynamic Program for Valid Mapping Problem Dyn-VMP """

INFINITY = float("inf")


class DynVMPAlgorithm(object):
    """
    Solve the Valid Mapping Problem using the DYN-VMP algorithm as presented in the paper. Note that the Optimized version
    is much quicker in computing solutions and that this vanilla implementation should probably never be used.
    To reiterate: Semantically, the optimized and the non-optimized versions are the same, but the pre-allocation of numpy.arrays
    and the pre-computation of indices in these arrays together with matrix operations makes the optimized one significantly (magnitudes!)
    faster.
    """

    def __init__(self, request, substrate):
        self.request = request
        self.substrate = substrate
        self.tree_decomposition = None
        self.tree_decomposition_arb = None
        self.cost_table = None
        self.mapping_table = None

    def preprocess_input(self):
        self.substrate.initialize_shortest_paths_costs()
        self.tree_decomposition = compute_tree_decomposition(self.request)
        self.tree_decomposition_arb = self.tree_decomposition.convert_to_arborescence()

    def compute_valid_mapping(self):
        """
        The code mostly follows the pseudocode of the DYN-VMP algorithm.

        Node mappings are represented as follows: Since the mapped request nodes are always well defined, the
        mapping is represented by a tuple of substrate nodes, where each substrate node in the tuple
        is the mapped location of the request node at that index in the *sorted* list of request nodes that are mapped.

        E.g.:
        For node_bag frozenset(["j", "i", "k"]), the tuple  ("u", "w", "v") represents the mapping
        { i -> u, j -> w, k -> v }

        :return:
        """
        self.initialize_tables()

        for t in self.tree_decomposition_arb.post_order_traversal():
            bag_nodes = self.tree_decomposition.node_bag_dict[t]
            for bag_mapping in self.cost_table[t].iterkeys():
                ind_cost = self._induced_edge_cost(sorted(bag_nodes), bag_mapping)
                if ind_cost != INFINITY:
                    children_valid = True
                    for t_child in self.tree_decomposition_arb.get_out_neighbors(t):
                        best_child_bag_mapping = None
                        best_child_mapping_cost = INFINITY
                        child_bag_nodes = self.tree_decomposition.node_bag_dict[t_child]
                        for child_bag_mapping, child_mapping_cost in self.cost_table[t_child].iteritems():
                            if child_mapping_cost == INFINITY:
                                continue
                            elif child_mapping_cost > best_child_mapping_cost:
                                """ 
                                I think it is better to check 2nd condition in Line 13 
                                before the for-loop condition in Line 12, as it is a simpler 
                                operation
                                """
                                continue
                            elif check_if_mappings_are_compatible(
                                    bag_nodes, bag_mapping,
                                    child_bag_nodes, child_bag_mapping
                            ):
                                best_child_bag_mapping = child_bag_mapping
                                best_child_mapping_cost = child_mapping_cost

                        if best_child_bag_mapping is not None:
                            bag_index_offset = 0
                            child_index_offset = 0
                            for i_idx, i in enumerate(sorted(self.request.nodes)):
                                if i in bag_nodes:
                                    bag_index_offset += 1
                                if i in child_bag_nodes:
                                    if i not in bag_nodes:
                                        child_mapped_node = best_child_bag_mapping[child_index_offset]
                                        self.mapping_table[t][bag_mapping][i_idx - bag_index_offset] = child_mapped_node
                                    child_index_offset += 1
                                if i not in bag_nodes and i not in child_bag_nodes:
                                    child_mapped_node = self.mapping_table[t_child][best_child_bag_mapping][i_idx - child_index_offset]
                                    if child_mapped_node is not None:
                                        self.mapping_table[t][bag_mapping][i_idx - bag_index_offset] = child_mapped_node
                        else:
                            # Check if mapping is not well-defined to avoid another indentation
                            children_valid = False
                            break

                    if children_valid:
                        # update cost table
                        self.cost_table[t][bag_mapping] = self._induced_edge_cost(
                            sorted(self.request.nodes - bag_nodes) + sorted(bag_nodes),
                            self.mapping_table[t][bag_mapping] + list(bag_mapping),
                        )

        return self._extract_mapping_from_table()

    def _extract_mapping_from_table(self):
        t_root = self.tree_decomposition_arb.root
        root_bag_nodes = self.tree_decomposition.node_bag_dict[t_root]
        root_bag_mapping = self._find_best_root_bag_mapping()

        if root_bag_mapping is None:
            return None  # TODO Or is it better to return an empty mapping object?

        # Convert to actual mapping...
        mapping_name = construct_name_tw_lp("dynvmp_mapping", req_name=self.request.name)
        result = solutions.Mapping(mapping_name, self.request, self.substrate, True)

        # Map request nodes according to mapping table:
        for i, u in zip(sorted(root_bag_nodes), root_bag_mapping):
            result.map_node(i, u)
        for i, u in zip(sorted(self.request.nodes - root_bag_nodes), self.mapping_table[t_root][root_bag_mapping]):
            result.map_node(i, u)

        # Map request edges:
        for i, j in self.request.edges:
            u = result.get_mapping_of_node(i)
            v = result.get_mapping_of_node(j)
            ij_demand = self.request.get_edge_demand((i, j))
            result.map_edge((i, j), self.shortest_path(u, v, ij_demand))

        return result

    def _find_best_root_bag_mapping(self):
        best_root_bag_mapping = None
        best_root_mapping_cost = INFINITY
        for root_bag_mapping, cost in self.cost_table[self.tree_decomposition_arb.root].iteritems():
            if cost < best_root_mapping_cost:
                best_root_mapping_cost = cost
                best_root_bag_mapping = root_bag_mapping
        return best_root_bag_mapping

    def initialize_tables(self):
        self.cost_table = {}
        self.mapping_table = {}

        for (t, bag_nodes) in self.tree_decomposition.node_bag_dict.iteritems():
            self.cost_table[t] = {}
            self.mapping_table[t] = {}
            for node_bag_mapping in mapping_space(self.request, self.substrate, sorted(bag_nodes)):
                self.cost_table[t][node_bag_mapping] = INFINITY
                self.mapping_table[t][node_bag_mapping] = [None] * (len(self.request.nodes) - len(bag_nodes))

    def _induced_edge_cost(self, req_nodes, sub_nodes):
        """
        Calculate minimal costs of mapping induced by partial node mapping.

        :param req_nodes: list of request nodes that are in the partial node mapping
        :param sub_nodes: list of substrate nodes to which the request nodes are mapped
        :return:
        """
        # TODO: might not be the nicest way to do this
        cost = 0
        for i_idx, i in enumerate(req_nodes):
            u = sub_nodes[i_idx]
            if u is None:
                continue  # ignore any nodes that are not mapped
            cost += self.request.get_node_demand(i) * self.substrate.get_node_type_cost(
                u, self.request.get_type(i)
            )
            for j_idx, j in enumerate(req_nodes):
                v = sub_nodes[j_idx]
                if v is None:
                    continue
                if (i, j) in self.request.edges:
                    ij_cost = self.substrate.get_shortest_paths_cost(u, v)
                    if ij_cost is None:
                        return INFINITY
                    cost += ij_cost
        return cost

    def shortest_path(self, start, target, min_capacity):
        # TODO: could add this to the alib by extending Graph.initialize_shortest_paths_costs()
        # Dijkstra algorithm (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Pseudocode)
        distance = {node: INFINITY for node in self.substrate.nodes}
        prev = {u: None for u in self.substrate.nodes}
        distance[start] = 0
        q = set(self.substrate.nodes)
        while q:
            u = min(q, key=lambda x: distance[x])
            if u == target:
                break
            q.remove(u)
            for uv in self.substrate.get_out_edges(u):
                if self.substrate.get_edge_capacity(uv) < min_capacity:
                    continue  # avoid using edges that are too small
                v = uv[1]
                new_dist = distance[u] + self.substrate.get_edge_cost(uv)
                if new_dist < distance[v]:
                    distance[v] = new_dist
                    prev[v] = u
        path = []
        u = target
        while u is not start:
            path.append((prev[u], u))
            u = prev[u]
        return list(reversed(path))


def check_if_mappings_are_compatible(bag_1, sub_nodes_1, bag_2, sub_nodes_2):
    intersection = bag_1 & bag_2
    intersection_mapping_1 = [u for (i, u) in zip(sorted(bag_1), sub_nodes_1) if i in intersection]
    intersection_mapping_2 = [u for (i, u) in zip(sorted(bag_2), sub_nodes_2) if i in intersection]
    return intersection_mapping_1 == intersection_mapping_2


""" Modelcreator: Treewidth LP & Decomposition Algorithm """

construct_name_tw_lp = modelcreator.build_construct_name([
    ("req_name", "req"),
    "type", "vnode", "snode", "vedge", "sedge", "other",
    ("bag", None, lambda b: "_".join(sorted(b))),
    ("bag_mapping", None, lambda bm: "_".join(sorted(bm))),
    ("sub_name", "substrate"),
    ("sol_name", "solution"),
])


class _TreewidthModelCreator(modelcreator.AbstractEmbeddingModelCreator):

    ''' Base for implementing a (not yet published) LP based on tree decompositions. Note that we expect the separation oracle
        based LP to be (nearly always) quicker than using this one.
        We have not really tested this implementation and you should probably not use it (without knowing what you are doing).
    '''
    def __init__(self,
                 scenario,
                 precomputed_tree_decompositions=None,  # Allow passing a dict of tree decompositions (mainly for testing)
                 gurobi_settings=None,
                 optimization_callback=modelcreator.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(_TreewidthModelCreator, self).__init__(scenario,
                                                     gurobi_settings=gurobi_settings,
                                                     optimization_callback=optimization_callback,
                                                     lp_output_file=lp_output_file,
                                                     potential_iis_filename=potential_iis_filename,
                                                     logger=logger)

        self.tree_decompositions = {}
        if precomputed_tree_decompositions is not None:
            self.tree_decompositions.update(precomputed_tree_decompositions)

    def preprocess_input(self):
        for req in self.scenario.requests:
            if req.name not in self.tree_decompositions:
                self.tree_decompositions[req.name] = compute_tree_decomposition(req)
            if not self.tree_decompositions[req.name].is_tree_decomposition(req):
                raise ValueError("Tree decomposition failed for request {}!".format(req))

    def create_variables_other_than_embedding_decision_and_request_load(self):
        self.y_vars = {}
        self.z_vars = {}

        for req in self.requests:
            self._create_y_variables(req)
            self._create_z_variables(req)

    def _create_y_variables(self, req):
        td = self.tree_decompositions[req.name]
        self.y_vars[req.name] = {}
        for bag_id, bag_nodes in td.node_bag_dict.items():
            self.y_vars[req.name][bag_id] = {}
            for a in mapping_space(req, self.substrate, sorted(bag_nodes)):
                variable_name = construct_name_tw_lp("bag_mapping", req_name=req.name, bag=bag_id, bag_mapping=a)
                self.y_vars[req.name][bag_id][a] = self.model.addVar(lb=0.0,
                                                                     ub=1.0,
                                                                     obj=0.0,
                                                                     vtype=GRB.BINARY,
                                                                     name=variable_name)

    def _create_z_variables(self, req):
        self.z_vars[req.name] = {}
        for ij in req.edges:
            self.z_vars[req.name][ij] = {}
            for uv in self.substrate.edges:
                self.z_vars[req.name][ij][uv] = {}
                for w in self.substrate.nodes:
                    variable_name = construct_name_tw_lp("flow", req_name=req.name, vedge=ij, sedge=uv, snode=w)
                    self.z_vars[req.name][ij][uv][w] = self.model.addVar(lb=0.0,
                                                                         ub=1.0,
                                                                         obj=0.0,
                                                                         vtype=GRB.BINARY,
                                                                         name=variable_name)

    def create_constraints_other_than_bounding_loads_by_capacities(self):
        for req in self.requests:
            self._create_constraints_embed_bags_equally(req)
            self._create_constraints_flows(req)

    def _create_constraints_embed_bags_equally(self, req):
        td = self.tree_decompositions[req.name]
        for bag in td.nodes:
            constr_name = construct_name_tw_lp("embed_bags_equally", req_name=req.name, bag=bag)
            expr = LinExpr(
                [(-1.0, self.var_embedding_decision[req])] +
                [(1.0, var) for bag_mapping, var in self.y_vars[req.name][bag].items()]
            )
            self.model.addConstr(LinExpr(expr), GRB.EQUAL, 0.0, constr_name)

    def _create_constraints_flows(self, req):
        td = self.tree_decompositions[req.name]
        for ij in req.edges:
            i, j = ij
            t_ij = td.get_any_covering_td_node(i, j)

            t_ij_nodes = sorted(td.node_bag_dict[t_ij])
            i_idx = t_ij_nodes.index(i)
            j_idx = t_ij_nodes.index(j)

            y_dict_for_t_ij = self.y_vars[req.name][t_ij]

            for w in self.substrate.nodes:
                for u in self.substrate.nodes:
                    if u == w:
                        continue
                    terms = ([(1.0, self.z_vars[req.name][ij][uv][w]) for uv in self.substrate.get_out_edges(u)]
                             + [(-1.0, self.z_vars[req.name][ij][vu][w]) for vu in self.substrate.get_in_edges(u)]
                             + [(-1.0, y_dict_for_t_ij[a]) for a in y_dict_for_t_ij if a[i_idx] == u and a[j_idx] == w])

                    constr_name = construct_name_tw_lp("flows", req_name=req.name, vedge=ij, snode=u, other=w)
                    expr = LinExpr(terms)
                    self.model.addConstr(LinExpr(expr), GRB.EQUAL, 0.0, constr_name)


def mapping_space(request, substrate, req_nodes):
    allowed_nodes_list = []
    for i in req_nodes:
        allowed_nodes = request.get_allowed_nodes(i)
        if allowed_nodes is None:
            allowed_nodes = list(substrate.nodes)
        allowed_nodes_list.append(allowed_nodes)
    for sub_node_tuple in itertools.product(*allowed_nodes_list):
        yield sub_node_tuple