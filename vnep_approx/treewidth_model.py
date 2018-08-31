import itertools
import os
import subprocess
from collections import deque
import numpy as np
from heapq import heappush, heappop


from gurobipy import GRB, LinExpr
from alib import datamodel, modelcreator, solutions

""" Some datastructures to represent tree decompositions """


class UndirectedGraph(object):
    def __init__(self, name):
        self.name = name
        self.nodes = set()
        self.edges = set()

        self.neighbors = {}
        self.incident_edges = {}

        # attribute dictionaries:
        self.graph = {}
        self.node = {}
        self.edge = {}

    def add_node(self, node):
        self.nodes.add(node)
        self.neighbors[node] = set()
        self.incident_edges[node] = set()

    def add_edge(self, i, j):
        if i not in self.nodes or j not in self.nodes:
            raise ValueError("Nodes not in graph!")
        new_edge = frozenset([i, j])
        if new_edge in self.edges:
            raise ValueError("Duplicate edge {new_edge}!")
        if len(new_edge) == 1:
            raise ValueError("Loop edges are not allowed ({i})")

        self.neighbors[i].add(j)
        self.neighbors[j].add(i)
        self.incident_edges[i].add(new_edge)
        self.incident_edges[j].add(new_edge)
        self.edges.add(new_edge)
        self.edge[new_edge] = {}
        return new_edge

    def remove_node(self, node):
        if node not in self.nodes:
            raise ValueError("Node not in graph.")

        edges_to_remove = list(self.incident_edges[node])

        for incident_edge in edges_to_remove:
            edge_as_list = list(incident_edge)
            self.remove_edge(edge_as_list[0], edge_as_list[1])

        del self.incident_edges[node]
        del self.neighbors[node]
        self.nodes.remove(node)

    def remove_edge(self, i, j):
        old_edge = frozenset([i, j])
        if i not in self.nodes or j not in self.nodes:
            raise ValueError("Nodes not in graph!")
        if old_edge not in self.edges:
            raise ValueError("Edge not in graph!")
        self.neighbors[i].remove(j)
        self.neighbors[j].remove(i)
        self.incident_edges[i].remove(old_edge)
        self.incident_edges[j].remove(old_edge)
        self.edges.remove(old_edge)
        del self.edge[old_edge]

    def get_incident_edges(self, node):
        return self.incident_edges[node]

    def get_neighbors(self, node):
        neighbors = set().union(*self.get_incident_edges(node))
        return neighbors - {node}


class TreeDecomposition(UndirectedGraph):
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
            raise ValueError("Empty or unspecified node bag: {node_bag}")
        if not isinstance(node_bag, frozenset):
            raise ValueError("Expected node bag as frozenset: {node_bag}")
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

        #     if req_node not in self.complete_request_node_to_tree_node_map:
        #         self.complete_request_node_to_tree_node_map[req_node] = []
        #     else:
        #         for other_tree_node in self.complete_request_node_to_tree_node_map[req_node]:
        #             edges_to_create.add(frozenset([other_tree_node,node]))
        #         self.complete_request_node_to_tree_node_mao[req_node].append(node)
        # for edge_to_create in edges_to_create:
        #     edge_as_list = list(edge_to_create)
        #     super(TreeDecomposition,self).add_edge(edge_as_list[0], edge_as_list[1])


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

    Leaf = 0
    Introduction = 1
    Forget = 2
    Join = 3
    Root = 4

class SmallSemiNiceTDArb(TreeDecomposition):

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

            neighbors_of_removed_node = self.get_neighbors(node_to_remove)

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
            # print("Splitting edge {} with repsective "
            #       "node bags {} and {} to introduce "
            #       "new node {} with bag {}".format(edge,
            #                                        tail_bag,
            #                                        head_bag,
            #                                        internode,
            #                                        intersection))


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


class ShortestValidPathsComputer(object):

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



class OptimizedDynVMPNode(object):

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


        #just for debugging; remove this later on!
        # other_result_list = []
        # print("\n\n\n === NEW ===")
        # self.fill_matrix_with_mapping_indices(meta_information, construction_rule, other_result_list)
        #
        # print other_result_list
        # print result_list
        #
        # if len(other_result_list) != len(result_list):
        #     raise ValueError("{}\n{}\n{}\n{}".format(other_result_list, result_list, construction_rule, meta_information))
        # for i in range(len(other_result_list)):
        #     if other_result_list[i] != result_list[i]:
        #         raise ValueError(
        #             "{}\n{}\n{}\n{}".format(other_result_list, result_list, construction_rule, meta_information))



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
                #if self.validity_array[current_value]: #TODO somehow does not work when only including valid mapping indices
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

        # debug = False
        # 
        # if debug:
        #     o_result = {}
        #     for o_mapping_index, mapping in enumerate(itertools.product(*self.list_of_ordered_allowed_nodes)):
        #         if o_mapping_index != mapping_index:
        #             continue
        #         for reqnode_index, reqnode in enumerate(self.contained_request_nodes):
        #             o_result[reqnode] = mapping[reqnode_index]
        #         break


        result_node_mapping = {}
        for reversed_reqnode_index in range(self.number_of_request_nodes):
            request_node = self._reversed_contained_request_nodes[reversed_reqnode_index]
            number_of_allowed_nodes_for_request_node =  self.number_of_allowed_nodes[request_node]
            snode_index = mapping_index % number_of_allowed_nodes_for_request_node
            snode = self.allowed_nodes[request_node][snode_index]
            result_node_mapping[request_node] = snode
            mapping_index /= number_of_allowed_nodes_for_request_node

        # if debug:
        #     print "Naively found vs. intelligently found:\n\t{}\n\t{}".format(result_node_mapping, o_result)
        # 
        #     if set(result_node_mapping.keys()) != set(o_result.keys()):
        #         raise ValueError("THis is bad!")
        #     for reqnode in result_node_mapping.keys():
        #         if result_node_mapping[reqnode] != o_result[reqnode]:
        #             raise ValueError("Really bad!")

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
                    if np.isnan(self.svpc.valid_sedge_costs[(reqedge_source, reqedge_target)][(mapping_of_source, mapping_of_target)]):

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


class OptimizedDynVMP(object):

    def __init__(self, substrate, request, ssntda, initial_snode_costs = None, initial_sedge_costs = None):
        self.substrate = substrate
        self.request = request
        if not isinstance(ssntda, SmallSemiNiceTDArb):
            raise ValueError("The Optimized DYNVMP Algorithm expects a small-semi-nice-tree-decomposition-arborescence")
        self.ssntda = ssntda

        self._initialize_costs(initial_snode_costs, initial_sedge_costs)

        self.vmrc = ValidMappingRestrictionComputer(substrate=substrate, request=request)
        self.svpc = ShortestValidPathsComputer(substrate=substrate, request=request, valid_mapping_restriction_computer=self.vmrc, edge_costs=self.sedge_costs)

    def _initialize_costs(self, snode_costs, sedge_costs):
        if snode_costs is None:
            self.snode_costs = {snode: 1.0 for snode in self.substrate.nodes}
        else:
            self.snode_costs = {snode: snode_costs[snode] for snode in snode_costs.keys()}

        if sedge_costs is None:
            self.sedge_costs = {sedge: 1.0 for sedge in self.substrate.edges}
        else:
            self.sedge_costs = {sedge: sedge_costs[sedge] for sedge in sedge_costs.keys()}

        self._max_demand = max([self.request.get_node_demand(reqnode) for reqnode in self.request.nodes] +
                               [self.request.get_edge_demand(reqedge) for reqedge in self.request.edges])

        self._max_cost = max(
            [cost for cost in self.snode_costs.values()] + [cost for cost in self.sedge_costs.values()])

        self._mapping_cost_bound = self._max_demand * self._max_cost * 2.0 * len(self.request.edges) * len(self.substrate.edges)



    def initialize_data_structures(self):
        self.vmrc.compute()
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
                current_edge_costs_array[node_pair_index] = self.svpc.valid_sedge_costs[edge_set_index][
                    (snode_1, snode_2)]

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
        reqedge_predecessors = self.svpc.valid_sedge_pred[reqedge][source_mapping]
        u = target_mapping
        path = []
        while u != source_mapping:
            pred = reqedge_predecessors[u]
            path.append((pred, u))
            u = pred
        path = list(reversed(path))
        return path

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

#
#   OLD
#





class NiceTDConversion(object): # TODO: Work in Progress
    LEAF = "L"
    JOIN = "J"
    INTRO = "I"
    FORGET = "F"

    def __init__(self, request, initial_td=None):
        if initial_td is None:
            initial_td = compute_tree_decomposition(request)
        self.initial_td = initial_td
        self.request = request
        self.nice_td = None

    def initialize(self, root=None):
        self.initial_arb = self.initial_td.convert_to_arborescence(root=root)
        self.nice_td = TreeDecomposition("{}_nice".format(self.initial_td.name))
        self.next_id = {
            self.LEAF: 1,
            self.JOIN: 1,
            self.FORGET: 1,
            self.INTRO: 1,
        }
        self.parent_dict = {}

    def make_nice_tree_decomposition(self):
        q = [self.initial_arb.root]
        while q:
            t = q.pop()
            t_bag = self.initial_td.node_bag_dict[t]
            self.nice_td.add_node(t, t_bag)
            if t in self.parent_dict:
                self.nice_td.add_edge(self.parent_dict[t], t)
            children = self.initial_arb.get_out_neighbors(t)

            num_children = len(children)
            if num_children == 0:
                t_bag = sorted(t_bag)
                self._make_forget_chain(t, frozenset([t_bag[-1]]), self.LEAF)
                continue

            if num_children == 1:
                connection_nodes = [t]  # connect directly to t
            else:
                # build binary tree of join nodes
                connection_nodes = self._make_join_node_tree(t)
            for parent, child in zip(connection_nodes, self.initial_arb.get_out_neighbors(t)):
                child_bag = self.initial_td.node_bag_dict[child]
                new_parent = self._make_forget_chain(parent, child_bag, self.FORGET)
                new_parent = self._make_intro_chain(new_parent, child_bag, self.INTRO)
                q.append(child)
                self.parent_dict[child] = new_parent
        return self.nice_td

    def _make_join_node_tree(self, t):
        num_children = len(self.initial_arb.get_out_neighbors(t))
        t_bag = self.nice_td.node_bag_dict[t]
        connection_nodes = deque([t])  # contains the leaves of the binary tree.
        while len(connection_nodes) < num_children:
            parent = connection_nodes.popleft()
            leftchild = self._next_node_id(self.JOIN)
            rightchild = self._next_node_id(self.JOIN)
            self.nice_td.add_node(leftchild, t_bag)
            self.nice_td.add_node(rightchild, t_bag)
            self.nice_td.add_edge(parent, leftchild)
            self.nice_td.add_edge(parent, rightchild)
            connection_nodes.append(leftchild)
            connection_nodes.append(rightchild)
        connection_nodes = list(connection_nodes)
        return connection_nodes

    def _make_forget_chain(self, start, target_bag, target_node_type):
        start_bag = self.nice_td.node_bag_dict[start]
        nodes_to_forget = sorted(start_bag - target_bag)
        parent = start
        new_node = None
        current_bag = set(start_bag)
        while nodes_to_forget:
            forget_node = nodes_to_forget.pop()
            current_bag.remove(forget_node)
            if len(current_bag) > 1:
                new_node = self._next_node_id(self.FORGET)
            else:
                new_node = self._next_node_id(target_node_type)
            self.nice_td.add_node(new_node, frozenset(current_bag))
            self.nice_td.add_edge(parent, new_node)
            parent = new_node
        return new_node

    def _make_intro_chain(self, start, target_bag, target_node_type):
        start_bag = self.nice_td.node_bag_dict[start]
        nodes_to_introduce = sorted(target_bag - start_bag)
        parent = start
        new_node = None
        current_bag = set(start_bag)
        while nodes_to_introduce:
            next_intro_node = nodes_to_introduce.pop()
            current_bag.add(next_intro_node)
            if len(current_bag) > 1:
                new_node = self._next_node_id(self.INTRO)
            else:
                new_node = self._next_node_id(target_node_type)
            self.nice_td.add_node(new_node, frozenset(current_bag))
            self.nice_td.add_edge(parent, new_node)
            parent = new_node
        return new_node

    def _next_node_id(self, node_type):
        result = "{}_{}".format(node_type, self.next_id[node_type])
        self.next_id[node_type] += 1
        return result


def is_nice_tree_decomposition(tree_decomp, arborescence):
    for t in tree_decomp.nodes:
        children = arborescence.get_out_neighbors(t)
        if len(children) == 0:
            # leaf node: May only contain one bag node
            if not _is_valid_leaf_node(tree_decomp, t):
                print "Node {} is not a valid leaf node".format(t)
                return False
        elif len(children) == 1:
            # introduce or forget node
            if not _is_valid_intro_or_forget_node(tree_decomp, t, next(iter(children))):
                print "Node {} is not a valid introduction or forget node".format(t)
                return False
        elif len(children) == 2:
            # join node: all children must have same bag set
            children = arborescence.get_out_neighbors(t)
            if not _is_valid_join_node(tree_decomp, t, children):
                print "Node {} is not a valid join node".format(t)
                return False
        else:
            print "Node {} has too many neighbors".format(t)
            return False
    return True


def _is_valid_leaf_node(tree_decomp, t):
    return len(tree_decomp.node_bag_dict[t]) == 1


def _is_valid_intro_or_forget_node(tree_decomp, t, child):
    bag_neighbor = tree_decomp.node_bag_dict[child]
    bag_t = tree_decomp.node_bag_dict[t]
    num_forgotten = len(bag_neighbor - bag_t)
    num_introduced = len(bag_t - bag_neighbor)
    is_forget_node = num_introduced == 0 and num_forgotten == 1
    is_intro_node = num_introduced == 1 and num_forgotten == 0
    return is_intro_node or is_forget_node


def _is_valid_join_node(tree_decomp, t, children):
    if len(children) != 2:
        return False
    node_bag = tree_decomp.node_bag_dict[t]
    c1 = children[0]
    c2 = children[1]
    return (node_bag == tree_decomp.node_bag_dict[c1] and
            node_bag == tree_decomp.node_bag_dict[c2])


""" Computing tree decompositions """


class TreeDecompositionComputation(object):
    """
    Use the exact tree decomposition algorithm implementation by Hisao Tamaki and Hiromu Ohtsuka, obtained
    from https://github.com/TCS-Meiji/PACE2017-TrackA, to compute tree deecompositions.

    It assumes that the path to the tree decomposition algorithm is stored in the environment variable PACE_TD_ALGORITHM_PATH
    """

    def __init__(self, graph):
        self.graph = graph
        self.map_nodes_to_numeric_id = None
        self.map_numeric_id_to_nodes = None

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
            p = subprocess.Popen("./tw-exact", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdoutdata, stderrdata = p.communicate(input=td_alg_input)
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


def compute_tree_decomposition(request):
    return TreeDecompositionComputation(request).compute_tree_decomposition()


""" Dynamic Program for Valid Mapping Problem Dyn-VMP """

INFINITY = float("inf")


class DynVMPAlgorithm(object):
    """
    Solve the Valid Mapping Problem using the DYN-VMP algorithm.
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


# TODO: inline & improve this function
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


class TreewidthModelCreator(modelcreator.AbstractEmbeddingModelCreator):
    def __init__(self,
                 scenario,
                 precomputed_tree_decompositions=None,  # Allow passing a dict of tree decompositions (mainly for testing)
                 gurobi_settings=None,
                 optimization_callback=modelcreator.gurobi_callback,
                 lp_output_file=None,
                 potential_iis_filename=None,
                 logger=None):
        super(TreewidthModelCreator, self).__init__(scenario,
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
