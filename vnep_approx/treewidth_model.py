import itertools
import os
import subprocess
from collections import deque

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
        return max(len(bag) for bag in self.node_bag_dict)

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
        self.post_order_traversal = None

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
            print("Splitting edge {} with repsective node bags {} and {} to introduce new node {} with bag {}".format(edge,
                                                                                                                      tail_bag,
                                                                                                                      head_bag,
                                                                                                                      internode,
                                                                                                                      intersection))


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
        self.post_order_traversal = []
        #root is set
        visited = set()
        queue = [self.root]
        while len(queue) > 0:
            current_node = queue.pop(0)
            print("Handling node {}".format(current_node))
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

            self.post_order_traversal.append(current_node)

        self.post_order_traversal = reversed(self.post_order_traversal)

        print("Nodes:\n{}".format(self.nodes))
        print("Edges:\n{}".format(self.edges))

        for node in self.nodes:
            if len(self.out_neighbors[node]) == 0:
                self.node_classification[node] = 'L'
            elif len(self.out_neighbors[node]) == 1:
                out_neighbor = self.out_neighbors[node][0]
                if self.node_bag_dict[out_neighbor].issubset(self.node_bag_dict[node]):
                    self.node_classification[node] = 'I'
                elif self.node_bag_dict[node].issubset(self.node_bag_dict[out_neighbor]):
                    self.node_classification[node] = 'F'
                else:
                    raise ValueError("Don't know what's happening here!")
            else:
                self.node_classification[node] = 'J'
                for out_neighbor in self.out_neighbors[node]:
                    if not self.node_bag_dict[out_neighbor].issubset(self.node_bag_dict[node]):
                        raise ValueError("Children should always be subsets of the join node!")









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
