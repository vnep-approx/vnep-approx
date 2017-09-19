# MIT License
#
# Copyright (c) 2016-2017 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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
#

from collections import deque, namedtuple
from random import Random

from alib import datamodel, util

random = Random("extended_cactus_graph")


class ExtendedCactusGraphError(Exception):
    pass


ExtendedGraphCycle = namedtuple("ExtendedGraphCycle",
                                "start_node end_node original_branches ext_graph_branches")
"""
This namedtuple represents a cycle in the extended graph.

* start_node/end_node: The start/end node of the cycle in the original request, as defined by the BFS
  ordering
* original_branches: A represention of the cycles's branches as a list of lists containing the
  original request edges for each branch
* ext_graph_branches: A representation of the branches in the extended graph as a list of branches.
  A branch is a dictionary, mapping each possible substrate end node to an ExtendedGraphPath
  instance (see description below)
"""

ExtendedGraphPath = namedtuple("ExtendedGraphPath",
                               "start_node end_node original_path extended_path extended_edges extended_nodes")
"""
This namedtuple represents a path in the extended graph.

* start_node/end_node: The start/end node of the cycle in the original request, as defined by the BFS 
  ordering
* original_path: A represention of the cycles's branches as a list containing the original request edges
  in the BFS ordering (i.e. some might be reversed)
* extended_path: A dictionary mapping each request node on the path to the extended graph edges associated
  with the node mapping (inter-layer edges) and each request edge on the path to the extended graph edges
  associated with the edge mapping (layer edges)
* extended_edges: A set containing each extended graph edge that is associated with the path.
* extended_nodes: A set containing each extended graph node that is associated with the path
"""


class ExtendedCactusGraph(datamodel.Graph):
    SOURCE_NODE = 1
    LAYER_NODE = 2
    SINK_NODE = 3

    def __init__(self, request, substrate):
        super(ExtendedCactusGraph, self).__init__("{}_ext".format(request.name))
        self.original_request = request
        self.reversed_request_edges = set()  #: this contains edges that point towards the root. edges are stored in their original orientation
        self.bfs_request = None  #: this will be a copy of the request, with all edges oriented according to the BFS orde
        self._original_substrate = substrate
        self._substrate_x = datamodel.SubstrateX(self._original_substrate)
        self._reversed_substrate = self._make_reversed_substrate()
        self._rev_substrate_x = datamodel.SubstrateX(self._reversed_substrate)
        if "root" in request.graph:
            root = request.graph["root"]
        else:
            root = random.choice(list(self.original_request.nodes))
        self.root = root

        self._cycles = None  #: for internal use during the preprocessing: store the request cycles as list of tuples of edgelists representing the branches in the original request.
        self._paths = None
        self._visited = None
        self.ecg_cycles = None
        self.ecg_paths = None
        self._nodes_to_explore = None
        self.layer_nodes = None
        self.cycle_branch_nodes = None
        # The following dictionaries each map a request "key" (an edge or node) to a dictionary mapping a substrate key to the corresponding nodes/edges in the extended graph
        self.source_nodes = None  #: "super source" nodes, on which outflow is induced
        self.sink_nodes = None  #: "super sink" nodes, on which inflow will be induced
        self.path_layer_nodes = None  #: "layer nodes" within paths (i.e. the nodes from "substrate copies")
        self.cycle_layer_nodes = None  #: "layer nodes" within cycles
        self._reverse_lookup_extended_nodes = None
        self._generate_extended_graph()

    def _generate_extended_graph(self):
        """
        To generate the topology of the extended graph, we need to
          - do a breadth-first search of the request, reorienting any edges to point away from the root,
              yielding the bfs_request. Simultaneously, a set of nodes to be explored in the next step
              is initialized with the request's leaf nodes (i.e. nodes without out-neighbors in the bfs graph)
          - decompose the bfs_request into paths and cycles. For simplicity, each edge that is not on a cycle
              is considered a path.
          - to generate the extended graph, the construction is applied as described in the
              technical report. The extended graphs of paths and cycles consist of "layers", where each layer is
              associated with a request edge for paths and cycles, and additionally with a target sink node
              for cycle layers. These layers are generated in _make_substrate_copy_for_layer.
              All the dictionaries of extended graph nodes defined in the initialize() method are maintained by
              the "_add_and_get_*_node" methods defined below. These are the only places where calls to
              Graph.add_node should be made.
        
        :return:
        """
        self.initialize()
        self._preprocessing()
        self._generate_extended_paths()
        self._generate_extended_cycles()

    def initialize(self):
        self._cycles = []  #: for internal use during the preprocessing: store the request cycles as list of tuples of edgelists representing the branches in the original request.
        self._paths = []
        self.ecg_cycles = []
        self.ecg_paths = []
        self._nodes_to_explore = None
        self.layer_nodes = set()
        self.cycle_branch_nodes = set()  #: set of nodes in cycles, from which another subgraph branches off.
        # The following dictionaries each map a request edge or node to a dictionary mapping substrate edges or nodes to the corresponding nodes/edges in the extended graph
        self.source_nodes = {}  #: "super source" nodes, on which outflow is induced
        self.sink_nodes = {}  #: "super sink" nodes, on which inflow will be induced
        self.path_layer_nodes = {}  #: "layer nodes" within paths (i.e. the nodes from "substrate copies")
        self.cycle_layer_nodes = {}  #: "layer nodes" within cycles

    def get_associated_original_resources(self, ext_node):
        if not hasattr(self, "_reverse_lookup_extended_nodes") or (self._reverse_lookup_extended_nodes is None):
            self._initialize_reverse_lookup()
        return self._reverse_lookup_extended_nodes[ext_node]

    def _preprocessing(self):
        self._initialize_bfs_request()
        self._request_decomposition_into_paths_and_cycles()

    def _initialize_bfs_request(self):
        self.bfs_request = datamodel.Request("{}_bfs".format(self.original_request.name))
        for node in self.original_request.nodes:
            demand = self.original_request.node[node]["demand"]
            ntype = self.original_request.node[node]["type"]
            allowed = self.original_request.node[node]["allowed_nodes"]
            self.bfs_request.add_node(node, demand=demand, ntype=ntype, allowed_nodes=allowed)
        self._add_bfs_edges_and_initialize_exploration_queue_with_leaf_nodes()

    def _add_bfs_edges_and_initialize_exploration_queue_with_leaf_nodes(self):
        self._visited = set()
        queue = deque([self.root])
        self._nodes_to_explore = deque()

        while queue:
            current_node = queue.popleft()
            is_leaf = True
            for out_neighbor in self.original_request.get_out_neighbors(current_node):
                if out_neighbor in self._visited:
                    continue
                is_leaf = False
                self._add_edge_to_bfs_request((current_node, out_neighbor), False)
                if out_neighbor not in queue:
                    queue.append(out_neighbor)
            for in_neighbor in self.original_request.get_in_neighbors(current_node):
                if in_neighbor in self._visited:
                    continue
                edge = (in_neighbor, current_node)
                self._add_edge_to_bfs_request(edge, True)
                is_leaf = False
                if in_neighbor not in queue:
                    queue.append(in_neighbor)
            if is_leaf and current_node not in self._nodes_to_explore:
                self._nodes_to_explore.append(current_node)
            self._visited.add(current_node)
        for node in self.original_request.nodes:
            if node not in self._visited:
                msg = "Request graph may have multiple components: Node {} was not visited by bfs.\n{}".format(node, self.original_request)
                raise ExtendedCactusGraphError(msg)

    def _add_edge_to_bfs_request(self, edge, is_reversed):
        if is_reversed:
            self.reversed_request_edges.add(edge)
            tail, head = edge
            new_edge = head, tail
        else:
            new_edge = edge
        new_tail, new_head = new_edge
        # print "BFS:   ", new_edge
        demand = self.original_request.edge[edge]["demand"]
        self.bfs_request.add_edge(new_tail, new_head, demand)

    def _request_decomposition_into_paths_and_cycles(self):
        self._visited = set()
        # print "Exploration Queue:", self._nodes_to_explore
        while self._nodes_to_explore:
            # print "   ", self._visited, "\n"
            current_node = self._nodes_to_explore.popleft()
            # print current_node, self._nodes_to_explore
            if current_node == self.root:
                continue  # ignore the root
            if current_node in self._visited:
                continue
            parents = self.bfs_request.in_neighbors[current_node]
            if len(parents) == 0:
                raise ExtendedCactusGraphError("Request graph may have multiple components: Node {} has no parent".format(current_node))
            elif len(parents) == 1:
                # print "    is_path"
                parent = parents[0]
                self._paths.append([(parent, current_node)])
                self._visited.add(current_node)
                if self._should_add_to_exploration_queue(parent):
                    self._nodes_to_explore.append(parent)
            elif len(parents) == 2:
                cycle = self._extract_cycle(current_node)
                self._mark_cycle_nodes_visited_and_extract_branch_nodes(cycle)
                self._visited.add(current_node)
                self._cycles.append(cycle)
                # add the top node of the cycle to the exploration queue:
                cycle_source = cycle[0][0][0]
                # print "    is_cycle with source", cycle_source
                # if cycle_source != self.root:
                if self._should_add_to_exploration_queue(cycle_source):
                    self._nodes_to_explore.append(cycle_source)
            else:
                raise ExtendedCactusGraphError("Request graph may not be a cactus graph: Node {} has parents {}".format(current_node, parents))

    def _should_add_to_exploration_queue(self, i):
        return (
            i not in self._visited and
            i not in self._nodes_to_explore and
            not self._has_unvisited_children(i) and
            i != self.root
        )

    def _has_unvisited_children(self, i):
        return any(j not in self._visited for j in self.bfs_request.out_neighbors[i])

    def _extract_cycle(self, sink_node):
        """
        Starting from a known sink node with exactly 2 parents, this function moves up towards the root
        along both branches until it finds a common ancestor. It returns a tuple of two edge lists, which
        correspond to the two branches of the cycle.
        
        :param sink_node: Node. This should be a sink node of a cycle, ie it has two parents corresponding to the left & right branch
        :return:
        """
        parents = self.bfs_request.in_neighbors[sink_node]
        if len(parents) != 2:
            raise ExtendedCactusGraphError("Expected a sink node with exactly two parents!")
        left_node = parents[0]
        right_node = parents[1]
        # represent each branch by the last explored node, a list of encountered edges,
        # and a dictionary mapping each potential common ancestor to the length of the edge list at that position
        current_node, current_branch, current_branch_visited = (left_node, [(left_node, sink_node)], {})
        other_node, other_branch, other_branch_visited = (right_node, [(right_node, sink_node)], {})
        while current_node not in other_branch_visited:
            switch_branches = False
            current_branch_visited[current_node] = len(current_branch)
            if current_node == self.root:
                switch_branches = True  # if current_node is the root, it is definitely the common ancestor!
            else:
                parents = self.bfs_request.in_neighbors[current_node]
                parent = parents[0]
                current_branch.append((parent, current_node))
                if self._has_unvisited_children(current_node):  # this may be a candidate for a common ancestor => switch to other branch
                    switch_branches = True
                current_node = parent
            if switch_branches:
                current_node, other_node = other_node, current_node
                current_branch, other_branch = other_branch, current_branch
                current_branch_visited, other_branch_visited = other_branch_visited, current_branch_visited
        other_branch = other_branch[:other_branch_visited[current_node]]  # reduce the other branch to the edges leading up to the common ancestor
        current_branch.reverse()
        other_branch.reverse()
        cycle = (current_branch, other_branch)
        return cycle

    def _mark_cycle_nodes_visited_and_extract_branch_nodes(self, cycle):
        start = cycle[0][0][0]
        end = cycle[0][-1][1]
        for branch in cycle:
            for ij in branch:
                for node in ij:
                    if node != start:
                        self._visited.add(node)
                        if node != end and len(self.bfs_request.get_out_neighbors(node)) > 1:
                            self.cycle_branch_nodes.add(node)

    def _make_reversed_substrate(self):
        """
        Generate a copy of the substrate with all edges reversed. All default substrate
        node & edge properties are preserved
        :return:
        """
        reversed_substrate = datamodel.Substrate("{}_reversed".format(self._original_substrate.name))
        for node in self._original_substrate.nodes:
            reversed_substrate.add_node(node,
                                        self._original_substrate.node[node]["supported_types"],
                                        self._original_substrate.node[node]["capacity"],
                                        self._original_substrate.node[node]["cost"])
            reversed_substrate.node[node] = self._original_substrate.node[node]
        for tail, head in self._original_substrate.edges:
            original_edge_properties = self._original_substrate.edge[(tail, head)]
            reversed_substrate.add_edge(head, tail,
                                        capacity=original_edge_properties["capacity"],
                                        cost=original_edge_properties["cost"],
                                        bidirected=False)
            # copy any additional entries of the substrate's edge dict:
            for key, value in self._original_substrate.edge[(tail, head)].iteritems():
                if key not in reversed_substrate.edge[(head, tail)]:
                    reversed_substrate.edge[(head, tail)][key] = value
        return reversed_substrate

    def _generate_extended_paths(self):
        for path in self._paths:
            start_node = path[0][0]
            end_node = path[-1][1]
            ext_path = ExtendedGraphPath(start_node=start_node, end_node=end_node, original_path=path, extended_path={}, extended_edges=set(), extended_nodes=set())
            valid_cycle_source_nodes = self._substrate_x.get_valid_nodes(
                self.original_request.get_type(start_node), self.original_request.get_node_demand(start_node)
            )
            connections_from_previous_layer = {u: self._add_and_get_super_source_node(start_node, u)
                                               for u in self.original_request.get_allowed_nodes(start_node)
                                               if u in valid_cycle_source_nodes}

            for edge in path:
                connections_from_previous_layer = self._make_substrate_copy_for_layer(edge, connections_from_previous_layer, ext_path)
            ext_path.extended_path[end_node] = []
            for u in connections_from_previous_layer:
                sink = self._add_and_get_super_sink_node(end_node, u)
                previous_layer_u = connections_from_previous_layer[u]
                self.add_edge(previous_layer_u, sink, bidirected=False, request_node=end_node, substrate_node=u)
                ext_path.extended_path[end_node].append((previous_layer_u, sink))
                ext_path.extended_edges.add((previous_layer_u, sink))
            self.ecg_paths.append(ext_path)

    def _generate_extended_cycles(self):
        for cycle in self._cycles:
            cycle_source = cycle[0][0][0]  # this looks confusing, but allows having more than 2 branches later on...
            cycle_target = cycle[0][-1][1]

            target_type = self.original_request.get_type(cycle_target)
            target_demand = self.original_request.get_node_demand(cycle_target)
            valid_cycle_target_node = self._substrate_x.get_valid_nodes(target_type, target_demand)
            cycle_out_nodes = set(self.original_request.get_allowed_nodes(cycle_target)).intersection(valid_cycle_target_node)

            # Sanity check: Ensure that all cycle branches end in same node!
            if any(cycle_source != b[0][0] or cycle_target != b[-1][1] for b in cycle):
                raise ExtendedCactusGraphError("Left and Right branch do not match in all branches:\n\t{}".format("\n\t".join(str(b) for b in cycle)))

            source_type = self.original_request.get_type(cycle_source)
            source_demand = self.original_request.get_node_demand(cycle_source)
            valid_cycle_source_nodes = self._substrate_x.get_valid_nodes(source_type, source_demand)

            cycle_source_nodes = {}
            # Obtain all possible starting nodes
            for u in self.original_request.get_allowed_nodes(cycle_source):
                if u in valid_cycle_source_nodes:
                    cycle_source_nodes[u] = self._add_and_get_super_source_node(cycle_source, u)

            ecg_branches = []
            for branch in cycle:
                ecg_branch = {}
                for w in cycle_out_nodes:
                    path = ExtendedGraphPath(start_node=cycle_source,
                                             end_node=cycle_target,
                                             original_path=branch,
                                             extended_path={},
                                             extended_edges=set(),
                                             extended_nodes=set())
                    connections_from_previous_layer = {k: v for k, v in cycle_source_nodes.iteritems()}
                    for edge in branch:
                        connections_from_previous_layer = self._make_substrate_copy_for_layer(edge,
                                                                                              connections_from_previous_layer,
                                                                                              path,
                                                                                              cycle_target_node=w)
                    if w not in connections_from_previous_layer:
                        raise ExtendedCactusGraphError("Sanity Check: Last branch layer should have connection to exit node.")
                    w_last_layer = connections_from_previous_layer[w]
                    w_sink_node = self._add_and_get_super_sink_node(cycle_target, w)
                    self.add_edge(w_last_layer, w_sink_node, bidirected=False, request_node=cycle_target, substrate_node=w)
                    if cycle_target not in path.extended_path:
                        path.extended_path[cycle_target] = []
                    ext_edge = (w_last_layer, w_sink_node)
                    path.extended_path[cycle_target].append(ext_edge)
                    path.extended_edges.add(ext_edge)
                    ecg_branch[w] = path
                ecg_branches.append(ecg_branch)
            ecg_cycle = ExtendedGraphCycle(start_node=cycle_source,
                                           end_node=cycle_target,
                                           original_branches=cycle,
                                           ext_graph_branches=ecg_branches)
            self.ecg_cycles.append(ecg_cycle)

    def _make_substrate_copy_for_layer(self, ij, connections_from_previous_layer, extended_path, cycle_target_node=None):
        """
        A copy of the substrate is associated with each request edge. This function generates such substrate copies
        for path and cycle layers.

        :param ij: request edge associated with the layer
        :param connections_from_previous_layer: dictionary: (substrate node u -> super node name of node u in previous layer). The nodes in the dictionary will be connected to the corresponding nodes in the new layer.
        :param cycle_target_node: IF this is a layer within a cycle, provide the target node associated with the path! If this is None, the edge is assumed to lie on a path.
        :return: A dictionary containing the super nodes that need to be connected to the next layer
        """
        sub = self._substrate_x
        i, j = ij
        ij_original_orientation = ij
        is_reversed_layer = (j, i) in self.reversed_request_edges  # Note: self.reversed_request_edges stores the edges in their *original* orientation
        if is_reversed_layer:
            ij_original_orientation = (j, i)
            sub = self._rev_substrate_x
        connections_to_next_layer = dict()

        j_type = self.original_request.get_type(j)
        allowed_nodes_j = set(self.original_request.get_allowed_nodes(j))
        valid_substrate_nodes = sub.get_valid_nodes(j_type, self.original_request.get_node_demand(j))

        out_nodes = allowed_nodes_j.intersection(valid_substrate_nodes)

        valid_substrate_edges = sub.get_valid_edges(self.original_request.get_edge_demand(ij_original_orientation))
        if is_reversed_layer:
            allowed_edges = self.original_request.get_allowed_edges(ij_original_orientation)
            if allowed_edges is not None:
                allowed_edges = {(v, u) for (u, v) in allowed_edges}
        else:
            allowed_edges = self.original_request.get_allowed_edges(ij_original_orientation)
        if allowed_edges is not None:
            valid_substrate_edges = valid_substrate_edges.intersection(allowed_edges)

        for u, previous_layer_u in connections_from_previous_layer.iteritems():  # add connections from the previous layer!
            u_layer = self._add_and_get_layer_node(ij, u, cycle_target_substrate_node=cycle_target_node)
            self.add_edge(previous_layer_u, u_layer, bidirected=False, request_node=i, substrate_node=u)
            extended_path.extended_nodes.add(u_layer)
            extended_path.extended_edges.add((previous_layer_u, u_layer))

            # remember that this edge corresponds to the mapping of request node i
            if i not in extended_path.extended_path:
                extended_path.extended_path[i] = []
            extended_path.extended_path[i].append((previous_layer_u, u_layer))

        for u in out_nodes:
            u_layer = self._add_and_get_layer_node(ij, u, cycle_target_substrate_node=cycle_target_node)
            extended_path.extended_nodes.add(u_layer)
            connections_to_next_layer[u] = u_layer

        extended_path.extended_path[ij] = []
        for uv in valid_substrate_edges:
            u, v = uv
            u_layer = self._add_and_get_layer_node(ij, u, cycle_target_substrate_node=cycle_target_node)
            v_layer = self._add_and_get_layer_node(ij, v, cycle_target_substrate_node=cycle_target_node)
            self.add_edge(u_layer, v_layer, bidirected=False, substrate_edge=(u, v), request_edge=ij)
            extended_path.extended_nodes.add(u_layer)
            extended_path.extended_nodes.add(v_layer)
            extended_path.extended_path[ij].append((u_layer, v_layer))
            extended_path.extended_edges.add((u_layer, v_layer))
        return connections_to_next_layer

    def _add_and_get_super_source_node(self, i, u):
        if i not in self.source_nodes:
            self.source_nodes[i] = {}
        new_node = ExtendedCactusGraph._super_node_name(i, u, "source")
        if u not in self.source_nodes[i]:
            self.source_nodes[i][u] = new_node
            self.add_node(new_node, request_node=i, substrate_node=u)
        return new_node

    def _add_and_get_super_sink_node(self, i, u):
        if i not in self.sink_nodes:
            self.sink_nodes[i] = {}
        new_node = ExtendedCactusGraph._super_node_name(i, u, "sink")
        if u not in self.sink_nodes[i]:
            self.sink_nodes[i][u] = new_node
            self.add_node(new_node, request_node=i, substrate_node=u)
        return new_node

    def _add_and_get_layer_node(self, edge, u, cycle_target_substrate_node=None):
        if cycle_target_substrate_node is None:
            return self._add_and_get_path_layer_node(edge, u)
        elif cycle_target_substrate_node in self._original_substrate.nodes:
            return self._add_and_get_cycle_layer_node(edge, u, cycle_target_substrate_node)
        else:
            raise ExtendedCactusGraphError("Invalid arguments: {} {} {}".format(edge, u, cycle_target_substrate_node))

    def _add_and_get_path_layer_node(self, ij, u):
        if ij in self.cycle_layer_nodes:
            msg = "Tried adding path edge {}, which is already part of a cycle: {}\n{}\n\n{}".format(
                ij,
                self.cycle_layer_nodes[ij],
                self.original_request,
                util.get_graph_viz_string(self.bfs_request))
            raise ExtendedCactusGraphError(msg)
        if ij not in self.path_layer_nodes:
            self.path_layer_nodes[ij] = {}
        new_node = ExtendedCactusGraph._super_node_name(ij, u, "layer")
        if u not in self.path_layer_nodes[ij]:
            self.add_node(new_node, request_edge=ij, substrate_node=u)
            self.path_layer_nodes[ij][u] = new_node
            self.layer_nodes.add(new_node)
        return new_node

    def _add_and_get_cycle_layer_node(self, ij, u, cycle_target_substrate_node):
        if ij in self.path_layer_nodes:
            msg = "Tried adding cycle edge {}, which is already part of a path: {}\n{}\n\n{}".format(
                ij,
                self.path_layer_nodes[ij],
                self.original_request,
                util.get_graph_viz_string(self.bfs_request))
            raise ExtendedCactusGraphError(msg)
        if ij not in self.cycle_layer_nodes:
            self.cycle_layer_nodes[ij] = {}
        if u not in self.cycle_layer_nodes[ij]:
            self.cycle_layer_nodes[ij][u] = {}
        new_node = ExtendedCactusGraph._super_node_name(ij, u, "layer_cycle", branch_substrate_node=cycle_target_substrate_node)
        if cycle_target_substrate_node not in self.cycle_layer_nodes[ij][u]:
            self.cycle_layer_nodes[ij][u][cycle_target_substrate_node] = new_node
            self.add_node(new_node, request_edge=ij, substrate_node=u, branch_substrate_node=cycle_target_substrate_node)
            self.layer_nodes.add(new_node)
        return new_node

    def _initialize_reverse_lookup(self):
        self._reverse_lookup_extended_nodes = {}

        for i in self.source_nodes:
            for u, u_i in self.source_nodes[i].iteritems():
                if u_i in self._reverse_lookup_extended_nodes:
                    raise ExtendedCactusGraphError("Sanity Check!")
                self._reverse_lookup_extended_nodes[u_i] = (i, u)

        for i in self.sink_nodes:
            for u, u_i in self.sink_nodes[i].iteritems():
                if u_i in self._reverse_lookup_extended_nodes:
                    raise ExtendedCactusGraphError("Sanity Check!")
                self._reverse_lookup_extended_nodes[u_i] = (i, u)

        for ij in self.path_layer_nodes:
            for u, u_ij in self.path_layer_nodes[ij].iteritems():
                if u_ij in self._reverse_lookup_extended_nodes:
                    raise ExtendedCactusGraphError("Sanity Check!")
                self._reverse_lookup_extended_nodes[u_ij] = (ij, u)

        for ij in self.cycle_layer_nodes:
            for u in self.cycle_layer_nodes[ij]:
                for w, u_ij_w in self.cycle_layer_nodes[ij][u].iteritems():
                    if u_ij_w in self._reverse_lookup_extended_nodes:
                        raise ExtendedCactusGraphError("Sanity Check!")
                    self._reverse_lookup_extended_nodes[u_ij_w] = (ij, u, w)

    @staticmethod
    def _super_node_name(request_key, substrate_node, node_type, branch_substrate_node=None):
        """
        :param request_key: request node (for source/sink nodes) or edge (for layer nodes) associated with the new node
        :param substrate_node: substrate node associated with the new node
        :param node_type: "source", "sink", "layer" or "layer_cycle" are valid.
        :param branch_substrate_node: the substrate node identifying the branch to which the layer node belongs
        :return: the super node's name as a string
        """
        if node_type == "source":
            return "{}_[{}]_+".format(substrate_node, request_key)
        elif node_type == "sink":
            return "{}_[{}]_-".format(substrate_node, request_key)
        elif node_type == "layer":
            i, j = request_key
            return "{}_[{}{}]".format(substrate_node, i, j)
        elif node_type == "layer_cycle":
            i, j = request_key
            if branch_substrate_node is None:
                raise ExtendedCactusGraphError("Need to provide a substrate node for the cycle's target when adding a cycle layer node!")
            return "{}_[{}{}]_[{}]".format(substrate_node, i, j, branch_substrate_node)
        else:
            raise ExtendedCactusGraphError("Invalid super node type {}!".format(node_type))
