import itertools
import os
import subprocess

from gurobipy import GRB, LinExpr

from alib import datamodel, modelcreator


########### Some datastructures to represent tree decompositions ###########


class UndirectedGraph(object):
    def __init__(self, name):
        self.name = name
        self.nodes = set()
        self.edges = set()

        # attribute dictionaries:
        self.graph = {}
        self.node = {}
        self.edge = {}
        self.neighbors = {}
        self.incident_edges = {}
        self.removed_tree_nodes = set()
        self.removed_contracted_nodes = set()

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

    def get_incident_edges(self, node):
        return self.incident_edges[node]

    def get_neighbors(self, node):
        neighbors = set().union(*self.get_incident_edges(node))
        return neighbors - {node}


class TreeDecomposition(UndirectedGraph):
    def __init__(self, name):
        super(TreeDecomposition, self).__init__(name)

        self.node_bag_dict = {}
        self.node_bags = set()
        self.req_node_map = {}

    def add_node(self, node, node_bag=None):
        if not node_bag:
            raise ValueError("Empty or unspecified node bag: {node_bag}")
        if not isinstance(node_bag, frozenset):
            raise ValueError("Expected node bag as frozenset: {node_bag}")
        super(TreeDecomposition, self).add_node(node)
        self.node_bags.add(node_bag)
        self.node_bag_dict[node] = node_bag
        for req_node in node_bag:
            self.req_node_map.setdefault(req_node, set()).add(node)

    def convert_to_arborescence(self, root_bag=None):
        if not self._verify_intersection_property() or not self._is_tree():
            raise ValueError("Cannot derive decomposition arborescence from invalid tree decomposition!")
        if root_bag is None:
            root_bag = next(iter(self.nodes))
        arborescence = TDArborescence(self.name, root_bag)
        q = {root_bag}
        arborescence.add_node(root_bag)
        visited = set()
        while q:
            bag = q.pop()
            visited.add(bag)
            for other in self.get_neighbors(bag):
                if other not in visited:
                    arborescence.add_node(other)
                    arborescence.add_edge(bag, other)
                    q.add(other)
        return arborescence

    def get_representative(self, *args):
        representatives = set(self.nodes)
        for i in args:
            representatives &= self.req_node_map[i]
        if not representatives:
            return None
        return sorted(representatives)[0]

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
        return set(self.req_node_map.keys()) == set(req.nodes)

    def _verify_all_edges_covered(self, req):
        for (i, j) in req.edges:
            # Check that there is some overlap in the sets of representative nodes:
            if not (self.req_node_map[i] & self.req_node_map[j]):
                return False
        return True

    def _verify_intersection_property(self):
        # Check that subtrees induced by each request node are connected
        for req_node, subtree_nodes in self.req_node_map.items():
            start_node = next(iter(subtree_nodes))
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

    def add_edge(self, tail, head, **kwargs):
        if self.get_in_edges(head):
            raise ValueError("Error: Arborescence must not contain confluences!")
        super(TDArborescence, self).add_edge(
            tail, head, bidirected=False
        )


########### Computing tree decompositions ###########


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

    def _convert_result_format_to_tree_decomposition(self, result):
        lines = result.split("\n")
        result = TreeDecomposition("{}_TD".format(self.graph.name))
        # TODO: Do we need to use the header line for something?
        for line in lines[1:]:
            line = [w.strip() for w in line.split() if w]
            if not line or line[0] == "c":  # ignore empty and comment lines
                continue
            elif line[0] == "b":
                bag_id = self._get_bagid(line[1])
                new_bag = frozenset([
                    self.map_numeric_id_to_nodes[i] for i in line[2:]
                ])
                result.add_node(bag_id, new_bag)
            else:
                assert len(line) == 2
                i, j = line
                result.add_edge(
                    self._get_bagid(i),
                    self._get_bagid(j),
                )
        return result

    def _get_bagid(self, numeric_id):
        return "bag_{}".format(numeric_id)


########### Modelcreator: Treewidth LP & Decomposition Algorithm ###########


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
                td_compute = TreeDecompositionComputation(req)
                self.tree_decompositions[req.name] = td_compute.compute_tree_decomposition()
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
            t_ij = td.get_representative(i, j)

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
