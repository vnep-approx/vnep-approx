from .experimental_label_optimization import super_duper_algorithm2 as optimize_bag, size as bag_size, residual as bag_residual, remove_subsets, format_label_sets, has_cycle

from alib import datamodel, util
import random
from .commutativity_model import CommutativityLabels


def generate_random_request_graph(size, prob, req_name="test_req"):
    req = datamodel.Request(req_name)
    for i in range(1, size + 1):
        node = "{}".format(i)
        neighbors = []
        for j in req.nodes:
            if random.random() <= prob:
                neighbors.append(j)
        req.add_node(node, 1.0, "t1", ["u"])
        for j in neighbors:
            req.add_edge(j, node, 1.0)

    visited = set()
    all_nodes = set(req.nodes)
    while visited != all_nodes:
        start_node = (all_nodes - visited).pop()
        if visited:
            req.add_edge(random.choice(list(visited)), start_node, 1.0)
        visited.add(start_node)
        stack = [start_node]
        while stack:
            i = stack.pop()
            neighbors = req.get_in_neighbors(i) + req.get_out_neighbors(i)
            for j in neighbors:
                if j in visited:
                    continue
                else:
                    stack.append(j)
                    visited.add(j)
    req.graph["root"] = random.choice(list(req.nodes))
    return req


def generate_random_acyclic_orientation(req):
    oriented = datamodel.Request(req.name + "_oriented")
    shuffle_nodes = list(req.nodes)
    random.shuffle(shuffle_nodes)
    # print shuffle_nodes
    root = "R"
    oriented.add_node(root, 1.0, "t1", ("u",))
    oriented.graph["root"] = root
    for i in shuffle_nodes:
        oriented.add_node(i, 1.0, "t1", ("u",))

    visited = set()
    for i in shuffle_nodes:
        visited.add(i)
        is_root = True
        for j in req.get_in_neighbors(i) + req.get_out_neighbors(i):
            if j in visited:
                is_root = False
                continue
            oriented.add_edge(i, j, 1.0)
        if is_root:
            oriented.add_edge(root, i, 1.0)
    # print util.get_graph_viz_string(oriented), "\n\n"
    # print util.get_graph_viz_string(req), "\n\n"
    return oriented


ITERATIONS = 500
ORIENTATION_ITERATIONS = 1000


def main():
    for iteration in range(ITERATIONS):
        req = generate_random_request_graph(15, 0.3)
        iteration_results = {}

        best_orientation = None
        best_max_bagsize_optimized = float("inf")
        best_max_bagsize = None
        oriented_requests = []

        for orientation_iteration in range(ORIENTATION_ITERATIONS):
            iteration_results[orientation_iteration] = {}
            oriented_req = generate_random_acyclic_orientation(req)
            oriented_requests.append(oriented_req)
            labels = CommutativityLabels.create_labels(oriented_req)

            max_bagsize = -1
            max_bagsize_optimized = -1

            for i in oriented_req.nodes:
                i_label_bags = labels.label_bags[i]
                iteration_results[orientation_iteration][i] = {}
                for bag_key, bag_edges in i_label_bags.items():
                    max_bagsize = max(max_bagsize, len(bag_key))
                    edge_label_sets = {frozenset(labels.get_edge_labels(ij)) for ij in bag_edges}
                    factored_labels, _ = optimize_bag(edge_label_sets)
                    without_subsets = remove_subsets(bag_residual(edge_label_sets, factored_labels))
                    size = bag_size(without_subsets) + len(factored_labels)
                    max_bagsize_optimized = max(max_bagsize_optimized, size)
                    iteration_results[orientation_iteration][i][bag_key] = (factored_labels, without_subsets, size)
                    # if factored_labels:
                    #     print "\n\n"
                    #     print format_label_sets(edge_label_sets), len(bag_key)
                    #     print "_".join(factored_labels), format_label_sets(without_subsets), size
            # print max_bagsize, max_bagsize_optimized
            if max_bagsize_optimized < best_max_bagsize_optimized:
                best_max_bagsize_optimized = max_bagsize_optimized
                best_max_bagsize = max_bagsize
                best_orientation = orientation_iteration

        best_oriented_req = oriented_requests[best_orientation]
        nodes_text = 'label="max_bagsize={}\\nmax_bagsize_optimized={}";\n'.format(best_max_bagsize, best_max_bagsize_optimized)
        has_factor = False
        for i in best_oriented_req.nodes:
            i_has_factor = False
            nodes_text += '"{0}" [label="{0}'.format(i)
            for bag_key in iteration_results[best_orientation][i]:
                factored_labels, without_subsets, size = iteration_results[best_orientation][i][bag_key]
                if factored_labels:
                    i_has_factor = True
                    has_factor = True
                print iteration, "_".join(factored_labels), format_label_sets(without_subsets), size
                if bag_key:
                    nodes_text += '\\nb={}/{} f={} ls={} s={}'.format("_".join(bag_key), len(bag_key), "_".join(factored_labels), format_label_sets(without_subsets), str(size))

            if i_has_factor:
                nodes_text += '",color="blue", penwidth="3'
            nodes_text += '"];\n'
        if has_factor:
            with open("out/{}.gv".format(iteration), "w") as f:
                f.write(util.get_graph_viz_string(
                    best_oriented_req,
                    get_edge_style=lambda (i, j): 'style="dashed"' if i == "R" else ""
                ).replace("digraph test_req_oriented {", "digraph test_req_oriented {\n" + nodes_text))
            with open("out/{}_orig.gv".format(iteration), "w") as f:
                f.write(util.get_graph_viz_string(req))


if __name__ == "__main__":
    main()
