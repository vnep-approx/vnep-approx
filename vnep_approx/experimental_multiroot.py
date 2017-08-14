from experimental_label_optimization import (
    optimize_bag,
    optimize_bag_leaves_first,
    size as bag_size,
    residual as bag_residual,
    remove_subsets,
    format_label_sets,
    has_cycle
)

import cPickle
import multiprocessing as mp
import time
import tqdm

from alib import datamodel, util
import random
from vnep_approx.commutativity_model import CommutativityLabels

start_time = time.time()


def generate_random_request_graph(size, prob, req_name="test_req"):
    req = datamodel.Request(req_name)
    for i in range(1, size + 1):
        node = "{:02}".format(i)
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


ITERATIONS = 5000

ORIENTATION_ITERATIONS = 1000

def execute(iteration):
    req = generate_random_request_graph(12, 0.5, req_name="req_{}".format(iteration))
    best_max_bagsize, best_max_bagsize_optimized, best_orientation, iteration_results, best_oriented_req = find_best_random_orientation(req, ORIENTATION_ITERATIONS)
    evaluate_best_orientation(req, iteration, best_max_bagsize, best_max_bagsize_optimized, iteration_results[best_orientation], best_oriented_req)
    print "Completed iteration ", iteration


def evaluate_best_orientation(req, iteration, best_max_bagsize, best_max_bagsize_optimized, best_iteration_result, best_oriented_req, use_time_diff=True, write_original=True):
    all_factored_labels = set()
    for i in best_oriented_req.nodes:
        for bag_key in best_iteration_result[i]:
            factored_labels, without_subsets, size = best_iteration_result[i][bag_key]  #
            all_factored_labels |= factored_labels

    nodes_text = 'label="max_bagsize={}\\nmax_bagsize_optimized={}";\n'.format(best_max_bagsize, best_max_bagsize_optimized)
    has_factor = False
    is_interesting = False
    for i in best_oriented_req.nodes:
        i_has_factor = False
        i_is_interesting = False
        nodes_text += '"{0}" [label="{0}'.format(i)
        for bag_key in best_iteration_result[i]:
            factored_labels, without_subsets, size = best_iteration_result[i][bag_key]
            for factored_label in factored_labels:
                children = best_oriented_req.get_out_neighbors(factored_label)
                i_is_interesting |= bool(children) and all(child not in factored_labels for child in children)
                is_interesting |= i_is_interesting
                if i_is_interesting:
                    break

            if factored_labels:
                i_has_factor = True
                has_factor = True
            # print iteration, "_".join(factored_labels), format_label_sets(without_subsets), size
            if bag_key:
                nodes_text += '\\nb={}/{} f={} ls={} s={}'.format("_".join(bag_key), len(bag_key), "_".join(factored_labels), format_label_sets(without_subsets), str(size))

        if i_has_factor and i in all_factored_labels:
            nodes_text += '",color="red", penwidth="3'
        elif i_has_factor:
            nodes_text += '",color="blue", penwidth="3'
        elif i in all_factored_labels:
            nodes_text += '",color="darkgreen", penwidth="3'
        if i_is_interesting:
            nodes_text += '",peripheries="2'
        nodes_text += '"];\n'

    # if has_factor and is_interesting:
    prefix = ""
    if use_time_diff:
        time_diff = int(time.time() - start_time)
        prefix = "{:05}_".format(time_diff)
    with open("out/output/{}{}.gv".format(prefix, iteration), "w") as f:
        f.write(util.get_graph_viz_string(
            best_oriented_req,
            get_edge_style=lambda (i, j): 'style="dashed"' if i == "R" else ""
        ).replace("\n", "\n" + nodes_text, 1))
    with open("out/output/{}{}_oriented_req.pickle".format(prefix, iteration), "w") as f:
        cPickle.dump(best_oriented_req, f)
    if write_original:
        with open("out/output/{}{}_orig.gv".format(prefix, iteration), "w") as f:
            f.write(util.get_graph_viz_string(req))
        with open("out/output/{}{}_orig_req.pickle".format(prefix, iteration), "w") as f:
            cPickle.dump(req, f)


def find_best_random_orientation(req, iterations=1000):
    iteration_results = {}
    best_orientation = None
    best_max_bagsize_optimized = float("inf")
    best_max_bagsize = None
    oriented_requests = []

    for orientation_iteration in range(iterations):
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
    return best_max_bagsize, best_max_bagsize_optimized, best_orientation, iteration_results, oriented_requests[best_orientation]


def find_best_random_orientation_multiple(req, iterations=1000, optimize_bag_functions=(optimize_bag,)):
    iteration_results = {optimize_bag: {} for optimize_bag in optimize_bag_functions}
    best_orientation = None
    best_max_bagsize_optimized = float("inf")
    best_max_bagsize = None
    oriented_requests = []

    for orientation_iteration in range(iterations):
        oriented_req = generate_random_acyclic_orientation(req)

        oriented_requests.append(oriented_req)
        labels = CommutativityLabels.create_labels(oriented_req)

        sizes = {}
        sizes_optimized = {}
        for optimize_bag in optimize_bag_functions:
            iteration_results[optimize_bag][orientation_iteration] = result_dict = {}
            max_bagsize = -1
            max_bagsize_optimized = -1

            for i in oriented_req.nodes:
                i_label_bags = labels.label_bags[i]
                result_dict[i] = {}

                for bag_key, bag_edges in i_label_bags.items():
                    max_bagsize = max(max_bagsize, len(bag_key))
                    edge_label_sets = {frozenset(labels.get_edge_labels(ij)) for ij in bag_edges}
                    factored_labels, _ = optimize_bag(oriented_req, edge_label_sets)
                    without_subsets = remove_subsets(bag_residual(edge_label_sets, factored_labels))
                    size = bag_size(without_subsets) + len(factored_labels)
                    max_bagsize_optimized = max(max_bagsize_optimized, size)
                    result_dict[i][bag_key] = (factored_labels, without_subsets, size)

            sizes[optimize_bag] = max_bagsize
            sizes_optimized[optimize_bag] = max_bagsize_optimized

        # if len(set(sizes_optimized.values())) != 1:
        for optimize_bag in optimize_bag_functions:
            evaluate_best_orientation(
                req,
                # "{}_{}_{}_{}".format(req.name, orientation_iteration,
                #                      optimize_bag.__name__,
                #                      sizes_optimized[optimize_bag]),
                "{}_{}_{}".format(req.name, orientation_iteration,
                                     optimize_bag.__name__),
                sizes[optimize_bag],
                sizes_optimized[optimize_bag],
                iteration_results[optimize_bag][orientation_iteration],
                oriented_req,
                write_original=False
            )


def evaluate_interesting():
    with open("out/interesting/1128_req.pickle", "r") as f:
        req = cPickle.load(f)

    best_max_bagsize, best_max_bagsize_optimized, best_orientation, iteration_results, best_oriented_req = find_best_random_orientation(req, 10000)
    evaluate_best_orientation(req, 0, best_max_bagsize, best_max_bagsize_optimized, iteration_results[best_orientation], best_oriented_req)


def compare_bag_algorithms(iteration):
    req = generate_random_request_graph(15, 0.3, req_name="req_{}".format(iteration))

    def optimize_bag_bla( req, label_sets):
        return optimize_bag(label_sets)
    optimize_bag_bla.__name__ = "optimize_bag"
    find_best_random_orientation_multiple(req, ORIENTATION_ITERATIONS, optimize_bag_functions=(optimize_bag_bla, optimize_bag_leaves_first))
    print "Completed iteration ", iteration


def main():
    pool = mp.Pool()
    pool.map(execute, range(ITERATIONS))

def main2():
    pool = mp.Pool()
    pool.map(compare_bag_algorithms, range(ITERATIONS))

    # compare_bag_algorithms(0)

    # for i in tqdm.tqdm(range(ITERATIONS)):
    #     i = 11
    #     random.seed(i)
    #     compare_bag_algorithms(i)

    # with open("out/output/00825_req_782_259_optimize_bag_leaves_first_8_oriented_req.pickle", "r") as f:
    #     oriented_req = cPickle.load(f)
    # labels = CommutativityLabels.create_labels(oriented_req)
    # bag_edges = labels.label_bags["R"].values()[0]
    # optimize_bag({frozenset(labels.get_edge_labels(ij)) for ij in bag_edges})
    # # optimize_bag_leaves_first(oriented_req, {frozenset(labels.get_edge_labels(ij)) for ij in bag_edges})

if __name__ == "__main__":
    # main()
    main2()
    # evaluate_interesting()
