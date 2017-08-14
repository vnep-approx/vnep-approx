from collections import Counter
from functools import partial

import copy
import math
from alib import datamodel, util

import experimental_multiroot
import tqdm
import itertools
import random
from heapq import heappush, heappop


def orient_request(req, node_order):
    oriented = datamodel.Request(req.name + "_oriented")
    # print shuffle_nodes
    root = "R"
    oriented.add_node(root, 1.0, "t1", ("u",))
    oriented.graph["root"] = root
    for i in node_order:
        oriented.add_node(i, 1.0, "t1", ("u",))

    visited = set()
    for i in node_order:
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


def orientation_id(req):
    out_neighbors = req.get_out_neighbors
    in_neighbors = req.get_in_neighbors
    result = []
    result_append = result.append
    grey = set()
    grey_add = grey.add
    black = set()
    black_add = black.add
    queue = [req.graph["root"]]
    while queue:
        i = heappop(queue)
        black_add(i)
        result_append(i)
        for j in out_neighbors(i):
            parents_visited = True
            for p in in_neighbors(j):
                if p not in black:
                    parents_visited = False
                    break
            if parents_visited and j not in grey:
                heappush(queue, j)
                grey_add(j)
    return tuple(result)


def has_roots(req):
    return any(not req.get_in_neighbors(i) for i in req.nodes)

def has_cycle(req):
    stack = [i for i in req.nodes if not req.get_in_neighbors(i)]
    grey = set(stack)
    black = set(stack)
    while stack:
        i = stack.pop()
        for j in req.get_out_neighbors(i):
            if j in grey:
                return True
            grey.add(j)
            stack.append(j)
    return False


def paper_algorithm(req):
    """
    Conte, Alessio, et al. "Listing acyclic orientations of graphs with single and multiple sources."
    Latin American Symposium on Theoretical Informatics. Springer, Berlin, Heidelberg, 2016.

    :param req:
    :return:
    """
    for oriented_req in tqdm.tqdm(alg_3(req, datamodel.Request("foo"), 0)):
        oriented_req.add_node("R", 1.0, "t1", ["u"])
        oriented_req.graph["root"] = "R"
        for node in oriented_req.nodes:
            if oriented_req.get_in_neighbors(node):
                continue
            if node == "R":
                continue
            oriented_req.add_edge("R", node, 1.0)
        yield oriented_req


def alg_3(original_graph, partial_orientation, i):
    if i >= len(original_graph.nodes):
        yield partial_orientation
    else:
        for Z in alg_4(original_graph, partial_orientation, i):
            orientation_extension = copy.deepcopy(partial_orientation)
            orientation_extension.add_node(sorted(original_graph.nodes)[i], 1.0,  "t1", ["u"])
            for ij in Z:
                orientation_extension.add_edge(ij[0], ij[1], 1.0)
            for blub in alg_3(original_graph, orientation_extension, i + 1):
                yield blub


def alg_4(original_graph, partial_orientation, i):
    for edges in generate(original_graph, partial_orientation, i, frozenset(), 0, frozenset(), frozenset()):
        yield edges


def generate(original_graph, partial_orientation, i, W, j, R, B):
    nodes = sorted(original_graph.nodes)
    v_i = nodes[i]
    # x_j = nodes[j]
    neighbors = set(original_graph.get_in_neighbors(v_i) + original_graph.get_out_neighbors(v_i))
    k = len(neighbors & set(nodes[:i]))
    if j >= k:
        yield W
        return
    x_j = sorted(neighbors)[j]

    X_i = {(x, v_i) for x in neighbors}
    Y_i = {(v_i, x) for x in neighbors}
    r, b = color_reachable(partial_orientation, v_i, W)
    if W & Y_i:
        R = R | r
    if x_j not in R:
        for edges in generate(original_graph, partial_orientation, i, W | {(x_j, v_i)}, j + 1, R, B):
            yield edges
    if W & X_i:
        B = B | b
    if x_j not in B:
        for edges in generate(original_graph, partial_orientation, i, W | {(v_i, x_j)}, j + 1, R, B):
            yield edges


def color_reachable(partial_orientation, v_i, W):
    W_out = {b for a, b in W if a == v_i}
    W_in = {a for a, b in W if b == v_i}
    red = {v_i}
    stack = list(W_out)
    while stack:
        node = stack.pop()
        red.add(node)
        for other in partial_orientation.get_out_neighbors(node):
            if other not in red:
                stack.append(other)

    black = {v_i}
    stack = list(W_in)
    while stack:
        node = stack.pop()
        black.add(node)
        for other in partial_orientation.get_in_neighbors(node):
            if other not in black:
                stack.append(other)

    return frozenset(red), frozenset(black)


def brute_force(req):
    nodes = sorted(req.nodes)
    permutation_count = math.factorial(len(nodes))
    for order in tqdm.tqdm(itertools.permutations(nodes), total=permutation_count):
        yield orient_request(req, order)

def count_orientations(req, alg):
    count = Counter()
    for oriented_req in alg(req):
        id = orientation_id(oriented_req)
        count[id] += 1
    return count


def compare(algorithms):
    req = experimental_multiroot.generate_random_request_graph(8, 0.5)
    count_dict = {}
    for alg in algorithms:
        print "\n\nEvaluating ", alg.__name__
        count_dict[alg] = count_orientations(req, alg)
        evaluate(count_dict[alg])
    keys = list(set(frozenset(k for k in counter.keys()) for counter in count_dict.values()))

    print "There were {} outputs".format(len(keys))
    # for duplicate in keys[0].symmetric_difference(keys[1]):
    #     print "_".join(duplicate)

    print util.get_graph_viz_string(req)
    print "BRUTE FORCE - PAPER"
    for id in sorted(set(count_dict[brute_force]) - set(count_dict[paper_algorithm])):
        print "_".join(id)

    print "\n\nPAPER - BRUTE FORCE"
    for id in sorted(set(count_dict[paper_algorithm]) - set(count_dict[brute_force])):
        print "_".join(id)
    print



def evaluate(count):
    s = 0
    m = len(str(count.most_common(1)[0][1]))
    permutation_count = sum(count.values())
    for id, c in sorted(count.most_common(), key=lambda x: (-x[1], x[0])):
        s += c
        print ("{} {:%d} {:7.2%%} {:7.2%%}" % m).format("_".join(id), c, float(c) / permutation_count, float(s) / permutation_count)
    print "Unique orientations: {} / {} ({} %)".format(
        len(count),
        permutation_count,
        (100.0 * len(count)) / permutation_count,
    )


def main():
    random.seed(0)
    compare([paper_algorithm, brute_force])


if __name__ == '__main__':
    main()
