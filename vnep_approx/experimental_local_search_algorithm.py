import random
from collections import Counter

from alib import datamodel

from experimental_label_optimization import get_bag_width
from experimental_multiroot import generate_random_request_graph, generate_random_acyclic_orientation
from experimental_orientation import paper_algorithm, orientation_id

random.seed(0)




def find_optimal_size(req):
    min_size = float('inf')
    counter = Counter()
    for orientation in paper_algorithm(req):
        size = get_bag_width(orientation)
        min_size = min(min_size, size)
        counter[size] += 1
    for i, size in sorted(counter.items()):
        print i, size
    return min_size


def is_dag(req):
    visited = set()
    queue = [req.graph["root"]]
    while queue:
        i = queue.pop()
        for ij in req.get_out_edges(i):
            j = ij[1]
            visited.add(ij)

            all_in_edges_visited = True
            for pj in req.get_in_edges(j):
                if pj not in visited:
                    all_in_edges_visited = False
                    break
            if all_in_edges_visited:
                queue.append(j)

    return req.edges == visited


def neighbor_orientations(oriented):
    for ij in oriented.edges:
        if "R" in ij:
            continue
        neighbor = datamodel.Request(oriented.name)
        for i in oriented.nodes:
            if i == "R":
                continue
            neighbor.add_node(i, 1.0, "t1", ["u"])
        for edge in oriented.edges:
            if "R" in edge:
                continue
            if edge == ij:
                continue
            tail, head = edge
            neighbor.add_edge(tail, head, 1.0)

        neighbor.add_edge(ij[1], ij[0], 1.0)
        neighbor.add_node("R", 1.0, "t1", ["u"])
        neighbor.graph["root"] = "R"

        for i in neighbor.nodes:
            if i == "R":
                continue
            if not neighbor.get_in_neighbors(i):
                neighbor.add_edge("R", i, 1.0)

        if is_dag(neighbor):
            yield neighbor


def local_search(req, search_depth=1000, cache={}):
    """
    Local search to find a "good" orientation:
      - start with a random orientation
      - consider all neighbor orientations obtained by reversing a single edge
      - continue with the neighbor orientations with the smallest bag width
    """
    best_orientation = generate_random_acyclic_orientation(req)
    min_size = get_bag_width(best_orientation)

    keep_looking = True
    cache_local = {}
    remaining_depth = search_depth
    while keep_looking and remaining_depth > 0:
        remaining_depth -= 1
        keep_looking = False
        print "    Local search, current best: ", min_size
        for neighbor in neighbor_orientations(best_orientation):
            id = orientation_id(neighbor)
            if id in cache_local:
                continue
            if id in cache:
                size = cache[id]
            else:
                size = get_bag_width(neighbor)
            cache[id] = size
            cache_local[id] = size
            # print "        neighbor", size
            if size <= min_size:
                if size < min_size:
                    remaining_depth = search_depth
                keep_looking = True
                best_orientation = neighbor
                min_size = size
    print len(cache_local), "calculations"
    print len(cache), "global calculations"
    return cache, cache_local


ITERATIONS = 500


def main():
    req = generate_random_request_graph(11, 0.4)
    opt_size = find_optimal_size(req)
    print "Optimal size: ", opt_size

    global_calculations = {}
    local_calculations = {}
    for iteration in range(ITERATIONS):
        global_calculations, local_calculations[iteration] = local_search(req, search_depth=10)
        local_optimum = min(local_calculations.values())

    sizes = Counter()
    calculations = Counter()
    for i, local_calculation in local_calculations.items():
        size = min(local_calculation.values())
        print "Iteration {} found local optimum of {} with {} calculations".format(
            i, size, len(local_calculation)
        )
        sizes[size] += 1
        calculations[len(local_calculation)] += 1
    print "Global number of calculations: {}".format(len(global_calculations))
    print "Optimal size: ", opt_size

    print "Local search size distribution:"
    for i in range(100):
        if i in sizes:
            print "   ", i, sizes[i]

    print "Local search effort distribution:"
    avg = 0.0
    for i, effort in sorted(calculations.most_common()):
        avg += i * effort
        if i in calculations:
            print "   ", i, calculations[i]
    avg /= sum(calculations.values())
    print "Average effort", avg

if __name__ == "__main__":
    main()
