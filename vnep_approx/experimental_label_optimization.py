import heapq
from heapq import heappush, heappop
import string
import sys
import timeit
from collections import deque, namedtuple
from itertools import combinations, chain

import random

# random.seed(0)

sys.stderr = sys.stdout


def is_bag(label_sets):
    label_sets = set(label_sets)
    bag = set(label_sets.pop())
    while label_sets:
        flag = False
        for labels in set(label_sets):
            if labels & bag:
                bag.update(labels)
                label_sets.remove(labels)
                flag = True
        if not flag:
            return False
    return True


def generate_labels(num_labels=10, num_edges=6):
    # return set(map(frozenset, ["ab", "bc", "cd", "de", "ea"]))
    # return set(map(frozenset, ["abc", "bcd", "cde", "dea", "eab"]))
    # return set(map(frozenset, ["abcde", "def", "fabc"]))
    # return set(map(frozenset, ["abcde", "def", "fa", "fb", "fc"]))
    labels = string.ascii_lowercase[:num_labels]
    while True:
        label_sets = {frozenset(random.sample(labels, random.randint(2, num_labels))) for i in range(num_edges)}
        if is_bag(label_sets) and has_cycle(remove_subsets(label_sets)):
            break
    return label_sets


def size(label_sets):
    if not label_sets:
        return 0
    return max(map(len, label_sets))


def residual(label_sets, extract):
    return {labels - extract for labels in label_sets if labels - extract}


def remove_subsets(label_sets):
    return {l1 for l1 in label_sets if not any(l1 < l2 for l2 in label_sets)}


def has_cycle(label_sets):
    if not label_sets:
        return False
    not_visited = set(label_sets)
    while not_visited:
        queue = deque([not_visited.pop()])
        visited = set(queue)
        visited_edges = set()
        while queue:
            labels = queue.popleft()
            # print("labels", labels)
            for other_labels in label_sets:
                # print("other", other_labels)
                if other_labels == labels or not (labels & other_labels):
                    continue
                edge = frozenset([labels, other_labels])
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)
                if other_labels in visited:
                    return True
                queue.append(other_labels)
                visited.add(other_labels)
        not_visited -= visited
    return False


# assert has_cycle(set(map(frozenset, ["ab", "bc", "cd", "de", "ea"])))
# assert not has_cycle(remove_subsets(set(map(frozenset, ["ab", "a"]))))
# assert not has_cycle(set(map(frozenset, ["ab", "ab"])))
# assert not has_cycle(set(map(frozenset, ["ab", "ac"])))


def format_label_sets(label_sets):
    return "{" + ", ".join(sorted("_".join(sorted(x)) for x in label_sets)) + "}"


def main():
    label_sets = generate_labels()
    print "\033[1m{}\n".format(format_label_sets(label_sets))
    all_labels = set(chain(*label_sets))
    for factor in (x for k in range(len(all_labels) + 1) for x in combinations(sorted(all_labels), k)):
        factor_set = frozenset(factor)
        residual_label_sets = remove_subsets(residual(label_sets, factor_set))
        if not has_cycle(residual_label_sets):
            sys.stdout.write("\033[34m")
        else:
            sys.stdout.write("\033[31m")
        print ("{:%s}  {:1}  {}  {}" % len(all_labels)).format("".join(factor), has_cycle(residual_label_sets), len(factor_set) + size(residual_label_sets), format_label_sets(residual_label_sets))
    factor, iterations = super_duper_algorithm(label_sets)
    print factor, iterations


def main2():
    label_sets = generate_labels()
    print "\033[1m{}\033[0m\n".format(format_label_sets(label_sets))
    # all_labels = set(chain(*label_sets))
    # for factor in (x for k in range(len(all_labels) + 1) for x in combinations(sorted(all_labels), k)):
    #     factor_set = frozenset(factor)
    #     residual_label_sets = remove_subsets(residual(label_sets, factor_set))
    #     if not has_cycle(residual_label_sets):
    #         sys.stdout.write("\033[34m")
    #     else:
    #         sys.stdout.write("\033[31m")
    #     print ("{:%s}  {:1}  {}  {}" % len(all_labels)).format("".join(factor), has_cycle(residual_label_sets), len(factor_set) + size(residual_label_sets), format_label_sets(residual_label_sets))

    factor, iterations = slow_algorithm(label_sets)
    residual_label_sets = remove_subsets(residual(label_sets, factor))
    # print "\033[1m"
    print "slow: {:10} {:4} {} {}".format("".join(sorted(factor)), iterations, has_cycle(residual_label_sets), len(factor) + size(residual_label_sets), format_label_sets(residual_label_sets))
    print "time:", min(timeit.repeat("slow_algorithm(label_sets)", "from __main__ import slow_algorithm\nlabel_sets={!r}".format(label_sets), number=100))

    factor, iterations = super_duper_algorithm(label_sets)
    residual_label_sets = remove_subsets(residual(label_sets, factor))
    # print "\033[1m"
    print "fast: {:10} {:4} {} {}".format("".join(sorted(factor)), iterations, has_cycle(residual_label_sets), len(factor) + size(residual_label_sets), format_label_sets(residual_label_sets))
    print "time:", min(timeit.repeat("super_duper_algorithm(label_sets)", "from __main__ import super_duper_algorithm\nlabel_sets={!r}".format(label_sets), number=100))

    factor, iterations = super_duper_algorithm2(label_sets)
    residual_label_sets = remove_subsets(residual(label_sets, factor))
    print "fast: {:10} {:4} {} {}".format("".join(sorted(factor)), iterations, has_cycle(residual_label_sets), len(factor) + size(residual_label_sets), format_label_sets(residual_label_sets))
    print "time:", min(timeit.repeat("super_duper_algorithm2(label_sets)", "from __main__ import super_duper_algorithm2\nlabel_sets={!r}".format(label_sets), number=100))


def slow_algorithm(label_sets):
    best_valid_factor = None
    best_size = float("inf")
    all_labels = set(chain(*label_sets))
    i = 0
    for i, factor in enumerate(x for k in range(len(all_labels) + 1) for x in combinations(sorted(all_labels), k)):
        factor_set = frozenset(factor)
        residual_label_sets = remove_subsets(residual(label_sets, factor_set))
        if not has_cycle(residual_label_sets):
            s = len(factor_set) + size(residual_label_sets)
            if s < best_size:
                best_size = s
                best_valid_factor = factor_set
                # if not has_cycle(residual_label_sets):
                #     sys.stdout.write("\033[34m")
                # else:
                #     sys.stdout.write("\033[31m")
                # print ("{:%s}  {:1}  {}  {}" % len(all_labels)).format("".join(factor), has_cycle(residual_label_sets), len(factor_set) + size(residual_label_sets), format_label_sets(residual_label_sets))
    return best_valid_factor, i


def super_duper_algorithm(label_sets):
    all_labels = set(chain(*label_sets))
    residual_label_sets = remove_subsets(label_sets)
    queue = [(size(residual_label_sets), has_cycle(residual_label_sets), 0, frozenset())]
    # e = (size, has_cycle, factor)
    visited = set()
    i = 1
    while queue:
        # print "\033[0m", i, queue
        s, hc, lf, factor = heapq.heappop(queue)
        if not hc:
            return factor, i
        for l in all_labels - factor:
            extended_factor = factor | {l}
            if extended_factor in visited:
                continue
            visited.add(extended_factor)
            residual_label_sets = remove_subsets(residual(label_sets, extended_factor))
            heapq.heappush(queue, (len(extended_factor) + size(residual_label_sets), has_cycle(residual_label_sets), len(extended_factor), extended_factor))
            i += 1


def optimize_bag(label_sets):
    all_labels = set(chain(*label_sets))
    residual_label_sets = remove_subsets(label_sets)
    queue = [(size(residual_label_sets), has_cycle(residual_label_sets), 0, frozenset(), residual_label_sets)]
    # e = (size, has_cycle, factor)
    visited = set()
    best_valid_size = float("inf")
    i = 1
    while queue:
        # print i, queue
        s, hc, lf, factor, label_sets = heappop(queue)
        # print "s={} hc={} fs={} f={} ls={}".format(
        #     s,
        #     hc,
        #     lf,
        #     "_".join(sorted(factor)),
        #     format_label_sets(label_sets),
        # )
        if not hc:
            return factor, i
        for l in all_labels - factor:
            extended_factor = factor | {l}
            if extended_factor in visited:
                continue
            visited.add(extended_factor)
            residual_label_sets = remove_subsets(residual(label_sets, {l}))
            new_size = len(extended_factor) + size(residual_label_sets)
            new_hc = has_cycle(residual_label_sets)
            if new_size < best_valid_size:
                heappush(queue, (new_size, new_hc, len(extended_factor), extended_factor, residual_label_sets))
                if not new_hc:
                    best_valid_size = new_size
            i += 1


class QueueElement(namedtuple("QueueElement", "size has_cycle factor_size factor label_sets candidates")):
    def __str__(self):
        return "s={} hc={} fs={} f={} ls={} cs={}".format(
            self.size,
            self.has_cycle,
            self.factor_size,
            "_".join(sorted(self.factor)),
            format_label_sets(self.label_sets),
            "_".join(sorted(self.candidates)),
        )


def is_candidate(req, i):
    if not req.get_out_neighbors(i):
        return True

    stack = [i]
    while stack:
        node = stack.pop()
        for other in req.get_out_neighbors(node):
            if len(req.get_in_neighbors(other)) > 1:
                return False
            stack.append(other)
    return True

def optimize_bag_leaves_first(req, label_sets):
    all_labels = set(chain(*label_sets))

    candidates = frozenset(i for i in all_labels if is_candidate(req, i))

    residual_label_sets = remove_subsets(label_sets)
    queue = [QueueElement(size(residual_label_sets), has_cycle(residual_label_sets), 0, frozenset(), residual_label_sets, candidates)]
    # e = (size, has_cycle, factor)
    visited = set()
    best_valid_size = float("inf")
    i = 1
    while queue:
        # print "\033[0m", i, queue
        s, hc, lf, factor, label_sets, candidates = qe = heappop(queue)
        # print qe
        if not hc:
            return factor, i
        for l in candidates:
            extended_factor = factor | {l}
            if extended_factor in visited:
                continue
            visited.add(extended_factor)
            residual_label_sets = remove_subsets(residual(label_sets, {l}))
            new_size = len(extended_factor) + size(residual_label_sets)
            if new_size < best_valid_size:
                new_hc = has_cycle(residual_label_sets)

                new_candidates = candidates - {l}
                for p in req.get_in_neighbors(l):
                    while len(req.get_in_neighbors(p)) == 1:
                        p = req.get_in_neighbors(p)[0]

                    if req.get_in_neighbors(p):
                        new_candidates = new_candidates | {p}

                heappush(queue, QueueElement(new_size, new_hc, len(extended_factor), extended_factor, residual_label_sets, new_candidates))
                if not new_hc:
                    best_valid_size = new_size
            i += 1
    print "ERROR!!!!"

    with open("out/output/req.gv", "w") as f:
        from alib import util
        f.write(util.get_graph_viz_string(req))

    with open("out/output/req.pickle", "w") as f:
        import cPickle
        cPickle.dump(req, f)


if __name__ == "__main__":
    main2()
