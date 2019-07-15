import numpy as np
from heapq import heappop, heappush
import traceback


def verify_correct_result(svpc):
    """ this function checks that:
        1) no path exceeds the latency limit
        2) the costs among a path is what is returned by costs
        3) every feasible node has an associated path and cost
        4) every infeasible node has associated costs of NaN and path of None
        5) dijkstra using latency metric is upper bound on costs
    """

    errors = []

    temp_latencies = np.full(svpc.number_of_nodes, -1, dtype=np.float64)
    temp_costs = np.full(svpc.number_of_nodes, -1, dtype=np.float64)

    try:

        for edgeset_index, source_node_dict in svpc.valid_sedge_paths.iteritems():
            for start_snode, target_dict in source_node_dict.iteritems():
                for target_snode, path in target_dict.iteritems():
                    total_latencies, total_costs = 0, 0
                    for edge in path:
                        total_latencies += svpc.edge_latencies[edge]
                        total_costs += svpc.edge_costs[edge]

                    # check criterium 1
                    if total_latencies > (1 + svpc.epsilon) * svpc.limit:
                        errors.append("Latency limit exceeded!\n  from {} to {}, edgeset {}"
                                      .format(start_snode, target_snode, edgeset_index))

                    # check criterium 2
                    if total_costs != svpc.get_valid_sedge_costs_from_edgesetindex(edgeset_index, (start_snode, target_snode)):
                        errors.append("Costs unequal to those along the path!\n  from {} to {}, edgeset {}"
                                      .format(start_snode, target_snode, edgeset_index))

                # one run of dijsktra using latency metric to find feasible nodes
                num_source_node = svpc.snode_id_to_num_id[start_snode]
                queue = [(0, 0, num_source_node)]
                temp_latencies.fill(np.inf)
                temp_costs.fill(np.inf)
                temp_latencies[num_source_node] = 0
                temp_costs[num_source_node] = 0

                while queue:
                    total_latencies, total_costs, num_current_node = heappop(queue)

                    for sedge in svpc.substrate.out_edges[svpc.num_id_to_snode_id[num_current_node]]:
                        if sedge in svpc.edge_set_id_to_edge_set[edgeset_index]:
                            num_endpoint = svpc.snode_id_to_num_id[sedge[1]]
                            lat = svpc.edge_latencies[sedge]
                            cost = svpc.edge_costs[sedge]
                            if total_latencies + lat < temp_latencies[num_endpoint]:
                                temp_latencies[num_endpoint] = total_latencies + lat
                                temp_costs[num_endpoint] = total_costs + cost
                                heappush(queue, (total_latencies + lat, total_costs + cost, num_endpoint))

                for num_node in range(svpc.number_of_nodes):
                    current_snode = svpc.num_id_to_snode_id[num_node]

                    if temp_latencies[num_node] < svpc.limit:
                        # this node is feasible, there has to be a path and costs

                        # check criterium 3
                        if np.isnan(svpc.get_valid_sedge_costs_from_edgesetindex(edgeset_index, (start_snode, current_snode))):
                            errors.append("Feasible node has NaN costs associated!\n  from {} to {}, edgeset {}"
                                          .format(start_snode, current_snode, edgeset_index))

                        if current_snode == start_snode:
                            continue

                        path = target_dict.get(current_snode, None)
                        if path is None:
                            errors.append("Feasible node has None path associated!\n  from {} to {}, edgeset {}"
                                          .format(start_snode, current_snode, edgeset_index))
                            continue

                        ref_costs = 0
                        for edge in path:
                            ref_costs += svpc.edge_costs[edge]

                        # check criterium 5
                        if ref_costs > temp_costs[num_node]:
                            errors.append("Dijkstra found cheaper path!\n  from {} to {}, edgeset {}"
                                          .format(start_snode, current_snode, edgeset_index))


                    else:
                        # this node is infeasible, the costs should be NaN and path should be None

                        # check criterium 4
                        if not np.isnan(svpc.get_valid_sedge_costs_from_edgesetindex(edgeset_index, (start_snode, current_snode))):
                            errors.append("Infeasible node has non-NaN costs associated!\n  from {} to {}, edgeset {}"
                                          .format(start_snode, current_snode, edgeset_index))

                        if target_dict.get(current_snode, None) is not None:
                            errors.append("Infeasible node has non-None path associated!\n  from {} to {}, edgeset {}"
                                          .format(start_snode, current_snode, edgeset_index))

            if not errors:
                print "Test finished, no errors!"
            else:
                print "Errors have been found:\n"
                for e in errors:
                    print e

    except Exception as e:
        print "An Exception occurred during testing. The structure may be wrong.\n\t", e
        traceback.print_exc()
