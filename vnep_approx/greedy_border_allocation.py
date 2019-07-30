# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2019 Yvonne-Anne Pignolet, Balazs Nemeth
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

"""
Created on Thu Jul 25 22:18:58 2019

@author: yvonne-annepignolet
"""

import networkx as nx
import time

from alib import solutions, modelcreator


class GreedyBorderAllocationResult(modelcreator.AlgorithmResult):

    def __init__(self, scenario, feasible, runtime, cost):
        self.scenario = scenario
        self.feasible = feasible
        self.runtime = runtime
        self.cost = cost
        self.alg_key = ("GBA", "0")
        self.solutions = {self.alg_key: []}

    def get_solution(self):
        return self.solutions[self.alg_key][0]

    def _get_solution_overview(self):
        return "GBA solution feasible: {}, cost: {}, runtime: {}".format(self.feasible, self.cost, self.runtime)

    def _cleanup_references_raw(self, original_scenario):
        self.scenario = original_scenario
        self.solutions[self.alg_key][0].scenario = original_scenario


class GreedyBorderAllocationForFogModel(object):

    ALGORITHM_ID = "GreedyBorderAllocationForFogModel"

    def __init__(self, scenario, logger, gurobi_settings=None):
        self.scenario = scenario
        self.logger = logger
        # required by the framework, it is designed for gurobi based algos
        self.gurobi_settings = gurobi_settings
        self.paths = {}
        self.AppGraphList, self.Substrate, self.LbNodes = None, None, None
        if self.scenario == None:
            return

        #
        self.substrate_resources = list(self.scenario.substrate.edges)
        self.substrate_edge_resources = list(self.scenario.substrate.edges)
        self.substrate_node_resources = []
        for ntype in self.scenario.substrate.get_types():
            for snode in self.scenario.substrate.get_nodes_by_type(ntype):
                self.substrate_node_resources.append((ntype, snode))
                self.substrate_resources.append((ntype, snode))

    def init_model_creator(self):
        """
        Converts scenario to internal networkx format

        :return:
        """
        substrate_alib = self.scenario.substrate
        if len(substrate_alib.types) > 1:
            raise NotImplementedError("GBA does not support NF types.")
        self.Substrate = nx.Graph()
        for ntype, snode in self.substrate_node_resources:
            self.Substrate.add_node(snode, weight=substrate_alib.node[snode]["capacity"][ntype])
        for u, v in self.substrate_edge_resources:
            self.Substrate.add_edge(u, v, weight=substrate_alib.edge[(u,v)]["capacity"])

        self.AppGraphList = []
        self.LbNodes = {}
        for request_alib in self.scenario.requests:
            AppGraph = nx.DiGraph()
            self.LbNodes[AppGraph] = {}
            for rnode in request_alib.nodes:
                AppGraph.add_node(rnode, weight=request_alib.get_node_demand(rnode))
                allowed_nodes = request_alib.get_allowed_nodes(rnode)
                if allowed_nodes is not None:
                    if len(allowed_nodes) > 1:
                        raise NotImplementedError("GBA does not support multiple allowed nodes")
                    self.LbNodes[AppGraph][rnode] = allowed_nodes[0]
            for i, j in request_alib.edges:
                AppGraph.add_edge(i, j, weight=request_alib.get_edge_demand((i, j)))
            self.AppGraphList.append(AppGraph)

    def calculate_solution_cost(self, node_mapping, link_mapping):
        """
        Calculates the cost of the allocation assuming 1.0 is the unit cost on both the nodes and links.

        :param node_mapping:
        :param link_mapping:
        :return:
        """
        for nf in node_mapping:
            subs_node = node_mapping[nf]
            
    def construct_results_object(self, result, feasible):
        """
        Constructs

        :param result:
        :param feasible:
        :return:
        """
        total_cost = 0.0
        if feasible:
            node_mapping, link_mapping = result

        # TODO: properly fill Mapping object
        mapping_alib = solutions.Mapping("GBA-mapping", self.scenario.requests[0], self.scenario.substrate, is_embedded=feasible)
        runtime = time.time() - self.start_time
        result_alib = GreedyBorderAllocationResult(self.scenario, feasible, runtime, total_cost)
        solution_alib = solutions.IntegralScenarioSolution(name="GBA-solution", scenario=self.scenario)
        solution_alib.add_mapping(self.scenario.requests[0], mapping_alib)
        result_alib.solutions[result_alib.alg_key].append(solution_alib)
        return result_alib

    def compute_integral_solution(self):
        """
        Calls GBA and converts the result matching the framework

        :return:
        """
        self.start_time = time.time()
        result = self.greedyBorderAllocation(self.AppGraphList, self.Substrate, self.LbNodes)
        if result is not None:
            feasible = True
        else:
            feasible = False
        return self.construct_results_object(result, feasible=feasible)

    # AppGraphList: container of networkx graphs with demand as weight attribute
    # Substrate: networkx graphs with capacity as weight attribute
    # LocationBound: dictionary AppGraphList graph to AppGraph node to  Substrate Node
    def greedyBorderAllocation(self, AppGraphList, Substrate, LbNodes):
        for e in Substrate.edges():
            Substrate[e[0]][e[1]]['free'] = Substrate[e[0]][e[1]]['weight']
        for u in Substrate.nodes():
            Substrate.node[u]['free'] = Substrate.node[u]['weight']
        self.paths = dict(nx.all_pairs_shortest_path(Substrate))
        #    2: µ ← new Mapping
        #    3: ADDALLLOCATIONBOUND(µ, LbNodes)
        mu = LbNodes
        #    4: sBE ← SORTEDBORDEREDGES(AppGraphList, µ)
        #    5: while ISINCOMPLETE(µ) do
        mu2 = {}
        num_appgraph_nodes = sum([len(graph.nodes()) for graph in AppGraphList])
        while len(mu) < num_appgraph_nodes:
            #    6: (a1, a2) ← NEXTEDGE(µ, sBE)
            (e, AppGraph) = self.nextEdge(AppGraphList, Substrate, mu)
            if e == None:
                return mu
            #    7: f ← CLOSESTFEASIBLEFOGNODE(a2, µ(a1))
            f = self.closestFeasibleFogNode(e, AppGraph, Substrate, mu)
            #    8: if ISDEFINED(f) then
            if f != None:
                #    9: ADDMAPPING(a2, f)
                mu[e[1]] = f
                self.updateSubstrate(e, mu[e[0]], mu[e[1]], AppGraph, Substrate, mu, mu2)
            #    10: UPDATEBORDEREDGES(a2)
            #    11: else return 0
            else:
                return None
        #    12: return µ
        return [mu, mu2]

    def nextEdge(self, AppGraphList, substrate, mu):
        #    14: (a1, a2) ← LARGESTEDGE(sBE)
        #    15: if (ISDEFINED(µ(a1)) then return (a1, a2)
        #    16: (a1, f) ← NEXTFROMDISJOINTPART(µ)
        #    17: if ISDEFINED(f) then
        #    18: ADDMAPPING(a1, f)
        #    19: UPDATEBORDEREDGES(a1)
        #    20: (a1, a2) ← LARGESTEDGE(µ, sBE)
        #    21: if ISUNDEFINED(b1) then goto 16
        #    22: return b1, b2
        #    23: else return 0
        #    16: (a1, f) ← NEXTFROMDISJOINTPART(µ)
        #    17: if ISDEFINED(f) then
        #    18: ADDMAPPING(a1, f)
        #    19: UPDATEBORDEREDGES(a1)
        #    20: (a1, a2) ← LARGESTEDGE(µ, sBE)
        #    21: if ISUNDEFINED(b1) then goto 16
        #    22: return b1, b2
        #    23: else return 0
        #    16: (a1, f) ← NEXTFROMDISJOINTPART(µ)
        #    17: if ISDEFINED(f) then

        #    14: (a1, a2) ← LARGESTEDGE(sBE)
        #    15: if (ISDEFINED(µ(a1)) then return (a1, a2)
        #    16: (a1, f) ← NEXTFROMDISJOINTPART(µ)
        sBE_with_AppGraph = self.sortedBorderEdges(AppGraphList, mu)
        while sBE_with_AppGraph == []:
            (u, AppGraph, v) = self.nextFromDisjointPart(AppGraphList, substrate, mu)
            #    17: if ISDEFINED(f) then
            if (v == None):
                return None
            #    18: ADDMAPPING(a1, f)
            mu[u] = v
            substrate.node[v]['free'] -= AppGraph.node[u]['weight']
            #    19: UPDATEBORDEREDGES(a1)
            sBE_with_AppGraph = self.sortedBorderEdges(AppGraphList, mu)
        return sBE_with_AppGraph[-1]

    def sortedBorderEdges(self, AppGraphList, mu):
        borderEdges = []
        for AppGraph in AppGraphList:
            for e in AppGraph.edges(data='weight'):
                if (e[0] in mu and not e[1] in mu):
                    borderEdges.append((e,AppGraph))
                if (not e[0] in mu and e[1] in mu):
                    borderEdges.append((e,AppGraph))
        return sorted(borderEdges, key=lambda x: x[0][2])

    def nextFromDisjointPart(self, AppGraphList, substrate, mu):
        for AppGraph in AppGraphList:
            not_mapped = set([x for x in AppGraph.nodes() if x not in mu])
            for u in not_mapped:
                if len(set(AppGraph[u]).intersection(set(mu.keys()))) == 0:
                    fit = [v for v in substrate.nodes() if substrate.node[v]['free'] >= AppGraph.node[u]['weight']]
                    if fit == []:
                        return (None, None, None)
                    return (u, AppGraph, fit[0])
        return (None, None, None)

    def fits(self, e, u, v, graph, Substrate):
        p = self.paths[u][v]
        for i in range(len(p) - 1):
            if Substrate[p[i]][p[i + 1]]['free'] < e[2]:
                return False
        return Substrate.node[v]['free'] >= graph.node[e[1]]['weight']


    def updateSubstrate(self, e, u, v, AppGraph, Substrate, mu, mu2):
        Substrate.node[v]['free'] -= AppGraph.node[e[1]]['weight']
        if u != v:
            p = self.paths[u][v]
            for i in range(len(p) - 1):
                Substrate[p[i]][p[i + 1]]['free'] -= e[2]
            mu2[e] = p

    # e[0] embedded on u, e[1] not yet
    def closestFeasibleFogNode(self, e, AppGraph, Substrate, mu):
        if e[0] in mu:
            sorted_nodes = sorted([v for v in Substrate.nodes()], key=lambda x: len(self.paths[mu[e[0]]][x]))
            for v in sorted_nodes:
                if self.fits(e, mu[e[0]], v, AppGraph, Substrate):
                    return v
        if e[1] in mu:
            sorted_nodes = sorted([v for v in Substrate.nodes()], key=lambda x: len(self.paths[mu[e[1]]][x]))
            for v in sorted_nodes:
                if self.fits(e, v, mu[e[1]], AppGraph, Substrate):
                    return v

    def feasibility(self, AppGraph, Substrate, mu_list):
        mu = mu_list[0]
        mu2 = mu_list[1]
        for node in Substrate.nodes():
            sum_node = 0
            for anode in [u for u in g.nodes() for g in AppGraph]:
                if mu[anode] == node:
                    sum_node += AppGraph.node[anode]['weight']
            if sum_node > Substrate.node[node]['weight']:
                return False
        for edge in Substrate.edges():
            sum_edge = 0
            for aedge in [u for u in g.edges(data='weight') for g in AppGraph]:
                if aedge in mu2 and edge in mu2[aedge]:
                    sum_edge += aedge[2]
            if sum_edge > Substrate[edge[0]][edge[1]]['weight']:
                return False
        return True


def main():
    global paths
    print("test heuristic!")
    for AppGraph1 in [nx.complete_graph(400), nx.ladder_graph(4), nx.complete_graph(4), ]:
        for Substrate in [nx.ladder_graph(4), nx.complete_graph(4), nx.complete_graph(300)]:
            AppGraph2 = nx.ladder_graph(4)
            for v in AppGraph1.nodes():
                AppGraph1.node[v]['weight'] = 2
            for e in AppGraph1.edges():
                AppGraph1[e[0]][e[1]]['weight'] = 2
            AppGraph1.node[0]['weight'] = 3
            for v in AppGraph2.nodes():
                AppGraph2.node[v]['weight'] = 2
            for e in AppGraph2.edges():
                AppGraph2[e[0]][e[1]]['weight'] = 2
            AppGraph2.node[0]['weight'] = 3
            for v in Substrate.nodes():
                Substrate.node[v]['weight'] = 4
                Substrate.node[v]['free'] = 4
            for e in Substrate.edges():
                Substrate[e[0]][e[1]]['weight'] = 4
                Substrate[e[0]][e[1]]['free'] = 4
            gba = GreedyBorderAllocationForFogModel(None, None)
            result = gba.greedyBorderAllocation([AppGraph1, AppGraph2], Substrate, {})
            if result != None and not gba.feasibility(AppGraph1, Substrate, result):
                print("problem")
            print("try next")
    print("done")
        