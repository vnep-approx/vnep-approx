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
        self.AppGraph, self.Substrate, self.LbNodes = None, None, None

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

        if len(self.scenario.requests) > 1:
            raise NotImplementedError("GBA does not support more requests")
        request_alib = self.scenario.requests[0]
        self.LbNodes = {}
        # TODO: make request graphs directed
        self.AppGraph = nx.Graph()
        for rnode in request_alib.nodes:
            self.AppGraph.add_node(rnode, weight=request_alib.get_node_demand(rnode))
            allowed_nodes = request_alib.get_allowed_nodes(rnode)
            if allowed_nodes is not None:
                if len(allowed_nodes) > 1:
                    raise NotImplementedError("GBA does not support multiple allowed nodes")
                self.LbNodes[rnode] = allowed_nodes[0]
        for i, j in request_alib.edges:
            self.AppGraph.add_edge(i, j, weight=request_alib.get_edge_demand((i,j)))

    def construct_results_object(self, result, feasible):
        """
        Constructs

        :param result:
        :param feasible:
        :return:
        """
        if feasible:
            node_mapping, link_mapping = result
        # TODO: properly fill Mapping object
        mapping_alib = solutions.Mapping("GBA-mapping", self.scenario.requests[0], self.scenario.substrate, is_embedded=feasible)
        runtime = time.time() - self.start_time
        result_alib = GreedyBorderAllocationResult(self.scenario, feasible, runtime, 10.0)
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
        result = self.greedyBorderAllocation(self.AppGraph, self.Substrate, self.LbNodes)
        if result is not None:
            feasible = True
        else:
            feasible = False
        return self.construct_results_object(result, feasible=feasible)

    # AppGraph, Substrate: networkx graphs with capacity/demand as attributes
    # LocationBound: dictionary AppGraph node to  Substrate Node
    def greedyBorderAllocation(self, AppGraph, Substrate, LbNodes):
        for e in Substrate.edges():
            Substrate[e[0]][e[1]]['free'] = Substrate[e[0]][e[1]]['weight']
        for u in Substrate.nodes():
            Substrate.node[u]['free'] = Substrate.node[u]['weight']
        self.paths = dict(nx.all_pairs_shortest_path(Substrate))
        #    2: µ ← new Mapping
        #    3: ADDALLLOCATIONBOUND(µ, LbNodes)
        mu = LbNodes
        #    4: sBE ← SORTEDBORDEREDGES(AppGraph, µ)
        #    5: while ISINCOMPLETE(µ) do
        mu2 = {}
        while len(mu) != len(AppGraph.nodes()):
            #    6: (a1, a2) ← NEXTEDGE(µ, sBE)
            e = self.nextEdge(AppGraph, Substrate, mu, self.sortedBorderEdges(AppGraph, mu))
            if e == None:
                return mu
            #    7: f ← CLOSESTFEASIBLEFOGNODE(a2, µ(a1))
            f = self.closestFeasibleFogNode(e, mu[e[0]], AppGraph, Substrate, mu)
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

    def nextEdge(self, AppGraph, substrate, mu, sBE):
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
        while sBE == []:
            (u, v) = self.nextFromDisjointPart(AppGraph, substrate, mu)
            #    17: if ISDEFINED(f) then
            if (v == None):
                return None
            #    18: ADDMAPPING(a1, f)
            mu[u] = v
            substrate.node[v]['free'] -= AppGraph.node[u]['weight']
            #    19: UPDATEBORDEREDGES(a1)
            sBE = self.sortedBorderEdges(AppGraph, mu)
        return sBE[-1]

    def sortedBorderEdges(self, AppGraph, mu):
        borderEdges = []
        for e in AppGraph.edges(data='weight'):
            if (e[0] in mu and not e[1] in mu):
                borderEdges.append(e)
            if (not e[0] in mu and e[1] in mu):
                borderEdges.append((e[1], e[0], e[2]))
        return sorted(borderEdges, key=lambda x: x[2])

    def nextFromDisjointPart(self, AppGraph, substrate, mu):
        not_mapped = set([x for x in AppGraph.nodes() if x not in mu])
        for u in not_mapped:
            if len(set(AppGraph[u]).intersection(set(mu.keys()))) == 0:
                fit = [v for v in substrate.nodes() if substrate.node[v]['free'] >= AppGraph.node[u]['weight']]
                if fit == []:
                    return (None, None)
                return (u, fit[0])
        return (None, None)

    def fits(self, e, u, v, AppGraph, Substrate):
        p = self.paths[u][v]
        for i in range(len(p) - 1):
            if Substrate[p[i]][p[i + 1]]['free'] < e[2]:
                return False
        return Substrate.node[v]['free'] >= AppGraph.node[e[1]]['weight']


    def updateSubstrate(self, e, u, v, AppGraph, Substrate, mu, mu2):
        Substrate.node[v]['free'] -= AppGraph.node[e[1]]['weight']
        if u != v:
            p = self.paths[u][v]
            for i in range(len(p) - 1):
                Substrate[p[i]][p[i + 1]]['free'] -= e[2]
            mu2[e] = p

    # e[0] embedded on u, e[1] not yet
    def closestFeasibleFogNode(self, e, u, AppGraph, Substrate, mu):
        sorted_nodes = sorted([v for v in Substrate.nodes()], key=lambda x: len(self.paths[u][x]))
        for v in sorted_nodes:
            if self.fits(e, u, v, AppGraph, Substrate):
                return v

    def feasibility(self, AppGraph, Substrate, mu_list):
        mu = mu_list[0]
        mu2 = mu_list[1]
        for node in Substrate.nodes():
            sum_node = 0
            for anode in AppGraph.nodes():
                if mu[anode] == node:
                    sum_node += AppGraph.node[anode]['weight']
            if sum_node > Substrate.node[node]['weight']:
                return False
        for edge in Substrate.edges():
            sum_edge = 0
            for aedge in AppGraph.edges(data='weight'):
                if aedge in mu2 and edge in mu2[aedge]:
                    sum_edge += aedge[2]
            if sum_edge > Substrate[edge[0]][edge[1]]['weight']:
                return False
        return True


def main():
    global paths
    print("test heuristic!")
    for AppGraph in [nx.complete_graph(400), nx.ladder_graph(4), nx.complete_graph(4), ]:
        for Substrate in [nx.ladder_graph(4), nx.complete_graph(4), nx.complete_graph(300)]:
            for v in AppGraph.nodes():
                AppGraph.node[v]['weight'] = 2
            for e in AppGraph.edges():
                AppGraph[e[0]][e[1]]['weight'] = 2
            AppGraph.node[0]['weight'] = 3
            for v in Substrate.nodes():
                Substrate.node[v]['weight'] = 4
                Substrate.node[v]['free'] = 4
            for e in Substrate.edges():
                Substrate[e[0]][e[1]]['weight'] = 4
                Substrate[e[0]][e[1]]['free'] = 4
            gba = GreedyBorderAllocationForFogModel(None, None)
            result = gba.greedyBorderAllocation(AppGraph, Substrate, {})
            if result != None and not gba.feasibility(AppGraph, Substrate, result):
                print("problem")