__author__ = 'Tom Koch (tkoch@inet.tu-berlin.de)'

from alib import datamodel
from vnep_approx.deferred import extendedgraph


def test_extendedgraph_generation():
    substrate = datamodel.Substrate("sub1")
    request = datamodel.LinearRequest("linearreq1")
    # REQUEST NODES AND EDGES
    request.add_node('i', 2, "FW")
    request.add_node('j', 2, "DPI")
    request.add_node('l', 2, "FW")
    request.graph['start_node'] = 'i'
    request.graph['end_node'] = 'j'
    request.add_edge('i', 'j', 2)
    request.add_edge('j', 'l', 2)
    # SUBSTRATE: - NODES
    substrate.add_node('u', ["FW"], {"FW": 1}, {"FW": 1})
    substrate.add_node('v', ["FW"], {"FW": 1}, {"FW": 1})
    substrate.add_node('w', ["DPI"], {"DPI": 1}, {"DPI": 1})
    #           - EDGES
    #           - EDGES
    substrate.add_edge('u', 'v')
    substrate.add_edge('v', 'w')
    substrate.add_edge('u', 'w')
    # generation
    ext_graph = extendedgraph.ExtendedGraph(request,
                                            substrate)
    assert len(ext_graph.edges) == 17
    print ext_graph
