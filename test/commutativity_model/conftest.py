import pytest

from alib import datamodel

i, j, k = "ijk"
ij = i, j
jk = j, k
ik = i, k
u, v, w = "uvw"
uv = u, v
vu = v, u
uw = u, w
wu = w, u
vw = v, w
wv = w, v


@pytest.fixture()
def substrate():
    substr = datamodel.Substrate("substrate")
    substr.add_node(u, ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substr.add_node(v, ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substr.add_node(w, ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substr.add_edge(u, v, capacity=10000, cost=1, bidirected=True)
    substr.add_edge(v, w, capacity=10000, cost=1, bidirected=True)
    substr.add_edge(u, w, capacity=10000, cost=1, bidirected=True)
    return substr


@pytest.fixture()
def tiny_substrate():
    substr = datamodel.Substrate("tiny_substrate")
    substr.add_node(u, ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substr.add_node(v, ["t1"], capacity={"t1": 10000}, cost={"t1": 1})
    substr.add_edge(u, v, capacity=10000, cost=1, bidirected=True)
    return substr


@pytest.fixture()
def triangle_request():
    req = datamodel.Request("test_req")
    req.add_node(i, 1.0, "t1", [u])
    req.add_node(j, 1.0, "t1", [v])
    req.add_node(k, 1.0, "t1", [u, v])
    req.add_edge(i, j, 1.0, None)
    req.add_edge(j, k, 1.0, None)
    req.add_edge(i, k, 1.0, None)
    req.graph["root"] = i
    return req
