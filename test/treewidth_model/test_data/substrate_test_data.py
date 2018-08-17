from alib import datamodel, scenariogeneration


def create_test_substrate():
    sub = datamodel.Substrate("test_sub")
    sub.add_node("u", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("v", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})
    sub.add_node("w", types=["t1"], capacity={"t1": 100}, cost={"t1": 1.0})

    sub.add_edge("u", "v", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("v", "w", capacity=100.0, cost=1.0, bidirected=False)
    sub.add_edge("w", "u", capacity=100.0, cost=1.0, bidirected=False)
    return sub
