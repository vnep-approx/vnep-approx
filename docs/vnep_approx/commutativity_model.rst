``commutativity_model``
=======================

.. automodule:: vnep_approx.commutativity_model


In the generalized commutativity model, it must be ensured that node mappings
are consistent. That means, if there are multiple paths between two request
nodes, the end node has to be mapped to the same substrate node on all paths.

For doing that, the model creator (:class:`CommutativityModelCreator`) uses a
labeling algorithm (:meth:`CommutativityLabels`).

In the context of the commutativity model, an *end node* is the end node of a
cycle in the DAG, having at least two in-neighbors. The *labels* of a node or
an edge are a set of end nodes. A node or edge is labeled with an end node if
there is a simple cycle (a cycle without repetitions) through the node or edge
ending in the end node.

For each edge, the labels are extracted. For each possible mapping of the edge
labels, a :class:`EdgeSubLP` is created.

A mapping of labels to substrate nodes is called *commutativity index* in
this context.


Labeling algorithm
------------------

The labeling is done in four steps in :meth:`CommutativityLabels.create_labels`:

1. Determine the reachable end nodes of every request node.
   Start from leaf nodes going up.
2. Check pairwise overlaps of the children of every request node that has
   multiple children. Identify end nodes that are in at least one overlap of
   the children.
3. Reduce the overlapping sets by removing dominated nodes so that only the
   actual cycle end nodes of simple cycles remain.

   If it is not possible to find a path from a possible start node to an end
   node from the overlapping set without visiting one avoided node, the
   possible start node is not a start node of the end node (dominated). Try
   other nodes from the overlapping set as avoid node.
4. Propagate the labels to all nodes on paths between the start and end nodes.


LP model
--------

Variables
~~~~~~~~~

``node_mapping_agg`` (req, i, u)
  aggregated node mappings
``node_mapping`` (req, i, u, commutativity index)
  node mappings

In edge sub-LPs:

``node_flow_source`` (req, ij, u, commutativity index)
  node flows of the sub-LP sources
``edge_flow`` (req, ij, uv, commutativity index)
  edge flows of the substrate copy
``node_flow_sink`` (req, ij, v, commutativity_index)
  node flows of the sub-LP sinks


Constraints
~~~~~~~~~~~

``force_node_mapping_for_embedded_request`` (req)
  force the embedding of the request
``node_mapping_aggregation`` (req, i, u, bag)
  aggregate node mappings
``node_mapping_in_edge_to_bag`` (req, i, u, ji, commutativity index, bag)
  connect sink variables of in-edges and node mapping variables
``node_mapping`` (req, i, u, ij, commutativity index)
  connect node mapping variables to source variables of out-edges
``track_node_load`` (req, u, type)
  track node loads
``track_edge_load`` (req, uv)
  track edge loads

In edge sub-LPs:

``flow_preservation`` (req, ij, u, commutativity index)
  flow preservation in an edge sub-LP


Classes
-------

.. autoexception:: vnep_approx.commutativity_model.CommutativityModelError

.. autoclass:: vnep_approx.commutativity_model.CommutativityModelCreator
  :members:
  :undoc-members:
  :private-members:

.. autoclass:: vnep_approx.commutativity_model.EdgeSubLP
  :members:
  :undoc-members:
  :private-members:

.. autoclass:: vnep_approx.commutativity_model.CommutativityLabels
  :members:
  :undoc-members:
  :private-members:
