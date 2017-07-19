# MIT License
#
# Copyright (c) 2016-2017 Matthias Rost, Elias Doehne, Tom Koch, Alexander Elvers
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
#

import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap as Basemap

from alib import scenariogeneration
from . import extendedgraph

MapCoordinate = namedtuple("MapCoordinate", "lon lat")
plt.rc('axes', linewidth=0)
plt.axis('off')
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')  # labels along the bottom edge are off

DPI = 200
MAP_HEIGHT = 1200
SUPER_NODE_MARGIN = 200
MAP_MARGIN = 0.1


class ExtendedGraphVisualizationError(Exception): pass


class ExtendedGraphVisualizer(object):
    def __init__(self):
        self._extended_graph = None
        # self._substrate = None
        self._substrate = None
        self._fig = None
        self._layer_axes = None
        self._layer_index_x_offset_list = None
        self._map_layer_count = None
        self._nodes_to_figure_coordinates = None
        self._nodes_to_data_coordinates = None
        self._map_width = None
        self._curved_edges = None
        self._background_map = None
        self._edge_weights = None

    def visualize(self,
                  extended_graph,
                  substrate,
                  edge_weights=None,
                  output_path=None,
                  curved_edges=True):
        self._curved_edges = curved_edges
        self._substrate = substrate

        if edge_weights is None:
            edge_weights = {e: 3.0 for e in extended_graph.edges}
        self._edge_weights = edge_weights
        self._setup(extended_graph)
        self._initialize_background_map()
        self._create_base_maps()
        self._add_inter_edges()
        self._save_or_show_figure(output_path)

    def _setup(self, extended_graph):
        self._extended_graph = extended_graph
        self._map_layer_count = max(self._extended_graph.node[u]["layer_index"] for u in self._extended_graph.nodes) - 1

        self._nodes_to_figure_coordinates = {}
        self._nodes_to_data_coordinates = {}
        self._layer_axes = {}

    def _initialize_background_map(self):  # calculate the correct aspect ratio by drawing a map of the substrate
        fig = plt.figure()
        ax = plt.subplot(111)
        nodes_to_map_coordinates = [MapCoordinate(lat=data['Latitude'], lon=data['Longitude'])
                                    for n, data in self._substrate.node.items()]
        lower_left, upper_right = ExtendedGraphVisualizer._get_corner_map_coordinates(nodes_to_map_coordinates)
        self._background_map = Basemap(resolution="c",
                                       projection='merc',
                                       anchor="W",
                                       ax=ax,
                                       llcrnrlat=lower_left.lat,
                                       urcrnrlat=upper_right.lat,
                                       llcrnrlon=lower_left.lon,
                                       urcrnrlon=upper_right.lon,
                                       fix_aspect=True)
        self._background_map.fillcontinents(ax=ax)
        (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        dx, dy = float(abs(xmax - xmin)), float(abs(ymax - ymin))
        self._map_width = (dx / dy) * MAP_HEIGHT
        plt.close(fig)

    def _create_base_maps(self):
        self.figure_width = self._map_width * self._map_layer_count + 2 * SUPER_NODE_MARGIN
        self._fig = plt.figure(figsize=(self.figure_width / float(DPI), MAP_HEIGHT / float(DPI)))
        self._fig.subplots_adjust(bottom=0, top=1, right=1, left=0)

        # first place the super nodes
        super_source_position = 0.1, 0.5
        super_sink_position = 0.9, 0.5

        super_ax_width = float(SUPER_NODE_MARGIN) / self.figure_width
        ax_left = super_ax_width
        ax_bottom = 0.0
        ax_width = self._map_width / self.figure_width
        ax_height = 1.0

        super_source_ax = self._fig.add_axes((0.0, ax_bottom, super_ax_width, ax_height))
        super_source_ax.set_xlim((0.0, 1.0))
        super_source_ax.set_ylim((0.0, 1.0))
        super_sink_ax = self._fig.add_axes((1.0 - super_ax_width, ax_bottom, super_ax_width, ax_height))
        super_sink_ax.set_xlim((0.0, 1.0))
        super_sink_ax.set_ylim((0.0, 1.0))

        self._layer_axes[0] = super_source_ax
        self._layer_axes[self._map_layer_count + 1] = super_sink_ax
        super_source_ax.plot([super_source_position[0]], [super_source_position[1]], "ro", markersize=5.0)
        super_sink_ax.plot([super_sink_position[0]], [super_sink_position[1]], "ro", markersize=5.0)

        self._nodes_to_data_coordinates[0] = {self._extended_graph.super_source: super_source_position}
        self._nodes_to_data_coordinates[self._map_layer_count + 1] = {
            self._extended_graph.super_sink: super_sink_position}

        for layer_index in range(1, self._map_layer_count + 1):
            new_ax = self._fig.add_axes((ax_left, ax_bottom, ax_width, ax_height))
            self._layer_axes[layer_index] = new_ax
            nodes_to_data_coordinates = self._add_nodes_and_map_to_axis(new_ax)
            self._nodes_to_data_coordinates[layer_index] = nodes_to_data_coordinates
            ax_left += ax_width

    def _add_nodes_and_map_to_axis(self, ax):
        nodes_to_map_coordinates = {n: MapCoordinate(lat=data['Latitude'], lon=data['Longitude'])
                                    for n, data in self._substrate.node.items()}

        nodes_to_data_coordinates = {node: self._background_map(mc.lon, mc.lat) for node, mc in
                                     nodes_to_map_coordinates.iteritems()}
        nodes = self._substrate.nodes
        xy = np.asarray([nodes_to_data_coordinates[v] for v in nodes])

        node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                     s=10,
                                     c="k",
                                     marker="o",
                                     linewidths=(0, 0))
        node_collection.set_zorder(2)
        self._background_map.fillcontinents(color='#9ACEEB', ax=ax)
        return nodes_to_data_coordinates

    def _add_inter_edges(self):
        edges = []
        for edge in self._extended_graph.edges:
            tail, head = edge
            tail_layer = self._extended_graph.node[tail]["layer_index"]
            head_layer = self._extended_graph.node[head]["layer_index"]
            if self._extended_graph.node[tail]["origin"] is None:
                tail_origin = self._extended_graph.super_source
            else:
                tail_origin = self._extended_graph.node[tail]["origin"]
            if self._extended_graph.node[head]["origin"] is None:
                head_origin = self._extended_graph.super_sink
            else:
                head_origin = self._extended_graph.node[head]["origin"]
            line = self._make_edge(edge, tail_origin, tail_layer, head_origin, head_layer)

            if line is not None:
                edges.append(line)
        self._fig.lines.extend(edges)

    def _make_edge(self, edge, u, layer_u, v, layer_v):
        if layer_u == layer_v:
            lon1 = self._substrate.node[u]['Longitude']
            lat1 = self._substrate.node[u]['Latitude']
            lon2 = self._substrate.node[v]['Longitude']
            lat2 = self._substrate.node[v]['Latitude']
            self._background_map.drawgreatcircle(lon1, lat1, lon2, lat2,
                                                 color="k",
                                                 ax=self._layer_axes[layer_u],
                                                 linewidth=self._edge_weights[edge],
                                                 # alpha = 0.8,
                                                 # dashes=(4,1),
                                                 zorder=1)
            return

        data_coordinates_u = self._nodes_to_data_coordinates[layer_u][u]
        data_coordinates_v = self._nodes_to_data_coordinates[layer_v][v]
        x_u, y_u = self._transform_from_data_to_figure_coordinates(data_coordinates_u,
                                                                   self._layer_axes[layer_u])
        x_v, y_v = self._transform_from_data_to_figure_coordinates(data_coordinates_v,
                                                                   self._layer_axes[layer_v])

        dx = abs(x_u - x_v)
        if x_u < x_v:
            x_left, y_left = x_u, y_u
            x_right, y_right = x_v, y_v
        else:
            x_left, y_left = x_v, y_v
            x_right, y_right = x_u, y_u
        n = 30
        x_values = np.linspace(x_left, x_right, n, endpoint=True)
        pi = 2 * np.arcsin(1.0)
        y_values = np.linspace(y_left, y_right, n, endpoint=True)

        if self._curved_edges and u != self._extended_graph.super_source and v != self._extended_graph.super_sink:
            y_values += 0.8 * MAP_MARGIN * np.sin((x_values - x_left) * pi / dx)

        return plt.Line2D(x_values, y_values,
                          transform=self._fig.transFigure,
                          linestyle="-",
                          color="k",
                          linewidth=self._edge_weights[edge],
                          zorder=1.0)

    def _transform_from_data_to_figure_coordinates(self, data_coordinates, ax):
        display_to_figure_trans = self._fig.transFigure.inverted()
        data_to_display_trans = ax.transData
        return display_to_figure_trans.transform(data_to_display_trans.transform(data_coordinates))

    @staticmethod
    def _get_corner_map_coordinates(map_coordinate_list):
        lower_left_lon = min(mc.lon for mc in map_coordinate_list)
        lower_left_lat = min(mc.lat for mc in map_coordinate_list)
        upper_right_lon = max(mc.lon for mc in map_coordinate_list)
        upper_right_lat = max(mc.lat for mc in map_coordinate_list)
        margin_lon = MAP_MARGIN * abs(upper_right_lon - lower_left_lon)
        margin_lat = MAP_MARGIN * abs(upper_right_lat - lower_left_lat)
        lower_left_lon -= margin_lon
        lower_left_lat -= margin_lat
        upper_right_lon += margin_lon
        upper_right_lat += margin_lat

        lower_left_lon = max(lower_left_lon, -179)
        upper_right_lon = min(upper_right_lon, 179)
        lower_left_lat = max(lower_left_lat, -70)
        upper_right_lat = min(upper_right_lat, 85)

        lower_left = MapCoordinate(lon=lower_left_lon, lat=lower_left_lat)
        upper_right = MapCoordinate(lon=upper_right_lon, lat=upper_right_lat)

        return lower_left, upper_right

    @staticmethod
    def _save_or_show_figure(output_path):
        if output_path is None:
            plt.show()
        else:
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            output_path = os.path.abspath(output_path)
            print "saving to", output_path
            plt.savefig(output_path, dpi=200)
        plt.close('all')


def example():
    from alib import datamodel
    reader = scenariogeneration.TopologyZooReader(path="../../github-alib/data/topologyZoo")
    substrate = reader.read_from_yaml(
        dict(topology="Aarnet",
             edge_capacity=1.0,
             node_types=["t1"],
             node_capacity=1.0,
             node_type_distribution=1.0),
        include_location=True
    )
    req = datamodel.LinearRequest("example_req")
    req.add_node("i", 1.0, "t1", allowed_nodes=["1"])
    req.add_node("j", 1.0, "t1", allowed_nodes=["2", "7"])
    req.add_node("l", 1.0, "t1", allowed_nodes=["3", "5"])
    req.add_edge("i", "j", 1.0)
    req.add_edge("j", "l", 1.0)

    ext_graph = extendedgraph.ExtendedGraph(req, substrate)
    visualizer = ExtendedGraphVisualizer()
    visualizer.visualize(ext_graph, substrate=substrate, output_path="./out/test_example.png")


if __name__ == "__main__":
    example()
