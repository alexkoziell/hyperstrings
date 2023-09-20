#    Copyright 2023 Alexander Koziell-Pipe

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Hypergraph drawing implementation."""
from matplotlib.patches import Circle, PathPatch, Rectangle  # type: ignore
from matplotlib.path import Path  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from hyperstrings.hypergraph.backend import backend, Array
from hyperstrings.hypergraph.backend import Vertex, Hyperedge, Label
from hyperstrings.hypergraph.composable import ComposableHypergraph


class DrawableHypergraph(ComposableHypergraph):
    """Drawable hypergraph implementation."""

    def __init__(self,
                 sources: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 targets: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 vertex_labels: Array = backend.array([], dtype=Label),
                 hyperedge_labels: Array = backend.array([], dtype=Label),
                 inputs=[],
                 outputs=[]) -> None:
        """Initialize a `DrawableHypergraph` Instance."""
        super().__init__(sources, targets,
                         vertex_labels, hyperedge_labels,
                         inputs, outputs)
        self.vertex_coords: Array = backend.zeros((self.num_vertices(), 2))
        self.hyperedge_coords: Array = backend.zeros(
            (self.num_hyperedges(), 2))

    def add_vertex(self, label: Label) -> Vertex:
        """Add a vertex to the hypergraph."""
        self.vertex_coords = backend.concat(
            (self.vertex_coords, backend.zeros((1, 2))))
        return super().add_vertex(label)

    def add_hyperedge(self, label: Label) -> Hyperedge:
        """Add a hyperedge to the hypergraph."""
        self.hyperedge_coords = backend.concat(
            (self.hyperedge_coords, backend.zeros((1, 2))))
        return super().add_hyperedge(label)

    def remove_vertices(self, *vertices: Vertex) -> None:
        """Remove vertices from the hypergraph."""
        keep_vertices = [v for v in self.vertices()
                         if v not in vertices]
        self.vertex_coords = self.vertex_coords[keep_vertices]
        return super().remove_vertices(*vertices)

    def remove_hyperedges(self, *hyperedges: Hyperedge) -> None:
        """Remove hyperedges from the hypergraph."""
        keep_hyperedges = [h for h in self.hyperedges()
                           if h not in hyperedges]
        self.hyperedge_coords = self.hyperedge_coords[keep_hyperedges]
        return super().remove_hyperedges(*hyperedges)

    def compute_coordinates(self):
        """Compute x and y coordinates based on a layer decomposition.

        Successive vertex and hyperedge layers are spaced a unit x-coordinate
        apart.

        Vertices and hyperedges in the same layer are spaced a
        unit y-coordinate apart.
        """
        hypergraph, layers = self.explicit_cycles().layer_decomposition()

        x = len(layers) / 2
        for vertex_layer, hyperedge_layer in zip(layers[::2], layers[1::2]):
            y = -len(vertex_layer) / 2
            for vertex in vertex_layer:
                hypergraph.vertex_coords[vertex] = (x, y)
                y += 1
            x += 1

            y = -len(hyperedge_layer) / 2
            for hyperedge in hyperedge_layer:
                hypergraph.hyperedge_coords[hyperedge] = (x, y)
                y += 1
            x += 1

        y = -len(layers[-1]) / 2
        for vertex in layers[-1]:
            hypergraph.vertex_coords[vertex] = (x, y)
            y += 1
        x += 1

        for vertex in hypergraph.inputs:
            x, y = hypergraph.vertex_coords[vertex]
            hypergraph.vertex_coords[vertex] = (x - 1, y)
        for vertex in hypergraph.outputs:
            x, y = hypergraph.vertex_coords[vertex]
            hypergraph.vertex_coords[vertex] = (x + 1, y)

        return hypergraph

    def draw_matplotlib(self, figsize: tuple[int, int] = (8, 4)):
        """Draw the hypergraph in matplotlib."""
        hypergraph = self.compute_coordinates()

        fig, ax = plt.subplots(figsize=figsize)

        for vertex, coords in enumerate(hypergraph.vertex_coords):
            if ((len(hypergraph.vertex_sources(vertex)) == 1 and
                len(hypergraph.vertex_targets(vertex)) == 1)
                    or vertex in hypergraph.inputs + hypergraph.outputs):
                radius = 1e-3
            else:
                radius = 0.05

            ax.add_patch(Circle(coords, radius, fc='black'))
            x, y = coords
            y -= 0.1
            ax.annotate(hypergraph.vertex_labels[vertex],
                        (x, y),
                        ha='center', va='center')

        for hyperedge, coords in enumerate(hypergraph.hyperedge_coords):
            label = hypergraph.hyperedge_labels[hyperedge]
            is_identity = label.startswith('_id_')
            is_cap = label.startswith('_cap_')
            is_cup = label.startswith('_cup_')
            is_spider = label.startswith('_spider_')
            box_width = 0 if is_identity or is_spider else 0.5
            box_height = 0 if is_identity or is_spider else 0.5
            cx, cy = coords
            x = cx - box_width / 2
            y = cy - box_height / 2
            if is_spider:
                ax.add_patch(Circle(coords, 0.05, fc='black'))
            else:
                ax.add_patch(Rectangle((x, y), box_width, box_height,
                                       alpha=0 if is_cap or is_cup else 1))
            if not (is_identity or is_cap or is_cup or is_spider):
                ax.annotate(hypergraph.hyperedge_labels[hyperedge],
                            (cx, cy),
                            ha='center', va='center')

            if is_cap or is_cup:
                start_y = cy - 0.4 * box_height
                end_y = cy + 0.4 * box_height
                start_x = x + box_width if is_cap else x
                control_x = x if is_cap else x + box_width
                path = Path([(start_x, start_y),  # start point
                             (control_x, start_y),
                             (control_x, end_y),
                             (start_x,
                              end_y)],  # end point
                            [Path.MOVETO] + [Path.CURVE4] * 3)
                wire = PathPatch(path, fc='none')
                ax.add_patch(wire)

            sources = hypergraph.hyperedge_sources(hyperedge)
            for port, vertex in enumerate(sources):
                start_x, start_y = hypergraph.vertex_coords[vertex]
                end_x = cx - box_width / 2
                if len(sources) > 1:
                    end_y = cy + (
                        0.8 * box_height
                        * (port / (len(sources) - 1) - 0.5)
                    )
                else:
                    end_y = cy
                dx = abs(start_x - end_x)

                # Create the Path object for the cubic Bezier curve
                path = Path([(start_x, start_y),  # start point
                            (start_x + dx * 0.4, start_y),  # control point 1
                            (end_x - dx * 0.4, end_y,),  # control point 2
                            (end_x, end_y)],  # end point
                            [Path.MOVETO] + [Path.CURVE4] * 3)
                wire = PathPatch(path, fc='none')
                ax.add_patch(wire)

            targets = hypergraph.hyperedge_targets(hyperedge)
            for port, vertex in enumerate(targets):
                start_x = cx + box_width / 2
                if len(targets) > 1:
                    start_y = cy + (
                        0.8 * box_height
                        * (port / (len(targets) - 1) - 0.5)
                    )
                else:
                    start_y = cy
                end_x, end_y = hypergraph.vertex_coords[vertex]
                dx = abs(start_x - end_x)

                # Create the Path object for the cubic Bezier curve
                path = Path([(start_x, start_y),  # start point
                            (start_x + dx * 0.4, start_y),  # control point 1
                            (end_x - dx * 0.4, end_y,),  # control point 2
                            (end_x, end_y)],  # end point
                            [Path.MOVETO] + [Path.CURVE4] * 3)
                wire = PathPatch(path, fc='none')
                ax.add_patch(wire)

        # Set the aspect ratio and auto-adjust limits of the plot
        ax.set_aspect('auto', 'box')
        ax.autoscale_view()

        # Hide the axis ticks and labels, but keep the bounding box
        plt.tick_params(axis='both', which='both',
                        bottom=False, left=False,
                        labelbottom=False, labelleft=False)

        # Invert the y axis
        ax.invert_yaxis()

        # Use all available space in plot window
        fig.tight_layout()
