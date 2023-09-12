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

from hyperstrings.hypergraph.immutable import Array
from hyperstrings.hypergraph.immutable import backend
from hyperstrings.hypergraph.immutable import Hyperedge, Vertex
from hyperstrings.hypergraph.immutable import Label
from hyperstrings.hypergraph.mutable import MutableHypergraph


class DrawableHypergraph(MutableHypergraph):
    """Drawable hypergraph implementation."""

    def __init__(self) -> None:
        """Initialize a `DrawableHypergraph` Instance."""
        super().__init__()
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

    def remove_vertices(self, vertices: list[Vertex]) -> None:
        """Remove vertices from the hypergraph."""
        keep_vertices = [v for v in self.vertices()
                         if v not in vertices]
        self.vertex_coords = self.vertex_coords.take(keep_vertices)
        return super().remove_vertices(vertices)

    def remove_hyperedges(self, hyperedges: list[Hyperedge]) -> None:
        """Remove hyperedges from the hypergraph."""
        keep_hyperedges = [h for h in self.hyperedges()
                           if h not in hyperedges]
        self.hyperedge_coords = self.hyperedge_coords.take(keep_hyperedges)
        return super().remove_hyperedges(hyperedges)

    def layer_decomposition(self) -> list[list[Vertex | Hyperedge]]:
        """Decompose the hypergraph into layers."""
        layers: list[list[Vertex | Hyperedge]] = [self.inputs.copy()]

        unplaced_vertices = set(self.vertices()
                                ).difference(self.inputs.copy())
        unplaced_hyperedges = set(self.hyperedges())

        # Input and zero-source vertices form the first layer
        zero_source_vertices = set()
        for vertex in unplaced_vertices:
            if len(self.vertex_sources(vertex)) == 0:
                layers[0].append(vertex)
                zero_source_vertices.add(vertex)
        unplaced_vertices -= zero_source_vertices

        # Place the remaining hyperedges and vertices in alternating layers
        while len(unplaced_vertices) + len(unplaced_hyperedges) > 0:
            ready_hyperedges = []  # Next hyperedge layer

            # If all source vertices are in previous layers, then an unplaced
            # hyperedge can be placed in the next layer
            for hyperedge in unplaced_hyperedges:
                if all(v not in unplaced_vertices
                       for v in self.hyperedge_sources(hyperedge)):
                    ready_hyperedges.append(hyperedge)

            # If a target hyperedge of any vertex in the most recent vertex
            # layer is not ready to be placed, traverse the layer with the
            # appropriate number of identity hyperedges
            for vertex in layers[-1]:
                for target, port in self.vertex_targets(vertex):
                    if target not in ready_hyperedges:
                        # Add a new vertex to be placed
                        # in the next vertex layer
                        vertex_label = self.vertex_labels[vertex]
                        new_vertex = self.add_vertex(vertex_label)
                        unplaced_vertices.add(new_vertex)
                        # Add an identity hyperedge to connect
                        # the orignal vertex to the new vertex
                        identity = self.add_hyperedge(f'_id_{vertex_label}')

                        # The only source vertex of this identity hyperedge is
                        # the original vertex that occured in the previous
                        # layer, hence it is ready to be placed
                        ready_hyperedges.append(identity)

                        # Update the connectivity information for the target
                        # hyperedge, new vertex and identity hyperedge
                        self.disconnect_source(vertex, target, port)
                        self.connect_source(vertex, identity, 0)
                        self.connect_target(new_vertex, identity, 0)
                        self.connect_source(new_vertex, target, port)

                # If any vertex in the most recent vertex layer is an output,
                # traverse the next hyperedge layer with an identity hyperedge
                for port, vertex in enumerate(self.outputs):
                    if vertex in layers[-1]:
                        # Add a new vertex to be placed
                        # in the next vertex layer
                        vertex_label = self.vertex_labels[vertex]
                        new_vertex = self.add_vertex(vertex_label)
                        unplaced_vertices.add(new_vertex)
                        # Add an identity hyperedge to connect
                        # the orignal vertex to the new vertex
                        identity = self.add_hyperedge(f'_id_{vertex_label}')

                        # The only source vertex of this identity hyperedge is
                        # the original vertex that occured in the previous
                        # layer, hence it is ready to be placed
                        ready_hyperedges.append(identity)

                        # Update the connectivity information
                        self.connect_source(vertex, identity, 0)
                        self.connect_target(new_vertex, identity, 0)

                        # Update hypergraph output information
                        self.outputs[port] = new_vertex

            # Next hyperedge layer is now finished,
            # so mark its hyperedges as placed
            unplaced_hyperedges.difference_update(ready_hyperedges)

            ready_vertices = []  # Next vertex layer

            # If all source hyperedges are in previous layers, then an
            # unplaced vertex can be placed in the next layer
            for vertex in unplaced_vertices:
                if all(h not in unplaced_hyperedges
                       for h, _ in self.vertex_sources(vertex)):
                    ready_vertices.append(vertex)

            # If a target vertex of any hyperedge in the most recent hyperedge
            # layer is not ready to be placed, traverse the layer with the
            # appropriate number of identity hyperedges
            for hyperedge in ready_hyperedges:
                for port, target in enumerate(
                        self.hyperedge_targets(hyperedge)):
                    if target not in ready_vertices:
                        # Add a new vertex to be placed
                        # in the next vertex layer
                        vertex_label = self.vertex_labels[target]
                        new_vertex = self.add_vertex(vertex_label)
                        # Add an identity hyperedge to connect
                        # the new vertex to the original vertex
                        identity = self.add_hyperedge(f'_id_{vertex_label}')
                        unplaced_hyperedges.add(identity)

                        # The only source of the new vertex is a ready
                        # hyperedge, hence it is ready to be placed into
                        # the next vertex layer
                        ready_vertices.append(new_vertex)

                        # Update the connectivity information
                        self.disconnect_target(target, hyperedge, port)
                        self.connect_target(new_vertex, hyperedge, port)
                        self.connect_source(new_vertex, identity, 0)
                        self.connect_target(target, identity, 0)

            # Next vertex layer is now finished,
            # so mark its vertices as placed
            unplaced_vertices.difference_update(ready_vertices)

            layers.append(ready_hyperedges)
            layers.append(ready_vertices)

        # Minimize wire crossings
        for layer_num, layer in zip(range(1, len(layers)), layers[1:]):
            scores: dict[int, float] = {}
            if layer_num % 2 == 0:  # vertex layer
                for vertex in layer:
                    # score = mean position of source hyperedges in
                    # previous layer
                    source_hyperedges = self.vertex_sources(vertex)
                    score = (sum(layers[layer_num - 1].index(hyperedge)
                                 for hyperedge, _
                                 in source_hyperedges)
                             / (len(source_hyperedges) + 1e-12))
                    scores[vertex] = score
            else:  # hyperedge layer
                for hyperedge in layer:
                    # score = mean position of source vertices in
                    # previous layer
                    source_vertices = self.hyperedge_sources(hyperedge)
                    score = (sum(layers[layer_num - 1].index(vertex)
                                 for vertex
                                 in source_vertices)
                             / (len(source_vertices) + 1e-12))
                    scores[hyperedge] = score
            # output order must be preserved
            if layer_num == len(layers):
                max_score = max(s for s in scores.values())
                scores = {k: s/max_score for k, s in scores.items()}

                def output_score(vertex: int) -> float:
                    if vertex in self.outputs:
                        score: float = self.outputs.index(vertex) + 1
                    else:
                        for i in range(len(self.outputs)):
                            if scores[vertex] <= self.outputs[i]:
                                score = i + scores[vertex]
                                return score
                        score = len(self.outputs) + scores[vertex]
                    return score

                layers[layer_num] = sorted(layer, key=output_score)
            else:
                layers[layer_num] = sorted(layer, key=lambda x: scores[x])

        return layers

    def compute_coordinates(self):
        """Compute x and y coordinates based on a layer decomposition.

        Successive vertex and hyperedge layers are spaced a unit x-coordinate
        apart.

        Vertices and hyperedges in the same layer are spaced a
        unit y-coordinate apart.
        """
        layers = self.layer_decomposition()
        x = len(layers) / 2
        for vertex_layer, hyperedge_layer in zip(layers[::2], layers[1::2]):
            y = -len(vertex_layer) / 2
            for vertex in vertex_layer:
                self.vertex_coords[vertex] = (x, y)
                y += 1
            x += 1

            y = -len(hyperedge_layer) / 2
            for hyperedge in hyperedge_layer:
                self.hyperedge_coords[hyperedge] = (x, y)
                y += 1
            x += 1

        y = -len(layers[-1]) / 2
        for vertex in layers[-1]:
            self.vertex_coords[vertex] = (x, y)
            y += 1
        x += 1

        for vertex in self.inputs:
            x, y = self.vertex_coords[vertex]
            self.vertex_coords[vertex] = (x - 1, y)
        for vertex in self.outputs:
            x, y = self.vertex_coords[vertex]
            self.vertex_coords[vertex] = (x + 1, y)

    def draw_matplotlib(self, figsize: tuple[int, int] = (8, 4)):
        """Draw the hypergraph in matplotlib."""
        self.compute_coordinates()

        fig, ax = plt.subplots(figsize=figsize)

        for vertex, coords in enumerate(self.vertex_coords):
            if ((len(self.vertex_sources(vertex)) == 1 and
                len(self.vertex_targets(vertex)) == 1)
                    or vertex in self.inputs + self.outputs):
                radius = 1e-3
            else:
                radius = 0.05

            ax.add_patch(Circle(coords, radius, fc='black'))
            x, y = coords
            y -= 0.1
            ax.annotate(self.vertex_labels[vertex],
                        (x, y),
                        ha='center', va='center')

        for hyperedge, coords in enumerate(self.hyperedge_coords):
            is_identity = self.hyperedge_labels[hyperedge].startswith('_id_')
            is_cap = self.hyperedge_labels[hyperedge].startswith('_cap_')
            is_cup = self.hyperedge_labels[hyperedge].startswith('_cup_')
            is_spider = self.hyperedge_labels[hyperedge].startswith('_spider_')
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
                ax.annotate(self.hyperedge_labels[hyperedge],
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

            sources = self.hyperedge_sources(hyperedge)
            for port, vertex in enumerate(sources):
                start_x, start_y = self.vertex_coords[vertex]
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

            targets = self.hyperedge_targets(hyperedge)
            for port, vertex in enumerate(targets):
                start_x = cx + box_width / 2
                if len(targets) > 1:
                    start_y = cy + (
                        0.8 * box_height
                        * (port / (len(targets) - 1) - 0.5)
                    )
                else:
                    start_y = cy
                end_x, end_y = self.vertex_coords[vertex]
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
