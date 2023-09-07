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
"""Hypergraph class defining a string diagram."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field

from matplotlib.patches import Circle, PathPatch, Rectangle  # type: ignore
from matplotlib.path import Path  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


@dataclass
class Hypergraph:
    """Hypergraph class, defining a string diagram in a monoidal category.

    This particular flavour of hypergraph is directed, with a total ordering
    on each of the input vertices and output vertices for each hyperedge,
    hence hyperedge inputs and outputs are stored as lists.
    """

    vertices: set[int]
    vertex_sources: dict[int, set[tuple[int, int]]]
    vertex_targets: dict[int, set[tuple[int, int]]]
    vertex_labels: dict[int, str]

    hyperedges: set[int]
    hyperedge_sources: dict[int, list[int]]
    hyperedge_targets: dict[int, list[int]]
    hyperedge_labels: dict[int, str]

    inputs: list[int]
    outputs: list[int]

    vertex_coords: dict[int, tuple[float, float]] = field(default_factory=dict)
    hyperedge_coords: dict[int, tuple[float, float]] = field(
        default_factory=dict)

    @property
    def normal_form(self) -> Hypergraph:
        """Remove all identity hyperedges."""
        normal_form = deepcopy(self)
        remove_hyperedges = set()
        for hyperedge in normal_form.hyperedges:
            if normal_form.hyperedge_labels[hyperedge].startswith('id'):
                source_vertex = normal_form.hyperedge_sources[hyperedge][0]
                target_vertex = normal_form.hyperedge_targets[hyperedge][0]
                normal_form.vertex_targets[source_vertex].remove(
                    (hyperedge, 0))
                normal_form.vertex_sources[target_vertex].remove(
                    (hyperedge, 0))
                normal_form.quotient_vertices(source_vertex, target_vertex)
                remove_hyperedges.add(hyperedge)
        for hyperedge in remove_hyperedges:
            normal_form.remove_hyperedge(hyperedge)
        normal_form.vertex_coords.clear()
        normal_form.hyperedge_coords.clear()
        return normal_form

    def quotient_vertices(self, vertex1: int, vertex2: int) -> None:
        """Merge vertex2 into vertex1."""
        vertex2_sources = self.vertex_sources.pop(vertex2)
        vertex2_targets = self.vertex_targets.pop(vertex2)

        for hyperedge, port in vertex2_sources:
            self.hyperedge_targets[hyperedge][port] = vertex1
        for hyperedge, port in vertex2_targets:
            self.hyperedge_sources[hyperedge][port] = vertex1

        self.vertex_sources[vertex1].update(vertex2_sources)
        self.vertex_targets[vertex1].update(vertex2_targets)

        self.vertex_labels.pop(vertex2)
        self.vertices.remove(vertex2)

    def add_vertex(self, label: str) -> int:
        """Add a vertex to the hypergraph."""
        # Give the vertex a unique integer identifier
        vertex_id = max(self.vertices) + 1
        self.vertices.add(vertex_id)
        self.vertex_sources[vertex_id] = set()
        self.vertex_targets[vertex_id] = set()
        self.vertex_labels[vertex_id] = label
        return vertex_id

    def add_hyperedge(self, label: str) -> int:
        """Add a hyperedge to the hypergraph."""
        # Give the hyperedge a unique integer identifier
        hyperedge_id = max(self.hyperedges) + 1
        self.hyperedges.add(hyperedge_id)
        self.hyperedge_sources[hyperedge_id] = []
        self.hyperedge_targets[hyperedge_id] = []
        self.hyperedge_labels[hyperedge_id] = label
        return hyperedge_id

    def remove_vertex(self, vertex: int) -> None:
        """Remove a vertex from the hypergraph."""
        self.vertices.remove(vertex)
        self.vertex_sources.pop(vertex)
        self.vertex_targets.pop(vertex)
        self.vertex_labels.pop(vertex)
        self.inputs = [i for i in self.inputs if i != vertex]
        self.outputs = [o for o in self.outputs if o != vertex]
        if vertex in self.vertex_coords.keys():
            self.vertex_coords.pop(vertex)

    def remove_hyperedge(self, hyperedge: int) -> None:
        """Remove a hyperedge from the hypergraph."""
        self.hyperedges.remove(hyperedge)
        self.hyperedge_sources.pop(hyperedge)
        self.hyperedge_targets.pop(hyperedge)
        self.hyperedge_labels.pop(hyperedge)
        if hyperedge in self.hyperedge_coords.keys():
            self.hyperedge_coords.pop(hyperedge)

    def set_hyperedge_sources(self, hyperedge: int,
                              vertices: list[int]) -> None:
        """Set the sources of a hyperedge."""
        self.hyperedge_sources[hyperedge] = vertices

    def set_hyperedge_targets(self, hyperedge: int,
                              vertices: list[int]) -> None:
        """Set the targets of a hyperedge."""
        self.hyperedge_targets[hyperedge] = vertices

    def set_vertex_sources(self, vertex: int,
                           hyperedges_and_ports: set[tuple[int, int]]) -> None:
        """Set the sources of a vertex."""
        self.vertex_sources[vertex] = hyperedges_and_ports

    def set_vertex_targets(self, vertex: int,
                           hyperedges_and_ports: set[tuple[int, int]]) -> None:
        """Set the targets of a vertex."""
        self.vertex_targets[vertex] = hyperedges_and_ports

    def layer_decomposition(self) -> list[list[int]]:
        """Decompose this hypergraph into layers.

        Decompose this hypergraph into alternating layers of vertices and
        hyperedges, in a way that is 'forward-directed': targets of any vertex
        or hyperedge lies in a subsequent layer.

        This method requires that the hypergraph is acyclic.
        """
        unplaced_vertices = self.vertices.difference(self.inputs)
        unplaced_hyperedges = self.hyperedges.copy()

        # Input and zero-source vertices form the first layer
        layers: list[list[int]] = [self.inputs.copy()]
        for vertex_id in unplaced_vertices:
            if len(self.vertex_sources[vertex_id]) == 0:
                layers[0].append(vertex_id)
                unplaced_vertices.remove(vertex_id)

        # Place the remaining hyperedges and vertices in alternating layers
        while len(unplaced_vertices) + len(unplaced_hyperedges) > 0:
            ready_hyperedges = []  # Next hyperedge layer

            # If all source vertices are in previous layers, then an unplaced
            # hyperedge can be placed in the next layer
            for hyperedge in unplaced_hyperedges:
                if all(vid not in unplaced_vertices
                       for vid in self.hyperedge_sources[hyperedge]):
                    ready_hyperedges.append(hyperedge)

            # If a target hyperedge of any vertex in the most recent vertex
            # layer is not ready to be placed, traverse the layer with the
            # appropriate number of identity hyperedges
            for vertex in layers[-1]:
                remove_targets: set[tuple[int, int]] = set()
                add_targets: set[tuple[int, int]] = set()
                for target, port in self.vertex_targets[vertex]:
                    if target not in ready_hyperedges:
                        # Add a new vertex to be placed
                        # in the next vertex layer
                        vertex_label = self.vertex_labels[vertex]
                        new_vertex = self.add_vertex(vertex_label)
                        unplaced_vertices.add(new_vertex)
                        # Add an identity hyperedge to connect
                        # the orignal vertex to the new vertex
                        identity = self.add_hyperedge(f'id_{vertex_label}')

                        # The only source vertex of this identity hyperedge is
                        # the original vertex that occured in the previous
                        # layer, hence it is ready to be placed
                        ready_hyperedges.append(identity)

                        # Update the connectivity information for the target
                        # hyperedge, new vertex and identity hyperedge
                        self.hyperedge_sources[target][port] = new_vertex
                        self.set_vertex_sources(new_vertex, {(identity, 0)})
                        self.set_vertex_targets(new_vertex, {(target, port)})
                        self.set_hyperedge_sources(identity, [vertex])
                        self.set_hyperedge_targets(identity, [new_vertex])

                        # Keep track of connectivity information of original
                        # vertex that needs to be updated
                        remove_targets.add((target, port))
                        add_targets.add((identity, 0))

                # Update connectivity information of original vertex
                self.vertex_targets[vertex].difference_update(remove_targets)
                self.vertex_targets[vertex].update(add_targets)

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
                        identity = self.add_hyperedge(f'id_{vertex_label}')

                        # The only source vertex of this identity hyperedge is
                        # the original vertex that occured in the previous
                        # layer, hence it is ready to be placed
                        ready_hyperedges.append(identity)

                        # Update the connectivity information for the original
                        # vertex, new vertex and identity hyperedge
                        self.set_vertex_targets(vertex, {(identity, 0)})
                        self.set_vertex_sources(new_vertex, {(identity, 0)})
                        self.set_hyperedge_sources(identity, [vertex])
                        self.set_hyperedge_targets(identity, [new_vertex])

                        # Update hypergraph output information
                        self.outputs[port] = new_vertex

            # Next hyperedge layer is now finished,
            # so mark its hyperedges as placed
            unplaced_hyperedges.difference_update(ready_hyperedges)

            ready_vertices = []  # Next vertex layer

            # If all source hyperedges are in previous layers, then an
            # unplaced vertex can be placed in the next layer
            for vertex in unplaced_vertices:
                if all(hid not in unplaced_hyperedges
                       for hid, _ in self.vertex_sources[vertex]):
                    ready_vertices.append(vertex)

            # If a target vertex of any hyperedge in the most recent hyperedge
            # layer is not ready to be placed, traverse the layer with the
            # appropriate number of identity hyperedges
            for hyperedge in ready_hyperedges:
                update_ports = []
                for port, target in enumerate(
                        self.hyperedge_targets[hyperedge]):
                    if target not in ready_vertices:
                        # Add a new vertex to be placed
                        # in the next vertex layer
                        vertex_label = self.vertex_labels[target]
                        new_vertex = self.add_vertex(vertex_label)
                        # Add an identity hyperedge to connect
                        # the new vertex to the original vertex
                        identity = self.add_hyperedge(f'id_{vertex_label}')
                        unplaced_hyperedges.add(identity)

                        # The only source of the new vertex is a ready
                        # hyperedge, hence it is ready to be placed into
                        # the next vertex layer
                        ready_vertices.append(new_vertex)

                        # Update the connectivity information for the target
                        # vertex, new vertex and identity hyperedge
                        self.vertex_sources[target].remove((hyperedge, port))
                        self.vertex_sources[target].add((identity, 0))
                        self.set_vertex_sources(new_vertex,
                                                {(hyperedge, port)})
                        self.set_vertex_targets(new_vertex, {(identity, 0)})
                        self.set_hyperedge_sources(identity, [new_vertex])
                        self.set_hyperedge_targets(identity, [target])

                        # Keep track of connectivity information of original
                        # hyperedge that needs to be updated
                        update_ports.append((port, new_vertex))

                # Update connectivity information of original hyperedge
                for port, new_vertex in update_ports:
                    self.hyperedge_targets[hyperedge][port] = new_vertex

            # Next vertex layer is now finished,
            # so mark its vertices as placed
            unplaced_vertices.difference_update(ready_vertices)

            layers.append(ready_hyperedges)
            layers.append(ready_vertices)

        return layers

    def compute_coordinates(self):
        """Compute x and y coordinates based on a layer decomposition.

        Successive vertex and hyperedge layers are spaced a unit x-coordinate
        apart.

        Vertices and hyperedges in the same layer are spaced a
        unit y-coordinate apart.
        """
        self.vertex_coords.clear()
        self.hyperedge_coords.clear()

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

    def draw_matplotlib(self, figsize: tuple[int, int] = (5, 3)):
        """Draw the hypergraph in matplotlib."""
        self.compute_coordinates()

        fig, ax = plt.subplots(figsize=figsize)

        for vertex, coords in self.vertex_coords.items():
            if (len(self.vertex_sources[vertex]) == 1 and
               len(self.vertex_targets[vertex]) == 1):
                radius = 1e-3
            else:
                radius = 0.05

            ax.add_patch(Circle(coords, radius, fc='black'))
            x, y = coords
            y -= 0.1
            ax.annotate(self.vertex_labels[vertex],
                        (x, y),
                        ha='center', va='center')

        for hyperedge, coords in self.hyperedge_coords.items():
            is_identity = self.hyperedge_labels[hyperedge].startswith('id')
            box_width = 0 if is_identity else 0.5
            box_height = 0 if is_identity else 0.5
            cx, cy = coords
            x = cx - box_width / 2
            y = cy - box_height / 2
            ax.add_patch(Rectangle((x, y), box_width, box_height))
            if not is_identity:
                ax.annotate(self.hyperedge_labels[hyperedge],
                            (cx, cy),
                            ha='center', va='center')

            sources = self.hyperedge_sources[hyperedge]
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

            targets = self.hyperedge_targets[hyperedge]
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

    def print(self):
        """Print layer decomposition of this hypergraph."""
        layers = self.layer_decomposition()
        print('v input v')
        for vertex_layer, hyperedge_layer in zip(layers[::2], layers[1::2]):
            print([self.vertex_labels[vid] for vid in vertex_layer])
            print([self.hyperedge_labels[hid] for hid in hyperedge_layer])
        print([self.vertex_labels[vid] for vid in layers[-1]])
        print('^ output ^')

    def __repr__(self) -> str:
        """Return term notation for layer decomposition of this hypergraph."""
        layers = self.layer_decomposition()

        layer_reprs = (
            ('(' if len(layer) > 1 else '')
            + ' ⨂ '.join(self.hyperedge_labels[hyperedge]
                         for hyperedge in layer)
            + (')' if len(layer) > 1 else '')
            for layer in layers[1::2])

        repr = ' ⨟ '.join(layer_reprs)

        return repr

    @classmethod
    def from_yarrow(cls, yarrow_diagram) -> Hypergraph:
        """Create a hypergraph from a yarrow diagram."""
        G = yarrow_diagram.G

        # Vertices and their labels
        vertices = set(range(G.wn.source))
        vertex_labels = {i: str(G.wn.table[i]) for i in range(G.wn.source)}

        # Hyperedges and their labels
        hyperedges = set(range(G.xn.source))
        hyperedge_labels = {i: str(G.xn.table[i]) for i in range(G.xn.source)}

        # Vertex -> hyperedge connections
        vertex_targets = {
            vertex: {
                (G.xi.table[i], G.pi.table[i])
                for i in range(G.xi.source) if G.wi.table[i] == vertex
            } for vertex in vertices
        }
        hyperedge_sources = {}
        for hyperedge in hyperedges:
            sources_and_ports = sorted(
                (
                    (G.wi.table[i], G.pi.table[i])
                    for i in range(G.wi.source)
                    if G.xi.table[i] == hyperedge
                ),
                key=lambda wp: wp[1]
            )
            hyperedge_sources[hyperedge] = [sp[0] for sp in sources_and_ports]

        # Hyperedge -> vertex connections
        vertex_sources = {
            vertex: {
                (G.xo.table[i], G.po.table[i])
                for i in range(G.xo.source) if G.wo.table[i] == vertex
            } for vertex in vertices
        }
        hyperedge_targets = {}
        for hyperedge in hyperedges:
            sources_and_ports = sorted(
                (
                    (G.wo.table[i], G.po.table[i])
                    for i in range(G.wo.source)
                    if G.xo.table[i] == hyperedge
                ),
                key=lambda wp: wp[1]
            )
            hyperedge_targets[hyperedge] = [sp[0] for sp in sources_and_ports]

        # Inputs and outputs
        s = yarrow_diagram.s
        t = yarrow_diagram.t
        inputs = [s.table[i] for i in range(s.source)]
        outputs = [t.table[i] for i in range(t.source)]

        return cls(
            vertices,
            vertex_sources,
            vertex_targets,
            vertex_labels,

            hyperedges,
            hyperedge_sources,
            hyperedge_targets,
            hyperedge_labels,

            inputs,
            outputs
        )
