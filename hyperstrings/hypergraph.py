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

    def __post_init__(self) -> None:
        """Perform post-initialization operations."""
        self.check_consistency()

    def sequential_compose(self, other: Hypergraph) -> Hypergraph:
        """Form the sequential composition self;other."""
        assert len(self.outputs) == len(other.inputs)
        assert all(self.vertex_labels[o] == other.vertex_labels[i]
                   for o, i in zip(self.outputs, other.inputs))
        composed, other = self.remove_index_collisions(other)
        old_outputs = composed.outputs.copy()
        composed.outputs = other.outputs.copy()
        for o, i in zip(old_outputs, other.inputs.copy()):
            composed.quotient_vertices(o, i)
        return composed

    def __rshift__(self, other: Hypergraph) -> Hypergraph:
        """Sequentially compose `self` with `other`."""
        return self.sequential_compose(other)

    def parallel_compose(self, other: Hypergraph) -> Hypergraph:
        """Form the parallel composition self⊗other."""
        composed, other = self.remove_index_collisions(other)
        composed.inputs += other.inputs.copy()
        composed.outputs += other.outputs.copy()
        return composed

    def __matmul__(self, other: Hypergraph) -> Hypergraph:
        """Parallel compose `self` with `other`."""
        return self.parallel_compose(other)

    def remove_index_collisions(self, other: Hypergraph
                                ) -> tuple[Hypergraph, Hypergraph]:
        """Change `other`'s vertex indices to avoid collisions with `self`.

        Used for composing hypergraphs.
        """
        max_vertex_index = max(self.vertices | other.vertices)
        max_hyperedge_index = max(self.hyperedges | other.hyperedges)
        other = deepcopy(other)
        for i, vertex in enumerate(other.vertices):
            new_index = max_vertex_index + i + 1
            other.change_vertex_index(vertex, new_index)
        for i, hyperedge in enumerate(other.hyperedges):
            new_index = max_hyperedge_index + i + 1
            other.change_hyperedge_index(hyperedge, new_index)
        composed = deepcopy(self)
        composed.vertices |= other.vertices
        composed.vertex_sources |= other.vertex_sources
        composed.vertex_targets |= other.vertex_targets
        composed.vertex_labels |= other.vertex_labels
        composed.hyperedges |= other.hyperedges
        composed.hyperedge_sources |= other.hyperedge_sources
        composed.hyperedge_targets |= other.hyperedge_targets
        composed.hyperedge_labels |= other.hyperedge_labels
        return composed, other

    def check_consistency(self) -> None:
        """Check consistency of connectivity information."""
        assert (self.vertices
                == set(self.vertex_sources.keys())
                == set(self.vertex_targets.keys())
                == set(self.vertex_labels.keys()))
        assert (self.hyperedges
                == set(self.hyperedge_sources.keys())
                == set(self.hyperedge_targets.keys())
                == set(self.hyperedge_labels.keys()))
        for vertex in self.vertices:
            assert all(self.hyperedge_sources[hyperedge][port] == vertex
                       for hyperedge, port in self.vertex_targets[vertex])
            assert all(self.hyperedge_targets[hyperedge][port] == vertex
                       for hyperedge, port in self.vertex_sources[vertex])
        for hyperedge in self.hyperedges:
            assert all((hyperedge, port) in self.vertex_sources[vertex]
                       for port, vertex
                       in enumerate(self.hyperedge_targets[hyperedge]))
            assert all((hyperedge, port) in self.vertex_targets[vertex]
                       for port, vertex
                       in enumerate(self.hyperedge_sources[hyperedge]))

    @classmethod
    def simple_init(cls,
                    vertex_labels: dict[int, str],
                    hyperedge_labels: dict[int, str],
                    hyperedge_sources: dict[int, list[int]],
                    hyperedge_targets: dict[int, list[int]],
                    inputs: list[int],
                    outputs: list[int]
                    ):
        """Create a Hypergraph instance with minimal information."""
        vertices = set(vertex_labels.keys())
        hyperedges = set(hyperedge_labels.keys())

        vertex_sources: dict[int, set[tuple[int, int]]]
        vertex_sources = {vertex: set() for vertex in vertices}
        vertex_targets: dict[int, set[tuple[int, int]]]
        vertex_targets = {vertex: set() for vertex in vertices}

        for hyperedge in hyperedges:
            for port, vertex in enumerate(hyperedge_sources[hyperedge]):
                vertex_targets[vertex].add((hyperedge, port))
            for port, vertex in enumerate(hyperedge_targets[hyperedge]):
                vertex_sources[vertex].add((hyperedge, port))

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

    def wires(self):
        """Return a set of hyperedge port to vertex connections.

        Wires are in the following format:
            `(vertex, hyperedge, port, direction)`
        If `direction` is -1, the wire connects a vertex to a hyperedge source
        port. If `direction` is 1, the wire connects a hyperedge target port
        to a vertex.
        """
        wires = set()
        for hyperedge in self.hyperedges:
            for port, vertex in enumerate(self.hyperedge_sources[hyperedge]):
                wires.add((vertex, hyperedge, port, -1))
            for port, vertex in enumerate(self.hyperedge_targets[hyperedge]):
                wires.add((vertex, hyperedge, port, 1))
        return wires

    def change_vertex_index(self, vertex: int, new_index: int) -> None:
        """Change the integer identitifier of a vertex."""
        assert new_index not in self.vertices
        self.vertices.remove(vertex)
        self.vertices.add(new_index)
        self.vertex_sources[new_index] = self.vertex_sources.pop(vertex)
        self.vertex_targets[new_index] = self.vertex_targets.pop(vertex)
        self.vertex_labels[new_index] = self.vertex_labels.pop(vertex)
        for hyperedge, port in self.vertex_sources[new_index]:
            self.hyperedge_targets[hyperedge][port] = new_index
        for hyperedge, port in self.vertex_targets[new_index]:
            self.hyperedge_sources[hyperedge][port] = new_index
        if vertex in self.inputs:
            self.inputs = [new_index if v == vertex else v
                           for v in self.inputs]
        if vertex in self.outputs:
            self.outputs = [new_index if v == vertex else v
                            for v in self.outputs]

    def change_hyperedge_index(self, hyperedge: int, new_index: int) -> None:
        """Change the integer identitifier of a hyperedge."""
        assert new_index not in self.hyperedges
        self.hyperedges.remove(hyperedge)
        self.hyperedges.add(new_index)
        sources = self.hyperedge_sources.pop(hyperedge)
        targets = self.hyperedge_targets.pop(hyperedge)
        self.hyperedge_sources[new_index] = sources
        self.hyperedge_targets[new_index] = targets
        self.hyperedge_labels[new_index] = self.hyperedge_labels.pop(hyperedge)
        for port, vertex in enumerate(sources):
            self.vertex_targets[vertex].remove((hyperedge, port))
            self.vertex_targets[vertex].add((new_index, port))
        for port, vertex in enumerate(targets):
            self.vertex_sources[vertex].remove((hyperedge, port))
            self.vertex_sources[vertex].add((new_index, port))

    def reset_indices(self) -> None:
        """Reset vertex and hyperedge identifiers to make them contigious."""
        vertices = sorted(self.vertices)
        for i, vertex in enumerate(vertices):
            if vertex != i:
                self.change_vertex_index(vertex, i)
        hyperedges = sorted(self.hyperedges)
        for i, hyperedge in enumerate(hyperedges):
            if hyperedge != i:
                self.change_hyperedge_index(hyperedge, i)

    def normal_form(self, in_place: bool = False) -> Hypergraph:
        """Remove all identity hyperedges."""
        normal_form = self if in_place else deepcopy(self)
        remove_hyperedges = set()
        for hyperedge in normal_form.hyperedges:
            if normal_form.hyperedge_labels[hyperedge].startswith('_id_'):
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

    def quotient_vertices(self, *vertices: int) -> None:
        """Merge vertex2 into vertex1."""
        vertex1 = vertices[0]
        for vertex2 in vertices[1:]:
            vertex2_sources = self.vertex_sources[vertex2]
            vertex2_targets = self.vertex_targets[vertex2]

            for hyperedge, port in vertex2_sources:
                self.hyperedge_targets[hyperedge][port] = vertex1
            for hyperedge, port in vertex2_targets:
                self.hyperedge_sources[hyperedge][port] = vertex1

            self.vertex_sources[vertex1].update(vertex2_sources)
            self.vertex_targets[vertex1].update(vertex2_targets)

            if vertex2 in self.inputs:
                self.inputs = [
                    vertex1 if v == vertex2 else v for v in self.inputs
                ]
            if vertex2 in self.outputs:
                self.outputs = [
                    vertex1 if v == vertex2 else v for v in self.outputs
                ]

            self.remove_vertex(vertex2)

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

    def is_source_monogamous(self) -> bool:
        """Check whether this hypergraph is source-monogamous.

        This means that all input vertices have zero sources and all
        non-input vertices have at most one source hyperedge.
        """
        return all(
            len(self.vertex_sources[vertex]) == 1
            if vertex not in self.inputs else
            len(self.vertex_sources[vertex]) == 0
            for vertex in self.vertices
        )

    def is_target_monogamous(self) -> bool:
        """Check whether this hypergraph is target-monogamous.

        This means that all output vertices have zero targets and all
        non-output vertices have at most one target hyperedge.
        """
        return all(
            len(self.vertex_targets[vertex]) <= 1
            if vertex not in self.outputs else
            len(self.vertex_targets[vertex]) == 0
            for vertex in self.vertices
        )

    def is_monogamous(self) -> bool:
        """Check whether this hypergraph is monogamous.

        This means that all input vertices have zero sources and exactly one
        target, all output vertices have exactly one source and zero targets,
        and all other vertices have at most one source and at most one target.
        """
        is_monogamous = True
        non_boundary_vertices = self.vertices.difference(
            self.inputs + self.outputs
        )
        # Non-boundary vertices
        is_monogamous &= all(
            (len(self.vertex_sources[vertex]) <= 1
             and len(self.vertex_targets[vertex]) <= 1)
            for vertex in non_boundary_vertices
        )
        # Input vertices
        is_monogamous &= all(
            (len(self.vertex_sources[vertex]) == 0
             and len(self.vertex_targets[vertex]) == 1)
            for vertex in self.inputs
        )
        # Output vertices
        is_monogamous &= all(
            (len(self.vertex_sources[vertex]) == 1
             and len(self.vertex_targets[vertex]) == 0)
            for vertex in self.outputs
        )
        return is_monogamous

    def make_spiders_explicit(self, in_place: bool = False) -> Hypergraph:
        """Make all non-monogamous vertices into explicit hyperedges.

        The resulting hypergraph is monogamous.
        """
        hypergraph = self if in_place else deepcopy(self)
        vertices = hypergraph.vertices.copy()
        for vertex in vertices:
            sources = hypergraph.vertex_sources[vertex]
            targets = hypergraph.vertex_targets[vertex]
            if len(sources) > 1 or len(targets) > 1:
                label = self.vertex_labels[vertex]
                spider = hypergraph.add_hyperedge(
                    f'_spider_{label}_{len(sources)}_{len(targets)}')
                for i, (hyperedge, port) in enumerate(sources):
                    new_vertex = hypergraph.add_vertex(label)
                    hypergraph.vertex_sources[new_vertex].add(
                        (hyperedge, port))
                    hypergraph.vertex_targets[new_vertex].add(
                        (spider, i))
                    hypergraph.hyperedge_targets[hyperedge][port] = new_vertex
                    hypergraph.hyperedge_sources[spider].append(new_vertex)
                for i, (hyperedge, port) in enumerate(targets):
                    new_vertex = hypergraph.add_vertex(label)
                    hypergraph.vertex_targets[new_vertex].add(
                        (hyperedge, port))
                    hypergraph.vertex_sources[new_vertex].add(
                        (spider, i))
                    hypergraph.hyperedge_sources[hyperedge][port] = new_vertex
                    hypergraph.hyperedge_targets[spider].append(new_vertex)
                if vertex in hypergraph.inputs:  # assumes zero input spider
                    new_vertex = hypergraph.add_vertex(label)
                    hypergraph.vertex_targets[new_vertex].add((spider, 0))
                    hypergraph.hyperedge_sources[spider].append(new_vertex)
                    hypergraph.inputs = [new_vertex if v == vertex else v
                                         for v in hypergraph.inputs]
                if vertex in hypergraph.outputs:  # assumes zero output spider
                    new_vertex = hypergraph.add_vertex(label)
                    hypergraph.vertex_sources[new_vertex].add((spider, 0))
                    hypergraph.hyperedge_targets[spider].append(new_vertex)
                    hypergraph.inputs = [new_vertex if v == vertex else v
                                         for v in hypergraph.outputs]
                hypergraph.remove_vertex(vertex)
        return hypergraph

    def make_spiders_implicit(self, in_place: bool = False) -> Hypergraph:
        """Replace all explicit spider boxes with vertices."""
        hypergraph = self if in_place else deepcopy(self)
        spiders = set(
            hyperedge for hyperedge in hypergraph.hyperedges
            if hypergraph.hyperedge_labels[hyperedge].startswith('_spider'))
        for spider in spiders:
            sources = hypergraph.hyperedge_sources[spider]
            targets = hypergraph.hyperedge_targets[spider]
            for port, vertex in enumerate(sources):
                hypergraph.vertex_targets[vertex].remove((spider, port))
            for port, vertex in enumerate(targets):
                hypergraph.vertex_sources[vertex].remove((spider, port))
            hypergraph.quotient_vertices(*(sources + targets))
            hypergraph.remove_hyperedge(spider)
        return hypergraph

    def children(self, vertex: int,
                 visited_children: set[int] | None = None) -> set[int]:
        """Return the set of children of a vertex."""
        if visited_children is None:
            visited_children = set()
        new_children: set[int] = set()
        target_hyperedges = set(
            hyperedge for hyperedge, _ in self.vertex_targets[vertex])
        for hyperedge in target_hyperedges:
            new_children.update(
                child for child in self.hyperedge_targets[hyperedge]
                if child not in visited_children
            )
        # if all vertices in 1-hop neighbourhood already visited, return
        if len(new_children) == 0:
            return visited_children
        else:
            visited_children.update(new_children)
            return visited_children.union(
                *(self.children(vertex, visited_children)
                  for vertex in new_children)
            )

    def is_acyclic(self) -> bool:
        """Return whether this hypergraph is acyclic."""
        for vertex in self.vertices:
            for child_vertex in self.children(vertex):
                if vertex in self.children(child_vertex):
                    return False
        return True

    def make_cycles_explicit(self, in_place: bool = False) -> Hypergraph:
        """Turn cycles into cups and caps.

        Currently, this method requires a monogamous hypergraph.
        If a cycle exists, introduce a cup, cap and 2 new vertices:
            - `new vertex 1` has source the cap and target the cup
            - `new vertex 2` has source the cap and target the old target
               of vertex for which a cycle was found.
            - vertex for which cycle was found has the same source but now
              has the cup as its new target
        This effectively makes original vertex 1 have no children.
        """
        assert self.is_monogamous()
        hypergraph = self if in_place else deepcopy(self)
        for vertex in hypergraph.vertices:
            for child_vertex in hypergraph.children(vertex):
                if vertex in hypergraph.children(child_vertex):
                    cap = hypergraph.add_hyperedge(
                        f'_cap_{hypergraph.vertex_labels[vertex]}')
                    cup = hypergraph.add_hyperedge(
                        f'_cup_{hypergraph.vertex_labels[vertex]}')
                    new_vertex1 = hypergraph.add_vertex(
                        hypergraph.vertex_labels[vertex])
                    new_vertex2 = hypergraph.add_vertex(
                        hypergraph.vertex_labels[vertex])
                    (hyperedge, port), = tuple(
                        hypergraph.vertex_targets[vertex])
                    hypergraph.vertex_targets[vertex] = {(cup, 1)}
                    hypergraph.vertex_targets[new_vertex1] = {(cup, 0)}
                    hypergraph.hyperedge_sources[cup] = [new_vertex1, vertex]
                    hypergraph.vertex_sources[new_vertex1] = {(cap, 0)}
                    hypergraph.vertex_sources[new_vertex2] = {(cap, 1)}
                    hypergraph.hyperedge_targets[cap] = [
                        new_vertex1, new_vertex2]
                    hypergraph.vertex_targets[new_vertex2].add(
                        (hyperedge, port))
                    hypergraph.hyperedge_sources[hyperedge][port] = new_vertex2
                    return hypergraph.make_cycles_explicit(in_place=True)
        return hypergraph

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
        zero_source_vertices = set()
        for vertex_id in unplaced_vertices:
            if len(self.vertex_sources[vertex_id]) == 0:
                layers[0].append(vertex_id)
                zero_source_vertices.add(vertex_id)
        unplaced_vertices -= zero_source_vertices

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
                        identity = self.add_hyperedge(f'_id_{vertex_label}')

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
                        identity = self.add_hyperedge(f'_id_{vertex_label}')

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
                        identity = self.add_hyperedge(f'_id_{vertex_label}')
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

        # Minimize wire crossings
        for layer_num, layer in zip(range(1, len(layers)), layers[1:]):
            scores: dict[int, float] = {}
            if layer_num % 2 == 0:  # vertex layer
                for vertex in layer:
                    # score = mean position of source hyperedges in
                    # previous layer
                    source_hyperedges = self.vertex_sources[vertex]
                    score = (sum(layers[layer_num - 1].index(hyperedge)
                                 for hyperedge, _
                                 in source_hyperedges)
                             / (len(source_hyperedges) + 1e-12))
                    scores[vertex] = score
            else:  # hyperedge layer
                for hyperedge in layer:
                    # score = mean position of source vertices in
                    # previous layer
                    source_vertices = self.hyperedge_sources[hyperedge]
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

        for vertex, coords in self.vertex_coords.items():
            if ((len(self.vertex_sources[vertex]) == 1 and
                len(self.vertex_targets[vertex]) == 1)
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

        for hyperedge, coords in self.hyperedge_coords.items():
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

    def term_decomposition(self) -> str:
        """Return term notation for layer decomposition of this hypergraph."""
        layers = self.layer_decomposition()

        layer_terms = (
            ('(' if len(layer) > 1 else '')
            + ' ⨂ '.join(self.hyperedge_labels[hyperedge]
                         for hyperedge in layer)
            + (')' if len(layer) > 1 else '')
            for layer in layers[1::2])

        term_decomposition = ' ⨟ '.join(layer_terms)

        return term_decomposition

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

    def to_yarrow(self, normal_form: bool = True):
        """Create a yarrow diagram from this hypergraph."""
        from yarrow import FiniteFunction, BipartiteMultigraph, Diagram
        hypergraph = self.normal_form() if normal_form else self
        hypergraph.reset_indices()

        num_vertices = len(hypergraph.vertices)  # G(W)
        num_hyperedges = len(hypergraph.hyperedges)  # G(X)

        # Inputs and outputs
        s = FiniteFunction(num_vertices, hypergraph.inputs)
        t = FiniteFunction(num_vertices, hypergraph.outputs)

        # Vertex labels
        vertex_labels = list(set(hypergraph.vertex_labels.values()))
        wn = FiniteFunction(
            len(vertex_labels),
            [vertex_labels.index(hypergraph.vertex_labels[vertex])
             for vertex in range(num_vertices)]
        )

        # Hyperedge labels
        hyperedge_labels = list(set(hypergraph.hyperedge_labels.values()))
        xn = FiniteFunction(
            len(hyperedge_labels),
            [hyperedge_labels.index(hypergraph.hyperedge_labels[hyperedge])
             for hyperedge in range(num_hyperedges)]
        )

        # Vertex -> hyperedge connections
        input_connections: list[tuple[int, int, int]] = []
        for hyperedge in hypergraph.hyperedges:
            for port, vertex in enumerate(
                    hypergraph.hyperedge_sources[hyperedge]):
                input_connections.append((vertex, hyperedge, port))
        wi = FiniteFunction(num_vertices,
                            [vhp[0] for vhp in input_connections])
        xi = FiniteFunction(num_hyperedges,
                            [vhp[1] for vhp in input_connections])
        pi = FiniteFunction(None, [vhp[2] for vhp in input_connections])

        # Hyperedge -> vertex connections
        output_connections: list[tuple[int, int, int]] = []
        for hyperedge in hypergraph.hyperedges:
            for port, vertex in enumerate(
                    hypergraph.hyperedge_targets[hyperedge]):
                output_connections.append((vertex, hyperedge, port))
        wo = FiniteFunction(num_vertices,
                            [vhp[0] for vhp in output_connections])
        xo = FiniteFunction(num_hyperedges,
                            [vhp[1] for vhp in output_connections])
        po = FiniteFunction(None, [vhp[2] for vhp in output_connections])

        G = BipartiteMultigraph(wi, wo, xi, xo, wn, pi, po, xn)

        yarrow_diagram = Diagram(s, t, G)

        # TODO: keep track of labels
        return yarrow_diagram

    @classmethod
    def from_discopy(cls, discopy_diagram) -> Hypergraph:
        """Create a hypergraph from a discopy diagram."""
        discopy_hypergraph = discopy_diagram.to_hypergraph()

        vertices: set[int] = set(discopy_hypergraph.wires)
        inputs = [*range(max(vertices) + 1,
                         max(vertices) + 1 + len(discopy_hypergraph.dom))]
        vertices.update(inputs)
        outputs = [*range(max(vertices) + 1,
                          max(vertices) + 1 + len(discopy_hypergraph.cod))]
        vertices.update(outputs)
        hyperedges = set(range(len(discopy_hypergraph.boxes)
                               + len(inputs) + len(outputs)))

        vertex_sources: dict[int, set[tuple[int, int]]] = {
            vertex: set() for vertex in vertices
        }
        vertex_targets: dict[int, set[tuple[int, int]]] = {
            vertex: set() for vertex in vertices
        }
        hyperedge_sources: dict[int, list[int]] = {
            hyperedge: [] for hyperedge in hyperedges
        }
        hyperedge_targets: dict[int, list[int]] = {
            hyperedge: [] for hyperedge in hyperedges
        }
        max_internal_spider = max(discopy_hypergraph.wires)
        vertex_labels = {
            vertex: discopy_hypergraph.spider_types[vertex].name
            for vertex in vertices if vertex <= max_internal_spider
        }
        num_boxes = len(discopy_hypergraph.boxes)
        hyperedge_labels = {
            hyperedge: discopy_hypergraph.boxes[hyperedge].name
            for hyperedge in hyperedges
            if hyperedge < num_boxes
        }

        for i, vertex in enumerate(inputs):
            wire = discopy_hypergraph.wires[i]
            label = discopy_hypergraph.spider_types[wire].name
            vertex_labels[vertex] = label
            hyperedge = len(discopy_hypergraph.boxes) + i
            vertex_sources[wire].add((hyperedge, 0))
            vertex_targets[vertex].add((hyperedge, 0))
            hyperedge_sources[hyperedge].append(vertex)
            hyperedge_targets[hyperedge].append(wire)
            hyperedge_labels[hyperedge] = f'_id_{label}'
        for i, vertex in enumerate(outputs):
            wire = discopy_hypergraph.wires[-len(outputs) + i]
            label = discopy_hypergraph.spider_types[wire].name
            vertex_labels[vertex] = label
            hyperedge = len(discopy_hypergraph.boxes) + len(inputs) + i
            vertex_sources[vertex].add((hyperedge, 0))
            vertex_targets[wire].add((hyperedge, 0))
            hyperedge_sources[hyperedge].append(wire)
            hyperedge_targets[hyperedge].append(vertex)
            hyperedge_labels[hyperedge] = f'_id_{label}'

        current_wire = len(discopy_hypergraph.dom)
        for hyperedge, box in enumerate(discopy_hypergraph.boxes):
            for input_port in range(len(box.dom)):
                wire = discopy_hypergraph.wires[current_wire]
                vertex_targets[wire].add((hyperedge, input_port))
                hyperedge_sources[hyperedge].append(wire)
                current_wire += 1
            for output_port in range(len(box.cod)):
                wire = discopy_hypergraph.wires[current_wire]
                vertex_sources[wire].add((hyperedge, output_port))
                hyperedge_targets[hyperedge].append(wire)
                current_wire += 1

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
        ).normal_form()

    def to_discopy(self, normal_form: bool = True):
        """Create a discopy hypergraph from this hypergraph."""
        from discopy.frobenius import Box, Ob, Ty
        from discopy.frobenius import Hypergraph as DCPHypergraph

        hypergraph = self.normal_form() if normal_form else self

        dom = Ty(*(Ob(hypergraph.vertex_labels[vertex])
                   for vertex in hypergraph.inputs))
        cod = Ty(*(Ob(hypergraph.vertex_labels[vertex])
                   for vertex in hypergraph.outputs))

        boxes = []
        wires = [vertex for vertex in hypergraph.inputs]

        for hyperedge in hypergraph.hyperedges:
            box_dom = Ty(*(Ob(hypergraph.vertex_labels[vertex])
                         for vertex
                         in hypergraph.hyperedge_sources[hyperedge]))
            box_cod = Ty(*(Ob(hypergraph.vertex_labels[vertex])
                         for vertex
                         in hypergraph.hyperedge_targets[hyperedge]))
            boxes.append(Box(hypergraph.hyperedge_labels[hyperedge],
                             box_dom, box_cod))

            wires += [vertex for vertex
                      in hypergraph.hyperedge_sources[hyperedge]]
            wires += [vertex for vertex
                      in hypergraph.hyperedge_targets[hyperedge]]

        wires += [vertex for vertex in hypergraph.outputs]

        spider_types = {vertex: Ty(hypergraph.vertex_labels[vertex])
                        for vertex in hypergraph.vertices}

        return DCPHypergraph(
            dom, cod, boxes, tuple(wires), spider_types
        )
