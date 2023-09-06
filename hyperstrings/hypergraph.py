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
from dataclasses import dataclass


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

    def add_vertex(self, label: str) -> int:
        """Add a vertex to the hypergraph."""
        # Give the vertex a unique integer identifier
        vertex_id = max(self.vertices) + 1
        self.vertices.add(vertex_id)
        self.vertex_labels[vertex_id] = label
        return vertex_id

    def add_hyperedge(self, label: str) -> int:
        """Add a hyperedge to the hypergraph."""
        # Give the hyperedge a unique integer identifier
        hyperedge_id = max(self.hyperedges) + 1
        self.hyperedges.add(hyperedge_id)
        self.hyperedge_labels[hyperedge_id] = label
        return hyperedge_id

    def set_hyperedge_sources(self, hyperedge: int,
                              vertices: list[int]) -> None:
        """Set the sources of a hyperedge.

        Vertex targets are updated appropriately.
        """
        self.hyperedge_sources[hyperedge] = vertices
        for port, vertex in enumerate(vertices):
            self.vertex_targets[vertex].add((hyperedge, port))

    def set_hyperedge_targets(self, hyperedge: int,
                              vertices: list[int]) -> None:
        """Set the targets of a hyperedge.

        Vertex sources are updated appropriately.
        """
        self.hyperedge_targets[hyperedge] = vertices
        for port, vertex in enumerate(vertices):
            self.vertex_sources[vertex].add((hyperedge, port))

    def set_vertex_sources(self, vertex: int,
                           hyperedges_and_ports: set[tuple[int, int]]) -> None:
        """Set the sources of a vertex.

        Hyperedge targets are updated appropriately.
        """
        self.vertex_sources[vertex] = hyperedges_and_ports
        for hyperedge, port in hyperedges_and_ports:
            self.hyperedge_targets[hyperedge][port] = vertex

    def set_vertex_targets(self, vertex: int,
                           hyperedges_and_ports: set[tuple[int, int]]) -> None:
        """Set the targets of a vertex.

        Hyperedge sources are updated appropriately.
        """
        self.vertex_targets[vertex] = hyperedges_and_ports
        for hyperedge, port in hyperedges_and_ports:
            self.hyperedge_sources[hyperedge][port] = vertex
