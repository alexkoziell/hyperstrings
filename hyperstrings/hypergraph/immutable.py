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
"""Immutable hypergraph class."""
from typing import Iterable, TypeAlias

import numpy as np

backend = np
backend.concat = backend.concatenate
Array: TypeAlias = np.ndarray

Vertex: TypeAlias = int
Hyperedge: TypeAlias = int
Port: TypeAlias = int
Label: TypeAlias = str

"""
NOTE: avoiding data-dependent output in `nonzero`
see
https://data-apis.org
 /array-api/2022.12/API_specification/generated/array_api.nonzero.html
 /array-api/2022.12/design_topics/data_dependent_output_shapes.html
"""


class ImmutableHypergraph:
    """Immutable Hypergraph implementation.

    Array usage follows Array API standard.

    Data structure:
        - sources: (num_hyperedges, num_vertices, max_num_source_ports) array
            sources[hyperedge, vertex, port] = 1 if vertex connected to input
            port of hyperedge else 0.
            Takes vertices to hyperedges.
        - targets: (num_vertices, num_hyperedges, max_num_target_ports) array
            sources[vertex, hyperedge, port] = 1 if vertex connected to output
            port of hyperedge else 0.
            Takes hyperedges to vertices.
        - vertex_labels: (num_vertices,) array
            contains labels indicating extra information about each vertex
        - hyperedge_labels: (num_hyperedges, ) array
            contains labels indicating extra information about each hyperedge
    """

    def __init__(self,
                 sources: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 targets: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 vertex_labels: Array = backend.array([], dtype=Label),
                 hyperedge_labels: Array = backend.array([], dtype=Label),
                 inputs=[],
                 outputs=[]
                 ) -> None:
        """Initialize a `Hypergraph` instance."""
        self.sources = sources
        self.targets = targets
        self.vertex_labels = vertex_labels
        self.hyperedge_labels = hyperedge_labels
        self.inputs = inputs
        self.outputs = outputs

    def num_vertices(self) -> int:
        """Return the number of vertices in the hypergraph."""
        return len(self.vertex_labels)

    def num_hyperedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        return len(self.hyperedge_labels)

    def max_source_ports(self) -> Port:
        """Return max number of hyperedge source ports in the hypergraph."""
        return self.sources.shape[2]

    def max_target_ports(self) -> Port:
        """Return max number of hyperedge target ports in the hypergraph."""
        return self.targets.shape[2]

    def num_source_ports(self, hyperedge: Hyperedge) -> int:
        """Return the number of source ports of `hyperedge`."""
        hyperedge_sources = self.sources[hyperedge, :, :]
        port_connected = backend.max(hyperedge_sources, axis=0)
        return int(backend.sum(port_connected))

    def num_target_ports(self, hyperedge: Hyperedge) -> int:
        """Return the number of source ports of `hyperedge`."""
        hyperedge_targets = self.targets[:, hyperedge, :]
        port_connected = backend.max(hyperedge_targets, axis=0)
        return int(backend.sum(port_connected))

    def vertices(self) -> Iterable[Vertex]:
        """Return the vertices in the hypergraph."""
        return range(self.num_vertices())

    def hyperedges(self) -> Iterable[Hyperedge]:
        """Return the hyperedges in the hypergraph."""
        return range(self.num_hyperedges())

    def vertex_sources(self, vertex: Vertex) -> list[tuple[Hyperedge, Port]]:
        """Return the sources of `vertex`."""
        return [(hyperedge, port)
                for hyperedge in self.hyperedges()
                for port, connected
                in enumerate(self.targets[vertex, hyperedge, :])
                if connected == 1]

    def vertex_targets(self, vertex: Vertex) -> list[tuple[Hyperedge, Port]]:
        """Return the targets of `vertex`."""
        return [(hyperedge, port)
                for hyperedge in self.hyperedges()
                for port, connected
                in enumerate(self.sources[hyperedge, vertex, :])
                if connected == 1]

    def hyperedge_sources(self, hyperedge: Hyperedge) -> list[Vertex]:
        """Return the list of sources of `hyperedge`."""
        sources = backend.argmax(self.sources[hyperedge, :, :], axis=0)
        sources = sources[:self.num_source_ports(hyperedge)]
        return sources

    def hyperedge_targets(self, hyperedge: Hyperedge) -> list[Vertex]:
        """Return the list of targets of `hyperedge`."""
        targets = backend.argmax(self.targets[:, hyperedge, :], axis=0)
        targets = targets[:self.num_target_ports(hyperedge)]
        return targets
