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
"""Mutable hypergraph class."""
from typing import Sequence

from hyperstrings.hypergraph.backend import backend
from hyperstrings.hypergraph.backend import Vertex, Hyperedge, Port, Label
from hyperstrings.hypergraph.immutable import ImmutableHypergraph


class MutableHypergraph(ImmutableHypergraph):
    """Mutable hypergraph implementation."""

    def add_vertex(self, label: Label) -> Vertex:
        """Add a vertex to the hypergraph."""
        self.sources = backend.concat(
            (self.sources,
             backend.zeros((self.num_hyperedges(), 1,
                           self.max_source_ports()))),
            axis=1
        )
        self.targets = backend.concat(
            (self.targets,
             backend.zeros((1, self.num_hyperedges(),
                           self.max_target_ports()))),
            axis=0
        )
        self.vertex_labels = backend.concat((self.vertex_labels,
                                             backend.array([label])))
        return self.num_vertices() - 1

    def add_hyperedge(self, label: Label) -> Vertex:
        """Add a hyperedge to the hypergraph."""
        self.sources = backend.concat(
            (self.sources,
             backend.zeros((1, self.num_vertices(),
                            self.max_source_ports()))),
            axis=0
        )
        self.targets = backend.concat(
            (self.targets,
             backend.zeros((self.num_vertices(), 1,
                            self.max_target_ports()))),
            axis=1
        )
        self.hyperedge_labels = backend.concat((self.hyperedge_labels,
                                                backend.array([label])))
        return self.num_hyperedges() - 1

    def remove_vertices(self, vertices: Sequence[Vertex]) -> None:
        """Remove vertices from the hypergraph."""
        keep_vertices = [v for v in self.vertices()
                         if v not in vertices]
        self.sources = self.sources.take(keep_vertices, axis=1)
        self.targets = self.targets.take(keep_vertices, axis=0)
        self.vertex_labels = self.vertex_labels[keep_vertices]
        self.inputs = [i - len([v for v in vertices if v < i])
                       for i in self.inputs if i not in vertices]
        self.outputs = [o - len([v for v in vertices if v < o])
                        for o in self.outputs if o not in vertices]

    def remove_hyperedges(self, hyperedges: Sequence[Hyperedge]) -> None:
        """Remove hyperedges from the hypergraph."""
        keep_hyperedges = [h for h in self.hyperedges()
                           if h not in hyperedges]
        self.sources = self.sources.take(keep_hyperedges, axis=0)
        self.targets = self.targets.take(keep_hyperedges, axis=1)
        self.hyperedge_labels = self.hyperedge_labels[keep_hyperedges]

    def connect_source(self, vertex: Vertex,
                       hyperedge: Hyperedge, port: Port) -> None:
        """Connect `vertex` to source port `port` of `hyperedge`."""
        if port < self.max_source_ports():
            if backend.any(self.sources[hyperedge, :, port]):
                already_connected = backend.argmax(
                    self.sources[hyperedge, :, port])
                raise ValueError(
                    f'Vertex {already_connected} already connected'
                    + f' to source port {port} of hyperedge {hyperedge}.')
        elif port >= self.max_source_ports():
            self.sources = backend.concat(
                (self.sources,
                 backend.zeros((self.num_hyperedges(),
                                self.num_vertices(), 1))),
                axis=2
            )
        self.sources[hyperedge, vertex, port] = 1

    def connect_target(self, vertex: Vertex,
                       hyperedge: Hyperedge, port: Port) -> None:
        """Connect `vertex` to target port `port` of `hyperedge`."""
        if port < self.max_target_ports():
            if backend.any(self.targets[:, hyperedge, port]):
                already_connected = backend.argmax(
                    self.targets[:, hyperedge, port])
                raise ValueError(
                    f'Vertex {already_connected} already connected'
                    + f' to target port {port} of hyperedge {hyperedge}.')
        elif port >= self.max_target_ports():
            self.targets = backend.concat(
                (self.targets,
                 backend.zeros((self.num_vertices(),
                                self.num_hyperedges(), 1))),
                axis=2
            )
        self.targets[vertex, hyperedge, port] = 1

    def disconnect_source(self, vertex: Vertex,
                          hyperedge: Hyperedge, port: Port) -> None:
        """Disconnect `vertex` from source port `port` of `hyperedge`."""
        if (port >= self.max_source_ports()
                or self.sources[hyperedge, vertex, port] == 0):
            raise ValueError(
                f'Vertex {vertex} already disconnected'
                + f' from source port {port} of hyperedge {hyperedge}.')
        self.sources[hyperedge, vertex, port] = 0

    def disconnect_target(self, vertex: Vertex,
                          hyperedge: Hyperedge, port: Port) -> None:
        """Disconnect `vertex` from target port `port` of `hyperedge`."""
        if (port >= self.max_target_ports()
                or self.targets[vertex, hyperedge, port] == 0):
            raise ValueError(
                f'Vertex {vertex} already disconnected'
                + f' from target port {port} of hyperedge {hyperedge}.')
        self.targets[vertex, hyperedge, port] = 0
