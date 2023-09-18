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
"""Composable hypergraph implementation."""
from __future__ import annotations
from hyperstrings.hypergraph.immutable import backend
from hyperstrings.hypergraph.immutable import Vertex
from hyperstrings.hypergraph.generator import GeneratorHypergraph


class ComposableHypergraph(GeneratorHypergraph):
    """Composable hypergraph class."""

    def combine(self, other: ComposableHypergraph) -> ComposableHypergraph:
        """Combine the vertices and hyperedges of `self` with `other`.

        This returns a combined hypergraph with no inputs or outputs.
        """
        sources = backend.zeros((
            self.num_hyperedges() + other.num_hyperedges(),
            self.num_vertices() + other.num_vertices(),
            backend.max((self.max_source_ports(), other.max_source_ports()))
        ))
        sources[:self.num_hyperedges(),
                :self.num_vertices(),
                :self.max_source_ports()] = self.sources.copy()
        sources[self.num_hyperedges():,
                self.num_vertices():,
                :other.max_source_ports()] = other.sources.copy()
        targets = backend.zeros((
            self.num_vertices() + other.num_vertices(),
            self.num_hyperedges() + other.num_hyperedges(),
            backend.max((self.max_target_ports(), other.max_target_ports()))
        ))
        targets[:self.num_vertices(),
                :self.num_hyperedges(),
                :self.max_target_ports()] = self.targets.copy()
        targets[self.num_vertices():,
                self.num_hyperedges():,
                :other.max_target_ports()] = other.targets.copy()
        vertex_labels = backend.concat(
            (self.vertex_labels, other.vertex_labels))
        hyperedge_labels = backend.concat(
            (self.hyperedge_labels, other.hyperedge_labels))
        combined = self.__class__(
            sources, targets, vertex_labels, hyperedge_labels
        )
        return combined

    def sequential_compose(self, other: ComposableHypergraph
                           ) -> ComposableHypergraph:
        """Form the sequential composition self;other."""
        assert len(self.outputs) == len(other.inputs)
        assert all(self.vertex_labels[o] == other.vertex_labels[i]
                   for o, i in zip(self.outputs, other.inputs))
        composed = self.combine(other)
        composed.inputs = self.inputs.copy()
        composed.outputs = [o + self.num_vertices()
                            for o in other.outputs]
        quotient_pairs = [[o, i + self.num_vertices()]
                          for o, i in zip(self.outputs, other.inputs)]
        for i in range(len(quotient_pairs)):
            for j in range(i+1, len(quotient_pairs)):
                for k in range(2):
                    if quotient_pairs[j][k] > quotient_pairs[i][1]:
                        quotient_pairs[j][k] -= 1
        for o, i in quotient_pairs:
            composed.quotient_vertices(o, i)
        return composed

    def __rshift__(self, other: ComposableHypergraph
                   ) -> ComposableHypergraph:
        """Sequentially compose `self` with `other`."""
        return self.sequential_compose(other)

    def parallel_compose(self, other: ComposableHypergraph
                         ) -> ComposableHypergraph:
        """Form the parallel composition self⊗other."""
        composed = self.combine(other)
        # TODO: array based
        composed.inputs = self.inputs + [
            i + self.num_vertices() for i in other.inputs]
        composed.outputs = self.outputs + [
            o + self.num_vertices() for o in other.outputs]
        return composed

    def __matmul__(self, other: ComposableHypergraph) -> ComposableHypergraph:
        """Parallel compose `self` with `other`."""
        return self.parallel_compose(other)

    def quotient_vertices(self, *vertices: Vertex) -> None:
        """Merge `vertices[1:]` into `vertices[0]`."""
        self.sources[:, vertices[0]] = self.sources[:, [*vertices]].any(axis=1)
        self.targets[vertices[0]] = self.targets[[*vertices]].any(axis=0)
        self.inputs = [vertices[0] if v in vertices else v
                       for v in self.inputs]
        self.outputs = [vertices[0] if v in vertices else v
                        for v in self.outputs]
        self.remove_vertices(vertices[1:])
