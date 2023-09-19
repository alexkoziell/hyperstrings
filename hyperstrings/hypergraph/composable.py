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
"""Composition and decomposition methods for hypergraphs."""
from __future__ import annotations

from hyperstrings.hypergraph.backend import backend
from hyperstrings.hypergraph.backend import Vertex, Hyperedge
from hyperstrings.hypergraph.mutable import MutableHypergraph


class ComposableHypergraph(MutableHypergraph):
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
