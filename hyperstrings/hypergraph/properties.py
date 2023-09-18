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
"""Methods testing useful properties of hypergraphs."""
from hyperstrings.hypergraph.immutable import backend, Vertex
from hyperstrings.hypergraph.immutable import ImmutableHypergraph


class PropertiesHypergraph(ImmutableHypergraph):
    """Property testing implementaions for hypergraphs."""

    def is_source_monogamous(self) -> bool:
        """Return whether this hypergraph is source-monogamous.

        This means that any vertex has at most one input wire.
        """
        return backend.all(backend.sum(self.targets, axis=(1, 2)) <= 1)

    def is_target_monogamous(self) -> bool:
        """Return whether this hypergraph is source-monogamous.

        This means that any vertex has at most one output wire.
        """
        return backend.all(backend.sum(self.sources, axis=(0, 2)) <= 1)

    def is_monogamous(self) -> bool:
        """Return whether this hypergraph is monogamous.

        This means that any vertex has at most one input wire and at most
        one output wire.
        """
        return self.is_source_monogamous() and self.is_target_monogamous()

    def children(self, vertex: Vertex) -> set[Vertex]:
        """Return the children of vertex."""
        transition_matrix = backend.matmul(
            backend.any(self.targets, axis=2),
            backend.any(self.sources, axis=2)
        )
        vertex_vector = backend.zeros(self.num_vertices())
        vertex_vector[vertex] = 1
        children: set[Vertex] = set()
        while True:
            vertex_vector = transition_matrix @ vertex_vector
            new_children = set(
                vertex for vertex, is_child in enumerate(vertex_vector)
                if is_child and vertex not in children
            )
            if len(new_children) == 0:
                break
            children.update(new_children)
        return children

    def is_cyclic(self) -> bool:
        """Return whether this hypergraph contains any cycles."""
        for vertex in self.vertices():
            if vertex in self.children(vertex):
                return True
        return False
