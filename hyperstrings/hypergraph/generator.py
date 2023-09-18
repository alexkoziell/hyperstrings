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
"""Generator methods for hypergraphs."""
from __future__ import annotations
from hyperstrings.hypergraph.immutable import Array, backend, Label, Vertex
from hyperstrings.hypergraph.mutable import MutableHypergraph


class GeneratorHypergraph(MutableHypergraph):
    """Generator-based methods for hypergraphs."""

    def __init__(self,
                 sources: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 targets: Array = backend.zeros((0, 0, 1),
                                                dtype=backend.int32),
                 vertex_labels: Array = backend.zeros((0), dtype=Label),
                 hyperedge_labels: Array = backend.zeros((0), dtype=Label),
                 inputs: list[Vertex] = [],
                 outputs: list[Vertex] = []) -> None:
        """Initialize a `GeneratorHypergraph` instance."""
        assert sources.shape[0] == targets.shape[1]
        assert sources.shape[1] == targets.shape[0]
        self.sources = sources
        self.targets = targets
        self.vertex_labels = vertex_labels
        self.hyperedge_labels = hyperedge_labels
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def generator(cls, input_labels: list[Label],
                  output_labels: list[Label],
                  label: Label) -> GeneratorHypergraph:
        """Create a generator with inputs, outputs and a label."""
        vertex_labels = backend.array([
            label for label in input_labels + output_labels])
        hyperedge_labels = backend.array([label])
        sources = backend.zeros((1, len(vertex_labels),
                                 len(input_labels)))
        targets = backend.zeros((len(vertex_labels), 1,
                                 len(output_labels)))
        sources[0, :len(input_labels)] = backend.eye(len(input_labels))
        targets[len(input_labels):, 0] = backend.eye(len(output_labels))
        inputs = list(range(len(input_labels)))
        outputs = list(range(len(input_labels),
                             len(input_labels) + len(output_labels)))
        return cls(
            sources,
            targets,
            vertex_labels,
            hyperedge_labels,
            inputs,
            outputs
        )
