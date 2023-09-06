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
    vertex_sources: dict[int, set[int]]
    vertex_targets: dict[int, set[int]]
    vertex_labels: dict[int, str]

    hyperedges: set[int]
    hyperedge_sources: dict[int, list[int]]
    hyperedge_targets: dict[int, list[int]]
    hyperedge_labels: dict[int, str]

    inputs: list[int]
    outputs: list[int]
