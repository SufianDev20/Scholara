"""Pure metric functions for evaluating the knowledge graph (Task 8).

No network, no database: the ``evaluate_kg`` command gathers the data and
the independent references, these functions only compute the numbers.
"""

import networkx as nx

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")


def normalise_doi(doi: str | None) -> str:
    """Lower-cased DOI without a ``https://doi.org/`` or ``doi:`` prefix ("" for none)."""
    value = (doi or "").strip().lower()
    for prefix in _DOI_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def reference_coverage(kg_count: int, reference_count: int) -> float | None:
    """Stored references / the reference source's count; None when the count is 0."""
    if not reference_count:
        return None
    return kg_count / reference_count


def citation_agreement(
    kg_pairs: set[tuple[str, str]], reference_pairs: set[tuple[str, str]]
) -> float | None:
    """Share of the graph's (citing, cited) DOI pairs the reference also lists."""
    def norm(pairs):
        return {(normalise_doi(a), normalise_doi(b)) for a, b in pairs}

    kg = norm(kg_pairs)
    if not kg:
        return None
    return len(kg & norm(reference_pairs)) / len(kg)


def topic_coherence(
    pairs: list[tuple[str, str]], subfield_of: dict[str, str | None]
) -> float | None:
    """Share of pairs whose two papers have the same subfield; unknown sides are skipped."""
    known = [(a, b) for a, b in pairs if subfield_of.get(a) and subfield_of.get(b)]
    if not known:
        return None
    return sum(1 for a, b in known if subfield_of[a] == subfield_of[b]) / len(known)


def precision_at_k(predicted: list[str], relevant: set[str], k: int = 5) -> float | None:
    """Share of the first ``k`` predictions that are relevant; None with no prediction."""
    top = predicted[:k]
    if not top:
        return None
    return sum(1 for p in top if p in relevant) / len(top)


def largest_component_share(paper_keys: list[str], cites_edges: list[tuple[str, str]]) -> float:
    """Share of papers in the largest weakly connected component of the CITES graph."""
    if not paper_keys:
        return 0.0
    graph = nx.DiGraph()
    graph.add_nodes_from(paper_keys)
    graph.add_edges_from((s, t) for s, t in cites_edges if s in graph and t in graph)
    largest = max(len(c) for c in nx.weakly_connected_components(graph))
    return largest / len(paper_keys)


def verdict(value: float | None, target: float | None, higher_is_better: bool = True) -> str:
    """"met" / "missed" against the target, "reported" without one, "not measurable" for None."""
    if value is None:
        return "not measurable"
    if target is None:
        return "reported"
    met = value >= target if higher_is_better else value <= target
    return "met" if met else "missed"
