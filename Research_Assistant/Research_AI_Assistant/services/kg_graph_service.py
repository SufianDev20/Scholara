"""Read the knowledge graph around seed papers.

Load the seeds' one-hop ``CITES`` neighbourhood from ``KGNode`` / ``KGEdge``,
then compute ``SIMILAR_TO`` links, Louvain clusters and citation degrees with
NetworkX per request. Nothing computed here is stored.
"""

import math
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx

from ..models import KGEdge, KGNode
from .kg_mapper import short_id

SIMILARITY_MIN = 0.15
SIMILARITY_TOP_K = 5
# One shared citer scores 1 / sqrt(1 * 1) = 1.0, so the prior works of a single
# seed would form a clique of maximum-weight links.
COCITATION_MIN_SHARED = 2
LIST_LIMIT = 10
LOCAL_PREFIX = "local:"

STAT_KEYS = {
    "paper": "papers",
    "author": "authors",
    "institution": "institutions",
    "venue": "venues",
    "topic": "topics",
    "keyword": "keywords",
}


class KGGraphError(Exception):
    """None of the requested seeds (OpenAlex or local) is a hydrated paper in the graph."""


class KGGraphService:
    @staticmethod
    def subgraph(
        seed_keys: list[str],
        edge_types: set[str] | None = None,
        similarity: bool = True,
        include_stubs: bool = False,
    ) -> dict:
        """The seeds' neighbourhood as plain ``nodes`` / ``edges`` lists.

        ``edge_types`` filters the stored edge types in the output (None means
        all); the scope is always computed from stored ``CITES`` edges.
        A ``local:`` seed is a seed itself and makes the OpenAlex paper it is
        ``SAME_AS`` a seed too. Local papers joined by ``SAME_AS`` to a scoped
        paper are added to the output; they have no citations of their own
        and get no cluster.
        """
        wanted = list(dict.fromkeys(short_id(k) for k in seed_keys))
        found = {
            n.key: n
            for n in KGNode.objects.filter(key__in=wanted, node_type="paper", is_stub=False)
        }
        seeds = [found[k] for k in wanted if k in found]
        if not seeds:
            raise KGGraphError(f"No seed paper is in the graph: {', '.join(wanted)}")
        local_seeds = [n for n in seeds if n.key.startswith(LOCAL_PREFIX)]
        if local_seeds:
            same_as = KGEdge.objects.filter(
                edge_type="SAME_AS", source__in=local_seeds,
                target__node_type="paper", target__is_stub=False,
            ).select_related("target")
            known = {n.key for n in seeds}
            for e in same_as:
                if e.target.key not in known:
                    known.add(e.target.key)
                    seeds.append(e.target)
        seed_set = {n.key for n in seeds}

        # One CITES hop from the seeds, in either direction.
        hop = KGEdge.objects.filter(edge_type="CITES")
        neighbours = [e.target for e in hop.filter(source__in=seeds).select_related("target")]
        neighbours += [e.source for e in hop.filter(target__in=seeds).select_related("source")]
        papers = {n.key: n for n in seeds}
        for n in neighbours:
            if n.node_type == "paper" and (include_stubs or not n.is_stub):
                papers.setdefault(n.key, n)

        stored = KGGraphService._load_edges(papers.values(), edge_types)
        for e in stored:
            if e.edge_type == "SAME_AS" and e.target.key in papers:
                papers.setdefault(e.source.key, e.source)
        cites = [(e.source.key, e.target.key) for e in stored if e.edge_type == "CITES"]
        hydrated = [k for k, n in papers.items() if not n.is_stub]
        refs = {k: set() for k in hydrated}
        citers = {k: set() for k in hydrated}
        for s, t in cites:
            if s in refs:
                refs[s].add(t)
            if t in citers:
                citers[t].add(s)

        nodes, edges = KGGraphService._output(papers, stored, edge_types)
        similar = KGGraphService.similarity_edges(refs, citers) if similarity else []
        edges += similar

        digraph = nx.DiGraph()
        digraph.add_nodes_from(papers)
        digraph.add_edges_from((s, t) for s, t in cites if s in papers and t in papers)
        clusters = (
            KGGraphService._clusters(sorted(k for k in papers if not k.startswith(LOCAL_PREFIX)),
                                     similar)
            if similarity else {}
        )

        cited_by_seeds = Counter(t for s, t in cites if s in seed_set)
        citing_seeds = Counter(s for s, t in cites if t in seed_set)
        result_nodes = []
        for key in [n.key for n in seeds] + sorted(set(papers) - seed_set):
            node = papers[key]
            result_nodes.append({
                "id": key, "type": "paper", "label": node.label, "is_seed": key in seed_set,
                "origin": "local" if key.startswith(LOCAL_PREFIX) else "openalex",
                "properties": node.properties,
                "metrics": {
                    "in_degree": digraph.in_degree(key),
                    "out_degree": digraph.out_degree(key),
                    "cluster": clusters.get(key),
                },
            })
        for key in sorted(nodes):
            node = nodes[key]
            result_nodes.append({
                "id": key, "type": node.node_type, "label": node.label,
                "is_seed": False, "properties": node.properties,
            })

        stats = {name: 0 for name in STAT_KEYS.values()}
        for n in result_nodes:
            stats[STAT_KEYS[n["type"]]] += 1
        stats["edges"] = len(edges)
        return {
            "nodes": result_nodes,
            "edges": edges,
            "prior_works": KGGraphService._top(cited_by_seeds, hydrated, seed_set),
            "derivative_works": KGGraphService._top(citing_seeds, hydrated, seed_set),
            "stats": stats,
        }

    @staticmethod
    def similarity_edges(refs: dict[str, set[str]], citers: dict[str, set[str]]) -> list[dict]:
        """``SIMILAR_TO`` edges from bibliographic coupling and co-citation.

        Weight = shared / sqrt(|A| * |B|) for each method; the larger wins
        (bibliographic coupling on a tie). Co-citation needs at least
        ``COCITATION_MIN_SHARED`` shared citers. An edge is kept when its weight is
        at least ``SIMILARITY_MIN`` and it is among the ``SIMILARITY_TOP_K``
        strongest edges of either endpoint.
        """
        best = {}
        for method, sets in (("bibliographic_coupling", refs), ("co_citation", citers)):
            holders = defaultdict(set)
            for paper, items in sets.items():
                for item in items:
                    holders[item].add(paper)
            shared = Counter()
            for group in holders.values():
                shared.update(combinations(sorted(group), 2))
            for (a, b), count in shared.items():
                if method == "co_citation" and count < COCITATION_MIN_SHARED:
                    continue
                weight = count / math.sqrt(len(sets[a]) * len(sets[b]))
                if weight >= SIMILARITY_MIN and weight > best.get((a, b), (0.0,))[0]:
                    best[(a, b)] = (weight, method, count)

        partners = defaultdict(list)
        for (a, b), (weight, _, _) in best.items():
            partners[a].append((-round(weight, 4), b))
            partners[b].append((-round(weight, 4), a))
        top = {p: {key for _, key in sorted(lst)[:SIMILARITY_TOP_K]} for p, lst in partners.items()}
        return [
            {
                "source": a, "target": b, "type": "SIMILAR_TO", "weight": round(weight, 4),
                "properties": {"method": method, "shared": count},
            }
            for (a, b), (weight, method, count) in sorted(best.items())
            if b in top[a] or a in top[b]
        ]

    @staticmethod
    def _load_edges(papers, edge_types):
        """Stored edges touching the scoped papers, plus their authors' affiliations, each once.

        ``CITES`` is always loaded (scope, similarity and degrees need it).
        """
        base = KGEdge.objects.select_related("source", "target")
        if edge_types is not None:
            base = base.filter(edge_type__in=set(edge_types) | {"CITES"})
        papers = list(papers)
        loaded = {}
        for edge in list(base.filter(source__in=papers)) + list(base.filter(target__in=papers)):
            loaded[edge.pk] = edge
        if edge_types is None or {"AUTHORED_BY", "AFFILIATED_WITH"} <= set(edge_types):
            authors = {e.target_id for e in loaded.values() if e.edge_type == "AUTHORED_BY"}
            for edge in base.filter(edge_type="AFFILIATED_WITH", source__in=authors):
                loaded[edge.pk] = edge
        return list(loaded.values())

    @staticmethod
    def _output(papers, stored, edge_types):
        """Non-paper nodes and stored edges for the output, filtered by ``edge_types``.

        A non-paper node is added when a requested edge attaches it to a scoped
        paper (institutions via an output author); an edge is kept when both of
        its ends are output nodes.
        """
        wanted = [e for e in stored if edge_types is None or e.edge_type in edge_types]
        nodes = {}
        for e in wanted:
            if e.source.key in papers and e.target.node_type != "paper":
                nodes[e.target.key] = e.target
        for e in wanted:
            if e.edge_type == "AFFILIATED_WITH" and e.source.key in nodes:
                nodes[e.target.key] = e.target
        present = set(papers) | set(nodes)
        edges = [
            {
                "source": e.source.key, "target": e.target.key, "type": e.edge_type,
                "weight": e.weight, "properties": e.properties,
            }
            for e in wanted
            if e.source.key in present and e.target.key in present
        ]
        edges.sort(key=lambda e: (e["type"], e["source"], e["target"]))
        return nodes, edges

    @staticmethod
    def _clusters(paper_keys, similar):
        """Louvain community index per paper; a paper with no link is its own community."""
        graph = nx.Graph()
        graph.add_nodes_from(paper_keys)
        graph.add_weighted_edges_from((e["source"], e["target"], e["weight"]) for e in similar)
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        return {key: index for index, members in enumerate(communities) for key in members}

    @staticmethod
    def _top(counts, hydrated, seed_keys):
        """Up to ``LIST_LIMIT`` non-seed hydrated papers by seed count, then key."""
        ranked = sorted((-counts[k], k) for k in hydrated if k not in seed_keys and counts[k])
        return [k for _, k in ranked[:LIST_LIMIT]]
