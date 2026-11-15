"""Build the knowledge graph from seed papers.

Fetch the seeds from OpenAlex, expand one hop (the references most seeds
share, and the works citing each seed), map every work with
``work_to_graph`` and upsert the nodes and edges in one transaction.
All fetching finishes before the transaction opens, so a failed OpenAlex
call writes nothing.
"""

import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from django.db import transaction

from ..models import KGEdge, KGNode
from .kg_mapper import EdgeSpec, NodeSpec, short_id, work_to_graph
# ID_BATCH (IDs per OpenAlex call) is defined once there and re-exported here
# with the other build limits.
from .openalex_service import ID_BATCH, OpenAlexAPIError, OpenAlexService  # noqa: F401

MAX_SEEDS = 25
MAX_PRIOR_WORKS = 50
PER_SEED_CITERS = 10
MAX_DERIVATIVE_WORKS = 50
FETCH_WORKERS = 4
# Checked only between OpenAlex calls; pyalex sets no timeout, so one hung call can outlast it.
BUILD_BUDGET_SECONDS = 20

EDGE_BATCH = 1000


class KGIngestError(Exception):
    """None of the requested seeds exist in OpenAlex."""


class KGIngestService:
    @classmethod
    def build(cls, seed_keys: list[str]) -> dict:
        """Fetch, expand and upsert the graph for `seed_keys`.

        Raises KGIngestError when no seed is found; lets OpenAlexAPIError propagate.
        """
        start = time.monotonic()
        api = OpenAlexService()

        keys = list(dict.fromkeys(short_id(k) for k in seed_keys if k))[:MAX_SEEDS]
        seeds = api.get_works_by_ids(keys) if keys else []
        if not seeds:
            raise KGIngestError("None of the seed papers were found in OpenAlex.")
        seed_ids = [short_id(w["id"]) for w in seeds]
        fetched = set(seed_ids)

        prior_ids = cls._most_shared_references(seeds, fetched)
        prior = api.get_works_by_ids(prior_ids) if prior_ids else []
        fetched.update(short_id(w["id"]) for w in prior)

        derivative, truncated = cls._fetch_citing(api, seed_ids, start, fetched)

        nodes, edges = cls._merge_specs(seeds + prior + derivative)
        cls._set_citation_properties(nodes, edges)
        edges_written = cls._write(nodes, edges)

        return {
            "seeds_found": len(seeds),
            "papers_hydrated": sum(
                1 for n in nodes.values() if n.node_type == "paper" and not n.is_stub
            ),
            "nodes_total": len(nodes),
            "edges_total": edges_written,
            "truncated": truncated,
        }

    @staticmethod
    def _most_shared_references(seeds: list[dict], exclude: set) -> list[str]:
        """The MAX_PRIOR_WORKS works cited by the most seeds, ties broken by key."""
        counts = Counter()
        for work in seeds:
            refs = {short_id(r) for r in work.get("referenced_works") or [] if r}
            counts.update(refs - exclude)
        ranked = sorted(counts, key=lambda k: (-counts[k], k))
        return ranked[:MAX_PRIOR_WORKS]

    @staticmethod
    def _fetch_citing(api, seed_ids, start, exclude) -> tuple[list[dict], bool]:
        """Citing works per seed, in a pool, until the build budget runs out.

        Workers only call OpenAlex (no ORM). Returns (works, truncated).
        """
        truncated = False
        futures = []
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            for seed in seed_ids:
                if time.monotonic() - start >= BUILD_BUDGET_SECONDS:
                    truncated = True
                    break
                futures.append(pool.submit(api.get_citing_works, [seed], limit=PER_SEED_CITERS))

        works: dict[str, dict] = {}
        seeds_cited = Counter()
        for future in futures:
            for work in future.result():  # re-raises OpenAlexAPIError from the worker
                key = short_id(work.get("id") or "")
                if not key or key in exclude:
                    continue
                works.setdefault(key, work)
                seeds_cited[key] += 1

        ranked = sorted(
            works,
            key=lambda k: (-seeds_cited[k], -(works[k].get("cited_by_count") or 0), k),
        )
        return [works[k] for k in ranked[:MAX_DERIVATIVE_WORKS]], truncated

    @staticmethod
    def _merge_specs(works: list[dict]) -> tuple[dict, dict]:
        """Map every work; a hydrated node spec wins over a stub for the same key."""
        nodes: dict[str, NodeSpec] = {}
        edges: dict[tuple, EdgeSpec] = {}
        for work in works:
            work_nodes, work_edges = work_to_graph(work)
            for spec in work_nodes:
                current = nodes.get(spec.key)
                if current is None or (current.is_stub and not spec.is_stub):
                    nodes[spec.key] = spec
            for spec in work_edges:
                edges.setdefault((spec.source_key, spec.target_key, spec.edge_type), spec)
        return nodes, edges

    @staticmethod
    def _set_citation_properties(nodes: dict, edges: dict) -> None:
        """Fill CITES properties where both papers are hydrated in this build."""
        authors = defaultdict(set)
        venue = {}
        for spec in edges.values():
            if spec.edge_type == "AUTHORED_BY":
                authors[spec.source_key].add(spec.target_key)
            elif spec.edge_type == "PUBLISHED_IN":
                venue[spec.source_key] = spec.target_key

        def hydrated(key):
            node = nodes.get(key)
            return node is not None and node.node_type == "paper" and not node.is_stub

        for edge_key, spec in edges.items():
            if spec.edge_type != "CITES":
                continue
            citing, cited = spec.source_key, spec.target_key
            if not (hydrated(citing) and hydrated(cited)):
                continue
            citing_year = nodes[citing].properties.get("year")
            cited_year = nodes[cited].properties.get("year")
            edges[edge_key] = EdgeSpec(citing, cited, "CITES", spec.weight, {
                "timespan_years": (
                    citing_year - cited_year
                    if citing_year is not None and cited_year is not None
                    else None
                ),
                "author_self_citation": bool(authors[citing] & authors[cited]),
                "venue_self_citation": (
                    citing in venue and venue[citing] == venue.get(cited)
                ),
            })

    @staticmethod
    def _write(nodes: dict, edges: dict) -> int:
        """Upsert nodes then edges in one transaction; returns edges written."""
        stubs = [n for n in nodes.values() if n.is_stub]
        hydrated = [n for n in nodes.values() if not n.is_stub]

        def to_model(spec):
            return KGNode(node_type=spec.node_type, key=spec.key, label=spec.label,
                          properties=spec.properties, is_stub=spec.is_stub)

        with transaction.atomic():
            KGNode.objects.bulk_create([to_model(n) for n in stubs], ignore_conflicts=True)
            KGNode.objects.bulk_create(
                [to_model(n) for n in hydrated],
                update_conflicts=True,
                unique_fields=["key"],
                update_fields=["label", "properties", "is_stub", "updated_at"],
            )
            # Re-read: objects returned by bulk_create(ignore_conflicts=True) have no PK.
            all_keys = {k for s in edges.values() for k in (s.source_key, s.target_key)}
            ids = dict(KGNode.objects.filter(key__in=all_keys).values_list("key", "id"))
            rows = [
                KGEdge(source_id=ids[s.source_key], target_id=ids[s.target_key],
                       edge_type=s.edge_type, weight=s.weight,
                       properties=dict(s.properties), provenance="openalex")
                for s in edges.values()
                if s.source_key in ids and s.target_key in ids
            ]
            KGEdge.objects.bulk_create(
                rows,
                update_conflicts=True,
                unique_fields=["source", "target", "edge_type"],
                update_fields=["weight", "properties", "provenance"],
                batch_size=EDGE_BATCH,
            )
        return len(rows)
