"""
Django management command that evaluates the knowledge graph against
independent sources (Crossref, OpenCitations, OpenAlex related_works) on a
frozen seed set and writes a Markdown and a JSON report.
Run: python manage.py evaluate_kg [--seeds PATH] [--out DIR]

A reference source that fails or returns no data for a paper makes that
paper "not measurable" for that metric: it is counted neither as agreement
nor as disagreement, and the report states how many papers each metric
covered. The OpenAlex email and the OpenCitations token are never logged or
written to the report.
"""

import json
import logging
import os
import random
import re
import statistics
import subprocess
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date
from itertools import combinations
from pathlib import Path
from urllib.parse import quote

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from Research_AI_Assistant.models import KGEdge, KGNode
from Research_AI_Assistant.services import openalex_service
from Research_AI_Assistant.services.kg_evaluation import (
    citation_agreement,
    largest_component_share,
    normalise_doi,
    precision_at_k,
    reference_coverage,
    topic_coherence,
    verdict,
)
from Research_AI_Assistant.services.kg_graph_service import (
    COCITATION_MIN_SHARED, KGGraphError, KGGraphService,
)
from Research_AI_Assistant.services.kg_ingest_service import (
    PER_SEED_CITERS,
    KGIngestError,
    KGIngestService,
)
from Research_AI_Assistant.services.kg_mapper import short_id
from Research_AI_Assistant.services.openalex_service import OpenAlexAPIError, OpenAlexService

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 10
CROSSREF_URL = "https://api.crossref.org/works/{doi}"
OPENCITATIONS_URL = "https://api.opencitations.net/index/v2/{op}/doi:{doi}"
RUNS = 3
SAMPLE_EDGES = 100
SAMPLE_SEED = 42
TOP_K = 5

# Targets set before the evaluation was run (task brief); never tuned afterwards.
TARGET_COVERAGE = 0.70
TARGET_AGREEMENT = 0.90
TARGET_COHERENCE_RATIO = 1.5
TARGET_ENTITY_SHARE = 0.90
TARGET_BUILD_SECONDS = 15.0
TARGET_SUBGRAPH_SECONDS = 2.0

WHY_MISSED = {
    "E1": "OpenAlex only keeps references it can match to a work; references to books, "
          "web pages and unindexed venues are dropped, so CITES edges fall short of the "
          "publisher's reference count.",
    "E2": "OpenAlex and OpenCitations resolve references independently; a reference matched "
          "to a different version of the same work (preprint vs published DOI) counts as "
          "disagreement.",
    "E3": "SIMILAR_TO pairs share a subfield less than 1.5 times as often as random pairs; "
          "with few seeds the links can be dominated by co-citation pairs that rest on only "
          "a couple of shared citers.",
    "E5": "OpenAlex has no author, institution, venue or topic record for some papers "
          "(often preprints, datasets and older works).",
    "E8": "OpenAlex responded slowly during the run; pyalex sets no request timeout, so one "
          "slow call stretches the build.",
    "E9": "OpenAlex data changed between the two builds (citing works are ranked by "
          "cited_by_count) or a build hit its time budget and was truncated.",
}

KNOWN_LIMITS = [
    "OpenAlex only keeps references it can match to a work, so CITES edges miss "
    "references to books, web pages and unindexed venues.",
    "Similarity needs shared references, so very new papers get few SIMILAR_TO links.",
    f"Co-citation only sees the stored citers: PER_SEED_CITERS = {PER_SEED_CITERS} per seed "
    "(E7f gives each method's share of SIMILAR_TO).",
    f"A co-citation link needs at least COCITATION_MIN_SHARED = {COCITATION_MIN_SHARED} "
    "shared citers; with one, two papers cited only by the same seed would score 1.0 "
    "(1/sqrt(1*1)).",
    "Local papers without a title match stay unconnected.",
    "Local matching is title+year only: the research server sends no DOI.",
    "Local papers have no CITES edges of their own; they reach the graph only through SAME_AS.",
]

_DOI_TOKEN = re.compile(r"(?:^|\s)doi:(\S+)")


# ---------------------------------------------------------------- reference sources

def _get_json(url: str, params: dict | None = None, headers: dict | None = None):
    """Decoded JSON of a 200 response, else None. Never logs params or headers."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as e:
        logger.warning("Reference source request failed (%s)", type(e).__name__)
        return None
    if response.status_code != 200:
        logger.warning("Reference source returned HTTP %s", response.status_code)
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _opencitations(op: str, doi: str):
    token = os.environ.get("OPENCITATIONS_TOKEN")
    return _get_json(
        OPENCITATIONS_URL.format(op=op, doi=quote(doi, safe="/")),
        headers={"authorization": token} if token else None,
    )


def crossref_reference_count(doi: str) -> int | None:
    """Crossref ``reference-count`` for a DOI; None when the call fails."""
    email = getattr(settings, "OPENALEX_EMAIL", "")
    body = _get_json(
        CROSSREF_URL.format(doi=quote(doi, safe="/")),
        params={"mailto": email} if email else None,
    )
    message = body.get("message") if isinstance(body, dict) else None
    count = message.get("reference-count") if isinstance(message, dict) else None
    return count if isinstance(count, int) else None


def opencitations_reference_count(doi: str) -> int | None:
    """OpenCitations ``reference-count`` for a DOI; None when the call fails."""
    body = _opencitations("reference-count", doi)
    try:
        return int(body[0]["count"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def opencitations_references(doi: str) -> set[str] | None:
    """DOIs OpenCitations lists as cited by ``doi``; None on failure or no data."""
    body = _opencitations("references", doi)
    if not isinstance(body, list):
        return None
    cited = {
        normalise_doi(match)
        for row in body if isinstance(row, dict)
        for match in _DOI_TOKEN.findall(str(row.get("cited") or ""))
    }
    return cited or None


def measure_coverage(papers: list[tuple[str, int]], fetch_count) -> tuple[float | None, int, list]:
    """Mean reference coverage over ``(doi, kg_count)`` papers whose count could be fetched.

    Returns (value, covered, rows) where rows holds ``(doi, kg_count, reference_count)``
    for every paper (reference_count None when not measurable). A paper whose
    fetch failed, or whose reference count is 0, is left out of the mean.
    """
    values, rows = [], []
    for doi, kg_count in papers:
        ref_count = fetch_count(doi)
        rows.append((doi, kg_count, ref_count))
        coverage = reference_coverage(kg_count, ref_count) if ref_count is not None else None
        if coverage is not None:
            values.append(coverage)
    return (sum(values) / len(values) if values else None), len(values), rows


# ---------------------------------------------------------------- helpers

@contextmanager
def count_openalex_calls():
    """Count every OpenAlex query made through ``openalex_service._get_with_retry``.

    Every KG lookup (works by id, citing works, title search) goes through it;
    a retry inside it is not counted again. Thread-safe (the build uses a pool).
    """
    counter = {"calls": 0}
    lock = threading.Lock()
    original = openalex_service._get_with_retry

    def counted(*args, **kwargs):
        with lock:
            counter["calls"] += 1
        return original(*args, **kwargs)

    openalex_service._get_with_retry = counted
    try:
        yield counter
    finally:
        openalex_service._get_with_retry = original


def commit_hash() -> str:
    """Short HEAD hash, plus "+ uncommitted changes" when the tree is dirty; else "unknown"."""
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=settings.BASE_DIR, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()

    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return f"{head} + uncommitted changes" if dirty else head


def _table_counts() -> tuple[int, int]:
    return KGNode.objects.count(), KGEdge.objects.count()


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _row(rid, metric, value, covered, target=None, higher_is_better=True,
         display=None, reason=None, target_text=None):
    result = verdict(value, target, higher_is_better)
    why = None
    if result == "not measurable":
        why = reason or "no data from the reference source"
    elif result == "missed":
        why = WHY_MISSED.get(rid.rstrip("abcde"))
    if target_text is None and target is not None:
        target_text = f"{'>=' if higher_is_better else '<='} {target}"
    return {
        "id": rid, "metric": metric, "value": value,
        "display": display if display is not None else _fmt(value),
        "covered": covered, "target": target_text or "none",
        "verdict": result, "why": why,
    }


# ---------------------------------------------------------------- command

class Command(BaseCommand):
    help = "Evaluate the knowledge graph on the frozen seed set and write the report."

    def add_arguments(self, parser):
        test_dir = Path(settings.BASE_DIR) / "Test"
        parser.add_argument("--seeds", default=str(test_dir / "kg_eval_seeds.json"))
        parser.add_argument("--out", default=str(test_dir))

    def handle(self, *args, **options):
        seed_file = json.loads(Path(options["seeds"]).read_text(encoding="utf-8"))
        paper_ids = [p["id"] for p in seed_file.get("paper_ids") or []]
        local_papers = seed_file.get("local_papers") or []
        seed_keys = paper_ids + [p["id"] for p in local_papers]

        builds, build_times, call_counts, counts_after = self._build(paper_ids, local_papers)
        graph, subgraph_times = self._subgraph(seed_keys)

        nodes = {n["id"]: n for n in graph["nodes"]}
        papers = [n for n in graph["nodes"] if n["type"] == "paper"]
        oa_papers = [n["id"] for n in papers if n["origin"] == "openalex"]
        seeds = [n["id"] for n in papers if n["is_seed"] and n["origin"] == "openalex"]
        edges = graph["edges"]
        cites = [(e["source"], e["target"]) for e in edges if e["type"] == "CITES"]
        similar = [e for e in edges if e["type"] == "SIMILAR_TO"]

        def doi_of(key):
            return normalise_doi(((nodes.get(key) or {}).get("properties") or {}).get("doi"))

        rows, details = [], {}
        self.stdout.write("E1: reference coverage (Crossref, OpenCitations)...")
        rows += self._coverage(seeds, doi_of, details)
        self.stdout.write("E2: citation agreement (OpenCitations)...")
        rows.append(self._agreement(cites, doi_of, details))
        rows.append(self._coherence(oa_papers, edges, nodes, similar, details))
        self.stdout.write("E4: precision@5 against OpenAlex related_works...")
        rows.append(self._precision(seeds, similar, details))
        rows += self._entities(oa_papers, edges, nodes)
        sheet = []
        rows.append(self._local(local_papers, builds[-1], sheet))
        rows += self._structure(papers, oa_papers, seeds, edges, cites, similar, details)
        rows += self._performance(builds, build_times, call_counts, subgraph_times, counts_after)

        report = {
            "date": date.today().isoformat(),
            "commit": commit_hash(),
            "seeds": seed_file,
            "rows": rows,
            "details": details,
            "manual_check_sheet": sheet,
            "known_limits": KNOWN_LIMITS,
        }
        out_dir = Path(options["out"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kg_evaluation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "kg_evaluation_report.md").write_text(self._markdown(report), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Report written to {out_dir}"))

    # ------------------------------------------------------------ runs (E8, E9)

    def _build(self, paper_ids, local_papers):
        builds, times, calls, counts = [], [], [], []
        for run in range(1, RUNS + 1):
            self.stdout.write(f"Build {run}/{RUNS}...")
            with count_openalex_calls() as counter:
                start = time.perf_counter()
                try:
                    builds.append(KGIngestService.build(paper_ids, local_papers))
                except (KGIngestError, OpenAlexAPIError) as e:
                    raise CommandError(f"Build failed: {type(e).__name__}") from e
                times.append(time.perf_counter() - start)
            calls.append(counter["calls"])
            counts.append(_table_counts())
        return builds, times, calls, counts

    def _subgraph(self, seed_keys):
        times, graph = [], None
        for _ in range(RUNS):
            start = time.perf_counter()
            try:
                graph = KGGraphService.subgraph(seed_keys)
            except KGGraphError as e:
                raise CommandError(str(e)) from e
            times.append(time.perf_counter() - start)
        return graph, times

    # ------------------------------------------------------------ metrics

    def _coverage(self, seeds, doi_of, details):
        with_doi = [k for k in seeds if doi_of(k)]
        kg_counts = {
            k: KGEdge.objects.filter(edge_type="CITES", source__key=k).count() for k in with_doi
        }
        papers = [(doi_of(k), kg_counts[k]) for k in with_doi]
        cr_value, cr_covered, cr_rows = measure_coverage(papers, crossref_reference_count)
        oc_value, oc_covered, oc_rows = measure_coverage(papers, opencitations_reference_count)
        details["reference_coverage"] = [
            {"id": k, "doi": doi, "kg_cites": kg, "crossref": cr, "opencitations": oc[2]}
            for k, (doi, kg, cr), oc in zip(with_doi, cr_rows, oc_rows)
        ]
        reason = ("no seed has a DOI" if not with_doi
                  else "the reference source failed or reported 0 references for every seed")
        return [
            _row("E1a", "Reference coverage vs Crossref", cr_value,
                 f"{cr_covered} of {len(with_doi)} seeds with a DOI", TARGET_COVERAGE,
                 reason=reason),
            _row("E1b", "Reference coverage vs OpenCitations", oc_value,
                 f"{oc_covered} of {len(with_doi)} seeds with a DOI", reason=reason),
        ]

    def _agreement(self, cites, doi_of, details):
        candidates = sorted({(doi_of(s), doi_of(t)) for s, t in cites if doi_of(s) and doi_of(t)})
        sample = random.Random(SAMPLE_SEED).sample(candidates, min(SAMPLE_EDGES, len(candidates)))
        fetched = {doi: opencitations_references(doi) for doi in sorted({s for s, _ in sample})}
        measured = {(s, t) for s, t in sample if fetched[s] is not None}
        reference = {(s, t) for s, cited in fetched.items() if cited for t in cited}
        value = citation_agreement(measured, reference)
        details["citation_agreement"] = {
            "sampled_edges": len(sample),
            "measured_edges": len(measured),
            "agreeing_edges": len(measured & reference),
            "disagreeing_edges": sorted(map(list, measured - reference)),
        }
        ok = sum(1 for v in fetched.values() if v is not None)
        reason = ("no CITES edge in scope has a DOI on both ends" if not sample
                  else "OpenCitations failed or listed no references for every sampled citing paper")
        return _row("E2", "Citation agreement with OpenCitations", value,
                    f"{len(measured)} of {len(sample)} sampled edges; "
                    f"{ok} of {len(fetched)} citing papers", TARGET_AGREEMENT, reason=reason)

    def _coherence(self, oa_papers, edges, nodes, similar, details):
        subfield_of = {k: None for k in oa_papers}
        for e in edges:
            if e["type"] == "HAS_TOPIC" and (e["properties"] or {}).get("is_primary"):
                topic = nodes.get(e["target"]) or {}
                subfield_of[e["source"]] = (topic.get("properties") or {}).get("subfield")
        pairs = [(e["source"], e["target"]) for e in similar]
        all_pairs = list(combinations(sorted(oa_papers), 2))
        random_pairs = random.Random(SAMPLE_SEED).sample(all_pairs, min(len(pairs), len(all_pairs)))
        coherence = topic_coherence(pairs, subfield_of)
        baseline = topic_coherence(random_pairs, subfield_of)
        ratio = coherence / baseline if coherence is not None and baseline else None
        known = sum(1 for a, b in pairs if subfield_of.get(a) and subfield_of.get(b))
        details["topic_coherence"] = {"similar_to": coherence, "random_baseline": baseline,
                                      "pairs": len(pairs), "random_pairs": len(random_pairs)}
        if not pairs:
            reason = "no SIMILAR_TO edge in scope"
        elif coherence is None:
            reason = "no SIMILAR_TO pair has a known subfield on both ends"
        else:
            reason = "the random baseline is 0 or has no pair with known subfields"
        return _row("E3", "Topic coherence of SIMILAR_TO / random baseline", ratio,
                    f"{known} of {len(pairs)} SIMILAR_TO pairs", TARGET_COHERENCE_RATIO,
                    display=(f"{_fmt(ratio)} ({_fmt(coherence)} vs random {_fmt(baseline)})"),
                    reason=reason, target_text=f">= {TARGET_COHERENCE_RATIO} x random")

    def _precision(self, seeds, similar, details):
        neighbours = {k: [] for k in seeds}
        for e in similar:
            for a, b in ((e["source"], e["target"]), (e["target"], e["source"])):
                if a in neighbours:
                    neighbours[a].append((-e["weight"], b))
        try:
            works = OpenAlexService().get_works_by_ids(seeds, fields=["id", "related_works"])
            related = {short_id(w["id"]): {short_id(r) for r in w.get("related_works") or []}
                       for w in works if w.get("id")}
            reason = "no seed had both SIMILAR_TO neighbours and OpenAlex related_works"
        except OpenAlexAPIError:
            related = {}
            reason = "the OpenAlex related_works lookup failed"
        values, per_seed = [], {}
        for k in seeds:
            predicted = [key for _, key in sorted(neighbours[k])]
            value = precision_at_k(predicted, related[k], TOP_K) if related.get(k) else None
            per_seed[k] = value
            if value is not None:
                values.append(value)
        details["precision_at_5"] = per_seed
        mean = sum(values) / len(values) if values else None
        return _row("E4", "Precision@5 against OpenAlex related_works", mean,
                    f"{len(values)} of {len(seeds)} seeds", reason=reason)

    def _entities(self, oa_papers, edges, nodes):
        scope = set(oa_papers)
        by_type = {t: {e["source"] for e in edges if e["type"] == t}
                   for t in ("AUTHORED_BY", "PUBLISHED_IN", "HAS_TOPIC", "AFFILIATED_WITH")}
        with_inst = {e["source"] for e in edges
                     if e["type"] == "AUTHORED_BY" and e["target"] in by_type["AFFILIATED_WITH"]}
        authors = [n for n in nodes.values() if n["type"] == "author"]
        orcid = sum(1 for n in authors if (n.get("properties") or {}).get("orcid"))

        def share(found):
            return len(found & scope) / len(scope) if scope else None

        covered = f"{len(scope)} hydrated papers"
        reason = "no hydrated OpenAlex paper in scope"
        return [
            _row("E5a", "Papers with >= 1 author", share(by_type["AUTHORED_BY"]), covered,
                 TARGET_ENTITY_SHARE, reason=reason),
            _row("E5b", "Papers with >= 1 institution (via an author)", share(with_inst),
                 covered, reason=reason),
            _row("E5c", "Papers with a venue", share(by_type["PUBLISHED_IN"]), covered,
                 reason=reason),
            _row("E5d", "Papers with >= 1 topic", share(by_type["HAS_TOPIC"]), covered,
                 TARGET_ENTITY_SHARE, reason=reason),
            _row("E5e", "Authors with an ORCID", orcid / len(authors) if authors else None,
                 f"{len(authors)} authors", reason="no author in scope"),
        ]

    def _local(self, local_papers, build, sheet):
        if not local_papers:
            return _row("E6", "Local resolution rate (title match)", None, "0 local papers",
                        reason="the seed file has no local papers")
        local = {p["id"]: p for p in local_papers}
        same_as = KGEdge.objects.filter(
            edge_type="SAME_AS", source__key__in=list(local)
        ).select_related("source", "target")
        matched_on = Counter()
        for edge in sorted(same_as, key=lambda e: e.source.key):
            paper, work = local[edge.source.key], edge.target
            kind = (edge.properties or {}).get("matched_on")
            matched_on[kind] += 1
            authors = KGEdge.objects.filter(edge_type="AUTHORED_BY", source=work).select_related("target")
            sheet.append({
                "local_id": paper["id"],
                "local_title": paper.get("title"),
                "local_year": paper.get("year"),
                "local_authors": paper.get("authors") or "not sent by the research server",
                "openalex_id": work.key,
                "openalex_title": work.label,
                "openalex_year": (work.properties or {}).get("year"),
                "openalex_authors": ", ".join(sorted(a.target.label for a in authors)),
                "matched_on": kind,
                "similarity": round(edge.weight, 3),
                "correct": "",
            })
        resolved = build["local"]["resolved"]
        split = ", ".join(f"{count} {kind}" for kind, count in sorted(matched_on.items())) or "none"
        return _row("E6", "Local resolution rate (title match)", resolved / len(local_papers),
                    f"{resolved} of {len(local_papers)} local papers ({split}); no DOI path")

    def _structure(self, papers, oa_papers, seeds, edges, cites, similar, details):
        by_type = Counter(e["type"] for e in edges)
        stubs = KGEdge.objects.filter(
            edge_type="CITES", source__key__in=oa_papers, target__is_stub=True
        ).values("target").distinct().count()
        citers = [KGEdge.objects.filter(edge_type="CITES", target__key=k).count() for k in seeds]
        co_citation = sum(1 for e in similar if (e["properties"] or {}).get("method") == "co_citation")
        details["edges_by_type"] = dict(sorted(by_type.items()))
        details["stored_citers_per_seed"] = dict(zip(seeds, citers))
        lcc = largest_component_share(oa_papers, cites) if oa_papers else None
        return [
            _row("E7a", "Papers in scope", len(papers), f"{len(seeds)} OpenAlex seeds",
                 display=f"{len(papers)} ({len(oa_papers)} OpenAlex, "
                         f"{len(papers) - len(oa_papers)} local)"),
            _row("E7b", "Edges by type", sum(by_type.values()), "scope",
                 display=", ".join(f"{t} {c}" for t, c in sorted(by_type.items()))),
            _row("E7c", "Stub share (stubs cited by scope papers)",
                 stubs / (stubs + len(oa_papers)) if stubs + len(oa_papers) else None,
                 f"{stubs} stubs, {len(oa_papers)} hydrated", reason="no paper in scope"),
            _row("E7d", "Share of papers in largest weakly connected CITES component", lcc,
                 f"{len(oa_papers)} hydrated papers", reason="no paper in scope"),
            _row("E7e", "Stored citers per seed (min / median / max)",
                 statistics.median(citers) if citers else None, f"{len(seeds)} seeds",
                 display=(f"{min(citers)} / {statistics.median(citers)} / {max(citers)}"
                          if citers else None),
                 reason="no OpenAlex seed in scope"),
            _row("E7f", "Share of SIMILAR_TO from co-citation",
                 co_citation / len(similar) if similar else None,
                 f"{co_citation} of {len(similar)} SIMILAR_TO edges",
                 reason="no SIMILAR_TO edge in scope"),
        ]

    def _performance(self, builds, build_times, call_counts, subgraph_times, counts_after):
        truncated = sum(1 for b in builds if b["truncated"])
        (n1, e1), (n2, e2) = counts_after[0], counts_after[1]
        return [
            _row("E8a", "Build time, median of 3 (s)", statistics.median(build_times),
                 f"{RUNS} runs; {truncated} truncated", TARGET_BUILD_SECONDS, False,
                 display=f"{statistics.median(build_times):.2f}"),
            _row("E8b", "OpenAlex calls per build, median of 3", statistics.median(call_counts),
                 f"{RUNS} runs", display=" / ".join(map(str, call_counts))
                 + f" (median {statistics.median(call_counts)})"),
            _row("E8c", "Subgraph time, median of 3 (s)", statistics.median(subgraph_times),
                 f"{RUNS} runs", TARGET_SUBGRAPH_SECONDS, False,
                 display=f"{statistics.median(subgraph_times):.2f}"),
            _row("E9", "Rebuild stability (node + edge rows changed by a 2nd build)",
                 abs(n2 - n1) + abs(e2 - e1), "table-wide row counts", 0, False,
                 display=f"nodes {n1} -> {n2}, edges {e1} -> {e2}", target_text="identical"),
        ]

    # ------------------------------------------------------------ output

    def _markdown(self, report) -> str:
        seeds = report["seeds"]
        lines = [
            "# Knowledge graph evaluation",
            "",
            f"- Date: {report['date']}",
            f"- Commit: {report['commit']}",
            f"- Seeds collected {seeds.get('collected')} for the query "
            f"\"{seeds.get('query')}\"",
            "",
            "## Seeds",
            "",
            "| # | ID | Title |",
            "|---|---|---|",
        ]
        entries = list(seeds.get("paper_ids") or []) + list(seeds.get("local_papers") or [])
        for i, p in enumerate(entries, 1):
            lines.append(f"| {i} | {_cell(p.get('id'))} | {_cell(p.get('title'))} |")

        lines += ["", "## Results", "",
                  "| # | Metric | Value | Covered | Target | Verdict |",
                  "|---|---|---|---|---|---|"]
        for r in report["rows"]:
            lines.append(f"| {r['id']} | {_cell(r['metric'])} | {_cell(r['display'])} | "
                         f"{_cell(r['covered'])} | {_cell(r['target'])} | {r['verdict']} |")

        why = [r for r in report["rows"] if r["why"]]
        if why:
            lines += ["", "## Why", ""]
            lines += [f"- **{r['id']}** ({r['verdict']}): {r['why']}" for r in why]

        coverage = report["details"].get("reference_coverage") or []
        if coverage:
            lines += ["", "## Reference coverage per seed (E1)", "",
                      "| ID | DOI | CITES stored | Crossref | OpenCitations |",
                      "|---|---|---|---|---|"]
            for c in coverage:
                lines.append(f"| {c['id']} | {_cell(c['doi'])} | {c['kg_cites']} | "
                             f"{_fmt(c['crossref'])} | {_fmt(c['opencitations'])} |")

        sheet = report["manual_check_sheet"]
        if sheet:
            lines += ["", "## E6 manual check sheet", "",
                      "Mark each pair correct or wrong by hand.", "",
                      "| Local ID | Local title | Local year | Local authors | OpenAlex ID | "
                      "OpenAlex title | OpenAlex year | OpenAlex authors | Matched on | "
                      "Similarity | Correct? |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in sheet:
                lines.append("| " + " | ".join(_cell(s[k]) for k in (
                    "local_id", "local_title", "local_year", "local_authors", "openalex_id",
                    "openalex_title", "openalex_year", "openalex_authors", "matched_on",
                    "similarity", "correct")) + " |")

        lines += ["", "## Known limits", ""]
        lines += [f"- {limit}" for limit in report["known_limits"]]
        return "\n".join(lines) + "\n"
