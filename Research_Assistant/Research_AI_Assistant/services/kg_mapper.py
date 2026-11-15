"""Map one raw OpenAlex work to knowledge-graph node and edge specs.

Pure functions: no database, no network. Keys are OpenAlex IDs with the
``https://openalex.org/`` prefix removed.
"""

from dataclasses import dataclass, field

_PREFIX = "https://openalex.org/"

# Documentation only: the SemOpenAlex / SPAR term each edge type stands for.
EDGE_VOCAB = {
    "CITES": "cito:cites",
    "AUTHORED_BY": "dcterms:creator",
    "AFFILIATED_WITH": "org:memberOf",
    "PUBLISHED_IN": "soa:hasHostVenue",
    "HAS_TOPIC": "soa:hasTopic",
    "HAS_KEYWORD": "soa:hasKeyword",
    "SAME_AS": "owl:sameAs",
}


@dataclass(frozen=True)
class NodeSpec:
    node_type: str
    key: str
    label: str = ""
    properties: dict = field(default_factory=dict)
    is_stub: bool = False


@dataclass(frozen=True)
class EdgeSpec:
    source_key: str
    target_key: str
    edge_type: str
    weight: float = 1.0
    properties: dict = field(default_factory=dict)


def short_id(openalex_url: str) -> str:
    """'https://openalex.org/W12' -> 'W12'; a value without the prefix is unchanged."""
    return openalex_url.removeprefix(_PREFIX)


def _name(obj: dict) -> str:
    return obj.get("display_name") or ""


def work_to_graph(work: dict) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    """Node and edge specs for one raw work. A work without an id maps to nothing.

    Nodes and edges are de-duplicated within the call. Missing paper
    properties are stored as None.
    """
    work_id = work.get("id")
    if not work_id:
        return [], []
    paper_key = short_id(work_id)

    nodes: dict[str, NodeSpec] = {}
    edges: dict[tuple, EdgeSpec] = {}

    def add_node(spec: NodeSpec) -> None:
        # First spec wins, so the hydrated paper is never replaced by a stub.
        nodes.setdefault(spec.key, spec)

    def add_edge(source, target, edge_type, weight=1.0, properties=None) -> None:
        edges.setdefault(
            (source, target, edge_type),
            EdgeSpec(source, target, edge_type, weight, properties or {}),
        )

    refs = work.get("referenced_works") or []
    year = work.get("publication_year")
    add_node(NodeSpec("paper", paper_key, work.get("title") or "", {
        "doi": work.get("doi"),
        "year": year,
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "fwci": work.get("fwci"),
        "referenced_works_count": len(refs),
    }))

    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        if not author.get("id"):
            continue
        author_key = short_id(author["id"])
        add_node(NodeSpec("author", author_key, _name(author),
                          {"orcid": author.get("orcid")}))
        add_edge(paper_key, author_key, "AUTHORED_BY", properties={
            "position": authorship.get("author_position"),
            "is_corresponding": authorship.get("is_corresponding"),
        })
        for inst in authorship.get("institutions") or []:
            if not inst.get("id"):
                continue
            inst_key = short_id(inst["id"])
            add_node(NodeSpec("institution", inst_key, _name(inst), {
                "country_code": inst.get("country_code"),
                "ror": inst.get("ror"),
                "type": inst.get("type"),
            }))
            add_edge(author_key, inst_key, "AFFILIATED_WITH",
                     properties={"year": year})

    source = (work.get("primary_location") or {}).get("source") or {}
    if source.get("id"):
        venue_key = short_id(source["id"])
        add_node(NodeSpec("venue", venue_key, _name(source), {
            "issn_l": source.get("issn_l"),
            "type": source.get("type"),
        }))
        add_edge(paper_key, venue_key, "PUBLISHED_IN")

    topics = [t for t in (work.get("topics") or []) if t.get("id")]
    primary_id = (work.get("primary_topic") or {}).get("id")
    if not primary_id and topics:
        primary_id = topics[0]["id"]
    for topic in topics:
        topic_key = short_id(topic["id"])
        add_node(NodeSpec("topic", topic_key, _name(topic), {
            "subfield": (topic.get("subfield") or {}).get("display_name"),
            "field": (topic.get("field") or {}).get("display_name"),
            "domain": (topic.get("domain") or {}).get("display_name"),
        }))
        score = topic.get("score")
        add_edge(paper_key, topic_key, "HAS_TOPIC",
                 weight=1.0 if score is None else score,
                 properties={"is_primary": topic["id"] == primary_id})

    for kw in work.get("keywords") or []:
        if not kw.get("id"):
            continue
        kw_key = short_id(kw["id"])
        add_node(NodeSpec("keyword", kw_key, _name(kw)))
        score = kw.get("score")
        add_edge(paper_key, kw_key, "HAS_KEYWORD",
                 weight=1.0 if score is None else score)

    for ref in refs:
        if not ref:
            continue
        ref_key = short_id(ref)
        add_node(NodeSpec("paper", ref_key, "", {}, is_stub=True))
        add_edge(paper_key, ref_key, "CITES")

    return list(nodes.values()), list(edges.values())
