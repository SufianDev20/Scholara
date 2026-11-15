"""
Knowledge graph API views. Business logic lives in the KG services; these
views validate input and map service exceptions to HTTP statuses.
"""

from rest_framework import serializers, status
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response

from .models import KGEdge
from .serializers import KGBuildRequestSerializer, MAX_GRAPH_SEEDS, normalise_graph_ids
from .services.kg_graph_service import KGGraphError, KGGraphService
from .services.kg_ingest_service import KGIngestError, KGIngestService
from .services.openalex_service import OpenAlexAPIError
from .throttles import GraphBuildRateThrottle, GraphRateThrottle


def _bad_request(details):
    return Response(
        {"error": "Invalid request.", "details": details},
        status=status.HTTP_400_BAD_REQUEST,
    )


@api_view(["POST"])
@throttle_classes([GraphBuildRateThrottle])
def graph_build(request):
    """
    POST /api/graph/build/
    Body: {"paper_ids": ["W123", "https://openalex.org/W456"],
           "local_papers": [{"id": "local:a.pdf", "title": "...", "year": 2023}]}
    Either list may be omitted, not both; at most 25 papers combined.

    Fetches and stores the seeds' neighbourhood, then returns it.
    200: {"build": {...}, "graph": {...}}
    400 invalid body, 404 no seed found, 502 OpenAlex failure.
    """
    serializer = KGBuildRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return _bad_request(serializer.errors)
    seeds = serializer.validated_data["paper_ids"]
    local_papers = serializer.validated_data["local_papers"]

    try:
        build = KGIngestService.build(seeds, local_papers=local_papers)
        graph = KGGraphService.subgraph(seeds + [p["id"] for p in local_papers])
    except (KGIngestError, KGGraphError) as exc:
        return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except OpenAlexAPIError as exc:
        return Response(
            {"error": f"OpenAlex request failed: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
    return Response({"build": build, "graph": graph})


def _flag(params, name, default):
    """Parse a 0/1 query flag; None when the value is anything else."""
    raw = params.get(name)
    if raw is None:
        return default
    return {"0": False, "1": True}.get(raw)


@api_view(["GET"])
@throttle_classes([GraphRateThrottle])
def graph_subgraph(request):
    """
    GET /api/graph/subgraph/?ids=W1,local:a.pdf&edge_types=CITES,HAS_TOPIC&similarity=0|1&include_stubs=0|1

    200: the subgraph. 400 bad parameters, 404 no seed paper in the graph.
    """
    params = request.query_params
    raw_ids = [i.strip() for i in params.get("ids", "").split(",") if i.strip()]
    if not raw_ids:
        return _bad_request({"ids": "This query parameter is required."})
    if len(raw_ids) > MAX_GRAPH_SEEDS:
        return _bad_request({"ids": f"At most {MAX_GRAPH_SEEDS} ids."})
    try:
        ids = normalise_graph_ids(raw_ids)
    except serializers.ValidationError as exc:
        return _bad_request({"ids": exc.detail})

    edge_types = None
    if "edge_types" in params:
        edge_types = {e.strip() for e in params["edge_types"].split(",") if e.strip()}
        valid = {code for code, _ in KGEdge.EDGE_TYPES}
        unknown = sorted(edge_types - valid)
        if unknown:
            return _bad_request({"edge_types": f"Unknown edge type(s): {', '.join(unknown)}"})

    similarity = _flag(params, "similarity", True)
    include_stubs = _flag(params, "include_stubs", False)
    if similarity is None or include_stubs is None:
        return _bad_request({"detail": "similarity and include_stubs must be 0 or 1."})

    try:
        graph = KGGraphService.subgraph(
            ids, edge_types=edge_types, similarity=similarity, include_stubs=include_stubs
        )
    except KGGraphError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    return Response(graph)
