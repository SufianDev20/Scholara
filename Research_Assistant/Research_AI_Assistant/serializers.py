import re

from rest_framework import serializers
from .models import ModelPerformance, QueryLog, ModelReliability, ResponseLog
from .services.extract_service import MAX_PDF_CANDIDATES
from .services.kg_mapper import short_id

# Stricter than pdf_service.MAX_PDF_URL_LENGTH (2000): this is an early
# request-validation cap, not the SSRF-guard's hard safety limit.
MAX_PDF_URL_INPUT_LENGTH = 500


class QueryLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryLog
        fields = ["query_text", "ranking_mode", "result_count", "created_at"]


class ModelPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPerformance
        fields = "__all__"


class ResponseLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResponseLog
        fields = [
            "model_name",
            "response_time",
            "success",
            "created_at",
            "error_message",
            "user_query_hash",
            "request_type",
            "response_length",
        ]


class ModelReliabilitySerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelReliability
        fields = [
            "model_name",
            "tier",
            "priority",
            "max_retries",
            "custom_temperature",
            "circuit_breaker_threshold",
            "last_updated",
        ]


class PDFUrlsValueField(serializers.Field):
    """One paper's PDF URL(s): a single URL string or a short ordered list. Always yields a list."""

    def to_internal_value(self, data):
        urls = [data] if isinstance(data, str) else data
        if not isinstance(urls, list) or not urls:
            raise serializers.ValidationError(
                "Expected a PDF URL or a non-empty list of PDF URLs."
            )
        if len(urls) > MAX_PDF_CANDIDATES:
            raise serializers.ValidationError(
                f"At most {MAX_PDF_CANDIDATES} PDF URLs per paper."
            )
        cleaned = []
        for url in urls:
            if not isinstance(url, str) or not url.strip():
                raise serializers.ValidationError("Each PDF URL must be a non-empty string.")
            if len(url.strip()) > MAX_PDF_URL_INPUT_LENGTH:
                raise serializers.ValidationError(
                    f"PDF URLs must be {MAX_PDF_URL_INPUT_LENGTH} characters or fewer."
                )
            cleaned.append(url.strip())
        return cleaned

    def to_representation(self, value):
        return value


class PaperMetaField(serializers.Field):
    """Display metadata for one paper. Lenient: bad parts are dropped, not rejected."""

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError("Expected an object.")
        title = data.get("title")
        authors = data.get("authors")
        year = data.get("year")
        return {
            "title": title.strip()[:300] if isinstance(title, str) else "",
            "authors": [
                a.strip()[:100] for a in authors[:10] if isinstance(a, str) and a.strip()
            ]
            if isinstance(authors, list)
            else [],
            "year": year if isinstance(year, int) else None,
        }

    def to_representation(self, value):
        return value


class QARequestSerializer(serializers.Serializer):
    """Validate an incoming Multi-Paper Q&A request (views.multi_paper_qa)."""

    paper_ids = serializers.ListField(
        child=serializers.CharField(min_length=1),
        min_length=1,
        max_length=10,
        help_text="OpenAlex work IDs of the selected papers (1-10).",
    )
    question = serializers.CharField(
        min_length=1, max_length=2000, trim_whitespace=True
    )
    pdf_urls = serializers.DictField(
        child=PDFUrlsValueField(),
        help_text="Mapping of paper_id -> direct PDF URL (or ordered list of up to 3 candidate URLs) for each selected paper.",
    )

    paper_meta = serializers.DictField(
        child=PaperMetaField(),
        required=False,
        help_text="Optional mapping of paper_id -> {title, authors, year}, used to name papers in the answer.",
    )

    def validate(self, attrs):
        """Ensure every paper_id has a matching pdf_url entry."""
        missing = [
            paper_id
            for paper_id in attrs["paper_ids"]
            if paper_id not in attrs["pdf_urls"]
        ]
        if missing:
            raise serializers.ValidationError(
                {"pdf_urls": f"Missing pdf_url for paper_id(s): {missing}"}
            )
        return attrs


MAX_GRAPH_SEEDS = 25
MAX_LOCAL_PAPERS = 10
LOCAL_ID_PREFIX = "local:"
LOCAL_ID_MAX_LENGTH = 120  # KGNode.key max_length
OPENALEX_WORK_ID_RE = re.compile(r"^(https://openalex\.org/)?W\d+$")


def normalise_work_ids(values):
    """Validate OpenAlex work IDs, return short keys with duplicates removed in order.

    Raises serializers.ValidationError naming the first bad ID. Shared by
    KGBuildRequestSerializer and the subgraph view's `ids` parameter.
    """
    for value in values:
        if not OPENALEX_WORK_ID_RE.match(value):
            raise serializers.ValidationError(
                f"Not an OpenAlex work ID: {value!r}"
            )
    return list(dict.fromkeys(short_id(v) for v in values))


def _check_local_id(value):
    if not value.startswith(LOCAL_ID_PREFIX) or value == LOCAL_ID_PREFIX:
        raise serializers.ValidationError(f"Not a local paper ID: {value!r}")
    if len(value) > LOCAL_ID_MAX_LENGTH:
        raise serializers.ValidationError(
            f"Local paper ID longer than {LOCAL_ID_MAX_LENGTH} characters: {value[:40]!r}..."
        )
    return value


def normalise_graph_ids(values):
    """Subgraph `ids`: OpenAlex work IDs (normalised) or `local:` paper IDs, deduplicated in order."""
    keys = []
    for value in values:
        if value.startswith(LOCAL_ID_PREFIX):
            keys.append(_check_local_id(value))
        else:
            keys.extend(normalise_work_ids([value]))
    return list(dict.fromkeys(keys))


class LocalPaperSerializer(serializers.Serializer):
    """One local-corpus paper: the university server sends no DOI or OpenAlex ID."""

    id = serializers.CharField()
    title = serializers.CharField(max_length=500, allow_blank=True)
    year = serializers.IntegerField(allow_null=True)

    def validate_id(self, value):
        return _check_local_id(value)


class KGBuildRequestSerializer(serializers.Serializer):
    """Validate a knowledge graph build request (views_graph.graph_build)."""

    paper_ids = serializers.ListField(
        child=serializers.CharField(min_length=1),
        required=False,
        default=list,
        max_length=MAX_GRAPH_SEEDS,
        help_text="OpenAlex work IDs of the seed papers (0-25).",
    )
    local_papers = serializers.ListField(
        child=LocalPaperSerializer(),
        required=False,
        default=list,
        max_length=MAX_LOCAL_PAPERS,
        help_text='Local-corpus papers {"id": "local:...", "title", "year"} (0-10).',
    )

    def validate_paper_ids(self, value):
        return normalise_work_ids(value)

    def validate_local_papers(self, value):
        unique = {}
        for paper in value:
            unique.setdefault(paper["id"], dict(paper))
        return list(unique.values())

    def validate(self, attrs):
        total = len(attrs["paper_ids"]) + len(attrs["local_papers"])
        if total == 0:
            raise serializers.ValidationError("Send paper_ids, local_papers or both.")
        if total > MAX_GRAPH_SEEDS:
            raise serializers.ValidationError(
                f"At most {MAX_GRAPH_SEEDS} papers in paper_ids and local_papers combined."
            )
        return attrs
