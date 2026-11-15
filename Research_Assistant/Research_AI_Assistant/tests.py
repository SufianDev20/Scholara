"""
Tests for Research AI Assistant.

Covers:
  - OpenAlexService: paper and author search
  - ExtractionService: metadata extraction, abstract reconstruction, concepts
  - OpenAlex views: search and author endpoints
  - OpenRouterService.complete(): success and failure paths
  - prompt_builder: build_user_message() and SYSTEM_PROMPT
  - Summarise view: valid input, missing fields, upstream errors
"""

import json
from collections import Counter
from datetime import timedelta
from unittest.mock import MagicMock, Mock, patch

from django.db import IntegrityError, transaction
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.utils import timezone

from .models import (
    KGEdge,
    KGNode,
    ModelPerformance,
    ModelReliability,
    PaperQAChunks,
    User,
)
from .services.llm_service import LLMError, STREAM_RESET
from .services.qa_chunk_cache import CHUNK_SCHEMA_VERSION, QAChunkCache
from .services.performance_tracker import (
    CIRCUIT_BREAKER_RESET_MIN,
    PerformanceTracker,
)

from .services.openalex_service import OpenAlexService, OpenAlexAPIError
from .services import kg_ingest_service
from .services.kg_ingest_service import KGIngestError, KGIngestService
from .services.kg_mapper import EDGE_VOCAB, EdgeSpec, short_id, work_to_graph
from .services.extract_service import ExtractionService
from django.core.cache import cache

from .services.openrouter_service import (
    OpenRouterService,
    OpenRouterAPIError,
    REASONING_MODELS_CACHE_KEY,
    fetch_free_models,
    reasoning_model_ids,
)
from .services.prompt_builder import (
    InvalidSummaryFormat,
    SUMMARY_MAX_TOKENS,
    ensure_references,
    require_all_papers,
    summary_max_tokens,
    system_prompt,
    build_user_message,
)


class ClerkAuthTestMixin:
    """
    Bypasses Clerk JWT verification for tests that exercise a view requiring
    auth (every DRF view except api_root/search/multi_paper_qa, per
    settings.py's DEFAULT_AUTHENTICATION_CLASSES), without needing a real
    Clerk token. Mix in before django.test.TestCase so MRO runs this setUp.
    """

    def setUp(self):
        super().setUp()
        self._test_user = User.objects.create(clerk_id="test-clerk-id")
        patcher = patch(
            "Research_AI_Assistant.authentication.ClerkAuthentication.authenticate",
            return_value=(self._test_user, None),
        )
        patcher.start()
        self.addCleanup(patcher.stop)


# ---------------------------------------------------------------------------
# OpenAlex service tests
# ---------------------------------------------------------------------------


class TestOpenAlexService(TestCase):
    """Test cases for OpenAlexService class."""

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_search_papers_success_default(self, mock_works_class):
        """Test successful paper search with default parameters."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        mock_search_result = Mock()
        mock_works_instance.similar.return_value = mock_search_result

        mock_search_result.filter.return_value = mock_search_result

        mock_results = [{"id": "1", "title": "Test Paper"}]
        mock_search_result.get.return_value = (
            mock_results,
            {"count": 1, "next_cursor": None},
        )

        service = OpenAlexService()
        result = service.search_papers("machine learning")

        # First page uses semantic search; Load More continues with keyword
        # from the start of its cursor chain ("*").
        self.assertEqual(
            result,
            {
                "results": mock_results,
                "meta": {"count": 1, "next_cursor": "*"},
                "search_type": "semantic",
            },
        )
        mock_works_class.assert_called_once()
        mock_works_instance.similar.assert_called_once_with("machine learning")
        mock_works_instance.search.assert_not_called()
        mock_search_result.filter.assert_called_once_with(is_retracted=False)
        mock_search_result.get.assert_called_once_with(return_meta=True, per_page=25)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_semantic_empty_falls_back_to_keyword(self, mock_works_class):
        """Zero semantic hits fall back to keyword search."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        semantic = Mock()
        semantic.filter.return_value = semantic
        semantic.get.return_value = ([], {"count": 0, "next_cursor": None})
        keyword = Mock()
        keyword.filter.return_value = keyword
        keyword_results = [{"id": "9"}]
        keyword.get.return_value = (keyword_results, {"count": 1, "next_cursor": "k2"})
        mock_works_instance.similar.return_value = semantic
        mock_works_instance.search.return_value = keyword

        result = OpenAlexService().search_papers("odd query", per_page=10)

        self.assertEqual(result["search_type"], "keyword")
        self.assertEqual(result["results"], keyword_results)
        self.assertEqual(result["meta"]["next_cursor"], "k2")
        keyword.get.assert_called_once_with(return_meta=True, per_page=10, cursor="*")

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_semantic_error_falls_back_to_keyword(self, mock_works_class):
        """A semantic API failure falls back to keyword search."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        semantic = Mock()
        semantic.filter.return_value = semantic
        semantic.get.side_effect = Exception("semantic down")
        keyword = Mock()
        keyword.filter.return_value = keyword
        keyword.get.return_value = ([{"id": "5"}], {"count": 1, "next_cursor": None})
        mock_works_instance.similar.return_value = semantic
        mock_works_instance.search.return_value = keyword

        result = OpenAlexService().search_papers("query")

        self.assertEqual(result["search_type"], "keyword")
        self.assertEqual(result["results"], [{"id": "5"}])

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_cursor_request_uses_keyword(self, mock_works_class):
        """Load More (cursor set) never uses semantic search."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance
        keyword = Mock()
        keyword.filter.return_value = keyword
        keyword.get.return_value = ([{"id": "7"}], {"count": 1, "next_cursor": "n"})
        mock_works_instance.search.return_value = keyword

        result = OpenAlexService().search_papers("query", cursor="*")

        mock_works_instance.similar.assert_not_called()
        self.assertEqual(result["search_type"], "keyword")

    def _semantic_keyword_mocks(self, mock_works_class, sem_results, kw_results=None):
        works = Mock()
        mock_works_class.return_value = works
        semantic = Mock()
        semantic.filter.return_value = semantic
        semantic.get.return_value = (sem_results, {"count": 999, "next_cursor": None})
        keyword = Mock()
        keyword.filter.return_value = keyword
        keyword.get.return_value = (
            kw_results or [],
            {"count": 42, "next_cursor": "kw2"},
        )
        works.similar.return_value = semantic
        works.search.return_value = keyword
        return semantic, keyword

    @staticmethod
    def _filter_kwargs(mock_query):
        return [c.kwargs for c in mock_query.filter.call_args_list]

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_default_year_range_is_not_a_filter(self, mock_works_class):
        """Frontend default 1900-2026 must stay on plain semantic search."""
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, [{"id": "1"}])

        result = OpenAlexService().search_papers("q", min_year=1900, max_year=2026)

        self.assertEqual(result["search_type"], "semantic")
        self.assertEqual(self._filter_kwargs(semantic), [{"is_retracted": False}])
        keyword.get.assert_not_called()

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_exact_year_uses_publication_year_on_semantic(self, mock_works_class):
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, [{"id": "1"}])

        result = OpenAlexService().search_papers("q", min_year=2023, max_year=2023)

        self.assertEqual(result["search_type"], "semantic")
        kwargs = self._filter_kwargs(semantic)
        self.assertIn({"publication_year": 2023}, kwargs)
        for k in kwargs:
            self.assertNotIn("from_publication_date", k)
            self.assertNotIn("to_publication_date", k)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_year_range_post_filters_semantic_pool(self, mock_works_class):
        pool = [{"id": str(y), "publication_year": y} for y in (2010, 2021, 2022, 2030)]
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, pool)

        result = OpenAlexService().search_papers(
            "q", per_page=2, min_year=2020, max_year=2024
        )

        self.assertEqual(result["search_type"], "semantic")
        self.assertEqual([w["id"] for w in result["results"]], ["2021", "2022"])
        self.assertEqual(result["meta"]["count"], 2)
        for k in self._filter_kwargs(semantic):
            self.assertNotIn("from_publication_date", k)
            self.assertNotIn("to_publication_date", k)
        keyword.get.assert_not_called()
        semantic.get.assert_called_once_with(return_meta=True, per_page=50)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_thin_semantic_pool_is_topped_up_from_keyword(self, mock_works_class):
        pool = [{"id": "a", "publication_year": 2021}]
        kw = [
            {"id": "a", "publication_year": 2021},
            {"id": "b", "publication_year": 2022},
            {"id": "c", "publication_year": 2023},
        ]
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, pool, kw)

        result = OpenAlexService().search_papers(
            "q", per_page=3, min_year=2020, max_year=2024
        )

        self.assertEqual(result["search_type"], "hybrid")
        self.assertEqual([w["id"] for w in result["results"]], ["a", "b", "c"])
        # next_cursor is "*", not the keyword page-2 cursor: Load More re-runs
        # keyword from the start and the frontend de-dupes, so keyword page-1
        # papers not used as top-up are not lost.
        self.assertEqual(result["meta"], {"count": 42, "next_cursor": "*"})
        kw_kwargs = self._filter_kwargs(keyword)
        self.assertIn({"from_publication_date": "2020-01-01"}, kw_kwargs)
        self.assertIn({"to_publication_date": "2024-12-31"}, kw_kwargs)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_oa_status_filtered_client_side_on_semantic(self, mock_works_class):
        pool = [
            {"id": "g", "open_access": {"oa_status": "gold"}},
            {"id": "b", "open_access": {"oa_status": "bronze"}},
            {"id": "n", "open_access": None},
        ]
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, pool)

        result = OpenAlexService().search_papers("q", per_page=1, oa_status="gold")

        self.assertEqual(result["search_type"], "semantic")
        self.assertEqual([w["id"] for w in result["results"]], ["g"])
        for k in self._filter_kwargs(semantic):
            self.assertNotIn("oa_status", k)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_post_filter_empty_falls_back_to_keyword(self, mock_works_class):
        pool = [{"id": "x", "publication_year": 1990}]
        kw = [{"id": "k", "publication_year": 2021}]
        semantic, keyword = self._semantic_keyword_mocks(mock_works_class, pool, kw)

        result = OpenAlexService().search_papers(
            "q", per_page=5, min_year=2020, max_year=2024
        )

        self.assertEqual(result["search_type"], "keyword")
        self.assertEqual(result["results"], kw)
        keyword.get.assert_called_once_with(return_meta=True, per_page=5, cursor="*")

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_search_papers_with_filters(self, mock_works_class):
        """Test paper search with all filters enabled."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        mock_search_result = Mock()
        mock_works_instance.search.return_value = mock_search_result

        mock_search_result.filter.return_value = mock_search_result

        mock_results = [{"id": "2", "title": "Open Access Paper"}]
        mock_search_result.get.return_value = (
            mock_results,
            {"count": 1, "next_cursor": None},
        )

        service = OpenAlexService()
        result = service.search_papers(
            query="AI",
            per_page=10,
            cursor="page3token",
            exclude_retracted=False,
            open_access_only=True,
            oa_status="gold",
        )

        self.assertEqual(result["results"], mock_results)
        self.assertEqual(mock_search_result.filter.call_count, 2)
        mock_search_result.filter.assert_any_call(is_oa=True)
        mock_search_result.filter.assert_any_call(oa_status="gold")
        mock_search_result.get.assert_called_once_with(
            return_meta=True, per_page=10, cursor="page3token"
        )

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_search_papers_no_filters(self, mock_works_class):
        """Test paper search with no filters."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        mock_search_result = Mock()
        mock_works_instance.search.return_value = mock_search_result

        mock_search_result.filter.return_value = mock_search_result

        mock_results = [{"id": "3", "title": "Any Paper"}]
        mock_search_result.get.return_value = (
            mock_results,
            {"count": 1, "next_cursor": None},
        )

        service = OpenAlexService()
        result = service.search_papers(
            query="physics",
            cursor="page2token",
            exclude_retracted=False,
            open_access_only=False,
            oa_status=None,
        )

        self.assertEqual(result["results"], mock_results)
        mock_search_result.filter.assert_not_called()
        mock_search_result.get.assert_called_once_with(
            return_meta=True, per_page=25, cursor="page2token"
        )

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_search_papers_api_error(self, mock_works_class):
        """Test that OpenAlexAPIError is raised on API failure."""
        mock_works_instance = Mock()
        mock_works_class.return_value = mock_works_instance

        mock_search_result = Mock()
        mock_works_instance.search.return_value = mock_search_result
        mock_search_result.filter.return_value = mock_search_result

        mock_search_result.get.side_effect = Exception("API timeout")

        service = OpenAlexService()

        with self.assertRaises(OpenAlexAPIError) as cm:
            service.search_papers("error query")

        self.assertIn("Failed to retrieve papers from OpenAlex", str(cm.exception))


# ---------------------------------------------------------------------------
# OpenAlex view tests
# ---------------------------------------------------------------------------


class TestOpenAlexViews(ClerkAuthTestMixin, TestCase):
    """
    Integration tests for the API endpoints defined in views.

    search_authors requires auth (fixed in the ponytail-audit pass: it had
    @throttle_classes but no @api_view, so DRF — and therefore auth — never
    ran); ClerkAuthTestMixin bypasses real Clerk verification. There is no
    "openalex_works_search" route — /api/search/ is named "search" and is
    intentionally open (AllowAny); it has no dedicated test here.
    """

    def setUp(self):
        super().setUp()
        self.client = Client()

    def test_authors_search_missing_query(self):
        resp = self.client.get(reverse("research_ai_assistant:openalex_authors_search"))
        self.assertEqual(resp.status_code, 400)
        content = resp.json()
        self.assertEqual(content["error"]["code"], "missing_query")

    @patch("Research_AI_Assistant.views.OpenAlexService.search_authors")
    def test_authors_search_success(self, mock_search_authors):
        mock_search_authors.return_value = [{"id": "A1"}]

        resp = self.client.get(
            reverse("research_ai_assistant:openalex_authors_search"), {"q": "foo"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"], [{"id": "A1"}])
        mock_search_authors.assert_called_once_with(query="foo", per_page=10, page=1)


# ---------------------------------------------------------------------------
# Extraction service tests
# ---------------------------------------------------------------------------


class TestExtractionService(TestCase):
    """Test cases for Extraction_Service class."""

    def test_extract_metadata_full(self):
        """Test extract_metadata with complete work data."""
        mock_work = {
            "id": "https://openalex.org/W123456789",
            "title": "Sample Research Paper",
            "authorships": [
                {
                    "author": {
                        "display_name": "John Doe",
                        "orcid": "0000-0000-0000-0001",
                    },
                    "institutions": [{"display_name": "University A"}],
                },
                {
                    "author": {"display_name": "Jane Smith", "orcid": None},
                    "institutions": [{"display_name": "University B"}],
                },
            ],
            "abstract_inverted_index": {
                "This": [0],
                "is": [1],
                "a": [2],
                "sample": [3],
                "abstract": [4],
            },
            "publication_year": 2023,
            "doi": "10.1234/sample.doi",
            "cited_by_count": 50,
            "concepts": [
                {"display_name": "Computer Science", "score": 0.9},
                {"display_name": "Artificial Intelligence", "score": 0.8},
                {"display_name": "Machine Learning", "score": 0.7},
                {"display_name": "Data Science", "score": 0.6},
                {"display_name": "Statistics", "score": 0.5},
                {"display_name": "Extra Concept", "score": 0.4},
            ],
            "primary_location": {"source": {"display_name": "Journal of AI"}},
            "best_oa_location": {"pdf_url": "https://example.com/fulltext.pdf"},
            "open_access": {"is_oa": True, "oa_status": "gold"},
        }

        result = ExtractionService.extract_metadata(mock_work)

        expected = {
            "openalex_id": "https://openalex.org/W123456789",
            "title": "Sample Research Paper",
            "authors": [
                {
                    "name": "John Doe",
                    "orcid": "0000-0000-0000-0001",
                    "institutions": ["University A"],
                },
                {"name": "Jane Smith", "orcid": "", "institutions": ["University B"]},
            ],
            "abstract": "This is a sample abstract",
            "publication_year": 2023,
            "doi": "10.1234/sample.doi",
            "cited_by_count": 50,
            "concepts": [
                {"name": "Computer Science", "score": 0.9},
                {"name": "Artificial Intelligence", "score": 0.8},
                {"name": "Machine Learning", "score": 0.7},
                {"name": "Data Science", "score": 0.6},
                {"name": "Statistics", "score": 0.5},
            ],
            "source": "Journal of AI",
            "is_open_access": True,
            "oa_status": "gold",
            "has_pdf_link": True,
            "full_text_url": "https://example.com/fulltext.pdf",
            "pdf_url": "https://example.com/fulltext.pdf",
            "pdf_candidates": ["https://example.com/fulltext.pdf"],
            "oa_url": None,
            "referenced_works": [],
            "referenced_works_count": 0,
        }

        self.assertEqual(result, expected)

    def test_extract_metadata_minimal(self):
        """Test extract_metadata with minimal work data."""
        mock_work = {"id": "https://openalex.org/W987654321", "title": "Minimal Paper"}

        result = ExtractionService.extract_metadata(mock_work)

        expected = {
            "openalex_id": "https://openalex.org/W987654321",
            "title": "Minimal Paper",
            "authors": [],
            "abstract": "",
            "publication_year": None,
            "doi": "",
            "cited_by_count": 0,
            "concepts": [],
            "source": None,
            "is_open_access": False,
            "oa_status": None,
            "has_pdf_link": False,
            "full_text_url": None,
            "pdf_url": None,
            "pdf_candidates": [],
            "oa_url": None,
            "referenced_works": [],
            "referenced_works_count": 0,
        }

        self.assertEqual(result, expected)

    def test_extract_authors(self):
        """Test _extract_authors method."""
        mock_work = {
            "authorships": [
                {
                    "author": {"display_name": "Author One", "orcid": "orcid123"},
                    "institutions": [
                        {"display_name": "Inst A"},
                        {"display_name": "Inst B"},
                    ],
                }
            ]
        }

        authors = ExtractionService._extract_authors(mock_work)

        expected = [
            {
                "name": "Author One",
                "orcid": "orcid123",
                "institutions": ["Inst A", "Inst B"],
            }
        ]

        self.assertEqual(authors, expected)

    def test_reconstruct_abstract(self):
        """Test _reconstruct_abstract method."""
        mock_work = {
            "abstract_inverted_index": {"Hello": [0], "world": [1], "test": [2]}
        }

        abstract = ExtractionService._reconstruct_abstract(mock_work)

        self.assertEqual(abstract, "Hello world test")

    def test_reconstruct_abstract_none(self):
        """Test _reconstruct_abstract with no abstract."""
        mock_work = {}

        abstract = ExtractionService._reconstruct_abstract(mock_work)

        self.assertEqual(abstract, "")

    def test_extract_concepts(self):
        """Test _extract_concepts method."""
        mock_work = {
            "concepts": [
                {"display_name": "Concept A", "score": 0.9},
                {"display_name": "Concept B", "score": 0.8},
                {"display_name": "Concept C", "score": 0.7},
                {"display_name": "Concept D", "score": 0.6},
                {"display_name": "Concept E", "score": 0.5},
                {"display_name": "Concept F", "score": 0.4},
            ]
        }

        concepts = ExtractionService._extract_concepts(mock_work)

        expected = [
            {"name": "Concept A", "score": 0.9},
            {"name": "Concept B", "score": 0.8},
            {"name": "Concept C", "score": 0.7},
            {"name": "Concept D", "score": 0.6},
            {"name": "Concept E", "score": 0.5},
        ]

        self.assertEqual(concepts, expected)


# ---------------------------------------------------------------------------
# Author search tests
# ---------------------------------------------------------------------------


class TestSearchAuthors(TestCase):
    @patch("Research_AI_Assistant.services.openalex_service.Authors")
    def test_search_authors_success(self, mock_authors_class):
        """Ensure author search produces expected results."""
        mock_authors_instance = Mock()
        mock_authors_class.return_value = mock_authors_instance
        mock_query = Mock()
        mock_authors_instance.search.return_value = mock_query
        mock_query.get.return_value = [{"id": "A1", "name": "Alice"}]

        service = OpenAlexService()
        result = service.search_authors("Alice", per_page=5, page=2)
        self.assertEqual(result, [{"id": "A1", "name": "Alice"}])
        mock_query.get.assert_called_once_with(per_page=5, page=2)

    @patch("Research_AI_Assistant.services.openalex_service.Authors")
    def test_search_authors_error(self, mock_authors_class):
        mock_instance = Mock()
        mock_authors_class.return_value = mock_instance
        mock_query = Mock()
        mock_instance.search.return_value = mock_query
        mock_query.get.side_effect = Exception("timeout")

        service = OpenAlexService()
        with self.assertRaises(OpenAlexAPIError):
            service.search_authors("fail")


# ---------------------------------------------------------------------------
# Shared fixtures for OpenRouter tests
# ---------------------------------------------------------------------------

MOCK_PAPERS = [
    {
        "openalex_id": "https://openalex.org/W1",
        "title": "Deep Learning for NLP",
        "authors": [
            {"name": "Jane Smith", "orcid": "0000-0001", "institutions": ["MIT"]},
            {"name": "John Doe", "orcid": "", "institutions": ["Stanford"]},
        ],
        "abstract": "We study deep learning methods for natural language processing.",
        "publication_year": 2022,
        "doi": "10.1234/dlnlp",
        "cited_by_count": 120,
        "concepts": [
            {"name": "Deep Learning", "score": 0.95},
            {"name": "NLP", "score": 0.88},
        ],
        "source": "Journal of AI Research",
        "is_open_access": True,
        "oa_status": "gold",
        "full_text_url": "https://example.com/paper1.pdf",
    },
    {
        "openalex_id": "https://openalex.org/W2",
        "title": "Transformer Architectures",
        "authors": [
            {"name": "Alice Wang", "orcid": "", "institutions": ["Oxford"]},
        ],
        "abstract": "An overview of transformer architectures in modern AI.",
        "publication_year": 2021,
        "doi": "10.5678/transformers",
        "cited_by_count": 300,
        "concepts": [
            {"name": "Transformers", "score": 0.99},
        ],
        "source": "Nature Machine Intelligence",
        "is_open_access": False,
        "oa_status": None,
        "full_text_url": None,
    },
]

MOCK_SUMMARY = (
    # Must start with "Paper N:" (no preamble) and carry the fixed field
    # labels — extract_final_summary() (prompt_builder.py) strips anything
    # before the first such block, and _passed_format_check
    # (performance_tracker.py) checks for these exact labels.
    "Paper 1: Deep Learning for NLP\n"
    "Authors: Smith, J.; Doe, J.\n"
    "Year: 2022\n"
    "Source: Journal of AI Research\n"
    "DOI: 10.1234/dlnlp\n"
    "Summary: Smith et al. demonstrate strong results for deep learning "
    "methods applied to natural language processing tasks.\n\n"
    "Paper 2: Transformer Architectures\n"
    "Authors: Wang, A.\n"
    "Year: 2021\n"
    "Source: Nature Machine Intelligence\n"
    "DOI: 10.5678/transformers\n"
    "Summary: Wang provides a comprehensive overview of transformer "
    "architectures used across modern AI systems.\n\n"
    "References:\n"
    "[1] Smith, J. and Doe, J. (2022) 'Deep Learning for NLP'. "
    "Journal of AI Research. doi: 10.1234/dlnlp\n"
    "[2] Wang, A. (2021) 'Transformer Architectures'. "
    "Nature Machine Intelligence. doi: 10.5678/transformers"
)


# ---------------------------------------------------------------------------
# OpenRouter service tests
# ---------------------------------------------------------------------------


@override_settings(OPENROUTER_API_KEY="test-key-123")
class TestOpenRouterService(TestCase):
    """Unit tests for OpenRouterService.complete()."""

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_success(self, mock_post):
        """Returns stripped content string on a valid 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-abc123",
            "model": "mistralai/mistral-7b-instruct:free",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"  {MOCK_SUMMARY}  "},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "total_tokens": 180,
            },
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        result = service.complete(
            system_prompt=system_prompt,
            user_message="test user message",
        )

        self.assertEqual(result, MOCK_SUMMARY)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(
            call_args[0][0], "https://openrouter.ai/api/v1/chat/completions"
        )

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_sends_correct_payload(self, mock_post):
        """Verifies model, messages structure, temperature, and max_tokens in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        service.complete(
            system_prompt="sys prompt",
            user_message="user msg",
            temperature=0.5,
            max_tokens=800,
        )

        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(payload["max_tokens"], 800)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "sys prompt")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "user msg")

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_sends_auth_header(self, mock_post):
        """Authorization header must use the configured API key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        service.complete(system_prompt="s", user_message="u")

        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key-123")
        self.assertEqual(headers["Content-Type"], "application/json")

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_empty_choices(self, mock_post):
        """Raises OpenRouterAPIError when choices list is empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_post.return_value = mock_response

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("no choices", str(cm.exception))

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_model_error_in_choice(self, mock_post):
        """Raises OpenRouterAPIError when choices[0].error is present."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "error": {"code": 429, "message": "Rate limit exceeded"},
                    "finish_reason": "error",
                }
            ]
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("model error", str(cm.exception).lower())

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_missing_content(self, mock_post):
        """Raises OpenRouterAPIError when message.content is None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}]
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("message.content", str(cm.exception))

    @override_settings(OPENROUTER_MODELS=["test/model:free"])
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_http_error(self, mock_post):
        """
        Raises OpenRouterAPIError on non-2xx HTTP status. complete() retries
        other free models before giving up, so OPENROUTER_MODELS pins the
        chain to one model to keep this test about a single failure, not the
        fallback chain. Real `requests` always puts the status text in
        raise_for_status's message, so the mock does too.
        """
        import requests as req_lib

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
            "401 Client Error: Unauthorized", response=mock_response
        )
        mock_post.return_value = mock_response

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("401", str(cm.exception))

    @override_settings(OPENROUTER_MODELS=["test/model:free"])
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_timeout(self, mock_post):
        """Raises OpenRouterAPIError with a timeout message on request timeout."""
        import requests as req_lib

        mock_post.side_effect = req_lib.exceptions.Timeout("Read timed out.")

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("timed out", str(cm.exception))

    @override_settings(OPENROUTER_MODELS=["test/model:free"])
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_raises_on_non_json_response(self, mock_post):
        """Raises OpenRouterAPIError when response body is not valid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("no json")
        mock_post.return_value = mock_response

        service = OpenRouterService()
        with self.assertRaises(OpenRouterAPIError) as cm:
            service.complete(system_prompt="s", user_message="u")

        self.assertIn("no json", str(cm.exception))

    def test_init_raises_without_api_key(self):
        """Raises OpenRouterAPIError at init time when API key is missing."""
        with self.settings(OPENROUTER_API_KEY=""):
            with self.assertRaises(OpenRouterAPIError) as cm:
                OpenRouterService()

            self.assertIn("OPENROUTER_API_KEY", str(cm.exception))

    @override_settings(
        OPENROUTER_API_KEY="key", OPENROUTER_MODEL="google/gemma-7b:free"
    )
    def test_init_uses_custom_model_from_settings(self):
        """OPENROUTER_MODEL setting overrides the default model."""
        service = OpenRouterService()
        self.assertEqual(service.model, "google/gemma-7b:free")


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder(TestCase):
    """Unit tests for build_user_message() and SYSTEM_PROMPT."""

    def test_system_prompt_contains_citation_instruction(self):
        """SYSTEM_PROMPT must instruct the model to use [N] inline markers."""
        self.assertIn("[N]", system_prompt)
        self.assertIn("Harvard", system_prompt)

    def test_system_prompt_forbids_fabrication(self):
        """SYSTEM_PROMPT must explicitly prohibit fabrication."""
        self.assertIn("fabricat", system_prompt.lower())

    def test_system_prompt_requires_references_section(self):
        """SYSTEM_PROMPT must instruct the model to output a References section."""
        self.assertIn("References", system_prompt)

    def test_build_user_message_numbers_papers(self):
        """Each paper must be numbered in the user message."""
        msg = build_user_message(MOCK_PAPERS, "deep learning")
        self.assertIn("Paper 1:", msg)
        self.assertIn("Paper 2:", msg)

    def test_build_user_message_includes_query(self):
        """The original search query must appear in the user message."""
        msg = build_user_message(MOCK_PAPERS, "deep learning NLP")
        self.assertIn("deep learning NLP", msg)

    def test_build_user_message_includes_title(self):
        """Each paper's title must appear in the user message."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("Deep Learning for NLP", msg)
        self.assertIn("Transformer Architectures", msg)

    def test_build_user_message_includes_doi(self):
        """Each paper's DOI must appear in the user message."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("10.1234/dlnlp", msg)
        self.assertIn("10.5678/transformers", msg)

    def test_build_user_message_includes_abstract(self):
        """Each paper's abstract must appear in the user message."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("natural language processing", msg)

    def test_build_user_message_includes_authors(self):
        """Author names must appear in the user message."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("Jane Smith", msg)
        self.assertIn("Alice Wang", msg)

    def test_build_user_message_includes_year(self):
        """Publication year must appear for each paper."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("2022", msg)
        self.assertIn("2021", msg)

    def test_build_user_message_missing_abstract_shows_na(self):
        """Papers with no abstract must say so explicitly, not show an empty line."""
        papers = [
            {
                "title": "No Abstract Paper",
                "authors": [],
                "abstract": "",
                "publication_year": 2020,
                "doi": "",
                "cited_by_count": 0,
                "concepts": [],
                "source": None,
                "is_open_access": False,
            }
        ]
        msg = build_user_message(papers, "test")
        self.assertIn("Abstract: Abstract not available", msg)

    def test_build_user_message_empty_papers(self):
        """Empty paper list returns a no-results message, not an error."""
        msg = build_user_message([], "test query")
        self.assertIn("No papers", msg)

    def test_build_user_message_includes_synthesis_instruction(self):
        """User message must instruct the model to summarise each paper."""
        msg = build_user_message(MOCK_PAPERS, "q")
        self.assertIn("summarize each paper", msg.lower())


@override_settings(OPENROUTER_API_KEY="test-key-123")
class TestSummariseView(ClerkAuthTestMixin, TestCase):
    """
    Integration tests for POST /api/summarise/.

    summarise requires auth (fixed in the ponytail-audit pass: it had
    @throttle_classes but no @api_view, so DRF — and therefore auth — never
    ran); ClerkAuthTestMixin bypasses real Clerk verification so these tests
    exercise the view's own logic rather than the auth layer.
    """

    def setUp(self):
        super().setUp()
        self.client = Client()

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_success(self, mock_service_class):
        """
        Returns 200 with the summary on valid input. The response body is
        just {"summary": ...} — it does not echo back paper_count or query.
        """
        mock_instance = MagicMock()
        mock_instance.complete.return_value = MOCK_SUMMARY
        mock_service_class.return_value = mock_instance

        payload = {"query": "deep learning", "papers": MOCK_PAPERS}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["summary"], MOCK_SUMMARY)

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_calls_complete_with_correct_args(self, mock_service_class):
        """View must pass SYSTEM_PROMPT and a user message built from the papers."""
        mock_instance = MagicMock()
        mock_instance.complete.return_value = MOCK_SUMMARY
        mock_service_class.return_value = mock_instance

        payload = {"query": "transformers", "papers": MOCK_PAPERS}
        self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )

        mock_instance.complete.assert_called_once()
        call_kwargs = mock_instance.complete.call_args[1]
        self.assertEqual(call_kwargs["system_prompt"], system_prompt)
        self.assertIn("transformers", call_kwargs["user_message"])

    def test_summarise_missing_query_returns_400(self):
        """Missing 'query' field must return 400."""
        payload = {"papers": MOCK_PAPERS}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("query", resp.json()["error"].lower())

    def test_summarise_missing_papers_returns_400(self):
        """Missing 'papers' field must return 400."""
        payload = {"query": "deep learning"}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("papers", resp.json()["error"].lower())

    def test_summarise_empty_papers_list_returns_400(self):
        """Empty 'papers' list must return 400."""
        payload = {"query": "deep learning", "papers": []}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_summarise_invalid_json_body_returns_400(self):
        """Malformed JSON body must return 400."""
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data="not json at all",
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("JSON", resp.json()["error"])

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_openrouter_api_error_returns_503(self, mock_service_class):
        """
        OpenRouterAPIError from complete() must return 503 (upstream
        unavailable). The exception's own message is deliberately not
        echoed to the client — only a fixed, generic message is — so this
        checks error_code rather than exception text.
        """
        mock_instance = MagicMock()
        mock_instance.complete.side_effect = OpenRouterAPIError("upstream failure")
        mock_service_class.return_value = mock_instance

        payload = {"query": "deep learning", "papers": MOCK_PAPERS}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["error_code"], "LLM_API_ERROR")

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_missing_api_key_returns_503(self, mock_service_class):
        """OpenRouterAPIError raised at service init (no key) must return 503."""
        mock_service_class.side_effect = OpenRouterAPIError(
            "OPENROUTER_API_KEY is not set"
        )

        payload = {"query": "deep learning", "papers": MOCK_PAPERS}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 503)

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_passes_all_papers_under_the_cap(self, mock_service_class):
        """
        Every paper under the cap (views.MAX_SUMMARY_PAPERS) is passed to
        build_user_message; none are silently dropped. The cap itself is
        covered by TestLLMInputCaps.
        """
        mock_instance = MagicMock()
        mock_instance.complete.return_value = MOCK_SUMMARY
        mock_service_class.return_value = mock_instance

        many_papers = [MOCK_PAPERS[0]] * 25
        payload = {"query": "q", "papers": many_papers}
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(resp.status_code, 200)
        call_kwargs = mock_instance.complete.call_args_list[0][1]
        self.assertEqual(call_kwargs["user_message"].count("Paper "), 25)

    def test_summarise_get_request_returns_405(self):
        """GET on /api/summarise/ must return 405 Method Not Allowed."""
        resp = self.client.get(reverse("research_ai_assistant:summarise"))
        self.assertEqual(resp.status_code, 405)


@override_settings(OPENROUTER_API_KEY="test-key-123")
class TestFetchFreeModels(TestCase):
    """
    fetch_free_models(): filters ':free' ids, excludes non-chat models using
    OpenRouter's own supported_parameters/output_modalities fields (no
    hand-maintained list), caches, and raises (no built-in fallback list)
    when the API call fails.
    """

    def setUp(self):
        cache.clear()

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_keeps_only_free_models(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: {
                "data": [
                    {"id": "a/model-one:free"},
                    {"id": "b/paid-model"},
                    {"id": "c/model-two:free"},
                ]
            }
        )
        models = fetch_free_models()
        self.assertIn("a/model-one:free", models)
        self.assertIn("c/model-two:free", models)
        self.assertNotIn("b/paid-model", models)
        headers = mock_get.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key-123")

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_result_is_cached(self, mock_get):
        mock_get.return_value = MagicMock(json=lambda: {"data": [{"id": "a/x:free"}]})
        fetch_free_models()
        fetch_free_models()
        self.assertEqual(mock_get.call_count, 1)

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_raises_when_api_unreachable(self, mock_get):
        """No FREE_MODELS fallback any more: a discovery failure must raise,
        not silently serve a list that can itself be stale."""
        import requests as req_lib

        mock_get.side_effect = req_lib.exceptions.ConnectionError("down")
        with self.assertRaises(OpenRouterAPIError):
            fetch_free_models()

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_excludes_model_with_no_generation_parameters(self, mock_get):
        """A model whose supported_parameters is present but has neither
        max_tokens nor temperature (typical of a classifier/embedder) is
        excluded, with no keyword list involved."""
        mock_get.return_value = MagicMock(
            json=lambda: {
                "data": [
                    {"id": "a/chat-model:free", "supported_parameters": ["temperature", "max_tokens"]},
                    {"id": "b/classifier:free", "supported_parameters": ["top_k"]},
                ]
            }
        )
        models = fetch_free_models()
        self.assertIn("a/chat-model:free", models)
        self.assertNotIn("b/classifier:free", models)

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_keeps_model_with_missing_supported_parameters(self, mock_get):
        """supported_parameters absent entirely: benefit of the doubt, kept."""
        mock_get.return_value = MagicMock(
            json=lambda: {"data": [{"id": "a/unknown-shape:free"}]}
        )
        models = fetch_free_models()
        self.assertIn("a/unknown-shape:free", models)

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_excludes_known_guardrail_model_despite_chat_shaped_parameters(
        self, mock_get
    ):
        """
        Regression test for a real case found by running initialize_models
        against the live API: nvidia/nemotron-3.5-content-safety:free
        reports supported_parameters=[max_tokens, temperature, ...] (so the
        generic filter alone would keep it) but is a moderation/guardrail
        model, not a chat model. KNOWN_NON_CHAT_MODEL_IDS excludes it by
        exact id.
        """
        mock_get.return_value = MagicMock(
            json=lambda: {
                "data": [
                    {
                        "id": "nvidia/nemotron-3.5-content-safety:free",
                        "supported_parameters": ["max_tokens", "temperature"],
                    },
                    {"id": "b/chat-model:free"},
                ]
            }
        )
        models = fetch_free_models()
        self.assertNotIn("nvidia/nemotron-3.5-content-safety:free", models)
        self.assertIn("b/chat-model:free", models)

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_excludes_model_with_non_text_output(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: {
                "data": [
                    {
                        "id": "a/image-model:free",
                        "architecture": {"output_modalities": ["image"]},
                    },
                    {"id": "b/chat-model:free"},
                ]
            }
        )
        models = fetch_free_models()
        self.assertNotIn("a/image-model:free", models)
        self.assertIn("b/chat-model:free", models)

    @patch("Research_AI_Assistant.services.openrouter_service.requests.get")
    def test_records_reasoning_models(self, mock_get):
        """A `reasoning` object or a reasoning entry in supported_parameters
        marks the model as reasoning; a model with neither is not."""
        mock_get.return_value = MagicMock(
            json=lambda: {
                "data": [
                    {"id": "a/thinker:free", "reasoning": {"mandatory": True}},
                    {
                        "id": "b/hybrid:free",
                        "supported_parameters": ["max_tokens", "reasoning"],
                    },
                    {"id": "c/plain:free", "supported_parameters": ["max_tokens"]},
                ]
            }
        )
        fetch_free_models()
        self.assertEqual(
            reasoning_model_ids(), {"a/thinker:free", "b/hybrid:free"}
        )


def _empty_choices_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"choices": []}
    return response


@override_settings(
    OPENROUTER_API_KEY="test-key-123",
    OPENROUTER_MODELS=[
        "r/one:free", "n/one:free", "r/two:free",
        "n/two:free", "n/three:free", "n/four:free",
    ],
    OPENROUTER_QA_MAX_MODELS=4,
    OPENROUTER_QA_DEADLINE_SECONDS=90,
    OPENROUTER_TIMEOUT_SECONDS=60,
)
@patch(
    "Research_AI_Assistant.services.openrouter_service.PerformanceTracker."
    "get_intelligent_model_order",
    side_effect=lambda models: list(models),
)
class TestOpenRouterQALimits(TestCase):
    """Q&A tries non-reasoning models first, caps the attempts, and stops
    walking the chain at a total deadline."""

    def setUp(self):
        cache.clear()
        cache.set(REASONING_MODELS_CACHE_KEY, ["r/one:free", "r/two:free"])

    def _posted_models(self, mock_post):
        return [c.kwargs["json"]["model"] for c in mock_post.call_args_list]

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_qa_tries_non_reasoning_first_and_caps_attempts(self, mock_post, _order):
        mock_post.return_value = _empty_choices_response()
        with self.assertRaises(OpenRouterAPIError):
            OpenRouterService().complete("s", "u", request_type="qa")
        self.assertEqual(
            self._posted_models(mock_post),
            ["n/one:free", "n/two:free", "n/three:free", "n/four:free"],
        )

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_non_qa_order_and_cap_unchanged(self, mock_post, _order):
        mock_post.return_value = _empty_choices_response()
        with self.assertRaises(OpenRouterAPIError):
            OpenRouterService().complete("s", "u", request_type="summary")
        self.assertEqual(
            self._posted_models(mock_post),
            ["r/one:free", "n/one:free", "r/two:free",
             "n/two:free", "n/three:free", "n/four:free"],
        )

    @patch("Research_AI_Assistant.services.openrouter_service.time")
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_qa_stops_at_deadline(self, mock_post, mock_time, _order):
        # Deadline set at t=0 (so 90); attempt 1 starts at t=0; by attempt 2
        # it is t=88, leaving less than MIN_ATTEMPT_SECONDS.
        mock_time.monotonic.side_effect = [0, 0, 88]
        mock_post.return_value = _empty_choices_response()
        with self.assertRaises(OpenRouterAPIError) as cm:
            OpenRouterService().complete("s", "u", request_type="qa")
        self.assertEqual(mock_post.call_count, 1)
        self.assertIn("time limit", str(cm.exception))
        # The single attempt's HTTP timeout was capped by the time left.
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 60)

    @patch("Research_AI_Assistant.services.openrouter_service.time")
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_qa_stream_stops_mid_stream_at_deadline(self, mock_post, mock_time, _order):
        chunk = 'data: {"choices":[{"delta":{"content":"tok"}}]}'
        response = MagicMock()
        response.iter_lines.return_value = iter([chunk, chunk, chunk])
        mock_post.return_value = response
        # deadline=90; attempt start t=0; first line t=10; second line t=95.
        mock_time.monotonic.side_effect = [0, 0, 10, 95]

        tokens = []
        with self.assertRaises(OpenRouterAPIError) as cm:
            for token in OpenRouterService().stream_complete(
                "s", "u", request_type="qa"
            ):
                tokens.append(token)
        self.assertEqual(tokens, ["tok"])
        self.assertEqual(mock_post.call_count, 1)
        response.close.assert_called_once()
        self.assertIn("time limit", str(cm.exception))


def _sse_response(*contents):
    """A streamed chat-completion response whose deltas are `contents`."""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
        for c in contents
    ] + ["data: [DONE]"]
    response = MagicMock()
    response.iter_lines.return_value = iter(lines)
    return response


def _reject_unless_valid_json(content):
    json.loads(content)


@override_settings(
    OPENROUTER_API_KEY="test-key-123",
    OPENROUTER_MODELS=["a/first:free", "b/second:free"],
)
@patch(
    "Research_AI_Assistant.services.openrouter_service.PerformanceTracker."
    "get_intelligent_model_order",
    side_effect=lambda models: list(models),
)
class TestOpenRouterStreamReset(TestCase):
    """When stream_complete falls through to another model after tokens
    were already yielded, it yields STREAM_RESET before the next model's."""

    def _stream(self):
        return list(OpenRouterService().stream_complete(
            "s", "u", request_type="qa",
            response_validator=_reject_unless_valid_json,
        ))

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_reset_between_a_failed_partial_stream_and_the_next_model(
        self, mock_post, _order
    ):
        mock_post.side_effect = [
            _sse_response('{"answer": "cut off'),
            _sse_response('{"answer": "ok"}'),
        ]
        self.assertEqual(
            self._stream(), ['{"answer": "cut off', STREAM_RESET, '{"answer": "ok"}']
        )

    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_no_reset_when_the_failed_model_sent_nothing(self, mock_post, _order):
        import requests as req_lib

        failed = MagicMock()
        failed.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
            "429 Too Many Requests", response=MagicMock(status_code=429)
        )
        mock_post.side_effect = [failed, _sse_response('{"answer": "ok"}')]
        self.assertEqual(self._stream(), ['{"answer": "ok"}'])


class TestSummaryCompleteness(TestCase):
    """A summary cut off before its References list must not be accepted,
    and the token budget grows with the number of papers."""

    COMPLETE = (
        "Paper 1: Title\nAuthors: Smith, J.\nYear: 2022\nSummary: A result.\n"
        "References:\nSmith, J. (2022) Title. Journal."
    )
    CUT_OFF = "Paper 1: Title\nAuthors: Smith, J.\nYear: 2022\nSummary: A res"

    PAPERS = [{
        "title": "Title", "publication_year": 2022, "source": "Journal",
        "doi": "10.1/x", "authors": [{"name": "Jane Smith"}],
    }]

    def test_complete_summary_is_accepted(self):
        require_all_papers(1)(self.COMPLETE)

    def test_summary_missing_a_paper_is_rejected(self):
        with self.assertRaises(InvalidSummaryFormat):
            require_all_papers(2)(self.COMPLETE)

    def test_missing_references_are_built_from_metadata(self):
        no_refs = self.COMPLETE.split("References:")[0].strip()
        out = ensure_references(no_refs, self.PAPERS)
        self.assertIn(
            "References:\n[1] Smith, J. (2022) 'Title'. Journal. doi: 10.1/x.", out
        )

    def test_existing_references_are_left_alone(self):
        self.assertEqual(ensure_references(self.COMPLETE, self.PAPERS), self.COMPLETE)

    def test_budget_grows_with_papers_and_is_capped(self):
        self.assertLess(summary_max_tokens(1), summary_max_tokens(5))
        self.assertEqual(summary_max_tokens(50), SUMMARY_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------

# A format-compliant summary body, so log_request_success scores a pass
# against the required prompt_builder labels.
COMPLIANT_SUMMARY = (
    "Authors: Smith, J.\nYear: 2022\nSource: Journal of AI\n"
    "DOI: 10.1234/x\nSummary: A result.\nReferences: Smith (2022)."
)


class TestCircuitBreaker(TestCase):
    """
    The circuit breaker must actually influence model selection:
    disabled models sort last, and recovered models are retried.
    """

    def _reliability(self, model_name, tier, priority=50, threshold=5):
        return ModelReliability.objects.create(
            model_name=model_name,
            tier=tier,
            priority=priority,
            circuit_breaker_threshold=threshold,
        )

    def test_inactive_model_sorts_after_active(self):
        """A tripped model loses to an active model with a lower tier score."""
        self._reliability("good:free", "emergency")
        self._reliability("bad:free", "primary")
        ModelPerformance.objects.create(
            model_name="bad:free",
            is_active=False,
            consecutive_failures=5,
            last_failure=timezone.now(),
        )

        order = PerformanceTracker.get_intelligent_model_order(
            ["good:free", "bad:free"]
        )

        # bad:free is 'primary' (base 1000) against 'emergency' (base 100),
        # so only the is_active partition can put it last.
        self.assertIn("bad:free", order, "inactive models must not be dropped")
        self.assertLess(order.index("good:free"), order.index("bad:free"))

    def test_reset_reenables_model_after_timeout(self):
        """A model idle-disabled for longer than the reset window comes back."""
        self._reliability("good:free", "emergency")
        self._reliability("bad:free", "primary")
        ModelPerformance.objects.create(
            model_name="bad:free",
            is_active=False,
            consecutive_failures=5,
            last_failure=timezone.now()
            - timedelta(minutes=CIRCUIT_BREAKER_RESET_MIN + 5),
        )

        PerformanceTracker.reset_circuit_breaker()

        perf = ModelPerformance.objects.get(model_name="bad:free")
        self.assertTrue(perf.is_active)
        self.assertEqual(perf.consecutive_failures, 0)

        # Re-enabled, its 'primary' tier score ranks it first again.
        order = PerformanceTracker.get_intelligent_model_order(
            ["good:free", "bad:free"]
        )
        self.assertLess(order.index("bad:free"), order.index("good:free"))

    def test_reset_keeps_recent_failure_disabled(self):
        """A model that failed a few minutes ago stays disabled and last."""
        self._reliability("good:free", "emergency")
        self._reliability("bad:free", "primary")
        ModelPerformance.objects.create(
            model_name="bad:free",
            is_active=False,
            consecutive_failures=5,
            last_failure=timezone.now() - timedelta(minutes=5),
        )

        PerformanceTracker.reset_circuit_breaker()

        perf = ModelPerformance.objects.get(model_name="bad:free")
        self.assertFalse(perf.is_active)
        self.assertEqual(perf.consecutive_failures, 5)

        order = PerformanceTracker.get_intelligent_model_order(
            ["good:free", "bad:free"]
        )
        self.assertEqual(order[-1], "bad:free")

    def test_reset_does_not_reenable_manually_deactivated_model(self):
        """A model an admin switched off (0 failures) must stay off."""
        self._reliability("bad:free", "primary")
        ModelPerformance.objects.create(
            model_name="bad:free",
            is_active=False,
            consecutive_failures=0,
            last_failure=timezone.now()
            - timedelta(minutes=CIRCUIT_BREAKER_RESET_MIN + 5),
        )

        PerformanceTracker.reset_circuit_breaker()

        perf = ModelPerformance.objects.get(model_name="bad:free")
        self.assertFalse(perf.is_active)

    def test_failure_trips_breaker_at_threshold_and_success_resets(self):
        """Threshold of 2: trips on the second failure, clears on success."""
        self._reliability("flaky:free", "primary", threshold=2)
        # log_request_failure only evaluates the threshold on an existing row.
        ModelPerformance.objects.create(model_name="flaky:free")

        PerformanceTracker.log_request_failure("flaky:free", "summary", 1.0, "boom")
        perf = ModelPerformance.objects.get(model_name="flaky:free")
        self.assertEqual(perf.consecutive_failures, 1)
        self.assertTrue(perf.is_active, "one failure must not trip the breaker")

        PerformanceTracker.log_request_failure(
            "flaky:free", "summary", 1.0, "boom again"
        )
        perf.refresh_from_db()
        self.assertEqual(perf.consecutive_failures, 2)
        self.assertFalse(perf.is_active)

        PerformanceTracker.log_request_success(
            "flaky:free", "summary", 1.0, COMPLIANT_SUMMARY
        )
        perf.refresh_from_db()
        self.assertTrue(perf.is_active)
        self.assertEqual(perf.consecutive_failures, 0)

    @override_settings(
        OPENROUTER_API_KEY="test-key-123", OPENROUTER_MODELS=["m/one:free"]
    )
    @patch("Research_AI_Assistant.services.openrouter_service.requests.post")
    def test_complete_retries_recovered_model_in_the_same_call(self, mock_post):
        """A model disabled long enough ago to clear the reset window must be
        usable again within the very call that triggers the reset, not just
        on a later request."""
        ModelReliability.objects.create(model_name="m/one:free", tier="primary")
        ModelPerformance.objects.create(
            model_name="m/one:free",
            is_active=False,
            consecutive_failures=5,
            last_failure=timezone.now()
            - timedelta(minutes=CIRCUIT_BREAKER_RESET_MIN + 5),
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        mock_post.return_value = mock_response

        service = OpenRouterService()
        result = service.complete(system_prompt="s", user_message="u")

        self.assertEqual(result, "ok")
        mock_post.assert_called_once()  # recovered model was actually tried, not skipped
        perf = ModelPerformance.objects.get(model_name="m/one:free")
        self.assertTrue(perf.is_active)


# ---------------------------------------------------------------------------
# Q&A chunk cache tests
# ---------------------------------------------------------------------------


class TestQAChunkCache(TestCase):
    """A cache hit must only be served when its source URL matches this
    request's candidates — paper_id alone is client-supplied and not proof
    the cached text belongs to it."""

    def test_cache_hit_served_when_source_matches(self):
        PaperQAChunks.objects.create(
            openalex_id="W1",
            qa_chunks=[{"paper_id": "W1", "page": 1, "text": "hello"}],
            qa_chunk_version=CHUNK_SCHEMA_VERSION,
            qa_source_pdf_url="https://good.test/a.pdf",
        )

        with patch.object(QAChunkCache, "_record_success") as mock_record:
            result = QAChunkCache.get_or_chunk("https://good.test/a.pdf", "W1")

        self.assertEqual(result["chunks"], [{"paper_id": "W1", "page": 1, "text": "hello"}])
        mock_record.assert_not_called()  # served from cache, no re-fetch

    def test_cache_hit_rejected_when_source_does_not_match(self):
        """A cached row built from a different URL must not be trusted for
        a request that never sent that URL for this paper_id."""
        PaperQAChunks.objects.create(
            openalex_id="W1",
            qa_chunks=[{"paper_id": "W1", "page": 1, "text": "poisoned"}],
            qa_chunk_version=CHUNK_SCHEMA_VERSION,
            qa_source_pdf_url="https://attacker.test/fake.pdf",
        )

        with patch(
            "Research_AI_Assistant.services.qa_chunk_cache.QAChunkService.fetch_and_chunk",
            return_value={
                "chunks": [{"paper_id": "W1", "page": 1, "text": "real"}],
                "page_count": 1,
            },
        ) as mock_fetch:
            result = QAChunkCache.get_or_chunk("https://real.test/paper.pdf", "W1")

        mock_fetch.assert_called_once()
        self.assertEqual(result["chunks"], [{"paper_id": "W1", "page": 1, "text": "real"}])


# ---------------------------------------------------------------------------
# Multi-paper Q&A auth tests
# ---------------------------------------------------------------------------


class TestMultiPaperQAAuth(TestCase):
    """multi_paper_qa must require a Clerk token (it spends LLM credit and
    fetches client-supplied URLs). Real ClerkAuthentication runs here; only
    the token check is replaced, so these tests exercise the auth layer."""

    def _post(self):
        return self.client.post(
            reverse("research_ai_assistant:multi_paper_qa"),
            data=json.dumps({"paper_ids": ["W1"], "question": "q", "pdf_urls": {"W1": "https://example.test/a.pdf"}}),
            content_type="application/json",
        )

    def test_unauthenticated_request_is_rejected(self):
        """No Authorization header: 401, before any PDF fetch or LLM call."""
        with patch(
            "Research_AI_Assistant.services.qa_chunk_cache.QAChunkCache.get_or_chunk"
        ) as mock_chunk:
            resp = self._post()
        self.assertEqual(resp.status_code, 401)
        mock_chunk.assert_not_called()

    @patch("Research_AI_Assistant.authentication.ClerkAuthentication.authenticate")
    def test_authenticated_request_reaches_view(self, mock_auth):
        """A verified token gets past auth; the view then runs (here it fails
        on the chunk step, which is what we want to show was reached)."""
        mock_auth.return_value = (User.objects.create(clerk_id="qa-user"), None)
        with patch(
            "Research_AI_Assistant.views.QAChunkCache.get_or_chunk",
            side_effect=Exception("reached view"),
        ) as mock_chunk:
            self._post()
        mock_chunk.assert_called_once()


class TestMultiPaperQALLMError(ClerkAuthTestMixin, TestCase):
    """An LLMError from get_llm() (bad LLM_BACKEND, missing Ollama config,
    etc.) must return 503 like summarise/generate_title do, not the generic
    500 from the catch-all Exception handler."""

    def _post(self):
        return self.client.post(
            reverse("research_ai_assistant:multi_paper_qa"),
            data=json.dumps(
                {
                    "paper_ids": ["W1"],
                    "question": "q",
                    "pdf_urls": {"W1": "https://example.test/a.pdf"},
                }
            ),
            content_type="application/json",
        )

    @patch("Research_AI_Assistant.views.get_llm")
    @patch("Research_AI_Assistant.views.QAChunkCache.get_or_chunk")
    def test_llm_error_returns_503_not_500(self, mock_chunk, mock_get_llm):
        mock_chunk.return_value = {
            "chunks": [{"paper_id": "W1", "page": 1, "text": "hello", "token_estimate": 5}]
        }
        mock_get_llm.side_effect = LLMError("Invalid LLM_BACKEND 'bogus'.")
        resp = self._post()
        self.assertEqual(resp.status_code, 503)


# ---------------------------------------------------------------------------
# Rate limiting for signed-in users
# ---------------------------------------------------------------------------


@override_settings(RESEARCH_API_URL="", RESEARCH_API_KEY="")
class TestSignedInUserThrottling(ClerkAuthTestMixin, TestCase):
    """Every API view except api_root requires sign-in, so the per-view rate
    limits must count signed-in users. (They used to subclass
    AnonRateThrottle, which DRF skips for any authenticated request.)"""

    def _title(self):
        return self.client.post(
            reverse("research_ai_assistant:generate_title"),
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            content_type="application/json",
        )

    def _mock_llm(self):
        patcher = patch("Research_AI_Assistant.views.get_llm")
        mock_get_llm = patcher.start()
        self.addCleanup(patcher.stop)
        mock_get_llm.return_value.complete.return_value = "A Research Title"
        return mock_get_llm.return_value

    def test_generate_title_is_limited_to_20_per_minute(self):
        self._mock_llm()
        statuses = [self._title().status_code for _ in range(21)]
        self.assertEqual(statuses, [200] * 20 + [429])

    def test_summarise_shares_the_llm_limit_with_generate_title(self):
        self._mock_llm()
        for _ in range(20):
            self._title()
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps({"query": "q", "papers": MOCK_PAPERS}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 429)

    def test_limit_is_counted_per_user(self):
        self._mock_llm()
        for _ in range(21):
            self._title()
        other_user = User.objects.create(clerk_id="other-clerk-id")
        with patch(
            "Research_AI_Assistant.authentication.ClerkAuthentication.authenticate",
            return_value=(other_user, None),
        ):
            self.assertEqual(self._title().status_code, 200)

    def test_multi_paper_qa_is_limited_to_10_per_minute(self):
        from .services.qa_pipeline import QAChunkError

        body = json.dumps(
            {
                "paper_ids": ["W1"],
                "question": "q",
                "pdf_urls": {"W1": "https://example.test/a.pdf"},
            }
        )
        with patch(
            "Research_AI_Assistant.views.QAChunkCache.get_or_chunk",
            side_effect=QAChunkError("blocked"),
        ):
            statuses = [
                self.client.post(
                    reverse("research_ai_assistant:multi_paper_qa"),
                    data=body,
                    content_type="application/json",
                ).status_code
                for _ in range(11)
            ]
        self.assertEqual(statuses, [422] * 10 + [429])

    @patch("Research_AI_Assistant.views.openalex_service.search_papers")
    def test_search_is_limited_to_60_per_minute(self, mock_search):
        mock_search.return_value = {"results": [], "meta": {}}
        url = reverse("research_ai_assistant:search")
        statuses = [self.client.get(url, {"q": "x"}).status_code for _ in range(61)]
        self.assertEqual(statuses, [200] * 60 + [429])

    def test_performance_api_is_limited_to_60_per_minute(self):
        self._test_user.is_staff = True
        self._test_user.save()
        url = reverse("research_ai_assistant:performance_stats")
        statuses = [self.client.get(url).status_code for _ in range(61)]
        self.assertEqual(statuses, [200] * 60 + [429])

    def test_per_view_counters_are_shared_between_worker_processes(self):
        """A per-process cache would multiply every limit by the number of
        gunicorn workers, so the counters live in the database."""
        from django.core.cache.backends.db import DatabaseCache

        from . import views

        for throttle in (
            views.SearchRateThrottle,
            views.GenerateTitle,
            views.MultiPaperQAThrottle,
        ):
            self.assertIsInstance(throttle.cache, DatabaseCache, throttle.__name__)


class TestLLMInputCaps(ClerkAuthTestMixin, TestCase):
    """summarise and generate_title forward client text to the LLM, so the
    amount they forward is bounded."""

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_rejects_more_than_50_papers(self, mock_get_llm):
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps({"query": "q", "papers": [MOCK_PAPERS[0]] * 51}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("50", resp.json()["error"])
        mock_get_llm.return_value.complete.assert_not_called()

    @patch("Research_AI_Assistant.views.get_llm")
    def test_summarise_accepts_50_papers(self, mock_get_llm):
        mock_get_llm.return_value.complete.return_value = MOCK_SUMMARY
        resp = self.client.post(
            reverse("research_ai_assistant:summarise"),
            data=json.dumps({"query": "q", "papers": [MOCK_PAPERS[0]] * 50}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

    @patch("Research_AI_Assistant.views.get_llm")
    def test_generate_title_truncates_long_messages(self, mock_get_llm):
        """Binder messages hold whole summaries; a title only needs the gist,
        so each message is cut to 2000 characters rather than rejected."""
        mock_get_llm.return_value.complete.return_value = "A Research Title"
        messages = [{"role": "user", "content": "§" * 10_000}] * 5
        resp = self.client.post(
            reverse("research_ai_assistant:generate_title"),
            data=json.dumps({"messages": messages}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        sent = mock_get_llm.return_value.complete.call_args[1]["user_message"]
        self.assertEqual(sent.count("§"), 5 * 2000)


class TestKGModels(TestCase):
    def test_node_key_is_unique(self):
        KGNode.objects.create(node_type="paper", key="W1", label="A")
        with self.assertRaises(IntegrityError), transaction.atomic():
            KGNode.objects.create(node_type="paper", key="W1", label="B")

    def test_edge_is_unique_per_type(self):
        a = KGNode.objects.create(node_type="paper", key="W1")
        b = KGNode.objects.create(node_type="paper", key="W2")
        KGEdge.objects.create(source=a, target=b, edge_type="CITES")
        with self.assertRaises(IntegrityError), transaction.atomic():
            KGEdge.objects.create(source=a, target=b, edge_type="CITES")

    def test_edge_defaults(self):
        a = KGNode.objects.create(node_type="paper", key="W1")
        b = KGNode.objects.create(node_type="author", key="A1")
        e = KGEdge.objects.create(source=a, target=b, edge_type="AUTHORED_BY")
        self.assertEqual((e.weight, e.provenance, e.properties), (1.0, "openalex", {}))


class TestOpenAlexGraphQueries(TestCase):
    """Bulk-by-ID, citing-works and by-DOI lookups used by the knowledge graph."""

    @staticmethod
    def _chain(mock_works, results=None):
        """Works() returns a query whose filter/filter_or/sort/select return itself."""
        query = Mock()
        for name in ("filter", "filter_or", "sort", "select"):
            getattr(query, name).return_value = query
        query.get.return_value = results if results is not None else []
        mock_works.return_value = query
        return query

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_works_by_ids_batches_by_50(self, mock_works):
        query = self._chain(mock_works)
        ids = [f"W{i}" for i in range(120)]
        OpenAlexService().get_works_by_ids(ids)
        self.assertEqual(query.get.call_count, 3)
        sizes = [c.kwargs["openalex_id"] for c in query.filter_or.call_args_list]
        self.assertEqual([len(s) for s in sizes], [50, 50, 20])
        self.assertEqual(
            [c.kwargs["per_page"] for c in query.get.call_args_list], [50, 50, 20]
        )

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_works_by_ids_empty_makes_no_call(self, mock_works):
        self.assertEqual(OpenAlexService().get_works_by_ids([]), [])
        mock_works.assert_not_called()

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_works_by_ids_fields_override(self, mock_works):
        query = self._chain(mock_works, [{"id": "W1"}])
        result = OpenAlexService().get_works_by_ids(
            ["W1"], fields=["id", "related_works"]
        )
        query.select.assert_called_once_with(["id", "related_works"])
        self.assertEqual(result, [{"id": "W1"}])

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_citing_works_uses_cites_filter(self, mock_works):
        query = self._chain(mock_works, [{"id": "W9"}])
        result = OpenAlexService().get_citing_works(["W1", "W2"], limit=5)
        query.filter.assert_called_once_with(cites="W1|W2")
        query.sort.assert_called_once_with(cited_by_count="desc")
        query.get.assert_called_once_with(per_page=5)
        self.assertEqual(result, [{"id": "W9"}])

    @patch("Research_AI_Assistant.services.openalex_service.time.sleep")
    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_citing_works_wraps_errors(self, mock_works, mock_sleep):
        import requests as req_lib

        query = self._chain(mock_works)
        response = Mock(status_code=503)
        query.get.side_effect = req_lib.exceptions.HTTPError(response=response)
        with self.assertRaises(OpenAlexAPIError):
            OpenAlexService().get_citing_works(["W1"])
        self.assertEqual(query.get.call_count, 3)

    @patch("Research_AI_Assistant.services.openalex_service.Works")
    def test_get_works_by_dois_normalises(self, mock_works):
        query = self._chain(mock_works, [{"id": "W1"}])
        OpenAlexService().get_works_by_dois(
            ["https://doi.org/10.1000/ABC", "10.2000/Def"]
        )
        query.filter_or.assert_called_once_with(doi=["10.1000/abc", "10.2000/def"])


OA = "https://openalex.org/"

FULL_WORK = {
    "id": OA + "W1",
    "doi": "https://doi.org/10.1000/x",
    "title": "Linked data",
    "publication_year": 2020,
    "type": "article",
    "cited_by_count": 7,
    "fwci": 1.5,
    "referenced_works": [OA + "W10", OA + "W11", OA + "W12"],
    "authorships": [
        {
            "author_position": "first",
            "is_corresponding": True,
            "author": {"id": OA + "A1", "display_name": "Ada", "orcid": "https://orcid.org/0000-0001"},
            "institutions": [
                {"id": OA + "I1", "display_name": "Uni One", "country_code": "GB",
                 "ror": "https://ror.org/1", "type": "education"}
            ],
        },
        {
            "author_position": "last",
            "is_corresponding": False,
            "author": {"id": OA + "A2", "display_name": "Bob", "orcid": None},
            "institutions": [],
        },
    ],
    "primary_location": {
        "source": {"id": OA + "S1", "display_name": "Journal One", "issn_l": "1234-5678", "type": "journal"}
    },
    "primary_topic": {"id": OA + "T2"},
    "topics": [
        {"id": OA + "T1", "display_name": "Topic One", "score": 0.9,
         "subfield": {"display_name": "SF"}, "field": {"display_name": "F"}, "domain": {"display_name": "D"}},
        {"id": OA + "T2", "display_name": "Topic Two", "score": 0.5,
         "subfield": {"display_name": "SF2"}, "field": {"display_name": "F2"}, "domain": {"display_name": "D2"}},
    ],
    "keywords": [
        {"id": OA + "keywords/semantic-web", "display_name": "Semantic web", "score": 0.7},
        {"id": OA + "keywords/rdf", "display_name": "RDF", "score": 0.4},
    ],
}


class TestKGMapper(TestCase):
    def test_short_id(self):
        self.assertEqual(short_id("https://openalex.org/W12"), "W12")
        self.assertEqual(short_id("https://openalex.org/keywords/semantic-web"), "keywords/semantic-web")
        self.assertEqual(short_id("W12"), "W12")

    def test_edge_vocab_has_six_terms(self):
        self.assertEqual(len(EDGE_VOCAB), 6)
        self.assertEqual(EDGE_VOCAB["CITES"], "cito:cites")

    def test_full_work_counts(self):
        nodes, edges = work_to_graph(FULL_WORK)
        by_type = Counter(n.node_type for n in nodes)
        self.assertEqual(by_type, {"paper": 4, "author": 2, "institution": 1, "venue": 1, "topic": 2, "keyword": 2})
        by_edge = Counter(e.edge_type for e in edges)
        self.assertEqual(by_edge, {"CITES": 3, "AUTHORED_BY": 2, "AFFILIATED_WITH": 1,
                                   "PUBLISHED_IN": 1, "HAS_TOPIC": 2, "HAS_KEYWORD": 2})

    def test_paper_node_is_hydrated(self):
        nodes, _ = work_to_graph(FULL_WORK)
        paper = next(n for n in nodes if n.key == "W1")
        self.assertFalse(paper.is_stub)
        self.assertEqual(paper.label, "Linked data")
        self.assertEqual(paper.properties, {
            "doi": "https://doi.org/10.1000/x", "year": 2020, "type": "article",
            "cited_by_count": 7, "fwci": 1.5, "referenced_works_count": 3})

    def test_referenced_works_are_stubs(self):
        nodes, edges = work_to_graph(FULL_WORK)
        stubs = [n for n in nodes if n.key in {"W10", "W11", "W12"}]
        self.assertEqual(len(stubs), 3)
        for n in stubs:
            self.assertTrue(n.is_stub)
            self.assertEqual(n.label, "")
            self.assertEqual(n.node_type, "paper")
        cites = [e for e in edges if e.edge_type == "CITES"]
        self.assertEqual({(e.source_key, e.target_key) for e in cites},
                         {("W1", "W10"), ("W1", "W11"), ("W1", "W12")})

    def test_topic_edge_carries_score_and_primary(self):
        _, edges = work_to_graph(FULL_WORK)
        topic = {e.target_key: e for e in edges if e.edge_type == "HAS_TOPIC"}
        self.assertEqual(topic["T1"].weight, 0.9)
        self.assertFalse(topic["T1"].properties["is_primary"])
        self.assertEqual(topic["T2"].weight, 0.5)
        self.assertTrue(topic["T2"].properties["is_primary"])

    def test_first_topic_is_primary_without_primary_topic(self):
        work = {**FULL_WORK, "primary_topic": None}
        _, edges = work_to_graph(work)
        topic = {e.target_key: e for e in edges if e.edge_type == "HAS_TOPIC"}
        self.assertTrue(topic["T1"].properties["is_primary"])
        self.assertFalse(topic["T2"].properties["is_primary"])

    def test_authorship_properties(self):
        nodes, edges = work_to_graph(FULL_WORK)
        authored = {e.target_key: e for e in edges if e.edge_type == "AUTHORED_BY"}
        self.assertEqual(authored["A1"].properties, {"position": "first", "is_corresponding": True})
        self.assertEqual(authored["A2"].properties, {"position": "last", "is_corresponding": False})
        aff = next(e for e in edges if e.edge_type == "AFFILIATED_WITH")
        self.assertEqual((aff.source_key, aff.target_key, aff.properties), ("A1", "I1", {"year": 2020}))

    def test_shared_institution_is_one_node(self):
        work = {**FULL_WORK, "authorships": [
            {"author": {"id": OA + "A1", "display_name": "Ada"}, "institutions": [{"id": OA + "I1", "display_name": "U"}]},
            {"author": {"id": OA + "A2", "display_name": "Bob"}, "institutions": [{"id": OA + "I1", "display_name": "U"}]},
        ]}
        nodes, edges = work_to_graph(work)
        self.assertEqual(sum(n.node_type == "institution" for n in nodes), 1)
        self.assertEqual(sum(e.edge_type == "AFFILIATED_WITH" for e in edges), 2)

    def test_duplicate_references_are_collapsed(self):
        work = {"id": OA + "W5", "referenced_works": [OA + "W9", OA + "W9"]}
        nodes, edges = work_to_graph(work)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)

    def test_sparse_work_does_not_raise(self):
        nodes, edges = work_to_graph({"id": OA + "W5", "title": None, "primary_location": None,
                                      "authorships": [], "referenced_works": None})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(edges, [])

    def test_null_source_and_no_id_work(self):
        nodes, edges = work_to_graph({"id": OA + "W5", "primary_location": {"source": None}})
        self.assertEqual((len(nodes), len(edges)), (1, 0))
        self.assertEqual(work_to_graph({"title": "x"}), ([], []))

    def test_authorship_without_author_id_is_skipped(self):
        work = {"id": OA + "W5", "authorships": [
            {"author": {"id": None, "display_name": "Ghost"}, "institutions": [{"id": OA + "I1"}]},
            {"author": None},
        ]}
        nodes, edges = work_to_graph(work)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(edges, [])


def _kg_work(key, year, refs=(), authors=(), venue=None):
    return {
        "id": OA + key,
        "title": f"Paper {key}",
        "publication_year": year,
        "cited_by_count": 0,
        "referenced_works": [OA + r for r in refs],
        "authorships": [{"author": {"id": OA + a, "display_name": a}} for a in authors],
        "primary_location": {"source": {"id": OA + venue}} if venue else None,
    }


# W1 cites W10, W11; W2 cites W10, W12; W20 cites W1. W1 and W10 share A1.
# W11 is not in OpenAlex, so it stays a stub.
KG_WORKS = {w["id"].removeprefix(OA): w for w in [
    _kg_work("W1", 2023, ["W10", "W11"], ["A1"], "S1"),
    _kg_work("W2", 2022, ["W10", "W12"], ["A2"], "S2"),
    _kg_work("W10", 2019, [], ["A1"], "S2"),
    _kg_work("W12", 2018),
    _kg_work("W20", 2024, ["W1"]),
]}


def _fake_get_works_by_ids(ids, fields=None):
    return [KG_WORKS[short_id(i)] for i in ids if short_id(i) in KG_WORKS]


def _fake_get_citing_works(ids, limit=50):
    targets = {OA + short_id(i) for i in ids}
    return [w for w in KG_WORKS.values() if targets & set(w["referenced_works"])][:limit]


class TestKGIngestService(TestCase):
    def setUp(self):
        by_ids = patch.object(OpenAlexService, "get_works_by_ids",
                              side_effect=_fake_get_works_by_ids)
        citing = patch.object(OpenAlexService, "get_citing_works",
                              side_effect=_fake_get_citing_works)
        self.by_ids = by_ids.start()
        self.citing = citing.start()
        self.addCleanup(by_ids.stop)
        self.addCleanup(citing.stop)

    def test_build_creates_nodes_and_edges(self):
        result = KGIngestService.build(["W1", "W2"])
        self.assertEqual(result["seeds_found"], 2)
        self.assertFalse(result["truncated"])
        self.assertTrue(KGEdge.objects.filter(source__key="W1", target__key="W10", edge_type="CITES").exists())
        self.assertTrue(KGEdge.objects.filter(source__key="W20", target__key="W1", edge_type="CITES").exists())
        self.assertEqual(result["nodes_total"], KGNode.objects.count())
        self.assertEqual(result["edges_total"], KGEdge.objects.count())
        # W1, W2, W10, W12, W20 hydrated; W11 is a stub.
        self.assertEqual(result["papers_hydrated"], 5)

    def test_most_shared_reference_is_hydrated_first(self):
        with patch.object(kg_ingest_service, "MAX_PRIOR_WORKS", 1):
            KGIngestService.build(["W1", "W2"])
        self.assertEqual(self.by_ids.call_args_list[1].args[0], ["W10"])

    def test_unfetched_reference_stays_stub(self):
        KGIngestService.build(["W1", "W2"])
        self.assertTrue(KGNode.objects.get(key="W11").is_stub)
        self.assertFalse(KGNode.objects.get(key="W10").is_stub)

    def test_rebuild_is_idempotent(self):
        KGIngestService.build(["W1", "W2"])
        counts = (KGNode.objects.count(), KGEdge.objects.count())
        KGIngestService.build(["W1", "W2"])
        self.assertEqual((KGNode.objects.count(), KGEdge.objects.count()), counts)
        self.assertFalse(KGNode.objects.get(key="W1").is_stub)

    def test_stub_does_not_overwrite_hydrated(self):
        KGIngestService.build(["W10"])
        with patch.object(kg_ingest_service, "MAX_PRIOR_WORKS", 0):
            KGIngestService.build(["W1"])
        node = KGNode.objects.get(key="W10")
        self.assertEqual(node.label, "Paper W10")
        self.assertFalse(node.is_stub)

    def test_citation_properties(self):
        KGIngestService.build(["W1", "W2"])
        e = KGEdge.objects.get(source__key="W1", target__key="W10", edge_type="CITES")
        self.assertEqual(e.properties, {"timespan_years": 4, "author_self_citation": True,
                                        "venue_self_citation": False})
        stub_edge = KGEdge.objects.get(source__key="W1", target__key="W11", edge_type="CITES")
        self.assertEqual(stub_edge.properties, {})

    def test_no_seed_found_raises(self):
        self.by_ids.side_effect = None
        self.by_ids.return_value = []
        with self.assertRaises(KGIngestError):
            KGIngestService.build(["W999"])
        self.assertEqual(KGNode.objects.count(), 0)

    def test_openalex_failure_writes_nothing(self):
        self.citing.side_effect = OpenAlexAPIError("boom")
        with self.assertRaises(OpenAlexAPIError):
            KGIngestService.build(["W1", "W2"])
        self.assertEqual(KGNode.objects.count(), 0)
        self.assertEqual(KGEdge.objects.count(), 0)

    def test_citing_works_fetched_once_per_seed(self):
        KGIngestService.build(["W1", "W2"])
        self.assertEqual(self.citing.call_count, 2)
        seen = set()
        for call in self.citing.call_args_list:
            self.assertEqual(len(call.args[0]), 1)
            self.assertEqual(call.kwargs["limit"], kg_ingest_service.PER_SEED_CITERS)
            seen.add(call.args[0][0])
        self.assertEqual(seen, {"W1", "W2"})

    def test_deadline_truncates_and_still_writes(self):
        with patch.object(kg_ingest_service, "BUILD_BUDGET_SECONDS", 0):
            result = KGIngestService.build(["W1", "W2"])
        self.assertTrue(result["truncated"])
        self.citing.assert_not_called()
        self.assertFalse(KGNode.objects.get(key="W1").is_stub)
        self.assertFalse(KGNode.objects.get(key="W10").is_stub)
        self.assertFalse(KGNode.objects.filter(key="W20").exists())

    def test_edges_use_reread_node_ids(self):
        real_bulk_create = KGNode.objects.bulk_create

        def bulk_create_without_pks(objs, **kwargs):
            objs = list(objs)
            real_bulk_create(objs, **kwargs)
            for obj in objs:
                obj.pk = None
            return [KGNode(key=obj.key) for obj in objs]

        with patch.object(KGNode.objects, "bulk_create", side_effect=bulk_create_without_pks):
            KGIngestService.build(["W1", "W2"])
        self.assertTrue(KGEdge.objects.filter(source__key="W1", target__key="W10", edge_type="CITES").exists())

    def test_edge_with_missing_node_is_dropped(self):
        def with_ghost_edge(work):
            nodes, edges = work_to_graph(work)
            return nodes, edges + [EdgeSpec(short_id(work["id"]), "W_GHOST", "CITES")]

        with patch.object(kg_ingest_service, "work_to_graph", side_effect=with_ghost_edge):
            result = KGIngestService.build(["W1", "W2"])
        self.assertFalse(KGEdge.objects.filter(target__key="W_GHOST").exists())
        self.assertEqual(result["edges_total"], KGEdge.objects.count())
