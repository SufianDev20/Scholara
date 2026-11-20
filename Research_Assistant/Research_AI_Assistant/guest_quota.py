"""
Guest Mode quotas, enforced on the server.

    @guest_quota("search")    GET  /api/search/          (first search AND Load More)
    @guest_quota("summary")   POST /api/summarise/       (automatic + retried summaries)
    @guest_quota("question")  POST /api/multi-paper-qa/  (Paper Analysis questions)

For a Clerk user the decorator does nothing. For a guest it:

  1. Vets the request before anything is reserved. Malformed requests are
     rejected here (fail closed), queries longer than MAX_GUEST_QUERY_CHARS
     are rejected, and summaries/questions may only use papers the server
     itself returned to this guest -- summaries only for a query this guest
     searched, compared in full -- with at most GUEST_MAX_SUMMARY_PAPERS
     papers per summary. The AI prompt is then built from the server's
     vetted query and stored paper copies, never the client's.
  2. Reserves one unit of the guest's quota with a single conditional UPDATE
     (`used = used + 1 WHERE used < limit`), so concurrent requests and
     multiple tabs can never exceed a limit.
  3. For AI kinds, also reserves the request's WORST-CASE number of upstream
     model attempts (ai_attempt_weight) against the global daily cap,
     GUEST_DAILY_AI_LIMIT, with the same kind of conditional UPDATE.
  4. Runs the view, then settles:
       search  -- refunded on any error (no AI is involved);
       AI kind -- refunded ONLY for the statuses in PRE_AI_REFUND_STATUSES,
                  which the views return before any AI call is made.
                  Success, 5xx (500/502/503) and exceptions are charged,
                  because the AI may already have been called -- possibly
                  several models -- so a failing request cannot be retried
                  for free.
  5. On a successful search, records the returned papers for step 1.
  6. Adds an X-Guest-Allowance header with what is left.
"""

import hashlib
import hmac
import json
from datetime import timedelta
from functools import wraps

from django.conf import settings
from django.db.models import F
from django.http import JsonResponse
from django.utils import timezone

from .models import GuestAllowance, GuestDailyUsage, GuestIssuedPaper
from .services.openrouter_service import DEFAULT_MAX_MODELS_PER_REQUEST, DEFAULT_QA_MAX_MODELS

KIND_FIELDS = {
    "search": "searches_used",
    "summary": "summaries_used",
    "question": "questions_used",
}
AI_KINDS = {"summary", "question"}

# Statuses the AI views return only BEFORE making any AI call:
#   summarise():      400 -- request validation
#   multi_paper_qa(): 400 -- serializer; 413 -- context too large;
#                     422 -- every selected PDF failed to process
# (tests.py asserts the AI is not called on these paths.)
PRE_AI_REFUND_STATUSES = {"summary": {400}, "question": {400, 413, 422}}

# Same as GuestIssuedPaper.query's max_length: a guest query is stored and
# compared in full, never truncated.
MAX_GUEST_QUERY_CHARS = 500

# Fields the summary prompt and Q&A need (prompt_builder / analysis.js).
# Large lists such as referenced_works are not stored.
STORED_PAPER_FIELDS = (
    "openalex_id",
    "title",
    "authors",
    "abstract",
    "publication_year",
    "doi",
    "source",
    "pdf_url",
    "oa_url",
    "pdf_candidates",
)

EXHAUSTED_MESSAGES = {
    "search": (
        "You've used all {limit} guest searches (Load More counts as a search). "
        "Sign in to keep researching. Your Recents will still be here."
    ),
    "summary": (
        "You've used all {limit} guest summaries. The papers are still listed; "
        "sign in to get AI summaries."
    ),
    "question": "You've used all {limit} guest questions. Sign in to keep asking about these papers.",
}


def limits():
    return {
        "search": settings.GUEST_SEARCH_LIMIT,
        "summary": settings.GUEST_SUMMARY_LIMIT,
        "question": settings.GUEST_QUESTION_LIMIT,
    }


def is_guest(request):
    return bool(getattr(getattr(request, "user", None), "is_guest", False))


def hash_ident(ident):
    """HMAC of a client identifier (IP), so raw addresses are never stored."""
    return hmac.new(
        settings.SECRET_KEY.encode(), str(ident or "").encode(), hashlib.sha256
    ).hexdigest()


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def reserve(allowance, kind):
    field = KIND_FIELDS[kind]
    updated = GuestAllowance.objects.filter(
        pk=allowance.pk,
        expires_at__gt=timezone.now(),
        **{f"{field}__lt": limits()[kind]},
    ).update(**{field: F(field) + 1})
    return updated == 1


def release(allowance, kind):
    field = KIND_FIELDS[kind]
    GuestAllowance.objects.filter(pk=allowance.pk, **{f"{field}__gt": 0}).update(
        **{field: F(field) - 1}
    )


def ai_attempt_weight(kind):
    """Worst-case upstream model attempts one guest request of `kind` can make.

    OpenRouter tries one model per attempt, up to OPENROUTER_MAX_MODELS_PER_REQUEST
    (summaries) or OPENROUTER_QA_MAX_MODELS (questions); the
    "ollama_then_openrouter" backend first tries Ollama once; summarise()
    may call the LLM twice (one format-repair attempt).
    """
    backend = (getattr(settings, "LLM_BACKEND", "openrouter") or "openrouter").strip()
    if backend == "ollama":
        per_call = 1
    else:
        if kind == "question":
            models = getattr(settings, "OPENROUTER_QA_MAX_MODELS", DEFAULT_QA_MAX_MODELS)
        else:
            models = getattr(settings, "OPENROUTER_MAX_MODELS_PER_REQUEST", DEFAULT_MAX_MODELS_PER_REQUEST)
        per_call = max(1, int(models)) + (1 if backend == "ollama_then_openrouter" else 0)
    calls = 2 if kind == "summary" else 1
    return per_call * calls


def reserve_daily_ai(weight):
    """Reserve `weight` upstream attempts of today's guest AI budget.

    Returns the day reserved, or None if the budget can't cover the worst case.
    """
    day = timezone.localdate()
    GuestDailyUsage.objects.get_or_create(day=day)
    updated = GuestDailyUsage.objects.filter(
        day=day, ai_calls__lte=settings.GUEST_DAILY_AI_LIMIT - weight
    ).update(ai_calls=F("ai_calls") + weight)
    return day if updated == 1 else None


def release_daily_ai(day, weight):
    GuestDailyUsage.objects.filter(day=day, ai_calls__gte=weight).update(
        ai_calls=F("ai_calls") - weight
    )


def allowance_payload(allowance):
    allowance.refresh_from_db(
        fields=["searches_used", "summaries_used", "questions_used", "expires_at"]
    )
    lim = limits()
    used = {kind: getattr(allowance, field) for kind, field in KIND_FIELDS.items()}
    return {
        "expires_at": allowance.expires_at.isoformat(),
        "limits": lim,
        "remaining": {kind: max(0, lim[kind] - used[kind]) for kind in lim},
        # The frontend sizes guest searches/summaries to this (see research.js).
        "max_summary_papers": settings.GUEST_MAX_SUMMARY_PAPERS,
    }


def prune_expired():
    """Drop guest data a day after it expires (issued papers cascade)."""
    now = timezone.now()
    GuestAllowance.objects.filter(expires_at__lt=now - timedelta(days=1)).delete()
    GuestDailyUsage.objects.filter(day__lt=timezone.localdate() - timedelta(days=30)).delete()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def _attach_allowance(response, allowance):
    try:
        response["X-Guest-Allowance"] = json.dumps(allowance_payload(allowance))
    except Exception:  # never let the header break a real response
        pass
    return response


def deny(status, code, message, allowance=None, kind=None):
    body = {"error": message, "code": code}
    if kind:
        body["quota"] = kind
    response = JsonResponse(body, status=status)
    if allowance is not None:
        _attach_allowance(response, allowance)
    return response


def _bad_request(allowance, message):
    return deny(400, "guest_invalid_request", message, allowance)


def _query_too_long(allowance):
    return deny(
        400,
        "guest_query_too_long",
        f"Guest searches are limited to {MAX_GUEST_QUERY_CHARS} characters. "
        "Shorten the topic, or sign in for full access.",
        allowance,
    )


def _not_allowed(allowance):
    return deny(
        403,
        "guest_paper_not_allowed",
        "Guest sessions can only use papers from their own searches. "
        "Run the search again, or sign in for full access.",
        allowance,
    )


# ---------------------------------------------------------------------------
# Strict checks: only what this guest was actually given
# ---------------------------------------------------------------------------


def _stored_paper(paper):
    return {key: paper.get(key) for key in STORED_PAPER_FIELDS if key in paper}


def vet_search(request, allowance):
    query = (request.query_params.get("q") or "").strip()
    if len(query) > MAX_GUEST_QUERY_CHARS:
        return _query_too_long(allowance)
    return None  # the view validates the rest


def record_search(request, allowance, response):
    try:
        payload = json.loads(response.content)
    except Exception:
        return
    papers = payload.get("papers") if isinstance(payload, dict) else None
    if not isinstance(papers, list):
        return
    # Full query, never truncated: vet_search capped it at MAX_GUEST_QUERY_CHARS.
    query = (request.query_params.get("q") or "").strip()
    rows = [
        GuestIssuedPaper(
            allowance=allowance,
            openalex_id=str(p["openalex_id"])[:100],
            query=query,
            paper=_stored_paper(p),
        )
        for p in papers
        if isinstance(p, dict) and p.get("openalex_id")
    ]
    if rows:
        GuestIssuedPaper.objects.bulk_create(rows, ignore_conflicts=True)


def _issued_by_id(allowance, ids):
    issued = {}
    for row in GuestIssuedPaper.objects.filter(allowance=allowance, openalex_id__in=set(ids)):
        issued.setdefault(row.openalex_id, row.paper)
    return issued


def vet_summary(request, allowance):
    try:
        data = json.loads(request.body)
    except (ValueError, TypeError):
        return _bad_request(allowance, "Invalid JSON body")
    if not isinstance(data, dict):
        return _bad_request(allowance, "Invalid JSON body")

    query, papers = data.get("query"), data.get("papers")
    if not isinstance(query, str) or not query.strip():
        return _bad_request(allowance, "Missing or invalid 'query' field")
    query = query.strip()
    if len(query) > MAX_GUEST_QUERY_CHARS:
        return _query_too_long(allowance)
    if not isinstance(papers, list) or not papers:
        return _bad_request(allowance, "Missing or invalid 'papers' field")
    limit = settings.GUEST_MAX_SUMMARY_PAPERS
    if len(papers) > limit:
        return deny(
            400,
            "guest_too_many_papers",
            f"Guest summaries can include at most {limit} papers. Sign in to summarise more.",
            allowance,
        )

    ids = [p.get("openalex_id") if isinstance(p, dict) else None for p in papers]
    if not all(isinstance(i, str) and i for i in ids):
        return _not_allowed(allowance)

    # The complete query must be one this guest searched (no prefix match).
    searched = GuestIssuedPaper.objects.filter(allowance=allowance, query=query).exists()
    issued = _issued_by_id(allowance, ids)
    if not searched or any(i not in issued for i in ids):
        return _not_allowed(allowance)

    # summarise() builds the prompt from these, never from the client's copies.
    request.guest_vetted_query = query
    request.guest_vetted_papers = [issued[i] for i in ids]
    return None


def _allowed_urls(paper):
    urls = {paper.get("pdf_url"), paper.get("oa_url")}
    candidates = paper.get("pdf_candidates")
    if isinstance(candidates, list):
        urls.update(candidates)
    return {u.strip() for u in urls if isinstance(u, str) and u.strip()}


def vet_question(request, allowance):
    data = request.data
    if not isinstance(data, dict):
        return _bad_request(allowance, "Invalid request.")
    paper_ids, pdf_urls = data.get("paper_ids"), data.get("pdf_urls")
    if not isinstance(paper_ids, list) or not isinstance(pdf_urls, dict) or not paper_ids:
        return _bad_request(allowance, "Invalid request.")

    if not all(isinstance(pid, str) and pid for pid in paper_ids):
        return _not_allowed(allowance)
    issued = _issued_by_id(allowance, paper_ids)
    for pid in paper_ids:
        paper = issued.get(pid)
        if paper is None:
            return _not_allowed(allowance)
        sent = pdf_urls.get(pid)
        sent = [sent] if isinstance(sent, str) else sent
        if not isinstance(sent, list) or not sent:
            return _bad_request(allowance, "Invalid request.")
        allowed = _allowed_urls(paper)
        if any(not isinstance(u, str) or u.strip() not in allowed for u in sent):
            return _not_allowed(allowance)

    # multi_paper_qa() names papers from the server's copy, not the client's.
    request.guest_paper_meta = {
        pid: {
            "title": (issued[pid].get("title") or "")[:300],
            "authors": [
                (a.get("name") or "")[:100]
                for a in (issued[pid].get("authors") or [])[:10]
                if isinstance(a, dict) and a.get("name")
            ],
            "year": issued[pid].get("publication_year")
            if isinstance(issued[pid].get("publication_year"), int)
            else None,
        }
        for pid in paper_ids
    }
    return None


VETTERS = {"search": vet_search, "summary": vet_summary, "question": vet_question}


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def _refundable(kind, status_code):
    if kind in AI_KINDS:
        return status_code in PRE_AI_REFUND_STATUSES[kind]
    return status_code >= 400  # search: no AI cost


def guest_quota(kind):
    if kind not in KIND_FIELDS:
        raise ValueError(f"Unknown guest quota kind: {kind}")

    def decorator(view):
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            if not is_guest(request):
                return view(request, *args, **kwargs)

            allowance = request.user.allowance

            denial = VETTERS[kind](request, allowance)
            if denial is not None:
                return denial

            if not reserve(allowance, kind):
                message = EXHAUSTED_MESSAGES[kind].format(limit=limits()[kind])
                return deny(403, "guest_quota_exhausted", message, allowance, kind)

            day = None
            weight = 0
            if kind in AI_KINDS:
                weight = ai_attempt_weight(kind)
                day = reserve_daily_ai(weight)
                if day is None:
                    release(allowance, kind)
                    return deny(
                        503,
                        "guest_capacity",
                        "Guest mode has reached today's limit. Sign in to continue, or try again tomorrow.",
                        allowance,
                    )

            try:
                response = view(request, *args, **kwargs)
            except Exception:
                # An AI request may have reached the provider before failing:
                # keep it charged. Searches have no AI cost.
                if kind not in AI_KINDS:
                    release(allowance, kind)
                raise

            if _refundable(kind, getattr(response, "status_code", 500)):
                release(allowance, kind)
                if day is not None:
                    release_daily_ai(day, weight)
            elif kind == "search" and response.status_code < 400:
                record_search(request, allowance, response)

            return _attach_allowance(response, allowance)

        return wrapped

    return decorator
