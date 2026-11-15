"""Match local-corpus papers to OpenAlex works by title and year.

The university research server sends only a ``local:<filename>`` id, a title
and a year (no DOI, no OpenAlex id), so a title search is the only link. A
wrong match is the costliest silent error, so a match needs title similarity
of at least ``TITLE_MATCH_MIN`` and, when both years are known, years at most
one apart (a preprint and its published version).
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from .openalex_service import FETCH_WORKERS, OpenAlexService

TITLE_MATCH_MIN = 0.90
MAX_YEAR_GAP = 1
# Read at call time so tests can patch it; a build's shared deadline can cut it shorter.
RESOLVE_BUDGET_SECONDS = 12
MIN_TITLE_LENGTH = 10
UNTITLED = "Untitled paper"  # LocalCorpusService's placeholder title

_NON_ALNUM = re.compile(r"[\W_]+")


def normalise_title(title: str) -> str:
    """Lower-cased, every run of non-alphanumerics collapsed to one space."""
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_title(a), normalise_title(b)).ratio()


def title_matches(local_title: str, local_year, work: dict) -> bool:
    if title_similarity(local_title, work.get("title") or "") < TITLE_MATCH_MIN:
        return False
    work_year = work.get("publication_year")
    if local_year is not None and work_year is not None:
        return abs(local_year - work_year) <= MAX_YEAR_GAP
    return True


def _searchable(paper: dict) -> bool:
    title = (paper.get("title") or "").strip()
    return len(title) >= MIN_TITLE_LENGTH and title != UNTITLED


def resolve_local_papers(
    local_papers: list[dict], deadline: float | None = None,
) -> tuple[dict[str, tuple[dict, float, str]], list[str], bool, set[str]]:
    """Returns (resolved, unresolved, truncated, searched).

    ``resolved`` maps a local id to (raw work, title similarity, matched_on),
    where matched_on is "title+year" when both years were known and checked,
    else "title". Papers not searched before RESOLVE_BUDGET_SECONDS ran out,
    or before ``deadline`` (a ``time.monotonic()`` value, the build's shared
    deadline) if that comes first, are unresolved and set ``truncated``. ``searched`` holds the ids that were
    actually looked up (found or not); papers skipped by the deadline or the
    title rule are not in it. Workers only call OpenAlex (no ORM);
    an OpenAlexAPIError from a worker propagates.
    """
    stop = time.monotonic() + RESOLVE_BUDGET_SECONDS
    if deadline is not None:
        stop = min(stop, deadline)
    api = OpenAlexService()

    def expired():
        return time.monotonic() >= stop

    def find(paper):
        # Queued calls re-check the clock: submit returns at once.
        if expired():
            return None, True
        return api.find_work_by_title(paper["title"], paper.get("year")), False

    truncated = False
    pending = []  # (paper, future or None), in request order
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for paper in local_papers:
            if not _searchable(paper):
                pending.append((paper, None))
            elif expired():
                truncated = True
                pending.append((paper, None))
            else:
                pending.append((paper, pool.submit(find, paper)))

    resolved, unresolved, searched = {}, [], set()
    for paper, future in pending:
        work, skipped = future.result() if future else (None, False)
        truncated = truncated or skipped
        if future and not skipped:
            searched.add(paper["id"])
        if work is None:
            unresolved.append(paper["id"])
            continue
        year_checked = paper.get("year") is not None and work.get("publication_year") is not None
        resolved[paper["id"]] = (
            work,
            title_similarity(paper["title"], work.get("title") or ""),
            "title+year" if year_checked else "title",
        )
    return resolved, unresolved, truncated, searched
