import { DOMManager } from './core.js';
import { MAX_ANALYZABLE_PAPERS } from './constants.js';
import { exhaustedMessage, guestErrorFrom, isGuestActive, maxSummaryPapers, remaining } from './guest.js';

// Guest Mode: the server accepts at most maxSummaryPapers() papers per guest
// summary; signed-in users are unaffected.
function capForGuest(papers) {
  return isGuestActive() ? papers.slice(0, maxSummaryPapers()) : papers;
}

// Guest Mode: an Error carrying a server (or known) guest refusal. Callers
// show err.message as a friendly notice with a sign-in link, not "Error: ...".
function guestError(guest, status) {
  const err = new Error(guest.message);
  err.guest = guest;
  err.status = status || 403;
  return err;
}

// A guest with none of `kind` left: refuse locally instead of sending a
// request the server would certainly refuse.
function guestQuotaSpent(kind) {
  return isGuestActive() && remaining(kind) === 0
    ? guestError({ code: "guest_quota_exhausted", message: exhaustedMessage(kind), quota: kind })
    : null;
}

// Extend DOMManager prototype with research-related methods

// ---------------------------------------------------------------------
//
// ---------------------------------------------------------------------
const ANALYSIS_SESSION_KEY = "scholara:analysisSession";

// `session` is the Recents session the research came from (null for a saved
// Binder). For a session the handoff also carries its earlier Paper Analysis
// turns, so follow-ups asked before are shown again instead of being lost,
// and `askOnOpen` tells analysis.js whether `question` is a new one to ask.
function writeAnalysisSession({ binder, session, question, papers, selectedIds }, domManager) {
  const payload = {
    researchId: binder.id || null,
    sourceQuery: binder.name || question || "",
    question: question || "",
    papers: Array.isArray(papers) ? papers : [],
    selectedIds: Array.isArray(selectedIds) ? selectedIds.filter(Boolean) : [],
    searchParams: (session && session.searchParams) || window.appState.searchParams || null,
    createdAt: Date.now(),
  };
  if (session) {
    // Explicit link back to Recents. A Binder saved from a session keeps the
    // session's id (researchId), so researchId alone must never be used to
    // route anything into Recents -- Binder handoffs don't carry this field.
    payload.recentsSessionId = session.id;
    payload.turns = Array.isArray(session.analysisTurns) ? session.analysisTurns : [];
    payload.askOnOpen = !!question;
  }

  const write = () => sessionStorage.setItem(ANALYSIS_SESSION_KEY, JSON.stringify(payload));
  try {
    write();
    return true;
  } catch (err) {
    // Recents share this tab's storage quota. The handoff takes priority:
    // shrink the Recents recovery copy to this session alone and retry.
    if (domManager && domManager.isStorageQuotaError(err)) {
      domManager.releaseRecentsStorage(session && session.id);
      try {
        write();
        return true;
      } catch (_retryErr) {
        /* fall through to the original error handling */
      }
    }
    console.error("Could not hand off research session to the analysis page:", err);
    alert("Couldn't open the analysis page (storage unavailable). Please try again.");
    return false;
  }
}

// Saves where the user was (scroll, opened paper) before the full-page
// navigation to Paper Analysis, so Back restores the same spot.
function rememberPositionBeforeAnalysis(domManager, session, paper) {
  if (!session) return;
  domManager.captureActiveSessionUi();
  session.lastOpenedPaperId = (paper && paper.openalex_id) || null;
  domManager.saveResearchSessions(); // mark changed so the flush writes it
  domManager.flushResearchSessions();
}

DOMManager.prototype.handleFollowUpQuestion = function() {
  // Blocks both the Send button and the Enter key: the button's disabled
  // attribute alone doesn't stop Enter (its keydown handler calls this
  // directly, bypassing .disabled), so the active-request check has to
  // live here too, not just on the button.
  if (window.appState.isThinking) return;

  const input = this.elements.researchInput;
  const question = input ? input.value.trim() : "";

  // No navigation for an empty submission, and nothing to carry over
  // without an active research session.
  if (!question || !window.appState.currentResearchBinder) return;

  const binder = window.appState.currentResearchBinder;
  const session = this.sessionForBinder(binder);
  const papers = Array.isArray(binder.papers) ? binder.papers : [];

  const selectedIds = papers
    .filter((p) => p && (p.pdf_url || p.oa_url))
    .slice(0, MAX_ANALYZABLE_PAPERS)
    .map((p) => p.openalex_id);

  rememberPositionBeforeAnalysis(this, session, null);
  if (!writeAnalysisSession({ binder, session, question, papers, selectedIds }, this)) return;

  if (input) input.value = "";
  window.location.href = "/analysis/";
};

DOMManager.prototype.openAnalysisPage = function(papers, clickedIndex) {
  const binder = window.appState.currentResearchBinder;
  if (!binder) return;
  const session = this.sessionForBinder(binder);

  const list = Array.isArray(papers) ? papers : [];
  const clicked = list[clickedIndex];
  const selectedIds = clicked && clicked.openalex_id ? [clicked.openalex_id] : [];

  rememberPositionBeforeAnalysis(this, session, clicked);
  if (!writeAnalysisSession({ binder, session, question: "", papers: list, selectedIds }, this)) return;

  window.location.href = "/analysis/";
};

DOMManager.prototype.renderReferences = function(papers) {
  if (!this.elements.referencesPanel || !this.elements.referencesList)
    return;

  this.elements.referencesList.innerHTML = "";
  // What the list currently shows; recents.js skips re-rendering identical lists.
  this._renderedReferences = { papers: papers, filter: window.appState.currentFilter || "relevance" };

  // Always show the panel - display empty state if no papers
  this.elements.referencesPanel.style.display = "flex";

  if (!papers || papers.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "references-empty";
    emptyState.textContent =
      "No papers retrieved yet. Research results will appear here.";
    this.elements.referencesList.appendChild(emptyState);
    return;
  }

  // Sort papers based on current filter
  const sortedPapers = this.sortPapers(papers);

  sortedPapers.forEach(
    function (paper, i) {
      var card = document.createElement("div");
      card.className = "source-card";
      // Focusable so keyboard users can open the paper in Paper Analysis
      // (Enter/Space), the same as clicking the card.
      card.tabIndex = 0;
      card.title = "Open in Paper Analysis";
      // Lets Back from Paper Analysis return focus to the paper opened.
      if (paper && paper.openalex_id) card.dataset.paperId = paper.openalex_id;

      // Paper metadata is untrusted text: escape every value before innerHTML.
      var doiUrl = this.paperDoiUrl(paper && paper.doi);
      var title = this.escapeHtml((paper && paper.title) || "Untitled");

      var linkHtml = doiUrl
        ? '<a class="source-link" href="' +
          this.escapeHtml(doiUrl) +
          '" target="_blank" rel="noopener noreferrer">' +
          title +
          "</a>"
        : title;

      var citationCount = Number(paper.cited_by_count) || 0;
      var meta = [paper.publication_year, paper.source]
        .filter(Boolean)
        .map((v) => this.escapeHtml(String(v)))
        .join(" · ");

      card.innerHTML =
        '<div class="source-number">' +
        (i + 1) +
        "</div>" +
        '<div class="source-body">' +
        '<div class="source-title">' +
        linkHtml +
        "</div>" +
        (meta ? '<div class="source-meta">' + meta + "</div>" : "") +
        '<div class="source-citation-count">' +
        citationCount.toLocaleString() +
        (citationCount === 1 ? " citation" : " citations") +
        "</div>" +
        "</div>";

      card.addEventListener("click", (e) => {
        if (e.target.closest("a.source-link")) return;
        e.preventDefault();
        this.openAnalysisPage(sortedPapers, i);
      });

      card.addEventListener("keydown", (e) => {
        if (e.target !== card) return; // the title link handles its own keys
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          this.openAnalysisPage(sortedPapers, i);
        }
      });

      this.elements.referencesList.appendChild(card);
    }.bind(this),
  );
};

DOMManager.prototype.setReferencesFilterVisibility = function(visible) {
  const filters = this.elements.referencesPanel?.querySelector(
    ".references-filters",
  );
  if (!filters) return;
  filters.style.display = visible ? "flex" : "none";
};

DOMManager.prototype.sortPapers = function(papers) {
  const activeFilter = window.appState.currentFilter || "relevance";
  const sorted = [...papers];

  if (activeFilter === "cited_by_count") {
    sorted.sort(
      (a, b) => (b.cited_by_count || 0) - (a.cited_by_count || 0),
    );
  } else {
    sorted.sort(
      (a, b) => (b.relevance_score || 0) - (a.relevance_score || 0),
    );
  }

  return sorted;
};

DOMManager.prototype.setupFilterListeners = function() {
  const filterBtns = document.querySelectorAll(".filter-btn");
  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (window.appState.isGeneratingResponse) return;
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      window.appState.currentFilter = btn.dataset.filter;
      const session = this.getActiveResearchSession();
      if (session) {
        session.currentFilter = btn.dataset.filter;
        this.saveResearchSessions();
      }

      if (window.appState.currentResearchBinder?.papers) {
        this.renderReferences(window.appState.currentResearchBinder.papers);
      }
    });
  });
};

// Runs the search (once) and the summary for the research session on screen.
//
// Ownership: the session and its own chat container are captured up front.
// Results always go into that session -- even if the user has since gone
// Home or switched to another topic -- and shared controls (references,
// load more, buttons, window.appState) are only touched while the session
// is still the one on screen. A session whose search already returned
// papers (summary retry after an error/interruption) skips the search.
DOMManager.prototype.generateResearchResponse = async function() {
  const binder = window.appState.currentResearchBinder;
  const session = this.sessionForBinder(binder);
  if (!session || session.pending.search) return;

  const container = session.container;
  const owns = () => this.ownsResearchView(binder);

  session.pending.search = true;
  session.status = "loading";
  session.errorMessage = "";
  session.loadMoreStatus = "";
  this.syncResearchUi(session);
  this.saveResearchSessions();

  const assistantDiv = this.addMessage("", false, container);

  // Loading state shown until the search + summary complete
  assistantDiv.innerHTML = this.loadingMessageHtml();

  try {
    const params = session.searchParams;

    if (!session.searched) {
      const searchData = await this.retrieveFromBackend(
        params.query,
        params.minYear,
        params.maxYear,
        params.mode,
        params.perPage,
      );

      // Store pagination info on the session (and globally while on screen)
      session.nextCursor = searchData.next_cursor;
      session.totalCount = searchData.total_count;
      binder.papers = Array.isArray(searchData.papers) ? searchData.papers : [];
      session.searched = true;
      session.summaryPaperCount = binder.papers.length;
      this.saveResearchSessions();

      if (owns()) {
        this.syncResearchUi(session);
        this.renderReferences(binder.papers);
      }

      // A 200 response with zero papers is a valid, successful search that
      // simply matched nothing — distinct from the search service being
      // unavailable (handled in the catch block below). Report it as such
      // and skip the summarise call: there is nothing to summarise.
      if (binder.papers.length === 0) {
        const noResultsMessage =
          "No papers matched this search. Try different keywords, a broader year range, or a different search mode.";
        assistantDiv.innerHTML = noResultsMessage;
        assistantDiv.style.opacity = "1";
        binder.messages = binder.messages.concat({
          role: "assistant",
          content: noResultsMessage,
        });
        session.status = "empty";
        return;
      }
    }

    const papersToUse = capForGuest(
      binder.papers.slice(0, session.summaryPaperCount || binder.papers.length),
    );

    const spent = guestQuotaSpent("summary");
    if (spent) throw spent;

    // Apply rate limiting before LLM call
    await this.waitForRateLimit();

    const response = await fetch("/api/summarise/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
          "X-CSRFToken": window.csrfToken,
      },
      body: JSON.stringify({
        query: params.query,
        papers: papersToUse,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      console.error("API error response:", err);
      const guest = guestErrorFrom(err);
      if (guest) throw guestError(guest, response.status);
      throw new Error(err.error || `Summarise error ${response.status}`);
    }

    const data = await response.json();
    const summary = data.summary || "No summary returned.";

    // Replace the loading placeholder with the actual response
    assistantDiv.innerHTML = this.markdownToHtml(summary);
    this.addPaperDetails(assistantDiv, papersToUse);
    assistantDiv.style.opacity = "1";

    binder.messages = binder.messages.concat({
      role: "assistant",
      content: summary,
    });
    // "ready" also reveals Save to Binder (see syncResearchUi), as before.
    session.status = "ready";
  } catch (err) {
    if (this.isPageUnloading()) {
      // The browser aborted the request because the page is going away.
      // Keep "loading": on restore it becomes "interrupted" with a retry.
      return;
    }
    console.error("Error in generateResearchResponse:", err);
    const message = err && err.message ? err.message : "Unknown error";
    session.status = "error";
    session.errorMessage = message;
    // Guest Mode limits read as a friendly notice with a sign-in link.
    session.errorGuest = !!(err && err.guest);
    session.loadMoreStatus = session.errorGuest ? "" : `Error: ${message}`;
    // Replace the loading placeholder with the error and an explicit retry
    this.renderSessionNotice(session, assistantDiv, session.errorGuest ? message : "Error: " + message);
  } finally {
    // Restores the follow-up Send button, reference filters, pagination and
    // load more for this session -- only if it is still on screen.
    session.pending.search = false;
    this.saveResearchSessions();
    this.syncResearchUi(session);
  }
};

DOMManager.prototype.loadMorePapers = async function() {
  const binder = window.appState.currentResearchBinder;
  if (!binder) return;
  // Recents session, or null for a saved Binder opened from the Binders
  // board (which keeps using the global pagination state, as before).
  const session = this.sessionForBinder(binder);
  const st = window.appState;

  const cursor = session ? session.nextCursor : st.nextCursor;
  const params = session ? session.searchParams : st.searchParams;
  const busy = session ? session.pending.loadMore : st.isLoadingMore;
  if (!cursor || busy) return;

  const owns = () => this.ownsResearchView(binder);
  const btn = this.elements.loadMoreBtn;
  const status = this.elements.loadMoreStatus;
  const container = session ? session.container : this.elements.researchChatContainer;
  const setStatus = (text) => {
    if (session) session.loadMoreStatus = text;
    if (owns() && status) status.textContent = text;
  };
  const setBusy = (value) => {
    if (session) {
      session.pending.loadMore = value;
      this.syncResearchUi(session);
    } else {
      st.isLoadingMore = value;
      if (btn) btn.disabled = value;
    }
  };

  // searchParams is only set by a fresh search; without it the request below
  // would throw before any feedback reached the user.
  if (!params?.query) {
    setStatus("Error: search context lost. Run the search again.");
    return;
  }

  // Guest Mode: Load More counts as a search; skip a certain refusal.
  const searchSpent = guestQuotaSpent("search");
  if (searchSpent) {
    setStatus(searchSpent.message);
    return;
  }

  let loadFailed = false;
  let firstNewNode = null;

  setStatus("Loading more papers...");
  setBusy(true);

  // Ensure button remains visible during loading
  if (owns() && this.elements.loadMoreContainer) {
    this.elements.loadMoreContainer.style.display = "flex";
  }

  // Keeps a saved Binder in the binders array in step with its research view.
  const syncSavedBinder = () => {
    if (binder.id.startsWith("temp-")) return;
    const binderIndex = st.binders.findIndex((b) => b.id === binder.id);
    if (binderIndex !== -1) {
      st.binders[binderIndex] = JSON.parse(JSON.stringify(binder));
      this.renderBinders(); // Update binder display
    }
  };

  try {
    // Quota filter = number of NEW papers each Load More must add. Load More is
    // keyword search (semantic search can't paginate), so its order can repeat
    // papers already shown from the semantic first page; over-fetch (API max 50),
    // then drop shown ids and trim back to the quota below.
    const quota = params.perPage || 25;
    const perPage = Math.min(quota * 2, 50);
    let yearParams = "";
    if (params.minYear) {
      yearParams += `&min_year=${encodeURIComponent(params.minYear)}`;
    }
    if (params.maxYear) {
      yearParams += `&max_year=${encodeURIComponent(params.maxYear)}`;
    }

    const response = await fetch(
      `/api/search/?q=${encodeURIComponent(params.query)}&mode=${params.mode}&cursor=${encodeURIComponent(cursor)}&load_more=true&per_page=${perPage}${yearParams}`,
    );

    if (!response.ok) {
      // Surface the server's own message (e.g. OpenAlex being unreachable)
      // rather than just the status code.
      let detail = "";
      let guest = null;
      try {
        const errBody = await response.json();
        detail = errBody.error || errBody.detail || "";
        guest = guestErrorFrom(errBody);
      } catch (_parseErr) {
        // Non-JSON body; the status code below is all we have.
      }
      if (guest) throw guestError(guest, response.status);
      throw new Error(detail || `Load more error ${response.status}`);
    }

    const data = await response.json();

    // Update cursor for next load -- on the session that asked for it.
    if (session) session.nextCursor = data.next_cursor;
    if (!session || owns()) st.nextCursor = data.next_cursor;

    // Drop papers already shown, then trim to the quota filter
    const shownIds = new Set(binder.papers.map((p) => p.openalex_id));
    data.papers = (data.papers || [])
      .filter((p) => !shownIds.has(p.openalex_id))
      .slice(0, quota);

    // Append new papers. A new array (not push) so a Binder saved earlier
    // from this session keeps its own paper list.
    binder.papers = binder.papers.concat(data.papers);
    if (session) this.saveResearchSessions();

    if (owns()) this.renderReferences(binder.papers);
    syncSavedBinder();

    const appendSummary = (summary) => {
      const labelDiv = document.createElement("div");
      labelDiv.textContent = "Summary of Additional Papers";
      labelDiv.className = "summary-label";
      container.appendChild(labelDiv);

      const summaryDiv = this.addMessage("", false, container);
      summaryDiv.innerHTML = this.markdownToHtml(summary);
      this.addPaperDetails(summaryDiv, data.papers);
      summaryDiv.style.opacity = "1";

      // Also add summary to messages array for persistence
      binder.messages = binder.messages.concat({
        role: "assistant",
        content: summary,
      });
      if (session) this.saveResearchSessions();
      return labelDiv;
    };

    const requestSummary = async () => {
      const summarySpent = guestQuotaSpent("summary");
      if (summarySpent) throw summarySpent;
      await this.waitForRateLimit();
      const summariseResponse = await fetch("/api/summarise/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
            "X-CSRFToken": window.csrfToken,
        },
        body: JSON.stringify({
          query: params.query,
          papers: capForGuest(data.papers),
        }),
      });
      if (!summariseResponse.ok) {
        const err = await summariseResponse.json().catch(() => ({}));
        const guest = guestErrorFrom(err);
        if (guest) throw guestError(guest, summariseResponse.status);
        throw new Error(
          err.error || `Summarise error ${summariseResponse.status}`,
        );
      }
      const summariseData = await summariseResponse.json();
      return summariseData.summary || "No summary returned.";
    };

    // Summarize additional papers
    try {
      firstNewNode = appendSummary(await requestSummary());
      syncSavedBinder();
    } catch (err) {
      console.warn("Summary of additional papers failed:", err);

      // Show user-friendly error message but continue the flow
      const errorDiv = document.createElement("div");
      errorDiv.className = "summary-error-message";
      errorDiv.setAttribute("role", "alert");
      errorDiv.innerHTML = `
        <p class="summary-error-message__title">Could not summarize additional papers</p>
        <p class="summary-error-message__detail"></p>
        <button type="button" class="summary-error-message__retry">Try Again</button>
      `;
      errorDiv.querySelector(".summary-error-message__detail").textContent = err.message;
      container.appendChild(errorDiv);
      firstNewNode = errorDiv;

      // Guest Mode limit: retrying would be refused again; offer sign-in.
      if (err.guest) {
        errorDiv.querySelector(".summary-error-message__title").textContent = "Summary not available in guest mode";
        const oldBtn = errorDiv.querySelector("button");
        const signIn = document.createElement("a");
        signIn.className = "summary-error-message__retry";
        signIn.href = "/sign-in/";
        signIn.textContent = "Sign in";
        if (oldBtn) oldBtn.replaceWith(signIn);
      }

      // Add retry functionality
      const retryBtn = err.guest ? null : errorDiv.querySelector("button");
      if (retryBtn) {
        retryBtn.onclick = async () => {
          retryBtn.textContent = "Retrying...";
          retryBtn.disabled = true;
          try {
            const summary = await requestSummary();
            // Replace error message with successful summary
            errorDiv.remove();
            appendSummary(summary);
            syncSavedBinder();
          } catch (retryErr) {
            retryBtn.textContent = "Retry Failed";
            retryBtn.disabled = false;
            console.warn("Summary retry failed:", retryErr);
          }
        };
      }

      // Continue without breaking the load more flow
      console.log("Load more flow continuing despite summary failure");
    }

    // Show message if available (e.g., "No more papers available")
    if (data.message) {
      const messageDiv = document.createElement("div");
      messageDiv.textContent = data.message;
      messageDiv.className = "load-more-message";
      container.appendChild(messageDiv);
    }
  } catch (err) {
    console.error("Load more failed:", err);
    loadFailed = true;
    setStatus(err && err.guest ? err.message : `Error: ${err.message}`);
  } finally {
    // Keep an error visible; only clear the "Loading..." text on success.
    if (!loadFailed) setStatus("");
    setBusy(false);
    if (owns()) {
      if (firstNewNode) firstNewNode.scrollIntoView({ behavior: "smooth", block: "start" });
      // Final pagination update to ensure button visibility is correct
      this.updatePaginationInfo();
    }
  }
};

DOMManager.prototype.waitForRateLimit = async function() {
  const MIN_INTERVAL = 3000; // 3 seconds between LLM calls
  const now = Date.now();
  const timeSinceLastCall = now - window.appState.lastLLMCall;

  if (timeSinceLastCall < MIN_INTERVAL) {
    const waitTime = MIN_INTERVAL - timeSinceLastCall;
    await new Promise((resolve) => setTimeout(resolve, waitTime));
  }

  window.appState.lastLLMCall = Date.now();
};

DOMManager.prototype.ensureElements = function() {
  // Re-cache elements if they're missing
  if (!this.elements.loadMoreContainer) {
    this.elements.loadMoreContainer =
      document.getElementById("loadMoreContainer");
  }
  if (!this.elements.loadMoreBtn) {
    this.elements.loadMoreBtn = document.getElementById("loadMoreBtn");
  }
  if (!this.elements.researchChatContainer) {
    this.elements.researchChatContainer = document.getElementById(
      "researchChatContainer",
    );
  }
};

// Renders pagination for the research view on screen (window.appState holds
// the on-screen session's values; see syncResearchUi in recents.js).
DOMManager.prototype.updatePaginationInfo = function() {
  // Ensure elements are available
  this.ensureElements();
  const binder = window.appState.currentResearchBinder;
  if (!binder) return;

  const totalLoaded = binder.papers?.length || 0;
  const totalCount = window.appState.totalCount || 0;
  const container = this.elements.researchChatContainer;

  // Find or create pagination info element (in this view's own container)
  let infoElement = container ? container.querySelector(".pagination-info") : null;
  if (!infoElement && container) {
    infoElement = document.createElement("div");
    infoElement.className = "pagination-info";
    container.appendChild(infoElement);
  }

  if (infoElement) {
    infoElement.innerHTML = `
      <div class="pagination-text">
        Showing ${totalLoaded} of ${totalCount.toLocaleString()} papers
        ${window.appState.nextCursor ? `(more available)` : "(all loaded)"}
      </div>
    `;
  }

  // Show or hide load more button based on nextCursor and loading state.
  // The button's click handler is attached once in core.js.
  if (this.elements.loadMoreContainer) {
    if (window.appState.nextCursor || window.appState.isLoadingMore) {
      this.elements.loadMoreContainer.style.display = "flex";
    } else {
      // Only hide when we know there are no more papers AND not loading
      this.elements.loadMoreContainer.style.display = "none";
    }
  }
};
