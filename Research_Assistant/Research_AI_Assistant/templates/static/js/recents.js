import { DOMManager } from './core.js';
import { isGuestActive, maxSummaryPapers } from './guest.js';

// ---------------------------------------------------------------------
// Recents: temporary research sessions for this browser tab.
//
// Every search creates one session. A session owns:
//   - binder: the object research.js / binders.js already treat as
//     window.appState.currentResearchBinder ({ id, name, color, messages,
//     papers }) -- same shape as before, so Save to Binder copies exactly
//     what it always did. Arrays on it are only ever replaced (concat),
//     never mutated in place, so a binder saved from a session never
//     changes when the session continues afterwards.
//   - its own search params, pagination cursor, UI state (scroll, sort,
//     load-more status) and any Paper Analysis follow-up turns.
//   - container: its own rendered chat element. Switching sessions swaps
//     that element in and out of the research view instead of rebuilding
//     it, and in-flight requests write into their own session's element,
//     so a late response can never land in a different topic.
//
// Storage: an in-memory Map is the source of truth while the page is
// open; a debounced copy goes to sessionStorage so a refresh (or a full
// reload when returning from Paper Analysis) can restore it. Nothing is
// sent to the server and nothing touches Binders.
// ---------------------------------------------------------------------

const STORAGE_KEY = "scholara:recents";
const STORAGE_VERSION = 1;
const ANALYSIS_SESSION_KEY = "scholara:analysisSession"; // written by research.js
const MAX_SESSIONS = 10;
const MAX_ANALYSIS_TURNS = 20; // matches analysis.js MAX_TURNS
const PERSIST_DELAY_MS = 300;
const HASH_PREFIX = "#research=";

const sessions = new Map(); // id -> session
let persistTimer = null;
let storageState = "ok"; // "ok" | "partial" | "unavailable"
let historyIndex = 0;
// Set once the page starts unloading (reload, or navigating to Paper
// Analysis): requests the browser aborts then are interruptions, not errors.
let pageUnloading = false;
// True when in-memory sessions differ from the last sessionStorage write.
let dirty = false;

const LOADING_HTML = `
        <div class="research-loading" role="status">
            <span class="research-loading__label">
                <span class="research-loading__pulse" aria-hidden="true"></span>
                Finding papers and drafting a summary&hellip;
            </span>
            <span class="research-loading__lines" aria-hidden="true"><span></span><span></span><span></span></span>
        </div>
    `;

// ==================== STORE ====================

function newSessionId() {
  // "temp-" prefix: research.js / paper.js treat temp- ids as unsaved research.
  return "temp-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
}

function sortedSessions() {
  return [...sessions.values()].sort((a, b) => b.lastAccessed - a.lastAccessed);
}

function isQuotaError(err) {
  return (
    !!err &&
    (err.name === "QuotaExceededError" ||
      err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      err.code === 22 ||
      err.code === 1014)
  );
}

function getStorage() {
  try {
    return window.sessionStorage;
  } catch (_err) {
    return null; // blocked (e.g. some privacy modes)
  }
}

function toStored(s, withTurns) {
  return {
    id: s.id,
    query: s.query,
    createdAt: s.createdAt,
    lastAccessed: s.lastAccessed,
    searchParams: s.searchParams,
    searched: s.searched,
    summaryPaperCount: s.summaryPaperCount,
    nextCursor: s.nextCursor,
    totalCount: s.totalCount,
    currentFilter: s.currentFilter,
    status: s.status,
    errorMessage: s.errorMessage,
    errorGuest: !!s.errorGuest,
    loadMoreStatus: s.loadMoreStatus,
    scrollY: s.scrollY,
    refScrollTop: s.refScrollTop,
    refsExpanded: s.refsExpanded,
    lastOpenedPaperId: s.lastOpenedPaperId,
    analysisTurns: withTurns ? s.analysisTurns : [],
    binder: {
      id: s.binder.id,
      name: s.binder.name,
      color: s.binder.color,
      messages: s.binder.messages,
      papers: s.binder.papers,
    },
  };
}

function fromStored(raw) {
  if (!raw || typeof raw !== "object" || typeof raw.id !== "string") return null;
  const b = raw.binder;
  if (!b || !Array.isArray(b.messages) || !Array.isArray(b.papers)) return null;
  if (!raw.searchParams || typeof raw.searchParams.query !== "string") return null;

  // A request can't survive a page unload: anything still loading when the
  // page went away is shown as interrupted, with an explicit retry -- never
  // silently re-run.
  const status = raw.status === "loading" ? "interrupted" : raw.status || "ready";

  return {
    id: raw.id,
    query: String(raw.query || raw.searchParams.query),
    createdAt: Number(raw.createdAt) || Date.now(),
    lastAccessed: Number(raw.lastAccessed) || Date.now(),
    searchParams: raw.searchParams,
    searched: !!raw.searched,
    summaryPaperCount: Number(raw.summaryPaperCount) || 0,
    nextCursor: raw.nextCursor || null,
    totalCount: raw.totalCount ?? null,
    currentFilter: raw.currentFilter || "relevance",
    status,
    errorMessage: raw.errorMessage || "",
    errorGuest: !!raw.errorGuest,
    loadMoreStatus: raw.loadMoreStatus || "",
    scrollY: Number(raw.scrollY) || 0,
    refScrollTop: Number(raw.refScrollTop) || 0,
    refsExpanded: !!raw.refsExpanded,
    lastOpenedPaperId: raw.lastOpenedPaperId || null,
    analysisTurns: Array.isArray(raw.analysisTurns) ? raw.analysisTurns : [],
    binder: { id: b.id || raw.id, name: b.name, color: b.color, messages: b.messages, papers: b.papers },
    container: null,
    pending: { search: false, loadMore: false },
  };
}

function loadStoredSessions() {
  const storage = getStorage();
  if (!storage) {
    storageState = "unavailable";
    return;
  }
  let parsed = null;
  try {
    parsed = JSON.parse(storage.getItem(STORAGE_KEY) || "null");
  } catch (_err) {
    parsed = null; // corrupt entry: start clean rather than fail
  }
  if (!parsed || parsed.v !== STORAGE_VERSION || !Array.isArray(parsed.sessions)) return;
  parsed.sessions.slice(0, MAX_SESSIONS).forEach((raw) => {
    const s = fromStored(raw);
    if (s && !sessions.has(s.id)) sessions.set(s.id, s);
  });
}

// Writes the store to sessionStorage. On a quota error it first drops the
// (largest, least essential) Paper Analysis transcripts, then the oldest
// sessions, until the write fits. The in-memory copies are untouched either
// way -- only refresh-recovery is reduced -- and the Recents panel says so.
function persistNow(onlyId) {
  clearTimeout(persistTimer);
  persistTimer = null;
  // Nothing changed since the last write: skip re-serializing every session
  // (pagehide, visibilitychange and the pre-navigation flush all land here).
  if (!dirty && !onlyId) return storageState !== "unavailable";
  dirty = false;

  const storage = getStorage();
  if (!storage) {
    storageState = "unavailable";
    return false;
  }

  const ordered = onlyId ? [sessions.get(onlyId)].filter(Boolean) : sortedSessions();
  let payload = ordered.map((s) => toStored(s, true));
  let reduced = !!onlyId && sessions.size > ordered.length;
  let strippedTurns = false;

  for (;;) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({ v: STORAGE_VERSION, sessions: payload }));
      storageState = reduced ? "partial" : "ok";
      return true;
    } catch (err) {
      if (!isQuotaError(err)) {
        storageState = "unavailable";
        return false;
      }
      if (!strippedTurns) {
        strippedTurns = true;
        if (payload.some((p) => p.analysisTurns.length)) {
          payload = payload.map((p) => ({ ...p, analysisTurns: [] }));
          reduced = true;
          continue;
        }
      }
      if (payload.length <= 1) {
        try {
          storage.removeItem(STORAGE_KEY);
        } catch (_e) {
          /* nothing more to free */
        }
        storageState = "unavailable";
        return false;
      }
      payload.pop(); // oldest last
      reduced = true;
    }
  }
}

function schedulePersist() {
  dirty = true;
  if (persistTimer) return;
  persistTimer = setTimeout(() => persistNow(), PERSIST_DELAY_MS);
}

// Pulls the follow-up transcript written by the Paper Analysis page back
// into the session that opened it, so it is carried into the next visit
// instead of being lost when another session opens analysis.
function importAnalysisTurns() {
  const storage = getStorage();
  if (!storage) return;
  let parsed = null;
  try {
    parsed = JSON.parse(storage.getItem(ANALYSIS_SESSION_KEY) || "null");
  } catch (_err) {
    return;
  }
  // Only handoffs made from a Recents session carry recentsSessionId; a
  // Binder's handoff (same researchId as its source session) is ignored.
  if (!parsed || typeof parsed.recentsSessionId !== "string" || !Array.isArray(parsed.turns)) return;
  const s = sessions.get(parsed.recentsSessionId);
  if (!s) return;
  s.analysisTurns = parsed.turns.filter((t) => t && t.id && t.question).slice(-MAX_ANALYSIS_TURNS);
  schedulePersist();
}

function evictOverflow() {
  if (sessions.size <= MAX_SESSIONS) return;
  const activeId = window.appState.activeSessionId;
  const candidates = sortedSessions()
    .reverse()
    .filter((s) => s.id !== activeId && !s.pending.search && !s.pending.loadMore);
  while (sessions.size > MAX_SESSIONS && candidates.length) {
    sessions.delete(candidates.shift().id);
  }
}

// ==================== HISTORY ====================

function historyState(view, sessionId) {
  return { scholara: 1, view, sessionId: sessionId || null, idx: historyIndex };
}

function urlFor(view, sessionId) {
  const base = location.pathname + location.search;
  return view === "research" ? base + HASH_PREFIX + encodeURIComponent(sessionId) : base;
}

function pushHistory(view, sessionId) {
  historyIndex += 1;
  try {
    history.pushState(historyState(view, sessionId), "", urlFor(view, sessionId));
  } catch (_err) {
    /* history unavailable (sandboxed frame): in-app navigation still works */
  }
}

function replaceHistory(view, sessionId) {
  try {
    history.replaceState(historyState(view, sessionId), "", urlFor(view, sessionId));
  } catch (_err) {
    /* see pushHistory */
  }
}

function ownHistoryState() {
  const s = history.state;
  return s && s.scholara ? s : null;
}

function sessionIdFromLocation() {
  if (location.hash.startsWith(HASH_PREFIX)) {
    try {
      return decodeURIComponent(location.hash.slice(HASH_PREFIX.length));
    } catch (_err) {
      return null;
    }
  }
  const s = ownHistoryState();
  return s && s.view === "research" ? s.sessionId : null;
}

// ==================== FORMATTING ====================

function relativeTime(ts) {
  const diff = Math.round((ts - Date.now()) / 1000);
  const abs = Math.abs(diff);
  if (abs < 45) return "Just now";
  let rtf = null;
  try {
    rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  } catch (_err) {
    rtf = null;
  }
  const fmt = (v, unit) => (rtf ? rtf.format(v, unit) : `${Math.abs(v)} ${unit}s ago`);
  if (abs < 3600) return fmt(Math.round(diff / 60), "minute");
  if (abs < 86400) return fmt(Math.round(diff / 3600), "hour");
  return fmt(Math.round(diff / 86400), "day");
}

const STATUS_LABELS = {
  loading: "Working…",
  interrupted: "Interrupted",
  error: "Error",
};

// ==================== DOMManager API ====================

DOMManager.prototype.getResearchSession = function(id) {
  return (id && sessions.get(id)) || null;
};

// The Recents session that owns `binder`, or null. A Binder saved from a
// session keeps the session's id but is a separate copy, so identity (not
// just id) decides: Binder views keep their original behaviour.
DOMManager.prototype.sessionForBinder = function(binder) {
  const s = binder ? this.getResearchSession(binder.id) : null;
  return s && s.binder === binder ? s : null;
};

DOMManager.prototype.getActiveResearchSession = function() {
  const s = this.getResearchSession(window.appState.activeSessionId);
  return s && this.ownsResearchView(s.binder) ? s : null;
};

// True while `binder` is the one on screen. Async work captures its binder
// at start and checks this before touching shared UI.
DOMManager.prototype.ownsResearchView = function(binder) {
  return !!binder && window.appState.isResearchView && window.appState.currentResearchBinder === binder;
};

DOMManager.prototype.saveResearchSessions = function() {
  schedulePersist();
};

DOMManager.prototype.flushResearchSessions = function() {
  this.captureActiveSessionUi();
  persistNow();
};

// Frees sessionStorage space for the Paper Analysis handoff by keeping only
// the given session's recovery copy. In-memory Recents are unaffected.
DOMManager.prototype.releaseRecentsStorage = function(keepId) {
  persistNow(keepId || "__none__");
};

DOMManager.prototype.isStorageQuotaError = isQuotaError;

DOMManager.prototype.isPageUnloading = function() {
  return pageUnloading;
};

DOMManager.prototype.readSearchParams = function(query) {
  const minYear = this.elements.yearMin?.value || null;
  const maxYear = this.elements.yearMax?.value || this.elements.yearFilter?.value || null;
  let perPage = parseInt(this.elements.quota?.value || "5");
  // Guest Mode: a guest summary may hold at most maxSummaryPapers() papers
  // (enforced by the server), so a guest search page is sized to match.
  if (isGuestActive()) perPage = Math.min(perPage, maxSummaryPapers());
  return {
    query,
    mode: this.elements.searchBy?.value || "best_match",
    minYear,
    maxYear,
    perPage,
  };
};

DOMManager.prototype.createResearchSession = function(query, searchParams) {
  const now = Date.now();
  const id = newSessionId();
  const session = {
    id,
    query,
    createdAt: now,
    lastAccessed: now,
    searchParams,
    searched: false,
    summaryPaperCount: 0,
    nextCursor: null,
    totalCount: null,
    currentFilter: window.appState.currentFilter || "relevance",
    status: "loading",
    errorMessage: "",
    loadMoreStatus: "",
    scrollY: 0,
    refScrollTop: 0,
    refsExpanded: false,
    lastOpenedPaperId: null,
    analysisTurns: [],
    // Same shape showResearchView has always created for the temporary binder.
    binder: {
      id,
      name: query.length > 35 ? query.substring(0, 32) + "..." : query,
      color: "#" + Math.floor(Math.random() * 16777215).toString(16),
      messages: [{ role: "user", content: query }],
      papers: [],
    },
    container: null,
    pending: { search: false, loadMore: false },
  };
  session.container = this.buildSessionContainer(session);
  sessions.set(id, session);
  evictOverflow();
  schedulePersist();
  return session;
};

DOMManager.prototype.createChatContainer = function() {
  const el = document.createElement("div");
  el.className = "research-chat-container";
  el.setAttribute("aria-live", "polite");
  return el;
};

// Rebuilds a session's chat from its stored messages. Only used when there is
// no live DOM for it (first creation, or after a page reload); never calls
// the backend.
DOMManager.prototype.buildSessionContainer = function(session) {
  const el = this.createChatContainer();
  const binder = session.binder;
  let assistantIndex = 0;

  binder.messages.forEach((msg) => {
    if (msg.role === "user") {
      this.addMessage(msg.content, true, el);
      return;
    }
    if (assistantIndex > 0) {
      const label = document.createElement("div");
      label.textContent = "Summary of Additional Papers";
      label.className = "summary-label";
      el.appendChild(label);
    }
    const div = this.addMessage("", false, el);
    if (session.status === "empty" && assistantIndex === 0) {
      div.textContent = msg.content;
    } else {
      div.innerHTML = this.markdownToHtml(msg.content);
      this.addPaperDetails(div, binder.papers);
    }
    div.style.opacity = "1";
    assistantIndex += 1;
  });

  if (session.status === "error") {
    const message = session.errorMessage || "Unknown error";
    this.renderSessionNotice(session, this.addMessage("", false, el), session.errorGuest ? message : "Error: " + message);
  } else if (session.status === "interrupted") {
    this.renderSessionNotice(
      session,
      this.addMessage("", false, el),
      session.searched
        ? "The summary was interrupted when the page was left or reloaded. The papers below were kept."
        : "This search was interrupted when the page was left or reloaded.",
    );
  }
  return el;
};

// Error / interrupted message with an explicit retry. Retrying never repeats
// a search that already returned papers -- it only re-requests the summary.
DOMManager.prototype.renderSessionNotice = function(session, div, text) {
  if (!div) return;
  div.classList.add("message--error");
  div.style.opacity = "1";
  div.textContent = text;

  // Guest Mode limit: a retry would be refused again, so offer sign-in.
  // Recents survive sign-in (same tab); once signed in (no guest session)
  // the same notice offers the normal retry instead.
  if (session.errorGuest && isGuestActive()) {
    const link = document.createElement("a");
    link.className = "research-retry-btn";
    link.href = "/sign-in/";
    link.textContent = "Sign in";
    div.appendChild(link);
    return;
  }

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "research-retry-btn";
  btn.textContent = session.searched ? "Retry summary" : "Search again";
  btn.addEventListener("click", () => {
    if (session.pending.search || !this.ownsResearchView(session.binder)) return;
    div.remove();
    this.generateResearchResponse();
  });
  div.appendChild(btn);
};

DOMManager.prototype.loadingMessageHtml = function() {
  return LOADING_HTML;
};

// Puts `container` into the research view in place of the current one. The
// active container always carries id="researchChatContainer".
DOMManager.prototype.mountChatContainer = function(container) {
  const current = this.elements.researchChatContainer || document.getElementById("researchChatContainer");
  if (current === container) return;
  if (current) current.removeAttribute("id");
  container.id = "researchChatContainer";
  if (current && current.parentNode) {
    current.replaceWith(container);
  } else {
    const host = this.elements.researchView?.querySelector(".research-main-content");
    if (host) host.prepend(container);
  }
  this.elements.researchChatContainer = container;
};

// Copies the live view state that isn't otherwise tracked (scroll, mobile
// references toggle) into the active session.
DOMManager.prototype.captureActiveSessionUi = function() {
  const s = this.getActiveResearchSession();
  if (!s) return;
  const scrollY = window.scrollY || window.pageYOffset || 0;
  const refScrollTop = this.elements.referencesList ? this.elements.referencesList.scrollTop : s.refScrollTop;
  const refsExpanded = this.elements.referencesPanel
    ? this.elements.referencesPanel.classList.contains("is-expanded")
    : s.refsExpanded;
  if (scrollY === s.scrollY && refScrollTop === s.refScrollTop && refsExpanded === s.refsExpanded) return;
  s.scrollY = scrollY;
  s.refScrollTop = refScrollTop;
  s.refsExpanded = refsExpanded;
  schedulePersist();
};

// Mirrors a session's flags and pagination into window.appState and the
// shared controls -- only when that session is the one on screen.
DOMManager.prototype.syncResearchUi = function(session) {
  if (!session || !this.ownsResearchView(session.binder)) return;
  const st = window.appState;
  const els = this.elements;

  st.isThinking = session.pending.search;
  st.isGeneratingResponse = session.pending.search;
  st.isLoadingMore = session.pending.loadMore;
  st.nextCursor = session.nextCursor;
  st.totalCount = session.totalCount;
  st.searchParams = session.searchParams;

  if (els.researchSendBtn) els.researchSendBtn.disabled = session.pending.search;
  this.setReferencesFilterVisibility(!session.pending.search);
  if (els.saveToBinderBtn) els.saveToBinderBtn.style.display = session.status === "ready" ? "block" : "none";
  if (els.loadMoreBtn) els.loadMoreBtn.disabled = session.pending.loadMore;
  if (els.loadMoreStatus) {
    els.loadMoreStatus.textContent = session.loadMoreStatus || (session.pending.loadMore ? "Loading more papers..." : "");
  }

  if (session.searched && !session.pending.search) {
    this.updatePaginationInfo();
  } else if (els.loadMoreContainer) {
    els.loadMoreContainer.style.display = "none";
  }
};

DOMManager.prototype.resetResearchFlags = function() {
  const st = window.appState;
  st.isThinking = false;
  st.isGeneratingResponse = false;
  st.isLoadingMore = false;
  if (this.elements.researchSendBtn) this.elements.researchSendBtn.disabled = false;
  if (this.elements.loadMoreBtn) this.elements.loadMoreBtn.disabled = false;
  if (this.elements.loadMoreStatus) this.elements.loadMoreStatus.textContent = "";
  if (this.elements.loadMoreContainer) this.elements.loadMoreContainer.style.display = "none";
  this.setReferencesFilterVisibility(true);
};

DOMManager.prototype.syncFilterButtons = function() {
  const active = window.appState.currentFilter || "relevance";
  document.querySelectorAll("#referencesPanel .filter-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.filter === active);
  });
};

// Shows a session in the research view without any network request.
DOMManager.prototype.renderSessionView = function(session, opts) {
  const options = opts || {};
  const els = this.elements;
  const previous = this.getActiveResearchSession();
  if (previous && previous !== session) this.captureActiveSessionUi();

  if (!session.container) session.container = this.buildSessionContainer(session);
  this.mountChatContainer(session.container);

  if (els.heroSection) els.heroSection.style.display = "none";
  if (els.bindersSection) els.bindersSection.style.display = "none";
  if (els.researchView) els.researchView.classList.add("show");
  if (els.researchQuery) els.researchQuery.textContent = session.query;

  window.appState.currentResearchBinder = session.binder;
  window.appState.isResearchView = true;
  window.appState.activeSessionId = session.id;
  window.appState.currentFilter = session.currentFilter || "relevance";
  this.syncFilterButtons();

  // Skip re-rendering the reference cards if they already show exactly
  // this papers array in this order (arrays are replaced, never mutated).
  const rendered = this._renderedReferences;
  if (!rendered || rendered.papers !== session.binder.papers || rendered.filter !== window.appState.currentFilter) {
    this.renderReferences(session.binder.papers);
  }

  if (els.researchInput && options.clearInput !== false) els.researchInput.value = "";

  session.lastAccessed = Date.now();
  this.syncResearchUi(session);
  schedulePersist();

  if (options.restoreScroll) this.restoreSessionScroll(session);
};

// Runs after research-view.js's open handler (a microtask that scrolls to
// the top and collapses references), once layout is ready.
DOMManager.prototype.restoreSessionScroll = function(session) {
  const y = session.scrollY || 0;
  const refTop = session.refScrollTop || 0;
  const expanded = !!session.refsExpanded;
  const paperId = session.lastOpenedPaperId;
  requestAnimationFrame(() => {
    if (!this.ownsResearchView(session.binder)) return;
    if (this.elements.researchView) {
      this.elements.researchView.dispatchEvent(
        new CustomEvent("research:restore-ui", { detail: { referencesExpanded: expanded } }),
      );
    }
    const list = this.elements.referencesList;
    if (list) list.scrollTop = refTop;
    window.scrollTo(0, y);
    if (paperId && list) {
      const card = [...list.querySelectorAll(".source-card")].find((c) => c.dataset.paperId === paperId);
      if (card) card.focus({ preventScroll: true });
    }
  });
};

DOMManager.prototype.renderHomeView = function() {
  const els = this.elements;
  this.captureActiveSessionUi();

  // Detach the session's preserved chat so nothing (e.g. opening a Binder,
  // which clears and refills the active container) can overwrite it.
  this.mountChatContainer(this.createChatContainer());

  if (els.researchView) els.researchView.classList.remove("show");
  if (els.heroSection) els.heroSection.style.display = "flex";
  if (els.bindersSection) els.bindersSection.style.display = "block";

  window.appState.isResearchView = false;
  window.appState.currentResearchBinder = null;
  window.appState.activeSessionId = null;
  this.resetResearchFlags();

  if (els.researchInput) els.researchInput.value = "";

  this.renderBinders();
  this.renderRecents();
  window.scrollTo(0, 0);
};

// ---- Navigation entry points ----

// New search from the home view.
DOMManager.prototype.startResearchSession = function(query) {
  const session = this.createResearchSession(query, this.readSearchParams(query));
  this.renderSessionView(session, { restoreScroll: false });
  pushHistory("research", session.id);
  return session;
};

DOMManager.prototype.openResearchSession = function(id) {
  const session = this.getResearchSession(id);
  if (!session) {
    this.renderRecents();
    return;
  }
  this.renderSessionView(session, { restoreScroll: true });
  const st = ownHistoryState();
  if (st && st.view === "research" && st.sessionId === id) replaceHistory("research", id);
  else pushHistory("research", id);
};

// "Go Home": back to Ask Scholara; the session stays in Recents.
DOMManager.prototype.goHome = function() {
  const wasResearch = window.appState.isResearchView;
  this.renderHomeView();
  const st = ownHistoryState();
  if (wasResearch && !(st && st.view === "home")) pushHistory("home");
  if (this.elements.queryInput) this.elements.queryInput.focus({ preventScroll: true });
};

// "Back": the previous in-app view (a previous session or home), via browser
// history so the browser's own Back button behaves the same way.
DOMManager.prototype.goBackFromResearch = function() {
  const st = ownHistoryState();
  if (st && st.view === "research" && st.idx > 0) {
    this.captureActiveSessionUi();
    history.back();
  } else {
    this.goHome();
  }
};

DOMManager.prototype.handleResearchPopState = function(event) {
  const st = event.state && event.state.scholara ? event.state : null;
  if (st && Number.isInteger(st.idx)) historyIndex = st.idx;
  const id = st ? (st.view === "research" ? st.sessionId : null) : sessionIdFromLocation();
  const session = this.getResearchSession(id);
  if (session) {
    if (!this.ownsResearchView(session.binder)) this.renderSessionView(session, { restoreScroll: true });
    if (!st) replaceHistory("research", session.id);
    return;
  }
  if (window.appState.isResearchView) this.renderHomeView();
  else this.renderRecents();
  if (!st || id) replaceHistory("home");
};

// ==================== RECENTS PANEL ====================

DOMManager.prototype.renderRecents = function() {
  const list = this.elements.recentsList;
  if (!list) return;
  const items = sortedSessions();

  if (this.elements.recentsCount) {
    this.elements.recentsCount.textContent = items.length + (items.length === 1 ? " session" : " sessions");
  }

  list.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("li");
    empty.className = "recents-empty";
    empty.textContent = "No recent research yet. Searches you run in this tab will appear here.";
    list.appendChild(empty);
  }

  items.forEach((s) => {
    const li = document.createElement("li");
    li.className = "recent-item";
    li.dataset.sessionId = s.id;

    const open = document.createElement("button");
    open.type = "button";
    open.className = "recent-item__open";
    open.dataset.action = "open";
    open.setAttribute("aria-label", "Reopen research: " + s.query);

    const icon = document.createElement("span");
    icon.className = "recent-item__icon";
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = '<i class="fa-solid fa-clock-rotate-left"></i>';

    const body = document.createElement("span");
    body.className = "recent-item__body";

    const topic = document.createElement("span");
    topic.className = "recent-item__topic";
    topic.textContent = s.query;

    const meta = document.createElement("span");
    meta.className = "recent-item__meta";
    const time = document.createElement("time");
    time.dateTime = new Date(s.lastAccessed).toISOString();
    time.title = new Date(s.lastAccessed).toLocaleString();
    time.textContent = relativeTime(s.lastAccessed);
    const n = s.binder.papers.length;
    const parts = [n + (n === 1 ? " paper" : " papers")];
    const turns = s.analysisTurns.length;
    if (turns) parts.push(turns + (turns === 1 ? " follow-up" : " follow-ups"));
    meta.append(time, document.createTextNode(" · " + parts.join(" · ")));

    body.append(topic, meta);

    const statusKey = s.pending.search ? "loading" : s.status;
    if (STATUS_LABELS[statusKey]) {
      const badge = document.createElement("span");
      badge.className = "recent-item__status recent-item__status--" + statusKey;
      badge.textContent = STATUS_LABELS[statusKey];
      body.appendChild(badge);
    }

    const action = document.createElement("span");
    action.className = "recent-item__action";
    action.setAttribute("aria-hidden", "true");
    action.innerHTML = '<span>Reopen</span><i class="fa-solid fa-arrow-right"></i>';

    open.append(icon, body, action);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "recent-item__remove";
    remove.dataset.action = "remove";
    remove.setAttribute("aria-label", "Remove from Recents: " + s.query);
    remove.title = "Remove from Recents";
    remove.innerHTML = '<i class="fa-solid fa-xmark" aria-hidden="true"></i>';

    li.append(open, remove);
    list.appendChild(li);
  });

  const note = this.elements.recentsStorageNote;
  if (note) {
    if (storageState === "ok") {
      note.hidden = true;
    } else {
      note.hidden = false;
      note.textContent =
        storageState === "partial"
          ? "Browser storage is full: some recent sessions won't survive a page refresh."
          : "Browser storage is unavailable: Recents will be lost if this page is refreshed.";
    }
  }
};

DOMManager.prototype.removeResearchSession = function(id) {
  if (window.appState.activeSessionId === id) return;
  sessions.delete(id);
  dirty = true;
  persistNow();
  this.renderRecents();
  // Keep keyboard focus in the list rather than dropping it to <body>.
  const next = this.elements.recentsList?.querySelector(".recent-item__open");
  if (next) next.focus();
  else if (this.elements.queryInput) this.elements.queryInput.focus({ preventScroll: true });
};

DOMManager.prototype.setupRecentsList = function() {
  const list = this.elements.recentsList;
  if (!list || list.dataset.wired) return;
  list.dataset.wired = "1";
  list.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    const item = e.target.closest(".recent-item");
    if (!btn || !item) return;
    if (btn.dataset.action === "open") this.openResearchSession(item.dataset.sessionId);
    else if (btn.dataset.action === "remove") this.removeResearchSession(item.dataset.sessionId);
  });
};

// ==================== INIT ====================

DOMManager.prototype.initResearchSessions = function() {
  loadStoredSessions();
  importAnalysisTurns();
  this.setupRecentsList();

  try {
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  } catch (_err) {
    /* not supported */
  }

  const st = ownHistoryState();
  historyIndex = st && Number.isInteger(st.idx) ? st.idx : 0;

  const session = this.getResearchSession(sessionIdFromLocation());
  if (session) {
    this.renderSessionView(session, { restoreScroll: true });
    replaceHistory("research", session.id);
  } else {
    this.renderRecents();
    replaceHistory("home");
  }

  window.addEventListener("popstate", (e) => this.handleResearchPopState(e));
  window.addEventListener("beforeunload", () => {
    pageUnloading = true;
  });
  window.addEventListener("pagehide", () => {
    pageUnloading = true;
    this.flushResearchSessions();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") this.flushResearchSessions();
  });
  // Back/forward cache restore: the DOM is intact; just pick up any follow-up
  // turns asked on the analysis page meanwhile.
  window.addEventListener("pageshow", (e) => {
    if (!e.persisted) return;
    pageUnloading = false;
    importAnalysisTurns();
    if (!window.appState.isResearchView) this.renderRecents();
  });
};
