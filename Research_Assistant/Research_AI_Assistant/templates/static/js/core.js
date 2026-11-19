import { getCsrfToken } from './utils.js';
import { clearGuestSession, exhaustedMessage, isGuestActive, remaining } from './guest.js';
import { loadClerk } from './auth.js';

window.csrfToken = getCsrfToken();

// ==================== DOM MANAGER ====================
export class DOMManager {
  constructor() {
    this.elements = this.cacheElements();
    this.setupEventListeners();
  }

  cacheElements() {
    return {
      // Search elements
      queryInput: document.getElementById("queryInput"),
      searchButton: document.querySelector(".search-button"),

      // Filter elements
      yearFilter: document.getElementById("yearFilter"),
      yearValue: document.getElementById("yearValue"),
      sliderTooltip: document.getElementById("sliderTooltip"),
      yearMin: document.getElementById("yearMin"),
      yearMax: document.getElementById("yearMax"),
      yearError: document.getElementById("wsYearError"),
      searchBy: document.getElementById("searchBy"),
      quota: document.getElementById("quota"),

      // Advanced options panel (attached below the search input)
      advancedToggle: document.getElementById("wsAdvancedToggle"),
      advancedPanel: document.getElementById("wsAdvancedPanel"),
      filtersContainer: document.getElementById("wsFiltersContainer"),

      // Binder elements
      bindersContainer: document.getElementById("bindersContainer"),
      binderCount: document.getElementById("binderCount"),

      // Recents (temporary research sessions, see recents.js)
      recentsList: document.getElementById("recentsList"),
      recentsCount: document.getElementById("recentsCount"),
      recentsStorageNote: document.getElementById("recentsStorageNote"),

      // Research view elements
      researchView: document.getElementById("researchView"),
      researchQuery: document.getElementById("researchQuery"),
      researchChatContainer: document.getElementById("researchChatContainer"),
      researchChatArea: document.querySelector(".research-chat-area"),
      researchInput: document.getElementById("research-input"),
      researchSendBtn: document.getElementById("research-send-btn"),
      researchBackBtn: document.getElementById("researchBackBtn"),
      researchHomeBtn: document.getElementById("researchHomeBtn"),
      saveToBinderBtn: document.getElementById("saveToBinderBtn"),

      // References panel elements
      sourcesPanel: document.getElementById("sourcesPanel"),
      sourcesList: document.getElementById("sourcesList"),
      referencesPanel: document.getElementById("referencesPanel"),
      referencesList: document.getElementById("referencesList"),

      // Load more elements
      loadMoreContainer: document.getElementById("loadMoreContainer"),
      loadMoreBtn: document.getElementById("loadMoreBtn"),
      loadMoreStatus: document.getElementById("loadMoreStatus"),

      // Profile elements: the account menu (#profileMenu) is entirely
      // owned by profile.js, including its own outside-click/Escape
      // close handling — nothing cached or wired here.

      // Section elements
      heroSection: document.querySelector(".hero-section"),
      bindersSection: document.querySelector(".binders-section"),
    };
  }

  setupEventListeners() {
    var elements = this.elements;

    // Search functionality - button uses onclick="performSearch()" in HTML
    // No additional event listener needed to avoid conflicts

    if (elements.queryInput) {
      elements.queryInput.addEventListener(
        "keydown",
        function (e) {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            this.handleSearch();
          }
        }.bind(this),
      );
    }

    // Filter listeners
    if (elements.yearFilter) {
      elements.yearFilter.addEventListener(
        "input",
        function () {
          this.updateYearLabel();
        }.bind(this),
      );
      elements.yearFilter.addEventListener(
        "mousedown",
        function () {
          this.showTooltip();
        }.bind(this),
      );
      elements.yearFilter.addEventListener(
        "mouseup",
        function () {
          this.hideTooltip();
        }.bind(this),
      );
      elements.yearFilter.addEventListener(
        "touchstart",
        function () {
          this.showTooltip();
        }.bind(this),
      );
      elements.yearFilter.addEventListener(
        "touchend",
        function () {
          this.hideTooltip();
        }.bind(this),
      );
    }

    // Years: two typeable/steppable number inputs (min=1900, max=2026).
    // Validated on blur/change rather than every keystroke, so a user
    // can freely type "19" while still building "1990" without it
    // being clamped mid-entry.
    if (elements.yearMin && elements.yearMax) {
      const onYearChange = function () {
        this.validateYearRange();
      }.bind(this);
      elements.yearMin.addEventListener("change", onYearChange);
      elements.yearMax.addEventListener("change", onYearChange);
      elements.yearMin.addEventListener(
        "blur",
        function () {
          this.validateYearRange();
        }.bind(this),
      );
      elements.yearMax.addEventListener(
        "blur",
        function () {
          this.validateYearRange();
        }.bind(this),
      );

      // Clears a stale correction message once the user starts editing
      // a year field again, rather than leaving it up until the next
      // blur (which may be on the other, unrelated field).
      const clearYearError = function () {
        if (elements.yearError) elements.yearError.hidden = true;
      };
      elements.yearMin.addEventListener("input", clearYearError);
      elements.yearMax.addEventListener("input", clearYearError);
    }

    this.setupAdvancedOptionsToggle();

    // Research view listeners. Follow-up submissions open the paper
    // analysis page (see handleFollowUpQuestion in research.js) instead
    // of continuing an in-page chat.
    if (elements.researchSendBtn) {
      elements.researchSendBtn.addEventListener(
        "click",
        function () {
          this.handleFollowUpQuestion();
        }.bind(this),
      );
    }

    if (elements.researchInput) {
      elements.researchInput.addEventListener(
        "keydown",
        function (e) {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            this.handleFollowUpQuestion();
          }
        }.bind(this),
      );
    }

    // Back: previous in-app view (previous session or home), keeping all
    // research progress. Go Home: straight to Ask Scholara; the session
    // stays available in Recents. Both live in recents.js.
    if (elements.researchBackBtn) {
      elements.researchBackBtn.addEventListener(
        "click",
        function () {
          this.goBackFromResearch();
        }.bind(this),
      );
    }

    if (elements.researchHomeBtn) {
      elements.researchHomeBtn.addEventListener(
        "click",
        function () {
          this.goHome();
        }.bind(this),
      );
    }

    if (elements.saveToBinderBtn) {
      elements.saveToBinderBtn.addEventListener(
        "click",
        function () {
          // Guest Mode: binders need an account. Only once Clerk confirms
          // the visitor is NOT signed in do we send them to sign-in (the
          // research stays in Recents, same tab); a signed-in user with a
          // stale guest credential saves normally and the credential is
          // discarded.
          if (isGuestActive()) {
            loadClerk()
              .then((clerk) => Boolean(clerk && clerk.user))
              .catch(() => false)
              .then((signedIn) => {
                if (signedIn) {
                  clearGuestSession();
                  this.saveToBinder();
                  return;
                }
                this.flushResearchSessions();
                window.location.href = "/sign-in/";
              });
            return;
          }
          this.saveToBinder();
        }.bind(this),
      );
    }

    // Load more button listener
    if (elements.loadMoreBtn) {
      elements.loadMoreBtn.addEventListener(
        "click",
        function () {
          this.loadMorePapers();
        }.bind(this),
      );
    }

    // Global keyboard shortcuts
    document.addEventListener("keydown", function (e) {
      if (e.metaKey && e.key === "k") {
        e.preventDefault();
        if (!window.appState.isResearchView && elements.queryInput) {
          elements.queryInput.focus();
        } else if (window.appState.isResearchView && elements.researchInput) {
          elements.researchInput.focus();
        }
      }
    });
  }

  // ==================== EVENT HANDLERS ====================
  handleSearch() {
    var query = this.elements.queryInput
      ? this.elements.queryInput.value.trim()
      : "";
    console.log("handleSearch called with query:", query);

    if (!query) {
      console.log("Empty query - showing alert");
      alert("Type a research question!");
      return;
    }

    // Guest Mode: with no searches left, explain instead of opening an empty
    // session and sending a request the server would refuse.
    const guestNotice = document.getElementById("wsGuestNotice");
    if (isGuestActive() && remaining("search") === 0) {
      if (guestNotice) {
        guestNotice.textContent = exhaustedMessage("search");
        guestNotice.hidden = false;
      }
      return;
    }
    if (guestNotice) guestNotice.hidden = true;

    console.log("Elements available:", {
      queryInput: !!this.elements.queryInput,
      researchView: !!this.elements.researchView,
      heroSection: !!this.elements.heroSection,
      bindersSection: !!this.elements.bindersSection,
    });

    try {
      console.log("Calling showResearchView with query:", query);
      this.showResearchView(query);
      this.elements.queryInput.value = "";
    } catch (error) {
      console.error("Error in handleSearch:", error);
      alert("Failed to start research. Please try again.");
    }
  }

  // ==================== DOM MANIPULATION ====================
  // Starts a new research session for `query` (recents.js keeps it as a
  // temporary session -- never a Binder) and runs its search + summary.
  showResearchView(query) {
    if (!query) return;
    if (!this.elements.researchView) {
      console.warn("Research view element not found");
      return;
    }

    let session = null;
    try {
      session = this.startResearchSession(query);
      this.generateResearchResponse();
    } catch (error) {
      console.error("Error in showResearchView:", error);
      alert("Failed to start research. Please try again.");
      if (session) session.pending.search = false;
    }
  }

  // Leaves the research view for the Ask Scholara home view. Kept under its
  // original name because Save to Binder (paper.js) calls it. The session is
  // no longer discarded: it stays in Recents.
  hideResearchView() {
    this.goHome();
  }

  addMessage(content, isUser = false, container) {
    if (!container) return null;

    var div = document.createElement("div");
    div.className = "message " + (isUser ? "user" : "assistant");

    if (isUser) {
      div.textContent = content;
    } else {
      div.innerHTML =
        '<div class="thinking-indicator">' +
        '<i class="fa-solid fa-brain thinking-icon"></i>' +
        "<span>Generating response...</span>" +
        "</div>";
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }
}
