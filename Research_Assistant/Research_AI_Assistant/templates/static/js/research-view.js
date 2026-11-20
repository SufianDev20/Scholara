// Presentation-only behaviour for the research conversation view
// (_research_view.html). Research state, fetching and rendering stay in
// core.js / research.js / binders.js; this module only watches the DOM
// they already produce:
//   - when the view opens: scroll to the top, collapse the mobile
//     references section, and move focus to the research title
//   - when it closes: return focus to the workspace search input
//   - keep the references count and the sort buttons' aria-pressed in sync
//   - expand/collapse the references section on narrow screens
(function () {
  "use strict";
  var view = document.getElementById("researchView");
  if (!view) return;

  var panel = document.getElementById("referencesPanel");
  var list = document.getElementById("referencesList");
  var count = document.getElementById("referencesCount");
  var toggle = document.getElementById("referencesToggle");
  var title = document.getElementById("researchQuery");

  function setExpanded(expanded) {
    if (!panel || !toggle) return;
    panel.classList.toggle("is-expanded", expanded);
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.setAttribute("aria-label", expanded ? "Hide references" : "Show references");
    var label = toggle.querySelector(".references-toggle__label");
    if (label) label.textContent = expanded ? "Hide" : "Show";
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      setExpanded(!panel.classList.contains("is-expanded"));
    });
  }

  // recents.js restores a session's mobile references toggle on Back/reopen
  // (dispatched after the open handler below has collapsed it).
  view.addEventListener("research:restore-ui", function (e) {
    if (e.detail && typeof e.detail.referencesExpanded === "boolean") {
      setExpanded(e.detail.referencesExpanded);
    }
  });

  var wasOpen = view.classList.contains("show");
  new MutationObserver(function () {
    var open = view.classList.contains("show");
    if (open === wasOpen) return;
    wasOpen = open;
    if (open) {
      window.scrollTo(0, 0);
      setExpanded(false);
      if (title) title.focus({ preventScroll: true });
    } else {
      var active = document.activeElement;
      if (!active || active === document.body || view.contains(active)) {
        var search = document.getElementById("queryInput");
        if (search) search.focus({ preventScroll: true });
      }
    }
  }).observe(view, { attributes: true, attributeFilter: ["class"] });

  if (list && count) {
    new MutationObserver(function () {
      var n = list.querySelectorAll(".source-card").length;
      count.textContent = n ? String(n) : "";
    }).observe(list, { childList: true });
  }

  if (panel) {
    panel.querySelectorAll(".filter-btn").forEach(function (btn) {
      function sync() {
        btn.setAttribute("aria-pressed", String(btn.classList.contains("active")));
      }
      sync();
      new MutationObserver(sync).observe(btn, { attributes: true, attributeFilter: ["class"] });
    });
  }
})();
