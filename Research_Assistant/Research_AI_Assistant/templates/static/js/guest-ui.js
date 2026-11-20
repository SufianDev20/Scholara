// Guest Mode presentation for the workspace (loaded by base.html).
// Shows the remaining-allowance chip in the navbar and relabels Save to
// Binder, which needs an account.
//
// Nothing guest-specific is shown until Clerk has confirmed the visitor is
// NOT signed in: someone who just signed in may still have the old guest
// credential in storage, which is then discarded instead (normal Binder
// saving, no chip, no guest requests).
import { loadClerk } from "./auth.js";
import {
  ALLOWANCE_EVENT,
  clearGuestSession,
  formatExpiry,
  getAllowance,
  getGuestSession,
  isGuestActive,
  refreshAllowance,
} from "./guest.js";

const chip = document.getElementById("guestChip");
const counts = document.getElementById("guestChipCounts");
// Narrow screens show this line instead of the chip (see styles.css).
const heroLine = document.getElementById("wsGuestAllowance");
const heroText = document.getElementById("wsGuestAllowanceText");
const saveBtn = document.getElementById("saveToBinderBtn");
const saveText = saveBtn ? saveBtn.querySelector("span") : null;
const SAVE_LABEL = saveText ? saveText.textContent : "";
const SAVE_TITLE = saveBtn ? saveBtn.getAttribute("title") : null;

let confirmedGuest = false;

const WORDS = {
  search: ["search", "searches"],
  summary: ["summary", "summaries"],
  question: ["question", "questions"],
};

function count(kind, n) {
  return `${n} ${WORDS[kind][n === 1 ? 0 : 1]}`;
}

function hide() {
  if (chip) chip.hidden = true;
  if (heroLine) heroLine.hidden = true;
}

function render() {
  if (!chip) return;
  if (!confirmedGuest || !isGuestActive()) {
    hide();
    restoreSave();
    return;
  }
  const left = getAllowance()?.remaining;
  const parts = left ? Object.keys(WORDS).map((kind) => count(kind, left[kind] ?? 0)) : [];
  if (counts) counts.textContent = parts.length ? parts.join(" · ") + " left" : "";
  if (heroLine && heroText) {
    heroText.textContent = "Guest access" + (parts.length ? ` · ${parts.join(" · ")} left` : "");
    heroLine.hidden = false;
  }
  const until = formatExpiry(getGuestSession()?.expires_at);
  const label =
    "Guest session" +
    (parts.length ? `: ${parts.join(", ")} left` : "") +
    (until ? `, until ${until}` : "") +
    ". Sign in for full access.";
  chip.setAttribute("aria-label", label);
  chip.title = label;
  chip.hidden = false;
  relabelSave();
}

function relabelSave() {
  if (!saveBtn) return;
  if (saveText) saveText.textContent = "Sign in to save";
  saveBtn.title = "Binders need an account. Sign in to save this research; it stays in Recents meanwhile.";
}

function restoreSave() {
  if (!saveBtn) return;
  if (saveText) saveText.textContent = SAVE_LABEL;
  if (SAVE_TITLE === null) saveBtn.removeAttribute("title");
  else saveBtn.title = SAVE_TITLE;
}

function startAsGuest() {
  confirmedGuest = true;
  render();
  refreshAllowance(); // re-sync with the server (no AI call)
}

if (isGuestActive()) {
  loadClerk()
    .then((clerk) => {
      if (clerk.user) {
        clearGuestSession(); // signed in: the guest credential is obsolete
        render();
      } else {
        startAsGuest();
      }
    })
    // No Clerk (not configured / failed to load): the gate already let this
    // guest in, so show the guest UI.
    .catch(startAsGuest);
}

window.addEventListener(ALLOWANCE_EVENT, render);
// Another tab used some of the shared allowance, or signed in / ended it.
window.addEventListener("storage", (e) => {
  if (e.key === "scholara:guest") render();
});
