import { loadClerk } from "./auth.js";
import { createLogger } from "./logger.js";
import { clearGuestSession, getGuestSession, startGuestSession } from "./guest.js";

const log = createLogger("sign-in");
const errorBox = document.getElementById("siError");
const noticeBox = document.getElementById("siNotice");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function showNotice(message) {
  if (!noticeBox) return;
  noticeBox.textContent = message;
  noticeBox.hidden = false;
}

log.info("sign-in page starting");
const params = new URLSearchParams(location.search);
if (params.get("error") === "config") {
  showError("We couldn't start sign-in. Please try again; if it keeps happening, the site is not configured yet.");
}
if (params.get("guest") === "ended") {
  showNotice("Your guest session has ended. Sign in to keep researching. Your recent research in this tab is still there.");
}

// Guest Mode button (rendered only when GUEST_MODE_ENABLED). A guest who is
// already active (e.g. came here to sign in) can go back to the workspace.
const guestBtn = document.getElementById("siGuestBtn");
if (guestBtn) {
  if (getGuestSession()) guestBtn.textContent = "Back to guest workspace";
  guestBtn.addEventListener("click", async () => {
    if (getGuestSession()) {
      window.location.replace("/workspace/");
      return;
    }
    guestBtn.disabled = true;
    guestBtn.textContent = "Starting guest session…";
    try {
      await startGuestSession();
      window.location.replace("/workspace/");
    } catch (err) {
      log.warn("guest session failed:", err);
      showError(err.message);
      guestBtn.disabled = false;
      guestBtn.textContent = "Continue as guest";
    }
  });
}

loadClerk()
  .then((clerk) => {
    if (clerk.user) {
      log.info("already signed in; going to workspace");
      clearGuestSession(); // Guest Mode: a signed-in account replaces any guest session
      window.location.replace("/workspace/");
      return;
    }
    // Guest Mode: the moment Clerk signs someone in here, drop the old guest
    // credential so the workspace never treats them as a guest. Temporary
    // Recents live in sessionStorage and are unaffected.
    clerk.addListener(({ user }) => {
      if (user) clearGuestSession();
    });
    log.info("mounting sign-in component");
    clerk.mountSignIn(document.getElementById("clerk-sign-in"), {
      forceRedirectUrl: "/workspace/",
      signUpForceRedirectUrl: "/workspace/",
    });
  })
  .catch((err) => {
    log.error("sign-in failed to start:", err);
    showError(err.message);
  });
