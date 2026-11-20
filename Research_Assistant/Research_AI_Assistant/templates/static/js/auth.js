// Clerk sign-in gate + token helper.
//
// Guest Mode (guest.js): a visitor with an unexpired guest session passes the
// gate and their API calls carry "Authorization: Guest <token>". A signed-in
// Clerk user always takes precedence (and clears any guest session).
import { createLogger } from "./logger.js";
import {
  GUEST_ENDED_CODES,
  clearGuestSession,
  guestAuthHeader,
  isGuestActive,
  readAllowanceHeader,
} from "./guest.js";

const log = createLogger("auth");
let clerkPromise = null;

function readPublishableKey() {
  const meta = document.querySelector('meta[name="clerk-publishable-key"]');
  return meta ? meta.content : "";
}

// pk_test_<base64(frontend-api-host + "$")> -> frontend API host
function frontendApiHost(key) {
  try {
    return atob(key.split("_").slice(2).join("_")).replace(/\$$/, "");
  } catch (err) {
    return "";
  }
}

export function loadClerk() {
  if (clerkPromise) return clerkPromise;
  clerkPromise = new Promise((resolve, reject) => {
    const key = readPublishableKey();
    if (!key) {
      reject(new Error("Sign-in is not configured (missing Clerk publishable key)."));
      return;
    }
    const host = frontendApiHost(key);
    if (!host) {
      reject(new Error("Sign-in is misconfigured (invalid Clerk publishable key)."));
      return;
    }
    log.info("loading Clerk SDK");
    const script = document.createElement("script");
    script.async = true;
    script.crossOrigin = "anonymous";
    script.setAttribute("data-clerk-publishable-key", key);
    script.src = `https://${host}/npm/@clerk/clerk-js@5/dist/clerk.browser.js`;
    script.onload = async () => {
      try {
        await window.Clerk.load();
        log.info("Clerk loaded; signed in:", Boolean(window.Clerk.user));
        resolve(window.Clerk);
      } catch (err) {
        log.error("Clerk failed to initialise:", err);
        reject(new Error("Could not start sign-in. Check your connection and Clerk settings."));
      }
    };
    script.onerror = () => {
      log.error("Clerk SDK script failed to load");
      reject(new Error("Could not load the sign-in service. Check your connection and try again."));
    };
    document.head.appendChild(script);
  });
  return clerkPromise;
}

/** Redirects to /sign-in/ when signed out. Resolves with Clerk when signed in. */
export async function requireSignIn() {
  // No publishable key: fail open; the API still enforces auth server-side.
  if (!readPublishableKey()) {
    log.error(
      "CLERK_PUBLISHABLE_KEY is not set - sign-in gate disabled. " +
        "Add it to Research_Assistant/.env to turn sign-in on."
    );
    return null;
  }
  try {
    const clerk = await loadClerk();
    if (!clerk.user) {
      if (isGuestActive()) {
        log.info("guest session active; continuing as guest");
        return null;
      }
      log.info("not signed in; redirecting to /sign-in/");
      window.location.replace("/sign-in/");
      return null;
    }
    clearGuestSession(); // signed in: full access replaces the guest session
    return clerk;
  } catch (err) {
    if (isGuestActive()) {
      log.warn("Clerk unavailable; continuing as guest:", err);
      return null;
    }
    log.error("sign-in gate failed:", err);
    window.location.replace("/sign-in/?error=config");
    return null;
  }
}

/**
 * Authorization header for API calls: the Clerk token when signed in, else the
 * guest token when a guest session is active, else {}. Never logs the token.
 */
export async function getAuthHeaders() {
  try {
    const clerk = await loadClerk();
    const token = clerk.session ? await clerk.session.getToken() : null;
    return token ? { Authorization: `Bearer ${token}` } : guestAuthHeader();
  } catch (err) {
    if (isGuestActive()) return guestAuthHeader();
    log.warn("could not get auth token:", err);
    return {};
  }
}

export async function signOut() {
  clearGuestSession();
  const clerk = await loadClerk();
  await clerk.signOut();
  window.location.replace("/sign-in/");
}

// Resolves a fetch input to a URL; returns it only for same-origin /api/ paths.
function sameOriginApiUrl(input) {
  let raw = "";
  if (typeof input === "string") raw = input;
  else if (input instanceof URL) raw = input.href;
  else if (input instanceof Request) raw = input.url;
  if (!raw) return null;
  let url;
  try {
    url = new URL(raw, window.location.href);
  } catch (err) {
    return null;
  }
  if (url.origin !== window.location.origin) return null;
  return url.pathname.startsWith("/api/") ? url : null;
}

// Wraps fetch: attaches the Clerk token to same-origin /api/ calls and logs them.
export function installFetchInterceptor() {
  if (window.__scholaraFetchPatched) return;
  window.__scholaraFetchPatched = true;
  const httpLog = createLogger("http");
  const originalFetch = window.fetch.bind(window);

  window.fetch = async (input, init) => {
    const apiUrl = sameOriginApiUrl(input);
    // Never attach the token to another origin or a non-API path.
    if (!apiUrl) return originalFetch(input, init);

    const opts = init || {};
    const isRequest = input instanceof Request;
    const rawUrl = apiUrl.pathname + apiUrl.search;
    const method = (opts.method || (isRequest && input.method) || "GET").toUpperCase();
    // fetch(request, { headers }) replaces the Request's headers, so merge them first.
    const headers = new Headers(isRequest ? input.headers : undefined);
    new Headers(opts.headers || {}).forEach((v, k) => headers.set(k, v));
    if (!headers.has("Authorization")) {
      const authHeaders = await getAuthHeaders();
      Object.entries(authHeaders).forEach(([k, v]) => headers.set(k, v));
    }
    const usedGuest = /^guest\s/i.test(headers.get("Authorization") || "");
    const started = performance.now();
    httpLog.info(`${method} ${rawUrl} started`);
    try {
      const response = await originalFetch(input, { ...opts, headers });
      const ms = Math.round(performance.now() - started);
      if (usedGuest) readAllowanceHeader(response);
      if (response.ok) {
        httpLog.info(`${method} ${rawUrl} -> ${response.status} (${ms} ms)`);
      } else {
        httpLog.warn(`${method} ${rawUrl} -> ${response.status} ${response.statusText} (${ms} ms)`);
        // Only 401 means "not signed in"; a 403 (including guest quota
        // limits) is an authorization error and stays on the page.
        if (response.status === 401) {
          if (usedGuest) {
            // A stale guest credential sent by someone who is in fact signed
            // in with Clerk: discard it and stay on the page -- never send a
            // signed-in user back to sign-in because of an old guest token.
            const clerk = await loadClerk().catch(() => null);
            if (clerk && clerk.user) {
              clearGuestSession();
              httpLog.info("discarded a stale guest credential; signed in with Clerk");
              return response;
            }
            // Only an ended guest session is discarded; a guest calling a
            // sign-in-only endpoint keeps their session for after sign-in.
            const body = await response.clone().json().catch(() => null);
            const ended = body && GUEST_ENDED_CODES.has(body.code);
            if (ended) clearGuestSession();
            httpLog.warn("guest session cannot continue; redirecting to sign-in");
            window.location.replace(ended ? "/sign-in/?guest=ended" : "/sign-in/");
          } else {
            httpLog.warn("session expired or missing; redirecting to sign-in");
            window.location.replace("/sign-in/");
          }
        }
      }
      return response;
    } catch (err) {
      if (err && err.name === "AbortError") {
        httpLog.info(`${method} ${rawUrl} aborted`);
      } else {
        httpLog.error(`${method} ${rawUrl} failed before a response arrived:`, err);
      }
      throw err;
    }
  };
}
