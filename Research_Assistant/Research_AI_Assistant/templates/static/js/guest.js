// Guest Mode, browser side.
//
// A guest session is a server-issued token (POST /api/guest/session/) kept in
// localStorage so every tab shares one allowance until it expires. The server
// is the only authority on quotas: the allowance stored here is just the last
// value the server reported (X-Guest-Allowance header) and is used for the
// indicator and to skip requests that would certainly be refused.
//
// Stored shape: { token, expires_at, allowance: { expires_at, limits, remaining } }

const STORAGE_KEY = "scholara:guest";
export const ALLOWANCE_EVENT = "scholara:guest-allowance";

// Server codes for guest-specific refusals (guest_auth.py / guest_quota.py).
export const GUEST_ENDED_CODES = new Set(["guest_expired", "guest_invalid", "guest_disabled"]);
const GUEST_LIMIT_CODES = new Set([
  "guest_quota_exhausted",
  "guest_capacity",
  "guest_paper_not_allowed",
  "guest_query_too_long",
  "guest_too_many_papers",
  "guest_invalid_request",
]);

// Server default for GUEST_MAX_SUMMARY_PAPERS; the server's own value (sent in
// every allowance payload) takes precedence. The server enforces it either way.
const DEFAULT_MAX_SUMMARY_PAPERS = 10;

function read() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (_err) {
    return null; // storage blocked or corrupt
  }
}

function write(value) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch (_err) {
    /* storage blocked: the session lasts only as long as this page */
  }
}

/** The current guest session, or null if there is none or it has expired. */
export function getGuestSession() {
  const s = read();
  if (!s || typeof s.token !== "string" || !s.token) return null;
  const expires = Date.parse(s.expires_at);
  if (!Number.isFinite(expires) || expires <= Date.now()) {
    clearGuestSession();
    return null;
  }
  return s;
}

export function isGuestActive() {
  return getGuestSession() !== null;
}

export function clearGuestSession() {
  let had = false;
  try {
    had = window.localStorage.getItem(STORAGE_KEY) !== null;
    window.localStorage.removeItem(STORAGE_KEY);
  } catch (_err) {
    /* nothing stored */
  }
  // Lets the guest UI (chip, Save label) switch back immediately.
  if (had) window.dispatchEvent(new CustomEvent(ALLOWANCE_EVENT, { detail: null }));
}

/** Papers a guest may put in one summary (and fetch per search page). */
export function maxSummaryPapers() {
  const n = getAllowance()?.max_summary_papers;
  return Number.isInteger(n) && n > 0 ? n : DEFAULT_MAX_SUMMARY_PAPERS;
}

/** Authorization header for a guest ({} when there is no guest session). */
export function guestAuthHeader() {
  const s = getGuestSession();
  return s ? { Authorization: `Guest ${s.token}` } : {};
}

export function getAllowance() {
  return getGuestSession()?.allowance || null;
}

/** Remaining units of one quota ("search" | "summary" | "question"), or null if unknown. */
export function remaining(kind) {
  const value = getAllowance()?.remaining?.[kind];
  return Number.isInteger(value) ? value : null;
}

function rememberAllowance(allowance) {
  const s = getGuestSession();
  if (!s || !allowance || typeof allowance !== "object") return;
  s.allowance = allowance;
  if (allowance.expires_at) s.expires_at = allowance.expires_at;
  write(s);
  window.dispatchEvent(new CustomEvent(ALLOWANCE_EVENT, { detail: allowance }));
}

/** Picks up the allowance the server attaches to guest responses. */
export function readAllowanceHeader(response) {
  const raw = response?.headers?.get?.("X-Guest-Allowance");
  if (!raw) return;
  try {
    rememberAllowance(JSON.parse(raw));
  } catch (_err) {
    /* malformed header: keep the previous value */
  }
}

/** Starts a guest session. Resolves with the session; rejects with a readable message. */
export async function startGuestSession() {
  let response;
  try {
    response = await fetch("/api/guest/session/", { method: "POST" });
  } catch (_networkErr) {
    throw new Error("Couldn't reach Scholara. Check your connection and try again.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || !data?.token) {
    throw new Error(data?.error || `Couldn't start a guest session (HTTP ${response.status}).`);
  }
  const session = { token: data.token, expires_at: data.expires_at, allowance: data.allowance || null };
  write(session);
  return session;
}

/** Refreshes the stored allowance from the server (one cheap call, no AI). */
export async function refreshAllowance() {
  if (!isGuestActive()) return null;
  try {
    const response = await fetch("/api/guest/status/", { headers: guestAuthHeader() });
    if (!response.ok) return null;
    const data = await response.json();
    rememberAllowance(data);
    return data;
  } catch (_err) {
    return null;
  }
}

/**
 * Guest-specific refusal in an API error body: { code, message, quota } or null.
 * Lets callers show a friendly "sign in" message instead of a generic error.
 */
export function guestErrorFrom(body) {
  if (!body || typeof body !== "object") return null;
  const code = body.code;
  if (!GUEST_LIMIT_CODES.has(code) && !GUEST_ENDED_CODES.has(code)) return null;
  return { code, message: body.error || "This isn't available in guest mode. Sign in for full access.", quota: body.quota || null };
}

export function exhaustedMessage(kind) {
  const limit = getAllowance()?.limits?.[kind];
  const n = Number.isInteger(limit) ? `all ${limit} ` : "all your ";
  if (kind === "search") {
    return `You've used ${n}guest searches (Load More counts as a search). Sign in to keep researching. Your Recents will still be here.`;
  }
  if (kind === "summary") return `You've used ${n}guest summaries. Sign in to get AI summaries.`;
  return `You've used ${n}guest questions. Sign in to keep asking about these papers.`;
}

export function formatExpiry(iso) {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  try {
    return new Date(t).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch (_err) {
    return "";
  }
}
