"""
Guest Mode authentication.

A guest holds a token issued by views_guest.issue_guest_session: a random
guest_id signed with SECRET_KEY (TimestampSigner, dedicated salt) that
expires after settings.GUEST_SESSION_HOURS. The browser sends it as

    Authorization: Guest <token>

GuestAuthentication is attached ONLY to the views that guests may use
(search, summarise, multi_paper_qa), listed BEFORE ClerkAuthentication:

  - no "Guest" header  -> returns None, so ClerkAuthentication runs exactly
                          as it always has;
  - "Guest" header     -> the token is verified here and Clerk is skipped.

Every other endpoint keeps Clerk only, so a guest token there gets a 401.
Guests are never stored as Users and never set the Postgres RLS context.
"""

import secrets
from datetime import timedelta

from django.conf import settings
from django.core import signing
from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from .models import GuestAllowance

TOKEN_SALT = "scholara.guest-session.v1"
AUTH_SCHEME = "guest"


class GuestUser:
    """Request principal for a guest. Not a Django User.

    `pk` is unique per guest so the existing per-user rate limits
    (throttles.py, keyed on request.user.pk) give every guest its own bucket.
    """

    is_authenticated = True
    is_anonymous = False
    is_active = True
    is_staff = False
    is_superuser = False
    is_guest = True

    def __init__(self, allowance):
        self.allowance = allowance
        self.guest_id = allowance.guest_id
        self.pk = self.id = f"guest-{allowance.guest_id}"

    def __str__(self):
        return self.pk

    def get_username(self):
        return self.pk

    def has_perm(self, perm, obj=None):
        return False

    def has_perms(self, perm_list, obj=None):
        return False

    def has_module_perms(self, app_label):
        return False


def session_lifetime():
    return timedelta(hours=settings.GUEST_SESSION_HOURS)


def new_guest_id():
    return secrets.token_hex(16)


def issue_guest_token(guest_id):
    return signing.TimestampSigner(salt=TOKEN_SALT).sign(guest_id)


def _fail(code, message):
    # Dict detail -> response body {"error": ..., "code": ...}, matching the
    # {"error": ...} shape the frontend already reads.
    raise AuthenticationFailed({"error": message, "code": code})


class GuestAuthentication(BaseAuthentication):
    def authenticate(self, request):
        header = request.META.get("HTTP_AUTHORIZATION", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != AUTH_SCHEME:
            return None  # not a guest request: let ClerkAuthentication handle it

        if not settings.GUEST_MODE_ENABLED:
            _fail("guest_disabled", "Guest access is turned off right now. Please sign in to continue.")

        token = token.strip()
        if not token:
            _fail("guest_invalid", "This guest session isn't valid. Start a new one or sign in.")

        try:
            guest_id = signing.TimestampSigner(salt=TOKEN_SALT).unsign(
                token, max_age=session_lifetime()
            )
        except signing.SignatureExpired:
            _fail("guest_expired", "Your guest session has ended. Sign in to keep researching.")
        except signing.BadSignature:
            _fail("guest_invalid", "This guest session isn't valid. Start a new one or sign in.")

        allowance = GuestAllowance.objects.filter(guest_id=guest_id).first()
        if allowance is None:
            _fail("guest_invalid", "This guest session isn't valid. Start a new one or sign in.")
        if allowance.expires_at <= timezone.now():
            _fail("guest_expired", "Your guest session has ended. Sign in to keep researching.")

        return (GuestUser(allowance), allowance)

    def authenticate_header(self, request):
        # DRF takes WWW-Authenticate from the FIRST authenticator on a view.
        # Guest is listed first on the guarded views, so return the same value
        # ClerkAuthentication does: responses to signed-in/out Clerk users on
        # those views stay byte-for-byte what they were (401, "Bearer").
        return "Bearer"
