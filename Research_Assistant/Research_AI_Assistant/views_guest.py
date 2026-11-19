"""
Guest Mode endpoints.

    POST /api/guest/session/  -> start a guest session (no sign-in required)
    GET  /api/guest/status/   -> remaining allowance for the current guest

Kept out of views.py so guest code stays separate from the Clerk-protected
views. See guest_auth.py and guest_quota.py for how guests are checked.
"""

from datetime import timedelta

from django.conf import settings
from django.core.cache import caches
from django.http import JsonResponse
from django.utils import timezone
from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
    throttle_classes,
)
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.throttling import AnonRateThrottle

from .guest_auth import GuestAuthentication, issue_guest_token, new_guest_id, session_lifetime
from .guest_ip import guest_client_ip
from .guest_quota import allowance_payload, hash_ident, prune_expired
from .models import GuestAllowance


class GuestIssueThrottle(AnonRateThrottle):
    """Burst limit on starting guest sessions, per client IP.

    The client IP comes from guest_ip.guest_client_ip -- the same identity
    the per-IP daily cap below uses -- which ignores X-Forwarded-For unless
    the request came through a proxy in settings.GUEST_TRUSTED_PROXIES.
    (DRF's default get_ident() would trust a client-supplied header.)
    """

    cache = caches["throttle"]
    scope = "guest_issue"
    rate = "10/hour"

    def get_ident(self, request):
        return guest_client_ip(request)


@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
@throttle_classes([GuestIssueThrottle])
def issue_guest_session(request):
    if not settings.GUEST_MODE_ENABLED:
        return JsonResponse(
            {"error": "Guest access is turned off right now. Please sign in.", "code": "guest_disabled"},
            status=403,
        )

    prune_expired()

    now = timezone.now()
    ip_hash = hash_ident(guest_client_ip(request))
    started_today = GuestAllowance.objects.filter(
        issued_ip_hash=ip_hash, created_at__gte=now - timedelta(hours=24)
    ).count()
    if started_today >= settings.GUEST_SESSIONS_PER_IP_PER_DAY:
        return JsonResponse(
            {
                "error": (
                    "The guest session limit for today has been reached on this network. "
                    "Sign in to keep researching."
                ),
                "code": "guest_issue_limit",
            },
            status=429,
        )

    guest_id = new_guest_id()
    allowance = GuestAllowance.objects.create(
        guest_id=guest_id,
        expires_at=now + session_lifetime(),
        issued_ip_hash=ip_hash,
    )
    return JsonResponse(
        {
            "token": issue_guest_token(guest_id),
            "expires_at": allowance.expires_at.isoformat(),
            "allowance": allowance_payload(allowance),
        },
        status=201,
    )


@api_view(["GET"])
@authentication_classes([GuestAuthentication])
@permission_classes([IsAuthenticated])
def guest_status(request):
    return JsonResponse(allowance_payload(request.user.allowance))
