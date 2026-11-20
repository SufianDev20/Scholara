"""
Client IP for Guest Mode limits (guest-session issuing and its throttle).

X-Forwarded-For is written by whoever sent the request, so it is only used
when the TCP peer (REMOTE_ADDR) is a proxy we trust, listed in
settings.GUEST_TRUSTED_PROXIES (IP addresses and/or CIDR ranges). Then the
header is read right to left -- each trusted proxy appends the address it
received the request from -- and the first address that is not a trusted
proxy is the client. Anything to the left of it was supplied by the client
and is ignored.

Fail-safe defaults:
  - GUEST_TRUSTED_PROXIES empty (the default) -> REMOTE_ADDR only; any
    X-Forwarded-For header is ignored.
  - REMOTE_ADDR not a trusted proxy -> REMOTE_ADDR only.
  - A malformed entry in the header -> stop there and use the last
    address a trusted proxy vouched for (never the malformed value).

DRF's NUM_PROXIES is deliberately not used: it counts hops without
checking that the request really came through those proxies.
"""

import ipaddress
import logging

logger = logging.getLogger(__name__)


def parse_trusted_proxies(entries):
    """IP networks from a list of addresses/CIDRs. Raises ValueError on a bad entry."""
    networks = []
    for entry in entries or ():
        text = str(entry).strip()
        if text:
            networks.append(ipaddress.ip_network(text, strict=False))
    return tuple(networks)


def _ip(value):
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None


def _is_trusted(ip, networks):
    return ip is not None and any(ip in net for net in networks)


def resolve_client_ip(remote_addr, forwarded_for, trusted_networks):
    """The client address for one request (see module docstring)."""
    remote = _ip(remote_addr or "")
    if remote is None:
        return str(remote_addr or "")
    if not trusted_networks or not _is_trusted(remote, trusted_networks):
        return str(remote)

    client = remote
    hops = [hop for hop in (forwarded_for or "").split(",") if hop.strip()]
    for hop in reversed(hops):
        ip = _ip(hop)
        if ip is None:
            break  # malformed: keep the last address a trusted proxy vouched for
        client = ip
        if not _is_trusted(ip, trusted_networks):
            break  # first untrusted address from the right is the client
    return str(client)


def guest_client_ip(request):
    """Client IP for a Django/DRF request, using settings.GUEST_TRUSTED_PROXIES."""
    from django.conf import settings

    try:
        networks = parse_trusted_proxies(getattr(settings, "GUEST_TRUSTED_PROXIES", ()))
    except ValueError as exc:
        # Misconfiguration must not open the limits up: trust no proxy.
        logger.error("Invalid GUEST_TRUSTED_PROXIES (%s); ignoring X-Forwarded-For.", exc)
        networks = ()
    meta = request.META
    return resolve_client_ip(
        meta.get("REMOTE_ADDR", ""), meta.get("HTTP_X_FORWARDED_FOR", ""), networks
    )
