"""API-key authentication for the Chest Xpert backend.

Auth is optional (CHESTXPERT_REQUIRE_AUTH=false) for demos. When enabled, all
mutating endpoints require a valid X-API-Key header. Credentials are read from
the environment only — never hardcoded.
"""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.config import SECURITY_CONFIG

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Validate the API key when auth is required. Returns None when auth off."""
    if not SECURITY_CONFIG.require_auth:
        return None

    expected = SECURITY_CONFIG.api_key
    if not expected:
        # Auth required but no key configured — reject all requests.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is required but no API key is configured on the server.",
        )

    if api_key is None or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return api_key
