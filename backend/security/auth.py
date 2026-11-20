"""Optional demo key or individually provisioned reviewer credentials."""
import hashlib
import hmac
import json
import os
from pathlib import Path
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from backend.config import SECURITY_CONFIG

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def reviewer_identity(key):
    filename = os.environ.get("CHESTXPERT_REVIEWERS_FILE")
    if not filename:
        return None
    try:
        records = json.loads(Path(filename).read_text())
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError("Reviewer records must be an array of objects")
        digest = hashlib.sha256((key or "").encode()).hexdigest()
        for record in records:
            if hmac.compare_digest(digest, record["key_sha256"]):
                if not all(record.get(k) for k in ("id", "name", "license")):
                    raise ValueError("Incomplete reviewer")
                return {k: record[k] for k in ("id", "name", "license")}
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(503, "Reviewer credentials are misconfigured.")
    return None

def get_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    if os.environ.get("CHESTXPERT_REVIEWERS_FILE"):
        if reviewer_identity(api_key) is None:
            raise HTTPException(401, "Invalid or missing reviewer key.")
        return api_key
    if not SECURITY_CONFIG.require_auth:
        return None
    expected = SECURITY_CONFIG.api_key
    if not expected:
        raise HTTPException(503, "Authentication is required but no API key is configured on the server.")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(401, "Invalid or missing API key.")
    return api_key
