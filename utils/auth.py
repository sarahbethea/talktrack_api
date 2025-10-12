"""
Lightweight auth dependency for FastAPI (MVP/demo).

Responsibilities:
- Accept API credentials via `x-api-key` or `Authorization: Bearer <key>`.
- Validate against in-memory `API_KEYS` and return a small user context.
- On failure, raise HTTP 401 so protected routes never execute.

Notes:
This is a demo module. Before production:
- Replace in-memory keys with a DB or secret manager (with rotation/expiry).
- Serve over HTTPS; avoid logging full secrets.
- Consider JWTs for self-contained claims (exp, sub, scopes).
- Enforce plan-based limits in routes using the injected user context.
"""
from fastapi import (
    Header, 
    HTTPException, 
    Request, 
    Depends
) 
from typing import Any

# Demo API keys and associated user info for testing.
# In production, use a secure database or key management system.
API_KEYS = {
    "demo-key-123": {"user_id":"demo_user1", "plan": "free"},
    "demo-key-456": {"user_id":"demo_user2", "plan": "premium"},
}  

async def verify_api_key( 
    x_api_key: str = Header(None),
    authorization: str = Header(None),
) -> dict[str, Any]:
    '''
    FastAPI dependency to verify API key from headers.

    Accepts either:
        - x-api-key: <key>
        - Authorization: Bearer <key>
    
    Returns:
        The associated user payload (e.g., {"user_id": "...", "plan": "..."}).
    
    Raises:
        HTTPException(401) if missing/invalid.
    '''
    key = x_api_key

    # Support Bearer token format
    # Example: "Authorization: Bearer demo-key-123"
    if not key and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            key = token.strip()

    if not key or key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    # Return user context so routes can branch on plan/limits
    return API_KEYS[key] 