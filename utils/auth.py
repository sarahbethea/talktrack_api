from fastapi import Header, HTTPException, Request, Depends # Header lets you grab HTTP headers from the request, HTTPException lets you reject a request with an error

API_KEYS = {
    "demo-key-123": {"user_id":"demo_user1", "plan": "free"},
    "demo-key-456": {"user_id":"demo_user2", "plan": "premium"},
}  # Example API key set, replace with your actual keys

async def verify_api_key( # This function is a fastAPI dependence that runs before a route executes, tries to get the API key from two headers: x-api-key (the custom header) and Authorization: Bearer
    x_api_key: str = Header(None),
    authorization: str = Header(None),
):
    key = x_api_key

    # Support Bearer token format
    if not key and authorization and authorization.startswith("Bearer "): # "if x-api-key was not provided, check the authorization header and pull the key from it"
        key = authorization.split(" ")[1] # splits up the header by spaces, and grabs the second token in Bearer demo-123, which is what we need

    if not key or key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return API_KEYS[key] # Return associated user info so your route can use it 