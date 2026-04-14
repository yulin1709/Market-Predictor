"""
data/auth.py — S&P Global authentication and request helpers.

Auth endpoint: POST https://api.ci.spglobal.com/auth/api
Body: username + password (form-encoded)
Returns: access_token (Bearer), refresh_token

Auto-refreshes token when expired or on 401.
"""
import os
import time
import requests
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(APP_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

AUTH_URL = "https://api.ci.spglobal.com/auth/api"
REFRESH_URL = "https://api.ci.spglobal.com/auth/api/refresh"
TOKEN_TTL = 3300  # refresh after 55 min (token valid 1 hour)

_access_token: str = ""
_refresh_token: str = os.getenv("SPGLOBAL_REFRESH_TOKEN", "")
_token_fetched_at: float = 0.0


def get_token() -> str:
    """Return a valid access token, refreshing if needed."""
    global _access_token, _refresh_token, _token_fetched_at

    # Still valid
    if _access_token and (time.time() - _token_fetched_at) < TOKEN_TTL:
        return _access_token

    # Try refresh token first (faster, no password needed)
    if _refresh_token:
        try:
            resp = requests.post(
                REFRESH_URL,
                headers={"accept": "application/json",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"refresh_token": _refresh_token},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                _access_token = data["access_token"]
                _refresh_token = data.get("refresh_token", _refresh_token)
                _token_fetched_at = time.time()
                print("  [auth] Token refreshed via refresh_token.")
                return _access_token
        except Exception:
            pass  # fall through to full login

    # Full login with username/password
    username = os.getenv("SPGLOBAL_USERNAME", "")
    password = os.getenv("SPGLOBAL_PASSWORD", "")
    if not username or not password:
        raise EnvironmentError(
            "Set SPGLOBAL_REFRESH_TOKEN or both SPGLOBAL_USERNAME and SPGLOBAL_PASSWORD in .env"
        )

    print("  [auth] Authenticating with S&P Global...")
    try:
        resp = requests.post(
            AUTH_URL,
            headers={"accept": "application/json",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"username": username, "password": password},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _access_token = data["access_token"]
        _refresh_token = data.get("refresh_token", "")
        _token_fetched_at = time.time()
        print("  [auth] Authentication successful.")
        return _access_token
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Auth failed ({e.response.status_code}): {e.response.text[:300]}")
    except Exception as e:
        raise RuntimeError(f"Auth error: {e}")


def get_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        "accept": "application/json",
    }


def api_get_response(
    url: str,
    params: dict | None = None,
    timeout: int = 30,
    retries: int = 1,
    accept: str = "application/json",
) -> requests.Response:
    """GET with automatic 401 token refresh and retry, returning the raw response."""
    global _access_token, _token_fetched_at
    for attempt in range(retries + 1):
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {get_token()}",
                "accept": accept,
            },
            params=params,
            timeout=timeout,
        )
        if resp.status_code == 401 and attempt < retries:
            _access_token = ""
            _token_fetched_at = 0.0
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"API GET {url} failed after retry exhaustion.")


def api_get(url: str, params: dict = None, retries: int = 1) -> dict:
    """GET with automatic 401 token refresh and retry."""
    try:
        return api_get_response(url, params=params, timeout=30, retries=retries).json()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"API GET {url} failed ({e.response.status_code}): {e.response.text[:300]}"
        )
