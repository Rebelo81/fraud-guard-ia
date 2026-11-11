from dataclasses import dataclass
from typing import Literal

import jwt
from fastapi import Depends, Header, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import AppError

RoleType = Literal["analyst", "admin"]

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject: str
    role: RoleType


def _resolve_principal(
    credentials: HTTPAuthorizationCredentials | None,
    x_api_key: str | None,
    settings: Settings,
) -> AuthenticatedPrincipal | None:
    token: str | None = None
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token:
        # In local_private development mode, allow analyst access when no token is passed,
        # while NEVER allowing anonymous admin access (PRD Section 3).
        if settings.API_AUTH_MODE == "local_private":
            return AuthenticatedPrincipal(subject="local-analyst", role="analyst")
        return None

    # Check explicit static tokens first
    if token == settings.ADMIN_API_TOKEN:
        return AuthenticatedPrincipal(subject="admin-token-user", role="admin")
    if token == settings.ANALYST_API_TOKEN:
        return AuthenticatedPrincipal(subject="analyst-token-user", role="analyst")

    # Check JWT token
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        role = payload.get("role", "analyst")
        sub = str(payload.get("sub", "jwt-user"))
        if role not in ("analyst", "admin"):
            raise AppError(
                code="FORBIDDEN",
                message="Invalid role in token.",
                status_code=status.HTTP_403_FORBIDDEN,
            )
        return AuthenticatedPrincipal(subject=sub, role=role)
    except jwt.PyJWTError:
        return None


def require_analyst(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedPrincipal:
    # If caller explicitly provided a token that is invalid, reject with 401
    explicit_token = (credentials and credentials.credentials) or x_api_key
    principal = _resolve_principal(credentials, x_api_key, settings)
    if principal is None or (
        explicit_token
        and principal.subject == "local-analyst"
        and explicit_token not in (settings.ANALYST_API_TOKEN, settings.ADMIN_API_TOKEN)
    ):
        raise AppError(
            code="UNAUTHORIZED",
            message="Valid authentication credentials are required.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    return principal


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedPrincipal:
    """PRD Section 3: Never expose administrative operations anonymously."""
    explicit_token = (credentials and credentials.credentials) or x_api_key
    if not explicit_token:
        raise AppError(
            code="UNAUTHORIZED",
            message="Administrative credentials are required.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    principal = _resolve_principal(credentials, x_api_key, settings)
    if principal is None or principal.subject == "local-analyst":
        raise AppError(
            code="UNAUTHORIZED",
            message="Invalid administrative credentials.",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    if principal.role != "admin":
        raise AppError(
            code="FORBIDDEN",
            message="Administrator role is required for this operation.",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return principal

