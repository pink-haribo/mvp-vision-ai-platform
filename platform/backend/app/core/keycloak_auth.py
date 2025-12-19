"""
Keycloak OIDC Token Validation Module

Keycloak에서 발급한 JWT Access Token을 검증합니다.
RS256 알고리즘과 JWKS를 사용하여 토큰 서명을 검증합니다.

Usage:
    from app.core.keycloak_auth import verify_keycloak_token

    payload = await verify_keycloak_token(token)
    print(payload.email, payload.sub)
"""

import httpx
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from jose.exceptions import JWKError
from pydantic import BaseModel

from app.core.config import settings


class KeycloakTokenPayload(BaseModel):
    """Keycloak JWT Access Token 페이로드"""

    # 표준 OIDC claims
    sub: str  # Subject (Keycloak User ID, UUID 형식)
    iss: str  # Issuer
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp

    # Audience - 단일 문자열 또는 리스트
    aud: str | list[str] = ""

    # Keycloak specific
    azp: Optional[str] = None  # Authorized party (client_id)
    typ: Optional[str] = None  # Token type (Bearer)
    session_state: Optional[str] = None

    # User info
    email: Optional[str] = None
    email_verified: bool = False
    preferred_username: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    name: Optional[str] = None

    # Roles (Keycloak 구조)
    realm_access: dict = {}  # {"roles": ["admin", "user"]}
    resource_access: dict = {}  # {"client-id": {"roles": [...]}}

    # Custom attributes (Keycloak User Attributes에서 매핑된 경우)
    department: Optional[str] = None
    company: Optional[str] = None

    class Config:
        extra = "ignore"  # 알 수 없는 필드 무시


class KeycloakJWKS:
    """
    Keycloak JWKS 관리 클래스

    - JWKS를 메모리에 캐싱 (1시간)
    - 키 로테이션 시 자동 갱신
    """

    def __init__(self, jwks_url: str, cache_duration: timedelta = timedelta(hours=1)):
        self._jwks_url = jwks_url
        self._jwks: Optional[dict] = None
        self._jwks_expires: Optional[datetime] = None
        self._cache_duration = cache_duration

    async def get_jwks(self) -> dict:
        """JWKS 가져오기 (캐시 우선)"""
        now = datetime.utcnow()

        # 캐시가 유효하면 캐시 반환
        if self._jwks and self._jwks_expires and now < self._jwks_expires:
            return self._jwks

        # JWKS 새로 가져오기
        async with httpx.AsyncClient() as client:
            response = await client.get(self._jwks_url, timeout=10.0)
            response.raise_for_status()
            self._jwks = response.json()
            self._jwks_expires = now + self._cache_duration

        return self._jwks

    def clear_cache(self):
        """캐시 초기화 (키 로테이션 시 호출)"""
        self._jwks = None
        self._jwks_expires = None

    def get_key_by_kid(self, jwks: dict, kid: str) -> Optional[dict]:
        """kid로 특정 키 찾기"""
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None


# 전역 JWKS 클라이언트 (싱글톤)
_jwks_client: Optional[KeycloakJWKS] = None


def get_jwks_client() -> KeycloakJWKS:
    """JWKS 클라이언트 싱글톤 반환"""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = KeycloakJWKS(settings.KEYCLOAK_JWKS_URL)
    return _jwks_client


async def verify_keycloak_token(token: str) -> KeycloakTokenPayload:
    """
    Keycloak Access Token 검증

    Args:
        token: Bearer 토큰 문자열 (Authorization 헤더에서 추출된 값)

    Returns:
        KeycloakTokenPayload: 검증된 토큰 페이로드

    Raises:
        JWTError: 토큰 검증 실패 시

    검증 항목:
        1. 서명 검증 (RS256, JWKS 사용)
        2. 만료 시간 (exp)
        3. Issuer (iss) - KEYCLOAK_ISSUER와 일치
        4. Audience (aud) - KEYCLOAK_CLIENT_ID 포함
    """
    jwks_client = get_jwks_client()

    try:
        # 1. 토큰 헤더에서 kid 추출
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise JWTError("Token header missing 'kid'")

        # 2. JWKS에서 해당 키 찾기
        jwks = await jwks_client.get_jwks()
        rsa_key = jwks_client.get_key_by_kid(jwks, kid)

        if not rsa_key:
            # 키가 없으면 캐시 갱신 후 재시도 (키 로테이션 대응)
            jwks_client.clear_cache()
            jwks = await jwks_client.get_jwks()
            rsa_key = jwks_client.get_key_by_kid(jwks, kid)

        if not rsa_key:
            raise JWTError(f"Unable to find key with kid: {kid}")

        # 3. 토큰 검증
        # audience 검증: azp(authorized party) 또는 aud에 client_id가 있어야 함
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            issuer=settings.KEYCLOAK_ISSUER,
            options={
                "verify_aud": False,  # audience는 수동 검증
                "verify_iss": True,
                "verify_exp": True,
            }
        )

        # 4. Audience 수동 검증 (azp 또는 aud 확인)
        azp = payload.get("azp")
        aud = payload.get("aud", [])
        if isinstance(aud, str):
            aud = [aud]

        valid_audience = (
            azp == settings.KEYCLOAK_CLIENT_ID or
            settings.KEYCLOAK_CLIENT_ID in aud or
            "account" in aud  # Keycloak 기본 audience
        )

        if not valid_audience:
            raise JWTError(
                f"Invalid audience. Expected '{settings.KEYCLOAK_CLIENT_ID}', "
                f"got azp='{azp}', aud={aud}"
            )

        return KeycloakTokenPayload(**payload)

    except JWKError as e:
        raise JWTError(f"JWK error: {str(e)}")
    except httpx.ConnectError:
        raise JWTError(
            f"Cannot connect to Keycloak server at {settings.KEYCLOAK_SERVER_URL}. "
            "Please ensure Keycloak is running."
        )
    except httpx.HTTPError as e:
        raise JWTError(f"Failed to fetch JWKS from Keycloak: {str(e)}")
    except Exception as e:
        if isinstance(e, JWTError):
            raise
        raise JWTError(f"Token validation failed: {str(e)}")


def extract_realm_roles(payload: KeycloakTokenPayload) -> list[str]:
    """Realm 역할 추출"""
    return payload.realm_access.get("roles", [])


def extract_client_roles(payload: KeycloakTokenPayload, client_id: str) -> list[str]:
    """특정 Client의 역할 추출"""
    client_access = payload.resource_access.get(client_id, {})
    return client_access.get("roles", [])


def has_realm_role(payload: KeycloakTokenPayload, role: str) -> bool:
    """Realm 역할 보유 여부"""
    return role in extract_realm_roles(payload)


def has_client_role(payload: KeycloakTokenPayload, client_id: str, role: str) -> bool:
    """Client 역할 보유 여부"""
    return role in extract_client_roles(payload, client_id)


def map_keycloak_to_system_role(payload: KeycloakTokenPayload) -> str:
    """
    Keycloak Realm 역할을 시스템 역할로 매핑

    Keycloak Realm Role → System Role 매핑:
        admin → ADMIN
        manager → MANAGER
        advanced_engineer → ADVANCED_ENGINEER
        standard_engineer → STANDARD_ENGINEER
        (default) → GUEST
    """
    from app.db.models import UserRole

    realm_roles = extract_realm_roles(payload)

    if "admin" in realm_roles:
        return UserRole.ADMIN
    elif "manager" in realm_roles:
        return UserRole.MANAGER
    elif "advanced_engineer" in realm_roles:
        return UserRole.ADVANCED_ENGINEER
    elif "standard_engineer" in realm_roles:
        return UserRole.STANDARD_ENGINEER
    else:
        return UserRole.GUEST
