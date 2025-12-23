# Phase 2: Backend OIDC 통합 - 세부 구현 계획

Keycloak OIDC 토큰을 FastAPI Backend에서 검증하고 사용자 인증을 처리하는 상세 구현 계획입니다.

**작성일**: 2025-12-19
**예상 소요 시간**: 4-6시간

---

## 현재 상태 분석

### 기존 인증 구조

```
platform/backend/app/
├── core/
│   ├── config.py        # JWT_SECRET, JWT_ALGORITHM 설정
│   ├── security.py      # 자체 JWT 생성/검증 (HS256)
│   └── service_jwt.py   # 서비스간 SSO 토큰
├── utils/
│   └── dependencies.py  # get_current_user (OAuth2PasswordBearer)
└── db/
    └── models.py        # User 모델 (keycloak_id 없음)
```

### 변경 필요 사항

| 파일 | 변경 내용 |
|------|----------|
| `config.py` | Keycloak 관련 설정 추가 |
| `pyproject.toml` | `httpx` 의존성 추가 |
| `keycloak_auth.py` | **신규** - OIDC 토큰 검증 모듈 |
| `models.py` | User 모델에 `keycloak_id` 필드 추가 |
| `dependencies.py` | 듀얼 인증 지원 (자체 JWT + Keycloak) |

---

## Step 2.1: Config에 Keycloak 설정 추가

**파일**: `platform/backend/app/core/config.py`

### 추가할 설정

```python
class Settings(BaseSettings):
    # ... 기존 설정 ...

    # ========================================
    # Keycloak OIDC Configuration
    # ========================================

    # Keycloak 활성화 여부 (False면 기존 자체 JWT 사용)
    KEYCLOAK_ENABLED: bool = False

    # Keycloak 서버 URL (예: http://localhost:8080)
    KEYCLOAK_SERVER_URL: str = "http://localhost:8080"

    # Realm 이름
    KEYCLOAK_REALM: str = "vision-ai"

    # Backend Client ID (토큰 검증용)
    KEYCLOAK_CLIENT_ID: str = "platform-backend"

    # Client Secret (Confidential Client인 경우)
    KEYCLOAK_CLIENT_SECRET: Optional[str] = None

    # OIDC Endpoints (자동 계산)
    @property
    def KEYCLOAK_ISSUER(self) -> str:
        """OIDC Issuer URL"""
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}"

    @property
    def KEYCLOAK_OPENID_CONFIG_URL(self) -> str:
        """OIDC Discovery URL"""
        return f"{self.KEYCLOAK_ISSUER}/.well-known/openid-configuration"

    @property
    def KEYCLOAK_JWKS_URL(self) -> str:
        """JWKS (JSON Web Key Set) URL"""
        return f"{self.KEYCLOAK_ISSUER}/protocol/openid-connect/certs"

    @property
    def KEYCLOAK_TOKEN_URL(self) -> str:
        """Token Endpoint URL"""
        return f"{self.KEYCLOAK_ISSUER}/protocol/openid-connect/token"

    @property
    def KEYCLOAK_USERINFO_URL(self) -> str:
        """UserInfo Endpoint URL"""
        return f"{self.KEYCLOAK_ISSUER}/protocol/openid-connect/userinfo"
```

### 환경 변수 (.env)

```env
# Keycloak OIDC
KEYCLOAK_ENABLED=true
KEYCLOAK_SERVER_URL=http://localhost:8080
KEYCLOAK_REALM=vision-ai
KEYCLOAK_CLIENT_ID=platform-backend
KEYCLOAK_CLIENT_SECRET=your-client-secret-here
```

### 체크포인트

- [ ] 환경변수 설정 완료
- [ ] `settings.KEYCLOAK_ISSUER` 정상 출력 확인
- [ ] OIDC Discovery URL 접근 가능 확인

```bash
# 테스트
curl http://localhost:8080/realms/vision-ai/.well-known/openid-configuration | jq
```

---

## Step 2.2: httpx 의존성 추가

**파일**: `platform/backend/pyproject.toml`

### 현재 상태

```toml
dependencies = [
    # ...
    "python-jose[cryptography]>=3.3.0",  # 이미 있음
    # httpx는 dev 의존성에만 있음
]

[project.optional-dependencies]
dev = [
    "httpx>=0.25.0",  # 테스트용으로만 있음
]
```

### 변경 사항

```toml
dependencies = [
    # ...
    "python-jose[cryptography]>=3.3.0",
    "httpx>=0.27.0",  # OIDC Discovery 및 JWKS 가져오기용 (추가)
]
```

### 설치

```bash
cd platform/backend
pip install httpx>=0.27.0
# 또는
uv add httpx
```

### 체크포인트

- [ ] `pip list | grep httpx` 로 설치 확인
- [ ] Python에서 `import httpx` 성공

---

## Step 2.3: OIDC 토큰 검증 모듈 생성

**파일**: `platform/backend/app/core/keycloak_auth.py` (신규)

### 전체 구조

```python
"""
Keycloak OIDC Token Validation Module

주요 기능:
1. JWKS (JSON Web Key Set) 캐싱 및 갱신
2. Access Token 검증 (RS256)
3. 토큰 페이로드 파싱
4. 역할 추출 및 매핑
"""
```

### 2.3.1 TokenPayload 모델

```python
from pydantic import BaseModel
from typing import Optional

class KeycloakTokenPayload(BaseModel):
    """Keycloak JWT Access Token 페이로드"""

    # 표준 OIDC claims
    sub: str                          # Subject (Keycloak User ID, UUID)
    iss: str                          # Issuer
    aud: str | list[str]              # Audience
    exp: int                          # Expiration timestamp
    iat: int                          # Issued at timestamp

    # Keycloak specific
    azp: Optional[str] = None         # Authorized party (client_id)
    typ: Optional[str] = None         # Token type (Bearer)

    # User info
    email: Optional[str] = None
    email_verified: bool = False
    preferred_username: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    name: Optional[str] = None

    # Roles (Keycloak 구조)
    realm_access: dict = {}           # {"roles": ["admin", "user"]}
    resource_access: dict = {}        # {"client-id": {"roles": [...]}}

    # Custom attributes (Keycloak에서 설정한 경우)
    department: Optional[str] = None
    company: Optional[str] = None
```

### 2.3.2 JWKS 캐싱 클래스

```python
import httpx
from datetime import datetime, timedelta
from typing import Optional

class KeycloakJWKS:
    """
    Keycloak JWKS 관리 클래스

    - JWKS를 메모리에 캐싱
    - 1시간마다 자동 갱신
    - 키 로테이션 시 수동 갱신 가능
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
```

### 2.3.3 토큰 검증 함수

```python
from jose import jwt, JWTError
from jose.exceptions import JWKError
from app.core.config import settings

# 전역 JWKS 클라이언트 (앱 시작 시 초기화)
_jwks_client: Optional[KeycloakJWKS] = None

def get_jwks_client() -> KeycloakJWKS:
    """JWKS 클라이언트 싱글톤"""
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = KeycloakJWKS(settings.KEYCLOAK_JWKS_URL)
    return _jwks_client


async def verify_keycloak_token(token: str) -> KeycloakTokenPayload:
    """
    Keycloak Access Token 검증

    Args:
        token: Bearer 토큰 문자열 (Authorization 헤더에서 추출)

    Returns:
        KeycloakTokenPayload: 검증된 토큰 페이로드

    Raises:
        JWTError: 토큰 검증 실패 시

    검증 항목:
        1. 서명 검증 (RS256, JWKS 사용)
        2. 만료 시간 (exp)
        3. Issuer (iss)
        4. Audience (aud) - KEYCLOAK_CLIENT_ID와 일치
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
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=settings.KEYCLOAK_CLIENT_ID,
            issuer=settings.KEYCLOAK_ISSUER,
            options={
                "verify_aud": True,
                "verify_iss": True,
                "verify_exp": True,
            }
        )

        return KeycloakTokenPayload(**payload)

    except JWKError as e:
        raise JWTError(f"JWK error: {str(e)}")
    except Exception as e:
        raise JWTError(f"Token validation failed: {str(e)}")
```

### 2.3.4 역할 추출 함수

```python
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
```

### 체크포인트

- [ ] `keycloak_auth.py` 파일 생성
- [ ] Python import 테스트: `from app.core.keycloak_auth import verify_keycloak_token`
- [ ] 단위 테스트 작성 및 통과

---

## Step 2.4: User 모델에 keycloak_id 필드 추가

**파일**: `platform/backend/app/db/models.py`

### 변경 사항

```python
class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # ========================================
    # Keycloak SSO 연동 필드 (Step 2.4)
    # ========================================
    keycloak_id = Column(
        String(36),          # Keycloak UUID 형식
        unique=True,
        nullable=True,       # 기존 사용자는 null
        index=True
    )

    email = Column(String(255), unique=True, nullable=False, index=True)

    # Keycloak 사용 시 hashed_password는 null 가능
    hashed_password = Column(String(255), nullable=True)  # nullable=False → True로 변경

    full_name = Column(String(255), nullable=True)
    # ... 나머지 필드 동일 ...
```

### 주의사항

1. `keycloak_id`는 Keycloak의 User UUID (36자, 예: `550e8400-e29b-41d4-a716-446655440000`)
2. 기존 사용자의 `keycloak_id`는 처음에 null → Keycloak 로그인 시 이메일로 매칭하여 연결
3. `hashed_password`를 `nullable=True`로 변경 (Keycloak 전용 사용자는 비밀번호 없음)

### 체크포인트

- [ ] 모델 변경 완료
- [ ] 마이그레이션 파일 생성 필요 (Step 2.5)

---

## Step 2.5: DB 마이그레이션 생성 및 실행

**도구**: Alembic

### 마이그레이션 파일 생성

```bash
cd platform/backend
alembic revision -m "add_keycloak_id_to_users"
```

### 마이그레이션 스크립트

**파일**: `migrations/versions/xxx_add_keycloak_id_to_users.py`

```python
"""add_keycloak_id_to_users

Revision ID: xxxx
Revises: previous_revision
Create Date: 2025-12-19
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = 'xxxx'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None


def upgrade():
    # 1. keycloak_id 컬럼 추가
    op.add_column(
        'users',
        sa.Column('keycloak_id', sa.String(36), nullable=True)
    )

    # 2. unique index 생성
    op.create_index(
        'ix_users_keycloak_id',
        'users',
        ['keycloak_id'],
        unique=True
    )

    # 3. hashed_password를 nullable로 변경
    # SQLite는 ALTER COLUMN을 지원하지 않으므로 조건부 실행
    bind = op.get_bind()
    if bind.dialect.name != 'sqlite':
        op.alter_column(
            'users',
            'hashed_password',
            existing_type=sa.String(255),
            nullable=True
        )


def downgrade():
    bind = op.get_bind()

    # 1. hashed_password를 non-nullable로 복원 (PostgreSQL만)
    if bind.dialect.name != 'sqlite':
        op.alter_column(
            'users',
            'hashed_password',
            existing_type=sa.String(255),
            nullable=False
        )

    # 2. index 삭제
    op.drop_index('ix_users_keycloak_id', 'users')

    # 3. keycloak_id 컬럼 삭제
    op.drop_column('users', 'keycloak_id')
```

### 마이그레이션 실행

```bash
cd platform/backend
alembic upgrade head
```

### 체크포인트

- [ ] 마이그레이션 파일 생성
- [ ] `alembic upgrade head` 성공
- [ ] DB에 `keycloak_id` 컬럼 확인

```bash
# SQLite 확인
sqlite3 data/db/vision_platform.db ".schema users"

# PostgreSQL 확인
psql -d platform -c "\d users"
```

---

## Step 2.6: Dependencies 수정 (듀얼 인증)

**파일**: `platform/backend/app/utils/dependencies.py`

### 전략: 듀얼 인증 모드

```
KEYCLOAK_ENABLED=true  → Keycloak 토큰 검증
KEYCLOAK_ENABLED=false → 기존 자체 JWT 검증
```

### 변경된 코드

```python
"""Authentication dependencies for FastAPI endpoints."""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_token
from app.db.database import get_db, get_user_db
from app.db import models

# 기존 OAuth2 스킴 (자체 JWT용)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/v1/auth/login",
    auto_error=False  # 토큰 없어도 에러 안 남 (Keycloak 폴백용)
)

# HTTP Bearer 스킴 (Keycloak용)
http_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    oauth2_token: Optional[str] = Depends(oauth2_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    user_db: Session = Depends(get_user_db)
) -> models.User:
    """
    현재 인증된 사용자 반환 (듀얼 인증 지원)

    KEYCLOAK_ENABLED=true: Keycloak 토큰 검증
    KEYCLOAK_ENABLED=false: 자체 JWT 검증
    """
    # 토큰 추출 (HTTPBearer 우선)
    token = bearer.credentials if bearer else oauth2_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if settings.KEYCLOAK_ENABLED:
        return await _get_user_from_keycloak_token(token, user_db)
    else:
        return _get_user_from_internal_token(token, user_db)


async def _get_user_from_keycloak_token(
    token: str,
    user_db: Session
) -> models.User:
    """Keycloak 토큰에서 사용자 조회 또는 생성 (JIT Provisioning)"""
    from app.core.keycloak_auth import (
        verify_keycloak_token,
        map_keycloak_to_system_role,
        KeycloakTokenPayload
    )

    # 1. 토큰 검증
    try:
        payload: KeycloakTokenPayload = await verify_keycloak_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate Keycloak token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. keycloak_id로 사용자 조회
    user = user_db.query(models.User).filter(
        models.User.keycloak_id == payload.sub
    ).first()

    # 3. 없으면 이메일로 기존 사용자 조회 (마이그레이션)
    if not user and payload.email:
        user = user_db.query(models.User).filter(
            models.User.email == payload.email
        ).first()

        if user:
            # 기존 사용자에 keycloak_id 연결
            user.keycloak_id = payload.sub
            user_db.commit()

    # 4. 그래도 없으면 새 사용자 생성 (JIT Provisioning)
    if not user:
        user = models.User(
            keycloak_id=payload.sub,
            email=payload.email or f"{payload.sub}@keycloak.local",
            hashed_password=None,  # Keycloak 사용자는 비밀번호 없음
            full_name=payload.name or payload.preferred_username or "Unknown",
            system_role=map_keycloak_to_system_role(payload),
            is_active=True,
            # 커스텀 속성 (Keycloak에서 설정된 경우)
            company=payload.company,
            department=payload.department,
        )
        user_db.add(user)
        user_db.commit()
        user_db.refresh(user)

    # 5. 활성 사용자 확인
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


def _get_user_from_internal_token(
    token: str,
    user_db: Session
) -> models.User:
    """기존 자체 JWT에서 사용자 조회"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        user_id_str: Optional[str] = payload.get("sub")
        token_type: Optional[str] = payload.get("type")

        if user_id_str is None or token_type != "access":
            raise credentials_exception

        user_id = int(user_id_str)
    except (JWTError, ValueError, TypeError):
        raise credentials_exception

    user = user_db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user


# 나머지 함수들은 기존과 동일
def get_current_active_user(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """활성 사용자 확인"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_current_superuser(
    current_user: models.User = Depends(get_current_user)
) -> models.User:
    """관리자 확인"""
    if current_user.system_role != models.UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user
```

### 체크포인트

- [ ] `dependencies.py` 수정 완료
- [ ] Import 에러 없음 확인
- [ ] `KEYCLOAK_ENABLED=false` 일 때 기존 로직 동작 확인

---

## Step 2.7: 테스트 및 검증

### 2.7.1 단위 테스트

**파일**: `platform/backend/tests/test_keycloak_auth.py`

```python
import pytest
from unittest.mock import patch, AsyncMock
from app.core.keycloak_auth import (
    verify_keycloak_token,
    KeycloakTokenPayload,
    map_keycloak_to_system_role,
    extract_realm_roles,
)

# Mock JWKS 응답
MOCK_JWKS = {
    "keys": [
        {
            "kid": "test-key-id",
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "n": "...",  # RSA public key modulus
            "e": "AQAB"
        }
    ]
}


@pytest.fixture
def mock_token_payload():
    return {
        "sub": "550e8400-e29b-41d4-a716-446655440000",
        "iss": "http://localhost:8080/realms/vision-ai",
        "aud": "platform-backend",
        "exp": 9999999999,
        "iat": 1700000000,
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "realm_access": {"roles": ["admin", "user"]}
    }


def test_extract_realm_roles():
    payload = KeycloakTokenPayload(
        sub="test",
        iss="test",
        aud="test",
        exp=9999999999,
        iat=1700000000,
        realm_access={"roles": ["admin", "manager"]}
    )

    roles = extract_realm_roles(payload)
    assert "admin" in roles
    assert "manager" in roles


def test_map_keycloak_to_system_role_admin():
    payload = KeycloakTokenPayload(
        sub="test",
        iss="test",
        aud="test",
        exp=9999999999,
        iat=1700000000,
        realm_access={"roles": ["admin"]}
    )

    from app.db.models import UserRole
    role = map_keycloak_to_system_role(payload)
    assert role == UserRole.ADMIN


def test_map_keycloak_to_system_role_guest():
    payload = KeycloakTokenPayload(
        sub="test",
        iss="test",
        aud="test",
        exp=9999999999,
        iat=1700000000,
        realm_access={"roles": []}
    )

    from app.db.models import UserRole
    role = map_keycloak_to_system_role(payload)
    assert role == UserRole.GUEST
```

### 2.7.2 통합 테스트

```python
import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_keycloak_auth_me_endpoint():
    """Keycloak 토큰으로 /auth/me 호출 테스트"""
    # 실제 Keycloak에서 발급받은 토큰 필요
    valid_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6..."

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {valid_token}"}
        )

        # 토큰이 유효하면 200, 아니면 401
        assert response.status_code in [200, 401]


@pytest.mark.asyncio
async def test_invalid_token_rejected():
    """잘못된 토큰 거부 테스트"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code == 401
```

### 2.7.3 수동 테스트 체크리스트

```bash
# 1. Keycloak에서 토큰 발급
TOKEN=$(curl -s -X POST "http://localhost:8080/realms/vision-ai/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=platform-frontend" \
  -d "username=admin@example.com" \
  -d "password=admin123" \
  | jq -r '.access_token')

echo $TOKEN

# 2. Backend API 호출
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN"

# 3. 토큰 디코딩 (jwt.io에서 확인)
echo $TOKEN | cut -d'.' -f2 | base64 -d | jq
```

### 체크포인트

- [ ] 단위 테스트 통과
- [ ] Keycloak 토큰으로 `/auth/me` 호출 성공
- [ ] 잘못된 토큰 거부 확인
- [ ] JIT Provisioning 동작 확인 (새 사용자 자동 생성)
- [ ] 기존 사용자 이메일 매칭 및 `keycloak_id` 연결 확인

---

## 구현 순서 요약

| Step | 작업 | 예상 시간 | 파일 |
|------|------|-----------|------|
| 2.1 | Config 설정 추가 | 15분 | `config.py` |
| 2.2 | httpx 의존성 추가 | 5분 | `pyproject.toml` |
| 2.3 | OIDC 토큰 검증 모듈 | 1-2시간 | `keycloak_auth.py` (신규) |
| 2.4 | User 모델 수정 | 15분 | `models.py` |
| 2.5 | DB 마이그레이션 | 30분 | `migrations/` |
| 2.6 | Dependencies 수정 | 1시간 | `dependencies.py` |
| 2.7 | 테스트 및 검증 | 1-2시간 | `tests/` |

**총 예상 시간**: 4-6시간

---

## 트러블슈팅

### 문제: "Token validation failed: Unable to find key"

```
원인: JWKS에 해당 kid가 없음
해결:
  1. Keycloak이 실행 중인지 확인
  2. JWKS URL 접근 가능 확인: curl {KEYCLOAK_JWKS_URL}
  3. 토큰이 올바른 Realm에서 발급되었는지 확인
```

### 문제: "Audience validation failed"

```
원인: 토큰의 aud와 KEYCLOAK_CLIENT_ID 불일치
해결:
  1. Keycloak Client 설정에서 audience 확인
  2. .env의 KEYCLOAK_CLIENT_ID 확인
  3. Client Scope에서 audience mapper 설정
```

### 문제: "Issuer validation failed"

```
원인: 토큰의 iss와 KEYCLOAK_ISSUER 불일치
해결:
  1. KEYCLOAK_SERVER_URL 확인 (http vs https)
  2. KEYCLOAK_REALM 이름 확인
  3. 토큰 디코딩하여 iss 값 확인
```

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-19 | 1.0 | 초기 문서 작성 |
