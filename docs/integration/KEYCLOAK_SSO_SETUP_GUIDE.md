# Keycloak SSO 구현 가이드

Vision AI Training Platform에 Keycloak을 사용한 SSO(Single Sign-On) 인증을 구현하기 위한 단계별 가이드입니다.

**작성일**: 2025-12-19
**대상 버전**: Keycloak 24.x / 25.x
**예상 구현 기간**: 5-7일

---

## 목차

1. [개요](#1-개요)
2. [아키텍처 설계](#2-아키텍처-설계)
3. [Phase 1: Keycloak 서버 설정](#3-phase-1-keycloak-서버-설정)
4. [Phase 2: Backend OIDC 통합](#4-phase-2-backend-oidc-통합)
5. [Phase 3: Frontend 인증 흐름](#5-phase-3-frontend-인증-흐름)
6. [Phase 4: 기존 인증 마이그레이션](#6-phase-4-기존-인증-마이그레이션)
7. [Phase 5: 고급 기능 구현](#7-phase-5-고급-기능-구현)
8. [Phase 6: 테스트 및 검증](#8-phase-6-테스트-및-검증)
9. [운영 고려사항](#9-운영-고려사항)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. 개요

### 1.1 현재 인증 구조

현재 플랫폼은 자체 JWT 기반 인증을 사용합니다:

```
Frontend → POST /auth/login → Backend (JWT 발급) → localStorage 저장
```

**현재 구현된 기능:**
- Email/Password 로그인 (`platform/backend/app/api/auth.py`)
- JWT 토큰 발급/갱신 (`platform/backend/app/core/security.py`)
- 서비스 간 SSO (`platform/backend/app/core/service_jwt.py`)
- Redis 세션 스토어 (`platform/backend/app/services/redis_session_store.py`)

### 1.2 Keycloak 도입 목표

| 목표 | 설명 |
|------|------|
| **중앙 집중식 인증** | 모든 마이크로서비스가 Keycloak을 통해 인증 |
| **표준 프로토콜** | OIDC/OAuth2 표준 준수 |
| **SSO** | 한번 로그인으로 모든 서비스 접근 |
| **외부 IdP 연동** | Google, Microsoft, SAML IdP 연동 가능 |
| **세밀한 권한 관리** | Realm/Client/Role 기반 권한 체계 |

### 1.3 Keycloak이란?

Keycloak은 Red Hat이 개발한 오픈소스 Identity and Access Management (IAM) 솔루션입니다:

- **OIDC/OAuth2/SAML 지원**: 표준 프로토콜 완벽 지원
- **Identity Brokering**: 외부 IdP (Google, GitHub, LDAP 등) 연동
- **User Federation**: LDAP/Active Directory 통합
- **세밀한 권한 관리**: Realm > Client > Role 계층 구조
- **관리 콘솔**: 웹 기반 관리 UI 제공

---

## 2. 아키텍처 설계

### 2.1 목표 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Frontend (Next.js)     │    │   Keycloak Server        │
│   - next-auth            │◄──►│   - Realm: vision-ai     │
│   - PKCE flow            │    │   - Client: platform     │
│   - Token management     │    │   - Client: labeler      │
└──────────────────────────┘    └──────────────────────────┘
                │                           │
                │ Bearer Token              │ Token Validation
                ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        API Gateway (Optional)                     │
└──────────────────────────────────────────────────────────────────┘
                │
        ┌───────┴───────┬───────────────┬────────────────┐
        ▼               ▼               ▼                ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐  ┌─────────────┐
│  Platform   │ │   Labeler   │ │  Training   │  │   Other     │
│  Backend    │ │   Backend   │ │  Services   │  │  Services   │
└─────────────┘ └─────────────┘ └─────────────┘  └─────────────┘
```

### 2.2 인증 흐름

```
1. 사용자가 로그인 버튼 클릭
2. Frontend → Keycloak 로그인 페이지로 리다이렉트
3. 사용자 인증 (Email/Password 또는 Social Login)
4. Keycloak → Frontend로 Authorization Code 전달
5. Frontend → Backend로 Code 전달 (또는 직접 Token 교환)
6. Token 검증 후 세션 생성
7. 이후 요청에 Access Token 포함
```

### 2.3 토큰 전략

| 토큰 유형 | 용도 | 만료 시간 |
|-----------|------|-----------|
| Access Token | API 인증 | 5-15분 |
| Refresh Token | Access Token 갱신 | 7일 |
| ID Token | 사용자 정보 | 5-15분 |

---

## 3. Phase 1: Keycloak 서버 설정

### 3.1 Keycloak 설치 (Docker)

**예상 소요 시간**: 2-3시간

#### Step 1.1: Docker Compose 설정 추가

`platform/infrastructure/docker-compose.yml`에 Keycloak 서비스 추가:

```yaml
services:
  # ... 기존 서비스들 ...

  keycloak:
    image: quay.io/keycloak/keycloak:24.0
    container_name: platform-keycloak
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: ${KEYCLOAK_ADMIN_PASSWORD:-admin123}
      KC_DB: postgres
      KC_DB_URL: jdbc:postgresql://postgres:5432/keycloak
      KC_DB_USERNAME: ${POSTGRES_USER:-admin}
      KC_DB_PASSWORD: ${POSTGRES_PASSWORD:-devpass}
      KC_HOSTNAME: localhost
      KC_HOSTNAME_PORT: 8080
      KC_HOSTNAME_STRICT: false
      KC_HTTP_ENABLED: true
      KC_HEALTH_ENABLED: true
    command: start-dev
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - keycloak_data:/opt/keycloak/data
    networks:
      - platform-network
    healthcheck:
      test: ["CMD-SHELL", "exec 3<>/dev/tcp/127.0.0.1/8080"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  keycloak_data:
```

#### Step 1.2: PostgreSQL에 Keycloak DB 생성

```sql
-- PostgreSQL에서 실행
CREATE DATABASE keycloak;
```

또는 init script 추가:

```yaml
# docker-compose.yml의 postgres 서비스
postgres:
  environment:
    POSTGRES_MULTIPLE_DATABASES: platform,keycloak
```

#### Step 1.3: Keycloak 시작 및 접속

```bash
cd platform/infrastructure
docker-compose up -d keycloak

# 상태 확인
docker-compose logs -f keycloak

# Admin 콘솔 접속
# http://localhost:8080/admin
# Username: admin
# Password: admin123
```

### 3.2 Realm 설정

**예상 소요 시간**: 1-2시간

#### Step 2.1: Realm 생성

1. Keycloak Admin Console 접속 (`http://localhost:8080/admin`)
2. 좌측 상단 드롭다운 → "Create Realm" 클릭
3. Realm 정보 입력:
   - **Realm name**: `vision-ai`
   - **Enabled**: ON

#### Step 2.2: Realm 설정 구성

**Login 탭:**
```
- User registration: ON (자체 회원가입 허용 시)
- Email as username: ON
- Login with email: ON
- Forgot password: ON
- Remember me: ON
```

**Tokens 탭:**
```
- Access Token Lifespan: 15 minutes
- Refresh Token Lifespan: 7 days
- Access Token Lifespan For Implicit Flow: 15 minutes
```

**Sessions 탭:**
```
- SSO Session Idle: 30 minutes
- SSO Session Max: 10 hours
- Remember Me Session Idle: 7 days
```

### 3.3 Client 설정

**예상 소요 시간**: 1-2시간

#### Step 3.1: Platform Client 생성

1. Clients → Create client
2. 기본 설정:
   - **Client ID**: `platform-frontend`
   - **Client Protocol**: `openid-connect`
   - **Root URL**: `http://localhost:3000`

3. Capability config:
   - **Client authentication**: OFF (Public client - SPA)
   - **Authorization**: OFF
   - **Standard flow**: ON (Authorization Code Flow)
   - **Direct access grants**: OFF

4. Login settings:
   ```
   Root URL: http://localhost:3000
   Home URL: http://localhost:3000
   Valid redirect URIs:
     - http://localhost:3000/*
     - http://localhost:3000/api/auth/callback/keycloak
   Valid post logout redirect URIs:
     - http://localhost:3000/*
   Web origins:
     - http://localhost:3000
     - +
   ```

#### Step 3.2: Backend Client 생성 (선택적)

Backend에서 직접 토큰을 검증하거나 서비스 계정이 필요한 경우:

1. Clients → Create client
2. 기본 설정:
   - **Client ID**: `platform-backend`
   - **Client Protocol**: `openid-connect`

3. Capability config:
   - **Client authentication**: ON (Confidential client)
   - **Service accounts roles**: ON (필요시)

4. Credentials 탭에서 Client Secret 확인/복사

### 3.4 Role 및 User 설정

#### Step 4.1: Realm Role 생성

Realm roles에서 다음 역할 생성:

| Role Name | Description |
|-----------|-------------|
| `admin` | 전체 시스템 관리자 |
| `manager` | 프로젝트/팀 관리자 |
| `advanced_engineer` | 고급 기능 사용 가능 |
| `standard_engineer` | 기본 기능 사용 |
| `guest` | 읽기 전용 |

#### Step 4.2: Client Role 생성 (선택적)

클라이언트별 세밀한 권한이 필요한 경우:

```
platform-frontend:
  - training:execute
  - training:monitor
  - models:read
  - models:deploy
  - datasets:manage
```

#### Step 4.3: 테스트 사용자 생성

1. Users → Add user
2. 사용자 정보:
   - **Username**: admin@example.com
   - **Email**: admin@example.com
   - **Email verified**: ON
   - **Enabled**: ON

3. Credentials 탭에서 비밀번호 설정:
   - **Password**: admin123
   - **Temporary**: OFF

4. Role mapping에서 역할 할당

### 3.5 Keycloak 설정 Export

설정을 코드로 관리하기 위해 Export:

```bash
# Container에서 Export
docker exec platform-keycloak \
  /opt/keycloak/bin/kc.sh export \
  --dir /opt/keycloak/data/export \
  --realm vision-ai

# 호스트로 복사
docker cp platform-keycloak:/opt/keycloak/data/export ./keycloak-config
```

**권장 파일 위치**: `platform/infrastructure/keycloak/realm-export.json`

---

## 4. Phase 2: Backend OIDC 통합

### 4.1 의존성 추가

**예상 소요 시간**: 30분

`platform/backend/pyproject.toml`에 추가:

```toml
[tool.poetry.dependencies]
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
httpx = "^0.27.0"  # 비동기 HTTP 클라이언트 (OIDC Discovery)
```

또는 기존 `pyjwt` 사용 시:
```toml
pyjwt = {extras = ["crypto"], version = "^2.8.0"}
```

```bash
cd platform/backend
poetry add "python-jose[cryptography]" httpx
```

### 4.2 OIDC 설정 추가

**`platform/backend/app/core/config.py`** 수정:

```python
class Settings(BaseSettings):
    # ... 기존 설정 ...

    # Keycloak OIDC 설정
    KEYCLOAK_ENABLED: bool = False
    KEYCLOAK_SERVER_URL: str = "http://localhost:8080"
    KEYCLOAK_REALM: str = "vision-ai"
    KEYCLOAK_CLIENT_ID: str = "platform-backend"
    KEYCLOAK_CLIENT_SECRET: str | None = None  # Confidential client인 경우

    # OIDC Discovery URL (자동 구성)
    @property
    def KEYCLOAK_ISSUER(self) -> str:
        return f"{self.KEYCLOAK_SERVER_URL}/realms/{self.KEYCLOAK_REALM}"

    @property
    def KEYCLOAK_OPENID_CONFIG_URL(self) -> str:
        return f"{self.KEYCLOAK_ISSUER}/.well-known/openid-configuration"

    @property
    def KEYCLOAK_JWKS_URL(self) -> str:
        return f"{self.KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
```

### 4.3 OIDC 토큰 검증 모듈

**`platform/backend/app/core/keycloak_auth.py`** 생성:

```python
"""
Keycloak OIDC Token Validation Module

Keycloak에서 발급한 JWT 토큰을 검증합니다.
JWKS (JSON Web Key Set)를 사용하여 토큰 서명을 검증합니다.
"""

import httpx
from datetime import datetime, timedelta
from typing import Any, Optional
from jose import jwt, JWTError
from jose.exceptions import JWKError
from pydantic import BaseModel
from functools import lru_cache

from app.core.config import settings


class TokenPayload(BaseModel):
    """Keycloak JWT 토큰 페이로드"""
    sub: str  # Subject (user ID in Keycloak)
    email: Optional[str] = None
    email_verified: bool = False
    preferred_username: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    name: Optional[str] = None
    realm_access: dict = {}  # Realm roles
    resource_access: dict = {}  # Client roles
    exp: int
    iat: int
    iss: str
    aud: str | list[str]
    azp: Optional[str] = None  # Authorized party (client_id)


class KeycloakJWKS:
    """Keycloak JWKS 관리 클래스"""

    def __init__(self):
        self._jwks: dict | None = None
        self._jwks_expires: datetime | None = None
        self._cache_duration = timedelta(hours=1)

    async def get_jwks(self) -> dict:
        """JWKS를 가져오거나 캐시된 값을 반환"""
        now = datetime.utcnow()

        if self._jwks and self._jwks_expires and now < self._jwks_expires:
            return self._jwks

        async with httpx.AsyncClient() as client:
            response = await client.get(
                settings.KEYCLOAK_JWKS_URL,
                timeout=10.0
            )
            response.raise_for_status()
            self._jwks = response.json()
            self._jwks_expires = now + self._cache_duration

        return self._jwks

    def clear_cache(self):
        """JWKS 캐시 초기화 (키 로테이션 시)"""
        self._jwks = None
        self._jwks_expires = None


# 전역 JWKS 인스턴스
_jwks_client = KeycloakJWKS()


async def verify_keycloak_token(token: str) -> TokenPayload:
    """
    Keycloak JWT 토큰 검증

    Args:
        token: Bearer 토큰 (access_token)

    Returns:
        TokenPayload: 검증된 토큰 페이로드

    Raises:
        JWTError: 토큰 검증 실패
    """
    try:
        # JWKS 가져오기
        jwks = await _jwks_client.get_jwks()

        # 토큰 헤더에서 kid 추출
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise JWTError("Token header missing 'kid'")

        # 해당 kid의 키 찾기
        rsa_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break

        if not rsa_key:
            # 키가 없으면 JWKS 갱신 후 재시도
            _jwks_client.clear_cache()
            jwks = await _jwks_client.get_jwks()
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    rsa_key = key
                    break

        if not rsa_key:
            raise JWTError(f"Unable to find matching key for kid: {kid}")

        # 토큰 검증
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

        return TokenPayload(**payload)

    except JWKError as e:
        raise JWTError(f"JWK error: {str(e)}")
    except Exception as e:
        raise JWTError(f"Token validation failed: {str(e)}")


def extract_roles(payload: TokenPayload) -> list[str]:
    """토큰에서 역할 목록 추출"""
    roles = []

    # Realm roles
    realm_roles = payload.realm_access.get("roles", [])
    roles.extend([f"realm:{role}" for role in realm_roles])

    # Client roles
    for client, access in payload.resource_access.items():
        client_roles = access.get("roles", [])
        roles.extend([f"{client}:{role}" for role in client_roles])

    return roles


def has_role(payload: TokenPayload, role: str, client: str | None = None) -> bool:
    """특정 역할 보유 여부 확인"""
    if client:
        # Client-specific role
        client_access = payload.resource_access.get(client, {})
        return role in client_access.get("roles", [])
    else:
        # Realm role
        return role in payload.realm_access.get("roles", [])
```

### 4.4 FastAPI 의존성 수정

**`platform/backend/app/utils/dependencies.py`** 수정:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.core.config import settings
from app.core.security import decode_token
from app.core.keycloak_auth import verify_keycloak_token, TokenPayload, has_role
from app.db.models import User
from app.db.database import get_db

# 기존 OAuth2 스킴 (자체 JWT용)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# HTTP Bearer 스킴 (Keycloak용)
http_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    bearer: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    db: Session = Depends(get_db)
) -> User:
    """
    현재 인증된 사용자 반환

    Keycloak이 활성화된 경우 Keycloak 토큰 검증,
    그렇지 않으면 기존 자체 JWT 검증
    """
    # 토큰 추출 (bearer 우선, 없으면 oauth2)
    actual_token = bearer.credentials if bearer else token

    if not actual_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if settings.KEYCLOAK_ENABLED:
        return await _get_user_from_keycloak_token(actual_token, db)
    else:
        return await _get_user_from_internal_token(actual_token, db)


async def _get_user_from_keycloak_token(token: str, db: Session) -> User:
    """Keycloak 토큰에서 사용자 조회/생성"""
    try:
        payload = await verify_keycloak_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Keycloak subject (user ID)로 사용자 조회
    user = db.query(User).filter(User.keycloak_id == payload.sub).first()

    if not user and payload.email:
        # 이메일로 기존 사용자 조회 (마이그레이션 케이스)
        user = db.query(User).filter(User.email == payload.email).first()
        if user:
            # Keycloak ID 연결
            user.keycloak_id = payload.sub
            db.commit()

    if not user:
        # 자동 사용자 생성 (JIT Provisioning)
        user = User(
            keycloak_id=payload.sub,
            email=payload.email or f"{payload.sub}@keycloak",
            full_name=payload.name or payload.preferred_username or "Unknown",
            is_active=True,
            # Keycloak 역할을 시스템 역할로 매핑
            system_role=_map_keycloak_role_to_system_role(payload),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


def _map_keycloak_role_to_system_role(payload: TokenPayload) -> str:
    """Keycloak 역할을 시스템 역할로 매핑"""
    from app.db.models import SystemRole

    if has_role(payload, "admin"):
        return SystemRole.ADMIN
    elif has_role(payload, "manager"):
        return SystemRole.MANAGER
    elif has_role(payload, "advanced_engineer"):
        return SystemRole.ADVANCED_ENGINEER
    elif has_role(payload, "standard_engineer"):
        return SystemRole.STANDARD_ENGINEER
    else:
        return SystemRole.GUEST


async def _get_user_from_internal_token(token: str, db: Session) -> User:
    """기존 자체 JWT에서 사용자 조회 (기존 로직)"""
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user
```

### 4.5 User 모델 수정

**`platform/backend/app/db/models.py`**에 Keycloak ID 필드 추가:

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Keycloak 연동 필드 추가
    keycloak_id = Column(String(36), unique=True, nullable=True, index=True)

    # 기존 필드들...
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Keycloak 사용 시 null 가능
    # ...
```

**마이그레이션 생성**:

```bash
cd platform/backend
poetry run alembic revision -m "add_keycloak_id_to_users"
```

```python
# migrations/versions/xxx_add_keycloak_id_to_users.py
def upgrade():
    op.add_column('users', sa.Column('keycloak_id', sa.String(36), nullable=True))
    op.create_index('ix_users_keycloak_id', 'users', ['keycloak_id'], unique=True)
    # hashed_password를 nullable로 변경
    op.alter_column('users', 'hashed_password', nullable=True)

def downgrade():
    op.drop_index('ix_users_keycloak_id', 'users')
    op.drop_column('users', 'keycloak_id')
    op.alter_column('users', 'hashed_password', nullable=False)
```

---

## 5. Phase 3: Frontend 인증 흐름

### 5.1 NextAuth.js 설정

**예상 소요 시간**: 2-3시간

#### Step 1: 의존성 설치

```bash
cd platform/frontend
pnpm add next-auth
```

#### Step 2: NextAuth 설정 파일 생성

**`platform/frontend/app/api/auth/[...nextauth]/route.ts`**:

```typescript
import NextAuth, { NextAuthOptions } from "next-auth";
import KeycloakProvider from "next-auth/providers/keycloak";

export const authOptions: NextAuthOptions = {
  providers: [
    KeycloakProvider({
      clientId: process.env.KEYCLOAK_CLIENT_ID!,
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET || "",
      issuer: process.env.KEYCLOAK_ISSUER,
      authorization: {
        params: {
          scope: "openid email profile",
        },
      },
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      // 초기 로그인 시 토큰에 정보 추가
      if (account) {
        token.accessToken = account.access_token;
        token.refreshToken = account.refresh_token;
        token.expiresAt = account.expires_at;
        token.idToken = account.id_token;
      }

      // 토큰 만료 확인 및 갱신
      if (Date.now() < (token.expiresAt as number) * 1000) {
        return token;
      }

      // 토큰 갱신
      return await refreshAccessToken(token);
    },
    async session({ session, token }) {
      // 세션에 accessToken 추가
      session.accessToken = token.accessToken as string;
      session.error = token.error as string | undefined;
      return session;
    },
  },
  events: {
    async signOut({ token }) {
      // Keycloak 세션도 로그아웃
      if (token?.idToken) {
        const logoutUrl = `${process.env.KEYCLOAK_ISSUER}/protocol/openid-connect/logout`;
        const params = new URLSearchParams({
          id_token_hint: token.idToken as string,
          post_logout_redirect_uri: process.env.NEXTAUTH_URL!,
        });
        await fetch(`${logoutUrl}?${params}`);
      }
    },
  },
  pages: {
    signIn: "/login",
    error: "/auth/error",
  },
};

async function refreshAccessToken(token: any) {
  try {
    const response = await fetch(
      `${process.env.KEYCLOAK_ISSUER}/protocol/openid-connect/token`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          client_id: process.env.KEYCLOAK_CLIENT_ID!,
          client_secret: process.env.KEYCLOAK_CLIENT_SECRET || "",
          grant_type: "refresh_token",
          refresh_token: token.refreshToken,
        }),
      }
    );

    const tokens = await response.json();

    if (!response.ok) {
      throw tokens;
    }

    return {
      ...token,
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token ?? token.refreshToken,
      expiresAt: Math.floor(Date.now() / 1000) + tokens.expires_in,
    };
  } catch (error) {
    console.error("Error refreshing access token", error);
    return {
      ...token,
      error: "RefreshAccessTokenError",
    };
  }
}

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

#### Step 3: 타입 확장

**`platform/frontend/types/next-auth.d.ts`**:

```typescript
import "next-auth";
import "next-auth/jwt";

declare module "next-auth" {
  interface Session {
    accessToken?: string;
    error?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    refreshToken?: string;
    expiresAt?: number;
    idToken?: string;
    error?: string;
  }
}
```

### 5.2 환경 변수 설정

**`platform/frontend/.env.local`**:

```env
# NextAuth
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-change-this

# Keycloak
KEYCLOAK_CLIENT_ID=platform-frontend
KEYCLOAK_CLIENT_SECRET=  # Public client면 비워두기
KEYCLOAK_ISSUER=http://localhost:8080/realms/vision-ai

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

### 5.3 AuthContext 통합

**`platform/frontend/contexts/AuthContext.tsx`** 수정:

```typescript
"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useSession, signIn, signOut } from "next-auth/react";

interface User {
  id: number;
  email: string;
  full_name: string;
  system_role: string;
  organization_id?: number;
  badge_color?: string;
  avatar_name?: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
  accessToken: string | null;

  // Keycloak SSO
  loginWithKeycloak: () => Promise<void>;
  logout: () => Promise<void>;

  // 기존 로컬 로그인 (폴백용)
  loginWithCredentials: (email: string, password: string) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { data: session, status } = useSession();
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loading = status === "loading";
  const accessToken = session?.accessToken ?? null;

  useEffect(() => {
    if (session?.accessToken) {
      fetchUserInfo(session.accessToken);
    } else if (status === "unauthenticated") {
      setUser(null);
    }
  }, [session, status]);

  async function fetchUserInfo(token: string) {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        setUser(null);
      }
    } catch (err) {
      console.error("Failed to fetch user info:", err);
      setUser(null);
    }
  }

  async function loginWithKeycloak() {
    setError(null);
    await signIn("keycloak", { callbackUrl: "/" });
  }

  async function logout() {
    setError(null);
    setUser(null);
    await signOut({ callbackUrl: "/login" });
  }

  async function loginWithCredentials(email: string, password: string) {
    // 기존 로컬 로그인 로직 (Keycloak 비활성화 시 폴백)
    setError(null);
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({ username: email, password }),
      });

      if (!response.ok) {
        throw new Error("Login failed");
      }

      const data = await response.json();
      // 로컬 토큰 저장 및 사용자 정보 로드
      localStorage.setItem("access_token", data.access_token);
      await fetchUserInfo(data.access_token);
    } catch (err) {
      setError("로그인에 실패했습니다.");
      throw err;
    }
  }

  async function register(data: RegisterData) {
    // 기존 회원가입 로직 유지
    // ...
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        error,
        accessToken,
        loginWithKeycloak,
        logout,
        loginWithCredentials,
        register,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
```

### 5.4 SessionProvider 추가

**`platform/frontend/app/providers.tsx`**:

```typescript
"use client";

import { SessionProvider } from "next-auth/react";
import { AuthProvider } from "@/contexts/AuthContext";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <AuthProvider>
        {children}
      </AuthProvider>
    </SessionProvider>
  );
}
```

**`platform/frontend/app/layout.tsx`** 수정:

```typescript
import { Providers } from "./providers";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
```

### 5.5 로그인 페이지 수정

**`platform/frontend/app/login/page.tsx`** 수정:

```typescript
"use client";

import { useAuth } from "@/contexts/AuthContext";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";

export default function LoginPage() {
  const { user, loading, loginWithKeycloak, loginWithCredentials } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const error = searchParams.get("error");

  useEffect(() => {
    if (user) {
      router.push("/");
    }
  }, [user, router]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-lg shadow">
        <h2 className="text-center text-3xl font-bold text-gray-900">
          Vision AI Platform
        </h2>

        {error && (
          <div className="bg-red-50 text-red-500 p-4 rounded">
            로그인 중 오류가 발생했습니다: {error}
          </div>
        )}

        {/* Keycloak SSO 버튼 */}
        <button
          onClick={() => loginWithKeycloak()}
          className="w-full flex justify-center py-3 px-4 border border-transparent
                     rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700
                     focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          SSO로 로그인
        </button>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-gray-300" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-gray-500">또는</span>
          </div>
        </div>

        {/* 로컬 로그인 폼 (선택적) */}
        <LocalLoginForm onSubmit={loginWithCredentials} />
      </div>
    </div>
  );
}

function LocalLoginForm({ onSubmit }: { onSubmit: (email: string, password: string) => Promise<void> }) {
  // 기존 로그인 폼 컴포넌트...
}
```

---

## 6. Phase 4: 기존 인증 마이그레이션

### 6.1 마이그레이션 전략

**예상 소요 시간**: 1일

```
Phase 4.1: 듀얼 인증 모드 (2주)
  ├── 기존 JWT 인증 유지
  ├── Keycloak 인증 병렬 지원
  └── 사용자가 선택 가능

Phase 4.2: Keycloak 전환 권장 (2주)
  ├── 로그인 시 Keycloak 우선 표시
  ├── 기존 사용자 마이그레이션 안내
  └── 사용자 피드백 수집

Phase 4.3: 기존 인증 제거 (1주)
  ├── 로컬 로그인 비활성화
  ├── Keycloak 전용 모드
  └── 마이그레이션 완료 확인
```

### 6.2 기존 사용자 마이그레이션

**옵션 A: 자동 마이그레이션 (이메일 기반)**

```python
# 첫 Keycloak 로그인 시 자동으로 기존 계정 연결
async def _get_user_from_keycloak_token(token: str, db: Session) -> User:
    payload = await verify_keycloak_token(token)

    # 1. Keycloak ID로 조회
    user = db.query(User).filter(User.keycloak_id == payload.sub).first()

    if not user and payload.email:
        # 2. 이메일로 기존 사용자 조회
        user = db.query(User).filter(User.email == payload.email).first()
        if user:
            # 기존 계정에 Keycloak ID 연결
            user.keycloak_id = payload.sub
            db.commit()

    # ...
```

**옵션 B: 수동 마이그레이션 (계정 연결 UI)**

```
1. 기존 로그인 → Keycloak 계정 연결 버튼 표시
2. 사용자가 Keycloak 인증 완료
3. 두 계정 정보 확인 후 연결
```

### 6.3 Keycloak으로 기존 사용자 Import

**Keycloak User Import JSON 생성**:

```python
# scripts/export_users_for_keycloak.py

import json
from app.db.database import SessionLocal
from app.db.models import User

def export_users_for_keycloak():
    db = SessionLocal()
    users = db.query(User).filter(User.is_active == True).all()

    keycloak_users = []
    for user in users:
        keycloak_users.append({
            "username": user.email,
            "email": user.email,
            "emailVerified": True,
            "enabled": True,
            "firstName": user.full_name.split()[0] if user.full_name else "",
            "lastName": " ".join(user.full_name.split()[1:]) if user.full_name else "",
            "attributes": {
                "platform_user_id": [str(user.id)],
                "company": [user.company or ""],
                "department": [user.department or ""],
            },
            "credentials": [
                {
                    "type": "password",
                    "temporary": True,  # 첫 로그인 시 비밀번호 변경 필요
                    "value": "changeme123"  # 임시 비밀번호
                }
            ],
            "realmRoles": [map_system_role(user.system_role)],
        })

    with open("keycloak-users-import.json", "w") as f:
        json.dump({"users": keycloak_users}, f, indent=2)

def map_system_role(role):
    mapping = {
        "admin": "admin",
        "manager": "manager",
        "advanced_engineer": "advanced_engineer",
        "standard_engineer": "standard_engineer",
        "guest": "guest",
    }
    return mapping.get(role, "standard_engineer")

if __name__ == "__main__":
    export_users_for_keycloak()
```

**Keycloak에서 Import**:
1. Realm Settings → Action → Partial import
2. JSON 파일 업로드
3. Import 실행

---

## 7. Phase 5: 고급 기능 구현

### 7.1 Social Login 연동

**예상 소요 시간**: 2-3시간

#### Google 연동

1. Keycloak Admin Console → Identity Providers → Add provider → Google
2. Google Cloud Console에서 OAuth 2.0 Client ID 생성
3. Keycloak에 Client ID/Secret 입력:
   ```
   Client ID: your-google-client-id.apps.googleusercontent.com
   Client Secret: your-google-client-secret
   ```

#### Microsoft 연동

1. Identity Providers → Add provider → Microsoft
2. Azure AD에서 App Registration 생성
3. Client ID/Secret 입력

### 7.2 LDAP/Active Directory 연동

```
User Federation → Add provider → LDAP

설정 예시:
- Vendor: Active Directory
- Connection URL: ldap://ad.company.com:389
- Users DN: cn=users,dc=company,dc=com
- Bind DN: cn=admin,dc=company,dc=com
- Sync Settings: Periodic Full Sync
```

### 7.3 2FA (Two-Factor Authentication)

1. Authentication → Required Actions
2. "Configure OTP" 활성화
3. Users → user 선택 → Required User Actions → "Configure OTP" 추가

또는 전체 Realm에 필수로 설정:
```
Authentication → Flows → browser → OTP Form → REQUIRED
```

### 7.4 Custom Claims 추가

Backend에서 사용할 커스텀 속성 추가:

1. Client Scopes → Create client scope → `platform-scope`
2. Mappers → Add mapper:
   ```
   Name: department
   Mapper Type: User Attribute
   User Attribute: department
   Token Claim Name: department
   Claim JSON Type: String
   Add to ID token: ON
   Add to access token: ON
   ```

3. Clients → platform-frontend → Client scopes → Add `platform-scope`

---

## 8. Phase 6: 테스트 및 검증

### 8.1 테스트 체크리스트

**인증 흐름 테스트:**
- [ ] Keycloak 로그인 성공
- [ ] 토큰 갱신 (Refresh Token)
- [ ] 로그아웃 (Frontend + Keycloak 세션)
- [ ] 권한 없는 사용자 접근 차단
- [ ] 비활성 사용자 접근 차단

**역할 기반 접근 제어:**
- [ ] Admin 역할 - 모든 기능 접근
- [ ] Manager 역할 - 관리 기능 접근
- [ ] Engineer 역할 - 기본 기능 접근
- [ ] Guest 역할 - 읽기 전용

**마이그레이션 테스트:**
- [ ] 기존 사용자 Keycloak 로그인
- [ ] 자동 계정 연결 (이메일 기반)
- [ ] 역할 매핑 정확성

**에러 처리:**
- [ ] 잘못된 토큰 거부
- [ ] 만료된 토큰 갱신 시도
- [ ] Keycloak 서버 다운 시 적절한 에러 메시지

### 8.2 테스트 스크립트

```bash
# Backend 테스트
cd platform/backend
poetry run pytest tests/test_keycloak_auth.py -v

# E2E 테스트
cd platform/frontend
pnpm test:e2e
```

### 8.3 통합 테스트 시나리오

```python
# tests/test_keycloak_auth.py

import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_keycloak_token_validation():
    """유효한 Keycloak 토큰으로 API 접근"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {VALID_KEYCLOAK_TOKEN}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data

@pytest.mark.asyncio
async def test_invalid_token_rejected():
    """잘못된 토큰 거부"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

@pytest.mark.asyncio
async def test_role_based_access():
    """역할 기반 접근 제어"""
    # Admin 토큰으로 관리 API 접근
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
        )
        assert response.status_code == 200

    # 일반 사용자 토큰으로 관리 API 접근 시도
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {USER_TOKEN}"}
        )
        assert response.status_code == 403
```

---

## 9. 운영 고려사항

### 9.1 Production 배포 체크리스트

**Keycloak 서버:**
- [ ] HTTPS 활성화 (TLS 인증서)
- [ ] 별도 PostgreSQL 데이터베이스 사용
- [ ] Admin 비밀번호 변경
- [ ] 클러스터링 구성 (HA)
- [ ] 백업 전략 수립

**보안:**
- [ ] CORS 설정 검토
- [ ] CSP (Content Security Policy) 설정
- [ ] Rate limiting 설정
- [ ] Brute force detection 활성화

**모니터링:**
- [ ] Health check 엔드포인트 설정
- [ ] 로그 수집 (ELK, Loki)
- [ ] 메트릭 수집 (Prometheus)
- [ ] 알림 설정 (로그인 실패 급증 등)

### 9.2 환경별 설정

**Development:**
```yaml
# docker-compose.override.yml
services:
  keycloak:
    command: start-dev
    environment:
      KC_HOSTNAME_STRICT: false
      KC_HTTP_ENABLED: true
```

**Production:**
```yaml
# docker-compose.prod.yml
services:
  keycloak:
    command: start
    environment:
      KC_HOSTNAME: auth.yourdomain.com
      KC_HOSTNAME_STRICT: true
      KC_HTTP_ENABLED: false
      KC_PROXY: edge  # 리버스 프록시 사용 시
```

### 9.3 백업 및 복구

```bash
# Realm 설정 Export
docker exec platform-keycloak \
  /opt/keycloak/bin/kc.sh export \
  --dir /tmp/export \
  --realm vision-ai \
  --users realm_file

# 복구
docker exec platform-keycloak \
  /opt/keycloak/bin/kc.sh import \
  --file /tmp/export/vision-ai-realm.json
```

---

## 10. 트러블슈팅

### 10.1 일반적인 문제

**문제: "Invalid redirect URI"**
```
원인: Keycloak Client 설정의 Valid Redirect URIs 불일치
해결:
  - http://localhost:3000/* 추가
  - 프로토콜 (http vs https) 확인
```

**문제: "CORS error"**
```
원인: Keycloak의 Web Origins 설정 누락
해결:
  - Client 설정 → Web Origins에 Frontend URL 추가
  - 또는 '+' 입력 (Valid Redirect URIs와 동일하게)
```

**문제: "Token validation failed"**
```
원인:
  1. Audience(aud) 불일치
  2. Issuer(iss) 불일치
  3. JWKS URL 접근 불가
해결:
  - Backend 설정의 KEYCLOAK_CLIENT_ID 확인
  - KEYCLOAK_ISSUER URL 확인
  - Keycloak 서버 접근성 확인
```

**문제: "User not found after Keycloak login"**
```
원인: JIT Provisioning 실패 또는 이메일 누락
해결:
  - Keycloak에서 email scope 활성화 확인
  - User의 email 필드 채워져 있는지 확인
  - Email verified 설정 확인
```

### 10.2 디버깅 팁

**Keycloak 로그 확인:**
```bash
docker-compose logs -f keycloak
```

**토큰 디코딩 (JWT.io):**
```
https://jwt.io 에서 Access Token 붙여넣기
→ Payload 확인 (iss, aud, exp, roles 등)
```

**OIDC Discovery 확인:**
```bash
curl http://localhost:8080/realms/vision-ai/.well-known/openid-configuration | jq
```

**JWKS 확인:**
```bash
curl http://localhost:8080/realms/vision-ai/protocol/openid-connect/certs | jq
```

---

## 부록

### A. 환경 변수 요약

| 변수 | 설명 | 예시 |
|------|------|------|
| `KEYCLOAK_ENABLED` | Keycloak 인증 활성화 | `true` |
| `KEYCLOAK_SERVER_URL` | Keycloak 서버 URL | `http://localhost:8080` |
| `KEYCLOAK_REALM` | Realm 이름 | `vision-ai` |
| `KEYCLOAK_CLIENT_ID` | Client ID | `platform-backend` |
| `KEYCLOAK_CLIENT_SECRET` | Client Secret | `xxxxx` |
| `NEXTAUTH_URL` | Frontend URL | `http://localhost:3000` |
| `NEXTAUTH_SECRET` | NextAuth 암호화 키 | `random-secret` |

### B. 관련 문서

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [NextAuth.js Keycloak Provider](https://next-auth.js.org/providers/keycloak)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [OIDC Specification](https://openid.net/specs/openid-connect-core-1_0.html)

### C. 구현 예상 일정

| Phase | 작업 | 예상 시간 |
|-------|------|-----------|
| 1 | Keycloak 서버 설정 | 4-5시간 |
| 2 | Backend OIDC 통합 | 4-6시간 |
| 3 | Frontend 인증 흐름 | 4-6시간 |
| 4 | 기존 인증 마이그레이션 | 4-8시간 |
| 5 | 고급 기능 (선택적) | 4-8시간 |
| 6 | 테스트 및 검증 | 4-6시간 |
| **총계** | | **24-39시간 (5-7일)** |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-12-19 | 1.0 | 초기 문서 작성 |
