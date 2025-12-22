# Keycloak Client 설정 체크리스트

## "restart login cookie not found" 에러 해결

### 1. Client Settings (필수)

```
Clients → [your-client-id] → Settings
```

- [ ] **Client ID**: 정확한 client ID 확인
- [ ] **Enabled**: ON
- [ ] **Client Protocol**: openid-connect
- [ ] **Access Type**: public (또는 confidential with secret)
- [ ] **Standard Flow Enabled**: ON
- [ ] **Direct Access Grants Enabled**: ON (optional)
- [ ] **Valid Redirect URIs**:
  ```
  https://your-domain.com/*
  https://your-domain.com/auth/callback/keycloak
  http://localhost:3000/*
  http://localhost:3000/auth/callback/keycloak
  ```
- [ ] **Valid Post Logout Redirect URIs**:
  ```
  https://your-domain.com/*
  http://localhost:3000/*
  ```
- [ ] **Web Origins**:
  ```
  https://your-domain.com
  http://localhost:3000
  +
  ```
- [ ] **Root URL**: `https://your-domain.com` (또는 비워두기)
- [ ] **Base URL**: `/`

### 2. Realm Settings

```
Realm Settings → Sessions
```

- [ ] **SSO Session Idle**: 30 minutes
- [ ] **SSO Session Max**: 10 hours
- [ ] **SameSite Cookie Value**: None (크로스 도메인) 또는 Lax (같은 도메인)

```
Realm Settings → Security Defenses → Headers
```

- [ ] **X-Frame-Options**: SAMEORIGIN
- [ ] **Content-Security-Policy**:
  ```
  frame-src 'self'; frame-ancestors 'self' https://your-app-domain.com;
  ```

### 3. 네트워크 환경 확인

#### Istio VirtualService 설정

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: frontend
spec:
  http:
  - match:
    - uri:
        prefix: /auth
    route:
    - destination:
        host: frontend-service
        port:
          number: 3000
    corsPolicy:
      allowOrigins:
      - regex: ".*"
      allowMethods:
      - GET
      - POST
      allowHeaders:
      - authorization
      - content-type
      - cookie
      allowCredentials: true
```

#### 환경 변수 확인

```bash
# Frontend .env
NEXTAUTH_URL=https://your-domain.com  # HTTPS 사용 (프로덕션)
KEYCLOAK_ISSUER=https://keycloak.example.com/realms/your-realm
KEYCLOAK_CLIENT_ID=your-client-id

# Keycloak과 Frontend가 같은 프로토콜 사용 (둘 다 HTTPS)
```

### 4. 일반적인 원인

#### ❌ 문제 상황
- Keycloak: `https://keycloak.example.com`
- Frontend: `http://your-domain.com` (HTTP)
→ Mixed content 에러, 쿠키 차단

#### ✅ 해결 방법
- 둘 다 HTTPS 사용
- 또는 개발 환경에서는 둘 다 HTTP 사용

#### ❌ 문제 상황
- Valid Redirect URIs: `https://your-domain.com/auth/callback/keycloak` (정확한 경로)
- 실제 callback: `https://your-domain.com/auth/callback/keycloak?code=xxx`
→ 쿼리 파라미터가 있으면 매칭 실패할 수 있음

#### ✅ 해결 방법
- Valid Redirect URIs에 와일드카드 사용: `https://your-domain.com/*`

### 5. 디버깅 방법

#### 브라우저 개발자 도구

1. **Network 탭**:
   - Keycloak 리다이렉트 요청 확인
   - Response Headers에서 `Set-Cookie` 확인
   - `AUTH_SESSION_ID` 쿠키가 설정되는지 확인

2. **Application 탭** (Chrome) / **Storage 탭** (Firefox):
   - Cookies → Keycloak 도메인
   - `AUTH_SESSION_ID`, `KC_RESTART` 쿠키 확인
   - `SameSite`, `Secure` 속성 확인

3. **Console 탭**:
   - CORS 에러, Mixed content 에러 확인

#### Keycloak 서버 로그

```bash
# Keycloak 로그 확인
kubectl logs -f deployment/keycloak -n keycloak

# 또는 Docker
docker logs -f keycloak
```

에러 메시지에서 다음 확인:
- Invalid redirect_uri
- Cookie not found
- CORS policy

### 6. 빠른 해결 방법

가장 빠른 해결책 (개발 환경):

1. **Keycloak Client Settings**:
   ```
   Valid Redirect URIs: *
   Web Origins: *
   ```
   ⚠️ 프로덕션에서는 사용하지 말 것!

2. **Realm Settings → Sessions**:
   ```
   SameSite Cookie Value: (empty) 또는 None
   ```

3. **환경 변수**:
   ```bash
   # 둘 다 HTTP 또는 둘 다 HTTPS
   NEXTAUTH_URL=http://localhost:3000
   KEYCLOAK_ISSUER=http://keycloak:8080/realms/your-realm
   ```

### 7. 확인 절차

1. ✅ Keycloak Client 설정 저장 후 브라우저 **완전히 새로고침** (Ctrl+Shift+R)
2. ✅ **Incognito/Private 모드**에서 테스트 (쿠키 캐시 문제 배제)
3. ✅ Keycloak Admin Console에서 Sessions 확인
   - `Realm Settings → Sessions → View all sessions`
   - 로그인 시도 시 새 세션이 생성되는지 확인
