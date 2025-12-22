import { NextRequest, NextResponse } from "next/server"
import { getToken } from "next-auth/jwt"

/**
 * Keycloak으로 직접 리다이렉트하는 커스텀 Middleware
 * NextAuth의 기본 리다이렉트(/auth/signin)를 거치지 않고 바로 Keycloak으로 이동
 */
export async function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl

  // NextAuth 엔드포인트는 통과
  if (pathname.startsWith("/api/auth/")) {
    return NextResponse.next()
  }

  // Error 페이지는 통과
  if (pathname.startsWith("/auth/error")) {
    return NextResponse.next()
  }

  // 토큰 확인
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  // 인증된 사용자는 통과
  if (token) {
    return NextResponse.next()
  }

  // 미인증 사용자 → Keycloak으로 직접 리다이렉트
  const keycloakIssuer = process.env.KEYCLOAK_ISSUER
  const clientId = process.env.KEYCLOAK_CLIENT_ID
  const nextAuthUrl = process.env.NEXTAUTH_URL || `${request.nextUrl.origin}`

  if (!keycloakIssuer || !clientId) {
    console.error("[Middleware] Missing Keycloak configuration")
    return NextResponse.redirect(new URL("/auth/error", request.url))
  }

  // Keycloak Authorization URL 생성
  const originalPath = pathname + search // 원래 요청 경로
  const redirectUri = `${nextAuthUrl}/api/auth/callback/keycloak`

  // NextAuth callback URL에 원래 경로 포함
  const callbackUrlWithPath = `${redirectUri}?callbackUrl=${encodeURIComponent(originalPath)}`

  const authUrl = new URL(`${keycloakIssuer}/protocol/openid-connect/auth`)
  authUrl.searchParams.set("client_id", clientId)
  authUrl.searchParams.set("redirect_uri", redirectUri) // Keycloak에는 base redirect_uri만 전달
  authUrl.searchParams.set("response_type", "code")
  authUrl.searchParams.set("scope", "openid profile email")
  authUrl.searchParams.set("state", encodeURIComponent(originalPath)) // state에도 경로 저장 (fallback)

  console.log("[Middleware] Redirecting to Keycloak:", authUrl.toString())
  return NextResponse.redirect(authUrl.toString())
}

export const config = {
  matcher: [
    /*
     * 인증이 필요한 모든 경로
     * 제외:
     * - api/auth (NextAuth 엔드포인트)
     * - auth/error (에러 페이지)
     * - _next/static (정적 파일)
     * - _next/image (이미지 최적화)
     * - favicon.ico
     */
    "/((?!api/auth|auth/error|_next/static|_next/image|favicon.ico).*)",
  ],
}
