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

  // NextAuth signIn 엔드포인트로 리다이렉트 (NextAuth가 state 관리)
  const originalPath = pathname + search // 원래 요청 경로
  const signInUrl = new URL(`${nextAuthUrl}/api/auth/signin/keycloak`)
  signInUrl.searchParams.set("callbackUrl", originalPath)

  console.log("[Middleware] Redirecting to NextAuth signin:", signInUrl.toString())
  return NextResponse.redirect(signInUrl.toString())
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
