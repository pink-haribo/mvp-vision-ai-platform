import { NextRequest, NextResponse } from "next/server"
import { getToken } from "next-auth/jwt"

/**
 * Authentication Middleware
 *
 * 미인증 사용자를 NextAuth signin 페이지로 리다이렉트합니다.
 * Custom signin page에서 자동으로 Keycloak으로 리다이렉트되며,
 * NextAuth가 state cookie를 생성하여 CSRF 보호를 제공합니다.
 */
export async function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl

  // NextAuth 엔드포인트는 통과
  if (pathname.startsWith("/api/auth/")) {
    return NextResponse.next()
  }

  // Error 페이지는 통과
  if (pathname.startsWith("/auth/error") || pathname.startsWith("/auth/logout-success")) {
    return NextResponse.next()
  }

  // Logout endpoint는 통과
  if (pathname === "/api/auth/logout") {
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

  // 미인증 사용자 → NextAuth signin 페이지로 리다이렉트
  // Custom signin page에서 자동으로 Keycloak으로 리다이렉트되며, state cookie가 생성됩니다
  const nextAuthUrl = process.env.NEXTAUTH_URL || `${request.nextUrl.origin}`

  // NextAuth signin 페이지로 리다이렉트 (원래 경로를 callbackUrl로 전달)
  const originalPath = pathname + search
  const signInUrl = new URL(`${nextAuthUrl}/api/auth/signin`)
  signInUrl.searchParams.set("callbackUrl", originalPath)

  console.log("[Middleware] Redirecting to signin page:", signInUrl.toString())
  return NextResponse.redirect(signInUrl.toString())
}

export const config = {
  matcher: [
    /*
     * 인증이 필요한 모든 경로
     * 제외:
     * - api/auth (NextAuth 엔드포인트)
     * - auth/error (에러 페이지)
     * - auth/logout-success (로그아웃 성공 페이지)
     * - _next/static (정적 파일)
     * - _next/image (이미지 최적화)
     * - favicon.ico
     */
    "/((?!api/auth|auth/error|auth/logout-success|_next/static|_next/image|favicon.ico).*)",
  ],
}
