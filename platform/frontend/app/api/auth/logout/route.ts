import { NextRequest, NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

/**
 * Custom Logout Endpoint
 *
 * NextAuth 세션 삭제 + Keycloak 세션 삭제 + 로그아웃 페이지로 리다이렉트
 */
export async function GET(request: NextRequest) {
  // 1. 현재 토큰 가져오기 (id_token 필요)
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  // 2. Keycloak 로그아웃 URL로 리다이렉트
  if (token?.idToken) {
    const keycloakIssuer = process.env.KEYCLOAK_ISSUER
    const logoutUrl = `${keycloakIssuer}/protocol/openid-connect/logout`

    // NEXTAUTH_URL 사용 (0.0.0.0 문제 방지)
    // Fallback: Host 헤더 또는 X-Forwarded-Host (Kubernetes Ingress)
    const baseUrl = process.env.NEXTAUTH_URL ||
                    `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('x-forwarded-host') || request.headers.get('host')}`
    const redirectUri = `${baseUrl}/`  // 홈으로 바로 리다이렉트

    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: redirectUri,
    })

    const finalUrl = `${logoutUrl}?${params.toString()}`

    // Keycloak 로그아웃 페이지로 리다이렉트
    // Keycloak이 로그아웃 후 홈(/)으로 리다이렉트
    return NextResponse.redirect(finalUrl)
  }

  // idToken이 없으면 바로 홈으로
  return NextResponse.redirect(
    new URL('/', request.nextUrl.origin)
  )
}
