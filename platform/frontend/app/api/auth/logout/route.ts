import { NextRequest, NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

/**
 * Custom Logout Endpoint
 *
 * Keycloak 세션 로그아웃 (브라우저 리다이렉트)
 * NextAuth 세션은 logout-success 페이지에서 정리
 */
export async function GET(request: NextRequest) {
  // NEXTAUTH_URL 사용 (0.0.0.0 문제 방지)
  const baseUrl = process.env.NEXTAUTH_URL ||
                  `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('x-forwarded-host') || request.headers.get('host')}`

  // 토큰 가져오기 (세션 쿠키가 아직 살아있음)
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  // Keycloak 로그아웃 URL로 리다이렉트
  if (token?.idToken) {
    const keycloakIssuer = process.env.KEYCLOAK_ISSUER
    const logoutUrl = `${keycloakIssuer}/protocol/openid-connect/logout`

    // 중간 페이지로 리다이렉트 (logout 파라미터 없이)
    const redirectUri = `${baseUrl}/auth/logout-success`

    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: redirectUri,
    })

    return NextResponse.redirect(`${logoutUrl}?${params.toString()}`)
  }

  // id_token이 없으면 바로 logout-success로 (baseUrl 사용)
  return NextResponse.redirect(new URL('/auth/logout-success', baseUrl))
}
