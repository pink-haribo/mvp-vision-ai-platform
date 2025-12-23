import { NextRequest, NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

/**
 * Keycloak Logout Redirect Endpoint
 *
 * NextAuth signOut() 이후 호출되어 Keycloak 브라우저 세션을 로그아웃시킴
 * NextAuth 세션은 이미 삭제된 상태
 */
export async function GET(request: NextRequest) {
  const keycloakIssuer = process.env.KEYCLOAK_ISSUER
  const logoutUrl = `${keycloakIssuer}/protocol/openid-connect/logout`

  // NEXTAUTH_URL을 최우선으로 사용 (0.0.0.0 문제 방지)
  const baseUrl = process.env.NEXTAUTH_URL ||
                  `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('x-forwarded-host') || request.headers.get('host')}`

  const redirectUri = `${baseUrl}/`

  // 토큰 가져오기 시도 (id_token_hint용)
  // NextAuth signOut() 직후라 토큰이 없을 수 있음
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  // id_token이 있으면 힌트와 함께 로그아웃
  if (token?.idToken) {
    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: redirectUri,
    })

    return NextResponse.redirect(`${logoutUrl}?${params.toString()}`)
  }

  // id_token이 없어도 Keycloak 로그아웃 시도 (hint 없이)
  const params = new URLSearchParams({
    post_logout_redirect_uri: redirectUri,
  })

  return NextResponse.redirect(`${logoutUrl}?${params.toString()}`)
}
