import { NextRequest, NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

/**
 * Keycloak Logout Redirect Endpoint
 *
 * NextAuth signOut() 이후 호출되어 Keycloak 브라우저 세션을 로그아웃시킴
 * NextAuth 세션은 이미 삭제된 상태
 */
export async function GET(request: NextRequest) {
  // 토큰 가져오기 (Keycloak logout에 필요한 id_token용)
  // NextAuth signOut() 직후라 토큰이 없을 수 있음
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  // Keycloak 로그아웃 URL로 브라우저 리다이렉트
  if (token?.idToken) {
    const keycloakIssuer = process.env.KEYCLOAK_ISSUER
    const logoutUrl = `${keycloakIssuer}/protocol/openid-connect/logout`

    const baseUrl = process.env.NEXTAUTH_URL ||
                    `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('x-forwarded-host') || request.headers.get('host')}`

    // 최종 리다이렉트 위치 (파라미터 없이 홈으로)
    const redirectUri = `${baseUrl}/`

    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: redirectUri,
    })

    return NextResponse.redirect(`${logoutUrl}?${params.toString()}`)
  }

  // id_token이 없으면 바로 홈으로
  return NextResponse.redirect(new URL('/', request.nextUrl.origin))
}
