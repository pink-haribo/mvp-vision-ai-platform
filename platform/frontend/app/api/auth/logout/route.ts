import { NextRequest, NextResponse } from 'next/server'
import { getToken } from 'next-auth/jwt'

/**
 * Custom Logout Endpoint
 *
 * NextAuth 세션 삭제 + Keycloak 세션 삭제 + 로그아웃 페이지로 리다이렉트
 */
export async function GET(request: NextRequest) {
  console.log('[Logout] Starting logout process...')

  // 1. 현재 토큰 가져오기 (id_token 필요)
  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
  })

  console.log('[Logout] Token found:', !!token)
  console.log('[Logout] ID Token found:', !!token?.idToken)

  // 2. NextAuth 세션 쿠키 삭제
  const response = NextResponse.redirect(
    new URL('/auth/logout-success', request.nextUrl.origin)
  )

  // NextAuth 세션 쿠키 삭제
  response.cookies.delete('next-auth.session-token')
  response.cookies.delete('__Secure-next-auth.session-token')

  // 3. Keycloak 로그아웃 URL로 리다이렉트
  if (token?.idToken) {
    const keycloakIssuer = process.env.KEYCLOAK_ISSUER
    const logoutUrl = `${keycloakIssuer}/protocol/openid-connect/logout`
    const redirectUri = `${request.nextUrl.origin}/auth/logout-success`

    console.log('[Logout] Keycloak Issuer:', keycloakIssuer)
    console.log('[Logout] Logout URL:', logoutUrl)
    console.log('[Logout] Redirect URI:', redirectUri)

    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: redirectUri,
    })

    const finalUrl = `${logoutUrl}?${params.toString()}`
    console.log('[Logout] Final Keycloak logout URL:', finalUrl)

    // Keycloak 로그아웃 페이지로 리다이렉트
    return NextResponse.redirect(finalUrl)
  }

  console.log('[Logout] No ID token, skipping Keycloak logout')
  return response
}
