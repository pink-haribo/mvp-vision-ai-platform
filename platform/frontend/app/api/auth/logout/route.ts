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

    const params = new URLSearchParams({
      id_token_hint: token.idToken as string,
      post_logout_redirect_uri: `${request.nextUrl.origin}/auth/logout-success`,
    })

    // Keycloak 로그아웃 페이지로 리다이렉트
    return NextResponse.redirect(`${logoutUrl}?${params.toString()}`)
  }

  return response
}
