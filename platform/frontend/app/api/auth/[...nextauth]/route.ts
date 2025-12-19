import NextAuth, { NextAuthOptions } from "next-auth"
import KeycloakProvider from "next-auth/providers/keycloak"

export const authOptions: NextAuthOptions = {
  providers: [
    KeycloakProvider({
      clientId: process.env.KEYCLOAK_CLIENT_ID!,
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET || "",
      issuer: process.env.KEYCLOAK_ISSUER,
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      // 초기 로그인 시 토큰에 access_token 저장
      if (account) {
        token.accessToken = account.access_token
        token.refreshToken = account.refresh_token
        token.expiresAt = account.expires_at
        token.idToken = account.id_token
      }

      // 토큰 만료 확인
      if (Date.now() < (token.expiresAt as number) * 1000) {
        return token
      }

      // 토큰 갱신
      return await refreshAccessToken(token)
    },
    async session({ session, token }) {
      // 세션에 accessToken 추가 (Backend API 호출용)
      session.accessToken = token.accessToken as string
      session.error = token.error as string | undefined
      return session
    },
  },
  events: {
    async signOut({ token }) {
      // Keycloak 세션도 함께 로그아웃
      if (token?.idToken) {
        const logoutUrl = `${process.env.KEYCLOAK_ISSUER}/protocol/openid-connect/logout`
        const params = new URLSearchParams({
          id_token_hint: token.idToken as string,
          post_logout_redirect_uri: process.env.NEXTAUTH_URL || "http://localhost:3000",
        })
        try {
          await fetch(`${logoutUrl}?${params}`)
        } catch (error) {
          console.error("Failed to logout from Keycloak:", error)
        }
      }
    },
  },
  pages: {
    error: "/auth/error",
  },
  session: {
    strategy: "jwt",
    maxAge: 60 * 60, // 1시간
  },
}

/**
 * Keycloak Refresh Token으로 Access Token 갱신
 */
async function refreshAccessToken(token: any) {
  try {
    const response = await fetch(
      `${process.env.KEYCLOAK_ISSUER}/protocol/openid-connect/token`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          client_id: process.env.KEYCLOAK_CLIENT_ID!,
          client_secret: process.env.KEYCLOAK_CLIENT_SECRET || "",
          grant_type: "refresh_token",
          refresh_token: token.refreshToken,
        }),
      }
    )

    const tokens = await response.json()

    if (!response.ok) {
      throw tokens
    }

    return {
      ...token,
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token ?? token.refreshToken,
      expiresAt: Math.floor(Date.now() / 1000) + tokens.expires_in,
    }
  } catch (error) {
    console.error("Error refreshing access token:", error)
    return {
      ...token,
      error: "RefreshAccessTokenError",
    }
  }
}

const handler = NextAuth(authOptions)
export { handler as GET, handler as POST }
