import NextAuth, { NextAuthOptions } from "next-auth"
import KeycloakProvider from "next-auth/providers/keycloak"

// 내부 통신용 Keycloak URL (서버 사이드에서 사용)
// K8s 환경에서는 서비스 이름, 개발 환경에서는 localhost 사용
const keycloakIssuerInternal =
  process.env.KEYCLOAK_ISSUER_INTERNAL || process.env.KEYCLOAK_ISSUER

export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  providers: [
    KeycloakProvider({
      clientId: process.env.KEYCLOAK_CLIENT_ID!,
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET || "",
      // 브라우저는 KEYCLOAK_ISSUER 사용, 서버는 KEYCLOAK_ISSUER_INTERNAL 사용
      issuer: keycloakIssuerInternal,
      authorization: {
        params: {
          // 브라우저 리다이렉트 시에는 외부 URL 사용
          // Keycloak authorization endpoint는 브라우저에서 접근
        },
      },
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
    async redirect({ url, baseUrl }) {
      // Middleware에서 설정한 callbackUrl로 리다이렉트
      // url이 상대 경로면 baseUrl과 결합
      if (url.startsWith("/")) return `${baseUrl}${url}`
      // url이 같은 origin이면 허용
      else if (new URL(url).origin === baseUrl) return url
      // 그 외에는 홈으로
      return baseUrl
    },
  },
  events: {
    async signOut({ token }) {
      // Keycloak 세션도 함께 로그아웃
      if (token?.idToken) {
        const logoutUrl = `${keycloakIssuerInternal}/protocol/openid-connect/logout`
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
    signIn: "/api/auth/signin", // Custom signin page (자동 Keycloak 리다이렉트)
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
    // 개발 환경에서 self-signed certificate 허용
    const fetchOptions: RequestInit = {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    }

    // Node.js 환경에서만 사용 가능
    if (process.env.NODE_ENV === "development" && typeof process !== "undefined") {
      // @ts-ignore - Node.js only
      process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"
    }

    const response = await fetch(
      `${keycloakIssuerInternal}/protocol/openid-connect/token`,
      {
        ...fetchOptions,
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
