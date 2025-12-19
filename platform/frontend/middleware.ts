import { withAuth } from "next-auth/middleware"

export default withAuth({
  pages: {
    signIn: "/api/auth/signin", // NextAuth 기본 로그인 (Keycloak 리다이렉트)
    error: "/auth/error",
  },
})

export const config = {
  matcher: [
    /*
     * 인증이 필요한 모든 경로
     * 제외:
     * - api/auth (NextAuth 엔드포인트)
     * - _next/static (정적 파일)
     * - _next/image (이미지 최적화)
     * - favicon.ico
     * - auth/error (에러 페이지)
     */
    "/((?!api/auth|_next/static|_next/image|favicon.ico|auth/error).*)",
  ],
}
