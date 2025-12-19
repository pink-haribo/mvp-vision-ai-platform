import { withAuth } from "next-auth/middleware"

export default withAuth({
  pages: {
    signIn: "/auth/signin", // 커스텀 로그인 페이지 (자동으로 Keycloak 리다이렉트)
    error: "/auth/error",
  },
})

export const config = {
  matcher: [
    /*
     * 인증이 필요한 모든 경로
     * 제외:
     * - auth (NextAuth 엔드포인트)
     * - _next/static (정적 파일)
     * - _next/image (이미지 최적화)
     * - favicon.ico
     */
    "/((?!auth|_next/static|_next/image|favicon.ico).*)",
  ],
}
