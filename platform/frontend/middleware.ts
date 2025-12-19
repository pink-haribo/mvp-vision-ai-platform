import { withAuth } from "next-auth/middleware"

export default withAuth({
  pages: {
    signIn: "/auth/signin/keycloak", // Keycloak으로 바로 리다이렉트
    error: "/auth/error",
  },
})

export const config = {
  matcher: [
    /*
     * 인증이 필요한 모든 경로
     * 제외:
     * - auth (NextAuth 엔드포인트 - /api/auth에서 /auth로 변경)
     * - _next/static (정적 파일)
     * - _next/image (이미지 최적화)
     * - favicon.ico
     */
    "/((?!auth|_next/static|_next/image|favicon.ico).*)",
  ],
}
