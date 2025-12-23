'use client'

import { useEffect } from 'react'
import { signOut } from 'next-auth/react'
import { useRouter } from 'next/navigation'

/**
 * Logout Success Page
 *
 * Keycloak 로그아웃 완료 후 이 페이지로 리다이렉트됨
 * NextAuth 클라이언트 세션을 정리하고 메인 페이지로 이동
 */
export default function LogoutSuccessPage() {
  const router = useRouter()

  useEffect(() => {
    // NextAuth 클라이언트 세션 정리
    signOut({ redirect: false }).then(() => {
      // 메인 페이지로 이동 (파라미터 없이)
      router.push('/')
    })
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-violet-600 mx-auto"></div>
        <p className="mt-6 text-lg text-gray-700">로그아웃 중입니다...</p>
      </div>
    </div>
  )
}
