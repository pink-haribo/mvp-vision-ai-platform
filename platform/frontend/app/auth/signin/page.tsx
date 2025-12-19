'use client'

import { useEffect } from 'react'
import { signIn } from 'next-auth/react'
import { useSearchParams } from 'next/navigation'

/**
 * 커스텀 로그인 페이지
 * NextAuth 기본 signin 페이지 대신 사용하여 바로 Keycloak으로 리다이렉트
 */
export default function SignInPage() {
  const searchParams = useSearchParams()
  const callbackUrl = searchParams.get('callbackUrl') || '/'

  useEffect(() => {
    // 페이지 로드 시 바로 Keycloak 로그인 시작
    signIn('keycloak', { callbackUrl })
  }, [callbackUrl])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">SSO 로그인 페이지로 이동 중...</p>
      </div>
    </div>
  )
}
