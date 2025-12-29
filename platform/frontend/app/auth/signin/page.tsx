'use client'

import { useEffect } from 'react'
import { signIn } from 'next-auth/react'
import { useSearchParams } from 'next/navigation'

/**
 * Custom NextAuth Signin Page
 *
 * NextAuth의 기본 signin 페이지 대신 사용하여 자동으로 Keycloak으로 리다이렉트합니다.
 * 사용자는 이 페이지를 거의 보지 못하며 (즉시 리다이렉트), state cookie가 정상적으로 생성됩니다.
 */
export default function SignInPage() {
  const searchParams = useSearchParams()
  const callbackUrl = searchParams.get('callbackUrl') || '/'

  useEffect(() => {
    // 페이지 로드 즉시 Keycloak 로그인 시작
    // NextAuth가 state cookie를 생성하고 CSRF 보호를 제공합니다
    signIn('keycloak', { callbackUrl })
  }, [callbackUrl])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-violet-600 mx-auto"></div>
        <p className="mt-6 text-lg font-medium text-gray-700">SSO 로그인 페이지로 이동 중...</p>
        <p className="mt-2 text-sm text-gray-500">잠시만 기다려주세요</p>
      </div>
    </div>
  )
}
