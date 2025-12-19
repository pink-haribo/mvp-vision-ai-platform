'use client'

import { useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils/cn'

export default function LoginPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { login, isAuthenticated, isLoading, error: authError } = useAuth()

  const error = searchParams.get('error')

  // 이미 로그인되어 있으면 홈으로 리다이렉트
  useEffect(() => {
    if (isAuthenticated) {
      router.push('/')
    }
  }, [isAuthenticated, router])

  const handleLogin = async () => {
    await login()
  }

  // 에러 메시지 매핑
  const getErrorMessage = (errorCode: string | null) => {
    if (!errorCode) return null

    const errorMessages: Record<string, string> = {
      'OAuthSignin': 'Keycloak 로그인을 시작할 수 없습니다.',
      'OAuthCallback': '인증 콜백 처리 중 오류가 발생했습니다.',
      'OAuthCreateAccount': '계정 생성 중 오류가 발생했습니다.',
      'EmailCreateAccount': '이메일 계정 생성 중 오류가 발생했습니다.',
      'Callback': '인증 처리 중 오류가 발생했습니다.',
      'OAuthAccountNotLinked': '이 이메일은 다른 로그인 방식으로 등록되어 있습니다.',
      'EmailSignin': '이메일 전송에 실패했습니다.',
      'CredentialsSignin': '로그인 정보가 올바르지 않습니다.',
      'SessionRequired': '로그인이 필요합니다.',
      'Default': '인증 중 오류가 발생했습니다.',
    }

    return errorMessages[errorCode] || errorMessages['Default']
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-600 mx-auto mb-4"></div>
          <p className="text-gray-600">로딩 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100">
      <div className="max-w-md w-full mx-4">
        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Logo/Title */}
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Vision AI Platform
            </h1>
            <p className="text-gray-600">로그인하여 시작하세요</p>
          </div>

          {/* Error Message */}
          {(error || authError) && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800 whitespace-pre-line">
                {getErrorMessage(error) || authError}
              </p>
            </div>
          )}

          {/* SSO Login Button */}
          <button
            onClick={handleLogin}
            className={cn(
              'w-full py-4 px-4 rounded-lg font-semibold',
              'bg-violet-600 hover:bg-violet-700 text-white',
              'transition-all duration-200',
              'shadow-lg hover:shadow-xl',
              'flex items-center justify-center gap-3'
            )}
          >
            <svg
              className="w-5 h-5"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
            SSO로 로그인
          </button>

          {/* Info */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800 font-medium mb-2">SSO 로그인 안내</p>
            <p className="text-xs text-blue-700">
              Keycloak을 통한 통합 인증으로 로그인합니다.<br />
              회사 계정으로 로그인해주세요.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
