'use client'

import { useSearchParams } from 'next/navigation'
import { signIn } from 'next-auth/react'

export default function AuthErrorPage() {
  const searchParams = useSearchParams()
  const error = searchParams.get('error')

  const getErrorMessage = (errorCode: string | null) => {
    const errorMessages: Record<string, string> = {
      'Configuration': '서버 설정 오류가 발생했습니다.',
      'AccessDenied': '접근이 거부되었습니다.',
      'Verification': '인증 링크가 만료되었거나 이미 사용되었습니다.',
      'OAuthSignin': '로그인을 시작할 수 없습니다.',
      'OAuthCallback': '인증 처리 중 오류가 발생했습니다.',
      'OAuthCreateAccount': '계정 생성 중 오류가 발생했습니다.',
      'OAuthAccountNotLinked': '이 이메일은 다른 방식으로 등록되어 있습니다.',
      'SessionRequired': '로그인이 필요합니다.',
      'Default': '인증 중 오류가 발생했습니다.',
    }
    return errorMessages[errorCode || 'Default'] || errorMessages['Default']
  }

  const handleRetry = () => {
    signIn('keycloak', { callbackUrl: '/' })
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100">
      <div className="max-w-md w-full mx-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>

          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            인증 오류
          </h1>

          <p className="text-gray-600 mb-6">
            {getErrorMessage(error)}
          </p>

          {error && (
            <p className="text-xs text-gray-400 mb-6">
              오류 코드: {error}
            </p>
          )}

          <button
            onClick={handleRetry}
            className="w-full py-3 px-4 bg-violet-600 hover:bg-violet-700 text-white font-semibold rounded-lg transition-colors"
          >
            다시 로그인
          </button>
        </div>
      </div>
    </div>
  )
}
