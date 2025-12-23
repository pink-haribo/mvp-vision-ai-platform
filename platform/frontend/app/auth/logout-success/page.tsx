'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

/**
 * Logout Success Page
 *
 * Keycloak 로그아웃 후 리다이렉트되는 페이지
 * 잠시 메시지를 보여준 후 로그인 페이지로 이동
 */
export default function LogoutSuccessPage() {
  const router = useRouter()

  useEffect(() => {
    // 2초 후 로그인 페이지로 리다이렉트
    const timer = setTimeout(() => {
      router.push('/')
    }, 2000)

    return () => clearTimeout(timer)
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100">
      <div className="max-w-md w-full mx-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>

          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            로그아웃 완료
          </h1>

          <p className="text-gray-600 mb-6">
            성공적으로 로그아웃되었습니다.
          </p>

          <div className="text-sm text-gray-500">
            잠시 후 로그인 페이지로 이동합니다...
          </div>
        </div>
      </div>
    </div>
  )
}
