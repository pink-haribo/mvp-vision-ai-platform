'use client'

import { useEffect, useState } from 'react'
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
  const [debugInfo, setDebugInfo] = useState<string[]>([])

  useEffect(() => {
    const log = (message: string) => {
      console.log(`[Logout Success] ${message}`)
      setDebugInfo(prev => [...prev, `${new Date().toISOString().split('T')[1]} - ${message}`])
    }

    log('Page mounted')

    // NextAuth callback-url 쿠키 삭제 (재로그인 시 이 페이지로 리다이렉트 방지)
    document.cookie = 'next-auth.callback-url=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT'
    document.cookie = '__Secure-next-auth.callback-url=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; secure'
    log('Cleared callback-url cookies')

    log('Starting signOut...')

    // NextAuth 클라이언트 세션 정리
    signOut({ redirect: false, callbackUrl: '/' })
      .then(() => {
        log('signOut completed successfully')
        log('Navigating to home...')
        // 메인 페이지로 이동 (히스토리에 남기지 않음)
        window.location.replace('/')
        log('window.location.replace called')
      })
      .catch((error) => {
        log(`signOut failed: ${error.message}`)
      })

    return () => {
      log('Component unmounting')
    }
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-violet-600 mx-auto"></div>
        <p className="mt-6 text-lg text-gray-700">로그아웃 중입니다...</p>

        {/* Debug Info */}
        {process.env.NODE_ENV === 'development' && (
          <div className="mt-8 p-4 bg-gray-100 rounded text-left max-w-md mx-auto">
            <p className="text-xs font-mono text-gray-600 mb-2">Debug Log:</p>
            {debugInfo.map((info, idx) => (
              <p key={idx} className="text-xs font-mono text-gray-800">{info}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
