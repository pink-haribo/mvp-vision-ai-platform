'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils/cn'

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      await login(email, password)
      router.push('/') // Redirect to home after successful login
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setIsLoading(false)
    }
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
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                이메일
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className={cn(
                  'w-full px-4 py-3 border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-gray-900'
                )}
                placeholder="your@email.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                비밀번호
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className={cn(
                  'w-full px-4 py-3 border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-gray-900'
                )}
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className={cn(
                'w-full py-3 px-4 rounded-lg font-semibold',
                'bg-violet-600 hover:bg-violet-700 text-white',
                'transition-all duration-200',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'shadow-lg hover:shadow-xl'
              )}
            >
              {isLoading ? '로그인 중...' : '로그인'}
            </button>
          </form>

          {/* Divider */}
          <div className="mt-8 pt-8 border-t border-gray-200">
            <p className="text-center text-sm text-gray-600">
              계정이 없으신가요?{' '}
              <Link
                href="/register"
                className="font-semibold text-violet-600 hover:text-violet-700"
              >
                회원가입
              </Link>
            </p>
          </div>

          {/* Demo Credentials */}
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800 font-medium mb-2">데모 계정:</p>
            <p className="text-xs text-blue-700">
              이메일: admin@example.com<br />
              비밀번호: admin123
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
