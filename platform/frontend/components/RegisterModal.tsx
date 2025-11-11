'use client'

import { useState } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils'
import Modal from './Modal'

interface RegisterModalProps {
  isOpen: boolean
  onClose: () => void
  onSwitchToLogin?: () => void
}

export default function RegisterModal({
  isOpen,
  onClose,
  onSwitchToLogin,
}: RegisterModalProps) {
  const { register } = useAuth()

  const [email, setEmail] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [fullName, setFullName] = useState('')

  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (username.length < 3) {
      setError('사용자명은 최소 3자 이상이어야 합니다')
      return
    }

    if (password.length < 8) {
      setError('비밀번호는 최소 8자 이상이어야 합니다')
      return
    }

    if (password !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다')
      return
    }

    setIsLoading(true)

    try {
      await register({
        email,
        username,
        password,
        full_name: fullName || undefined,
      })
      // Reset form
      setEmail('')
      setUsername('')
      setPassword('')
      setConfirmPassword('')
      setFullName('')
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSwitchToLogin = () => {
    setEmail('')
    setUsername('')
    setPassword('')
    setConfirmPassword('')
    setFullName('')
    setError('')
    onSwitchToLogin?.()
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="회원가입" size="md">
      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Register Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Email */}
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
            이메일 *
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className={cn(
              'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'text-white placeholder-gray-500'
            )}
            placeholder="your@email.com"
          />
        </div>

        {/* Username */}
        <div>
          <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
            사용자명 *
          </label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            minLength={3}
            className={cn(
              'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'text-white placeholder-gray-500'
            )}
            placeholder="username"
          />
          <p className="mt-1 text-xs text-gray-500">
            영문, 숫자, 언더스코어만 사용 가능 (최소 3자)
          </p>
        </div>

        {/* Full Name */}
        <div>
          <label htmlFor="fullName" className="block text-sm font-medium text-gray-300 mb-2">
            이름
          </label>
          <input
            id="fullName"
            type="text"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className={cn(
              'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'text-white placeholder-gray-500'
            )}
            placeholder="홍길동"
          />
        </div>

        {/* Password */}
        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
            비밀번호 *
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            className={cn(
              'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'text-white placeholder-gray-500'
            )}
            placeholder="••••••••"
          />
          <p className="mt-1 text-xs text-gray-500">최소 8자 이상</p>
        </div>

        {/* Confirm Password */}
        <div>
          <label
            htmlFor="confirmPassword"
            className="block text-sm font-medium text-gray-300 mb-2"
          >
            비밀번호 확인 *
          </label>
          <input
            id="confirmPassword"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            minLength={8}
            className={cn(
              'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'text-white placeholder-gray-500'
            )}
            placeholder="••••••••"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className={cn(
            'w-full py-3 px-4 rounded-lg font-medium transition-all',
            'bg-gradient-to-r from-violet-600 to-fuchsia-600',
            'hover:from-violet-700 hover:to-fuchsia-700',
            'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:ring-offset-2',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'text-white'
          )}
        >
          {isLoading ? '가입 중...' : '회원가입'}
        </button>

        {/* Switch to Login */}
        <div className="text-center pt-4">
          <p className="text-sm text-gray-400">
            이미 계정이 있으신가요?{' '}
            <button
              type="button"
              onClick={handleSwitchToLogin}
              className="text-violet-400 hover:text-violet-300 font-medium transition-colors"
            >
              로그인
            </button>
          </p>
        </div>
      </form>
    </Modal>
  )
}
