'use client'

import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils'
import Modal from './Modal'

interface ProfileModalProps {
  isOpen: boolean
  onClose: () => void
}

const COMPANIES = ['삼성전자', '협력사', '직접 입력']
const DIVISIONS = ['생산기술연구소', 'MX', 'VD', 'DA', 'SR', '직접 입력']

export default function ProfileModal({
  isOpen,
  onClose,
}: ProfileModalProps) {
  const { user } = useAuth()

  const [fullName, setFullName] = useState('')
  const [company, setCompany] = useState('')
  const [companyCustom, setCompanyCustom] = useState('')
  const [division, setDivision] = useState('')
  const [divisionCustom, setDivisionCustom] = useState('')
  const [department, setDepartment] = useState('')

  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPasswordChange, setShowPasswordChange] = useState(false)

  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // Load user data when modal opens
  useEffect(() => {
    if (isOpen && user) {
      setFullName(user.full_name || '')
      setCompany(user.company || '')
      setDivision(user.division || '')
      setDepartment(user.department || '')
      setError('')
      setSuccess('')
    }
  }, [isOpen, user])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSuccess('')

    // Validate custom fields
    if (company === '직접 입력' && !companyCustom) {
      setError('회사명을 입력해주세요')
      return
    }

    if (division === '직접 입력' && !divisionCustom) {
      setError('사업부명을 입력해주세요')
      return
    }

    // Validate password if changing
    if (showPasswordChange) {
      if (newPassword.length > 0 && newPassword.length < 8) {
        setError('비밀번호는 최소 8자 이상이어야 합니다')
        return
      }

      if (newPassword !== confirmPassword) {
        setError('비밀번호가 일치하지 않습니다')
        return
      }
    }

    setIsLoading(true)

    try {
      const token = localStorage.getItem('access_token')
      if (!token) {
        throw new Error('인증 토큰이 없습니다')
      }

      const updateData: any = {
        full_name: fullName || null,
        company: company || null,
        company_custom: company === '직접 입력' ? companyCustom : null,
        division: division || null,
        division_custom: division === '직접 입력' ? divisionCustom : null,
        department: department || null,
      }

      if (showPasswordChange && newPassword) {
        updateData.password = newPassword
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(updateData)
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || '프로필 업데이트에 실패했습니다')
      }

      setSuccess('프로필이 성공적으로 업데이트되었습니다')
      setShowPasswordChange(false)
      setNewPassword('')
      setConfirmPassword('')

      // Close modal after 1.5 seconds
      setTimeout(() => {
        onClose()
        window.location.reload() // Reload to update user info in sidebar
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : '프로필 업데이트에 실패했습니다')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="프로필 설정" size="md">
      <div className="max-h-[70vh] overflow-y-auto pr-2">
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Success Message */}
        {success && (
          <div className="mb-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
            <p className="text-sm text-green-400">{success}</p>
          </div>
        )}

        {/* Read-only Information */}
        <div className="mb-6 space-y-3">
          <div className="p-4 bg-gray-800 rounded-lg">
            <p className="text-xs text-gray-400 mb-1">이메일</p>
            <p className="text-sm text-white">{user?.email}</p>
          </div>
          <div className="p-4 bg-gray-800 rounded-lg">
            <p className="text-xs text-gray-400 mb-1">사용자명</p>
            <p className="text-sm text-white">{user?.username}</p>
          </div>
        </div>

        {/* Profile Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-300">기본 정보</h3>

            <div>
              <label htmlFor="fullName" className="block text-sm font-medium text-gray-300 mb-2">
                이름 (선택사항)
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
              <p className="mt-1 text-xs text-gray-500">
                입력하지 않으면 사용자명이 표시됩니다
              </p>
            </div>
          </div>

          {/* Organization Information */}
          <div className="space-y-4 pt-4 border-t border-gray-800">
            <h3 className="text-sm font-semibold text-gray-300">조직 정보</h3>

            <div>
              <label htmlFor="company" className="block text-sm font-medium text-gray-300 mb-2">
                회사
              </label>
              <select
                id="company"
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                className={cn(
                  'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-white'
                )}
              >
                <option value="">선택하세요</option>
                {COMPANIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {company === '직접 입력' && (
              <div>
                <label htmlFor="companyCustom" className="block text-sm font-medium text-gray-300 mb-2">
                  회사명 입력
                </label>
                <input
                  id="companyCustom"
                  type="text"
                  value={companyCustom}
                  onChange={(e) => setCompanyCustom(e.target.value)}
                  className={cn(
                    'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-white placeholder-gray-500'
                  )}
                  placeholder="회사명을 입력하세요"
                />
              </div>
            )}

            <div>
              <label htmlFor="division" className="block text-sm font-medium text-gray-300 mb-2">
                사업부
              </label>
              <select
                id="division"
                value={division}
                onChange={(e) => setDivision(e.target.value)}
                className={cn(
                  'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-white'
                )}
              >
                <option value="">선택하세요</option>
                {DIVISIONS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            {division === '직접 입력' && (
              <div>
                <label htmlFor="divisionCustom" className="block text-sm font-medium text-gray-300 mb-2">
                  사업부명 입력
                </label>
                <input
                  id="divisionCustom"
                  type="text"
                  value={divisionCustom}
                  onChange={(e) => setDivisionCustom(e.target.value)}
                  className={cn(
                    'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-white placeholder-gray-500'
                  )}
                  placeholder="사업부명을 입력하세요"
                />
              </div>
            )}

            <div>
              <label htmlFor="department" className="block text-sm font-medium text-gray-300 mb-2">
                부서
              </label>
              <input
                id="department"
                type="text"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                className={cn(
                  'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-white placeholder-gray-500'
                )}
                placeholder="개발팀"
              />
            </div>
          </div>

          {/* Password Change */}
          <div className="space-y-4 pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-300">비밀번호 변경</h3>
              <button
                type="button"
                onClick={() => setShowPasswordChange(!showPasswordChange)}
                className="text-xs text-violet-400 hover:text-violet-300"
              >
                {showPasswordChange ? '취소' : '변경하기'}
              </button>
            </div>

            {showPasswordChange && (
              <>
                <div>
                  <label htmlFor="newPassword" className="block text-sm font-medium text-gray-300 mb-2">
                    새 비밀번호 (최소 8자)
                  </label>
                  <input
                    id="newPassword"
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className={cn(
                      'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                      'text-white placeholder-gray-500'
                    )}
                    placeholder="••••••••"
                  />
                </div>

                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-300 mb-2">
                    비밀번호 확인
                  </label>
                  <input
                    id="confirmPassword"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className={cn(
                      'w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                      'text-white placeholder-gray-500'
                    )}
                    placeholder="••••••••"
                  />
                </div>
              </>
            )}
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
            {isLoading ? '업데이트 중...' : '프로필 업데이트'}
          </button>
        </form>
      </div>
    </Modal>
  )
}
