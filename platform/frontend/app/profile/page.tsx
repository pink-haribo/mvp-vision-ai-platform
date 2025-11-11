'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils/cn'

const COMPANIES = ['삼성전자', '협력사', '직접 입력']
const DIVISIONS = ['생산기술연구소', 'MX', 'VD', 'DA', 'SR', '직접 입력']

export default function ProfilePage() {
  const router = useRouter()
  const { user, isAuthenticated, isLoading: authLoading } = useAuth()

  const [fullName, setFullName] = useState('')
  const [company, setCompany] = useState('')
  const [companyCustom, setCompanyCustom] = useState('')
  const [division, setDivision] = useState('')
  const [divisionCustom, setDivisionCustom] = useState('')
  const [department, setDepartment] = useState('')
  const [phoneNumber, setPhoneNumber] = useState('')
  const [bio, setBio] = useState('')

  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPasswordChange, setShowPasswordChange] = useState(false)

  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
    }
  }, [authLoading, isAuthenticated, router])

  // Load user data
  useEffect(() => {
    if (user) {
      setFullName(user.full_name || '')
      setCompany(user.company || '')
      setDivision(user.division || '')
      setDepartment(user.department || '')
      setPhoneNumber(user.phone_number || '')
      setBio(user.bio || '')
    }
  }, [user])

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
        phone_number: phoneNumber || null,
        bio: bio || null,
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

      // Reload user data
      window.location.reload()
    } catch (err) {
      setError(err instanceof Error ? err.message : '프로필 업데이트에 실패했습니다')
    } finally {
      setIsLoading(false)
    }
  }

  if (authLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">로딩 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 to-indigo-100 py-12">
      <div className="max-w-2xl mx-auto px-4">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">프로필 설정</h1>
            <p className="text-gray-600">개인 정보를 관리합니다</p>
          </div>

          {/* User Info Card */}
          <div className="mb-8 p-4 bg-violet-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">이메일</p>
                <p className="font-semibold text-gray-900">{user.email}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">권한</p>
                <p className="font-semibold text-violet-600">{user.system_role}</p>
              </div>
            </div>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {success && (
            <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">{success}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">기본 정보</h3>

              <div>
                <label htmlFor="fullName" className="block text-sm font-medium text-gray-700 mb-2">
                  이름
                </label>
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900'
                  )}
                  placeholder="홍길동"
                />
              </div>
            </div>

            {/* Organization Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">조직 정보</h3>

              <div>
                <label htmlFor="company" className="block text-sm font-medium text-gray-700 mb-2">
                  회사
                </label>
                <select
                  id="company"
                  value={company}
                  onChange={(e) => {
                    setCompany(e.target.value)
                    if (e.target.value !== '직접 입력') {
                      setCompanyCustom('')
                    }
                  }}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900 bg-white'
                  )}
                >
                  <option value="">선택하세요</option>
                  {COMPANIES.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              {company === '직접 입력' && (
                <div>
                  <label htmlFor="companyCustom" className="block text-sm font-medium text-gray-700 mb-2">
                    회사명 입력
                  </label>
                  <input
                    id="companyCustom"
                    type="text"
                    value={companyCustom}
                    onChange={(e) => setCompanyCustom(e.target.value)}
                    className={cn(
                      'w-full px-4 py-3 border border-gray-300 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                      'text-gray-900'
                    )}
                    placeholder="회사명을 입력하세요"
                  />
                </div>
              )}

              <div>
                <label htmlFor="division" className="block text-sm font-medium text-gray-700 mb-2">
                  사업부
                </label>
                <select
                  id="division"
                  value={division}
                  onChange={(e) => {
                    setDivision(e.target.value)
                    if (e.target.value !== '직접 입력') {
                      setDivisionCustom('')
                    }
                  }}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900 bg-white'
                  )}
                >
                  <option value="">선택하세요</option>
                  {DIVISIONS.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>

              {division === '직접 입력' && (
                <div>
                  <label htmlFor="divisionCustom" className="block text-sm font-medium text-gray-700 mb-2">
                    사업부명 입력
                  </label>
                  <input
                    id="divisionCustom"
                    type="text"
                    value={divisionCustom}
                    onChange={(e) => setDivisionCustom(e.target.value)}
                    className={cn(
                      'w-full px-4 py-3 border border-gray-300 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                      'text-gray-900'
                    )}
                    placeholder="사업부명을 입력하세요"
                  />
                </div>
              )}

              <div>
                <label htmlFor="department" className="block text-sm font-medium text-gray-700 mb-2">
                  부서
                </label>
                <input
                  id="department"
                  type="text"
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900'
                  )}
                  placeholder="AI 개발팀"
                />
              </div>
            </div>

            {/* Contact & Bio */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">연락처 & 소개</h3>

              <div>
                <label htmlFor="phoneNumber" className="block text-sm font-medium text-gray-700 mb-2">
                  전화번호
                </label>
                <input
                  id="phoneNumber"
                  type="tel"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900'
                  )}
                  placeholder="010-1234-5678"
                />
              </div>

              <div>
                <label htmlFor="bio" className="block text-sm font-medium text-gray-700 mb-2">
                  소개
                </label>
                <textarea
                  id="bio"
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  rows={3}
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900 resize-none'
                  )}
                  placeholder="컴퓨터 비전 엔지니어"
                />
              </div>
            </div>

            {/* Password Change */}
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b pb-2">
                <h3 className="text-lg font-semibold text-gray-900">비밀번호 변경</h3>
                <button
                  type="button"
                  onClick={() => setShowPasswordChange(!showPasswordChange)}
                  className="text-sm text-violet-600 hover:text-violet-700"
                >
                  {showPasswordChange ? '취소' : '변경하기'}
                </button>
              </div>

              {showPasswordChange && (
                <>
                  <div>
                    <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 mb-2">
                      새 비밀번호
                    </label>
                    <input
                      id="newPassword"
                      type="password"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      minLength={8}
                      className={cn(
                        'w-full px-4 py-3 border border-gray-300 rounded-lg',
                        'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                        'text-gray-900'
                      )}
                      placeholder="최소 8자 이상"
                    />
                  </div>

                  <div>
                    <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-2">
                      비밀번호 확인
                    </label>
                    <input
                      id="confirmPassword"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className={cn(
                        'w-full px-4 py-3 border border-gray-300 rounded-lg',
                        'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                        'text-gray-900'
                      )}
                      placeholder="비밀번호 재입력"
                    />
                  </div>
                </>
              )}
            </div>

            <div className="flex gap-4">
              <button
                type="submit"
                disabled={isLoading}
                className={cn(
                  'flex-1 py-3 px-4 rounded-lg font-semibold',
                  'bg-violet-600 hover:bg-violet-700 text-white',
                  'transition-all duration-200',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'shadow-lg hover:shadow-xl'
                )}
              >
                {isLoading ? '저장 중...' : '변경사항 저장'}
              </button>

              <button
                type="button"
                onClick={() => router.push('/')}
                className={cn(
                  'px-6 py-3 rounded-lg font-semibold',
                  'bg-gray-200 hover:bg-gray-300 text-gray-700',
                  'transition-all duration-200'
                )}
              >
                취소
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
