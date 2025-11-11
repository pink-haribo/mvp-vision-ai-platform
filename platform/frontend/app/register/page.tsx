'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils/cn'

const COMPANIES = ['삼성전자', '협력사', '직접 입력']
const DIVISIONS = ['생산기술연구소', 'MX', 'VD', 'DA', 'SR', '직접 입력']

export default function RegisterPage() {
  const router = useRouter()
  const { register } = useAuth()

  // Basic fields
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [fullName, setFullName] = useState('')

  // Organization fields
  const [company, setCompany] = useState('')
  const [companyCustom, setCompanyCustom] = useState('')
  const [division, setDivision] = useState('')
  const [divisionCustom, setDivisionCustom] = useState('')
  const [department, setDepartment] = useState('')

  // Contact & bio
  const [phoneNumber, setPhoneNumber] = useState('')
  const [bio, setBio] = useState('')

  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (password.length < 8) {
      setError('비밀번호는 최소 8자 이상이어야 합니다')
      return
    }

    if (password !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다')
      return
    }

    if (company === '직접 입력' && !companyCustom) {
      setError('회사명을 입력해주세요')
      return
    }

    if (division === '직접 입력' && !divisionCustom) {
      setError('사업부명을 입력해주세요')
      return
    }

    setIsLoading(true)

    try {
      await register({
        email,
        password,
        full_name: fullName || undefined,
        company: company || undefined,
        company_custom: company === '직접 입력' ? companyCustom : undefined,
        division: division || undefined,
        division_custom: division === '직접 입력' ? divisionCustom : undefined,
        department: department || undefined,
        phone_number: phoneNumber || undefined,
        bio: bio || undefined,
      })
      router.push('/')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-indigo-100 py-12">
      <div className="max-w-2xl w-full mx-4">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Vision AI Platform</h1>
            <p className="text-gray-600">새로운 계정을 만드세요</p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">기본 정보</h3>

              <div>
                <label htmlFor="fullName" className="block text-sm font-medium text-gray-700 mb-2">
                  이름 (선택)
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

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                  이메일 <span className="text-red-500">*</span>
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
                  비밀번호 <span className="text-red-500">*</span>
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
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
                  비밀번호 확인 <span className="text-red-500">*</span>
                </label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className={cn(
                    'w-full px-4 py-3 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-gray-900'
                  )}
                  placeholder="비밀번호 재입력"
                />
              </div>
            </div>

            {/* Organization Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">조직 정보 (선택)</h3>

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
              <h3 className="text-lg font-semibold text-gray-900 border-b pb-2">연락처 & 소개 (선택)</h3>

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
              {isLoading ? '가입 중...' : '회원가입'}
            </button>
          </form>

          <div className="mt-8 pt-8 border-t border-gray-200">
            <p className="text-center text-sm text-gray-600">
              이미 계정이 있으신가요?{' '}
              <Link
                href="/login"
                className="font-semibold text-violet-600 hover:text-violet-700"
              >
                로그인
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
