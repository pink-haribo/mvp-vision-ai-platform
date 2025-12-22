'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'
import { useSession, signIn, signOut } from 'next-auth/react'

interface User {
  id: number
  email: string
  full_name: string | null
  is_active: boolean
  system_role: string
  badge_color?: string | null
  company?: string | null
  division?: string | null
  department?: string | null
  phone_number?: string | null
  bio?: string | null
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  accessToken: string | null
  login: () => Promise<void>
  logout: () => Promise<void>
  error: string | null
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { data: session, status } = useSession()
  const [user, setUser] = useState<User | null>(null)
  const [error, setError] = useState<string | null>(null)

  const isLoading = status === 'loading'
  const accessToken = session?.accessToken ?? null

  // 세션 변경 시 사용자 정보 가져오기
  useEffect(() => {
    if (session?.accessToken) {
      fetchUserInfo(session.accessToken)
    } else if (status === 'unauthenticated') {
      setUser(null)
    }
  }, [session, status])

  // 세션 에러 처리 (토큰 갱신 실패 등)
  useEffect(() => {
    if (session?.error === 'RefreshAccessTokenError') {
      setError('세션이 만료되었습니다. 다시 로그인해주세요.')
      // 자동 로그아웃
      signOut({ callbackUrl: '/' })
    }
  }, [session?.error])

  async function fetchUserInfo(token: string) {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })

      if (response.ok) {
        const userData = await response.json()
        setUser(userData)
        setError(null)
      } else {
        // Backend에서 사용자 정보 가져오기 실패
        const errorData = await response.json().catch(() => ({}))
        console.error('Failed to fetch user info:', errorData)
        setUser(null)

        if (response.status === 401) {
          setError('인증이 만료되었습니다.')
        }
      }
    } catch (err) {
      console.error('Error fetching user info:', err)
      setUser(null)
      setError('서버에 연결할 수 없습니다.')
    }
  }

  async function login() {
    setError(null)
    // Keycloak 로그인 페이지로 리다이렉트
    await signIn('keycloak', { callbackUrl: '/' })
  }

  async function logout() {
    setError(null)
    setUser(null)
    // Custom logout endpoint로 리다이렉트 (NextAuth + Keycloak 세션 모두 삭제)
    window.location.href = '/api/auth/logout'
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user && !!session,
        isLoading,
        accessToken,
        login,
        logout,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
