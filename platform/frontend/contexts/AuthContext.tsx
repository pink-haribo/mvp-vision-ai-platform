'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'

interface User {
  id: number
  email: string
  full_name: string | null
  is_active: boolean
  system_role: string  // '''guest''' | '''standard_engineer''' | '''advanced_engineer''' | '''manager''' | '''admin'''
  badge_color?: string | null
  company?: string | null
  division?: string | null
  department?: string | null
  phone_number?: string | null
  bio?: string | null
}

interface RegisterData {
  email: string
  password: string
  full_name?: string
  company?: string
  company_custom?: string
  division?: string
  division_custom?: string
  department?: string
  phone_number?: string
  bio?: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => void
  refreshToken: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Check for existing token on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('access_token')
      if (token) {
        try {
          // Fetch current user info
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          })

          if (response.ok) {
            const userData = await response.json()
            setUser(userData)
          } else {
            // Token invalid, clear it
            localStorage.removeItem('access_token')
            localStorage.removeItem('refresh_token')
          }
        } catch (error) {
          console.error('Error checking auth:', error)
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
        }
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const login = async (email: string, password: string) => {
    // OAuth2 format requires FormData with 'username' and 'password' fields
    const formData = new FormData()
    formData.append('username', email)  // OAuth2 uses 'username' field for email
    formData.append('password', password)

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/login`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Login failed')
    }

    const data = await response.json()

    // Store tokens
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)

    // Fetch user info
    const userResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${data.access_token}`
      }
    })

    if (userResponse.ok) {
      const userData = await userResponse.json()
      setUser(userData)
    }
  }

  const register = async (data: RegisterData) => {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Registration failed')
    }

    // Auto-login after registration
    await login(data.email, data.password)
  }

  const logout = () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setUser(null)
  }

  const refreshToken = async () => {
    const refresh = localStorage.getItem('refresh_token')
    if (!refresh) {
      logout()
      return
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ refresh_token: refresh })
      })

      if (response.ok) {
        const data = await response.json()
        localStorage.setItem('access_token', data.access_token)
        localStorage.setItem('refresh_token', data.refresh_token)
      } else {
        logout()
      }
    } catch (error) {
      console.error('Error refreshing token:', error)
      logout()
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        refreshToken
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
