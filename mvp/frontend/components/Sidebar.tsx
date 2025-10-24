'use client'

import { useState, useEffect, useRef } from 'react'
import { User, FolderIcon, PlusIcon, Settings, LogOut } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { cn } from '@/lib/utils/cn'
import { useAuth } from '@/contexts/AuthContext'

interface Project {
  id: number
  name: string
  description: string | null
  task_type: string | null
  created_at: string
  updated_at: string
  experiment_count: number
}

interface SidebarProps {
  onProjectSelect?: (projectId: number) => void
  selectedProjectId?: number | null
  onCreateProject?: () => void
  onOpenLogin?: () => void
  onOpenRegister?: () => void
  onOpenProfile?: () => void
}

export default function Sidebar({
  onProjectSelect,
  selectedProjectId,
  onCreateProject,
  onOpenLogin,
  onOpenRegister,
  onOpenProfile,
}: SidebarProps) {
  const router = useRouter()
  const { user: authUser, isAuthenticated, logout } = useAuth()
  const [projects, setProjects] = useState<Project[]>([])
  const [loadingProjects, setLoadingProjects] = useState(false)
  const [showUserMenu, setShowUserMenu] = useState(false)
  const userMenuRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  // Fetch recent projects only when authenticated
  useEffect(() => {
    if (isAuthenticated) {
      fetchRecentProjects()
    } else {
      setProjects([])
    }
  }, [isAuthenticated])

  const fetchRecentProjects = async () => {
    setLoadingProjects(true)
    try {
      const token = localStorage.getItem('access_token')
      if (!token) {
        setProjects([])
        setLoadingProjects(false)
        return
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        // Get top 5 most recent projects (excluding "Uncategorized")
        const filtered = data
          .filter((p: Project) => p.name !== 'Uncategorized')
          .sort((a: Project, b: Project) =>
            new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
          )
          .slice(0, 5)
        setProjects(filtered)
      } else if (response.status === 401) {
        // Unauthorized - clear projects
        setProjects([])
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
      setProjects([])
    } finally {
      setLoadingProjects(false)
    }
  }

  const handleProjectClick = (projectId: number) => {
    // Select the project (workspace will update automatically)
    onProjectSelect?.(projectId)
  }

  const handleLogout = () => {
    logout()
    setShowUserMenu(false)
    router.push('/')  // 메인 페이지로 이동 (로그아웃 상태의 플랫폼 화면)
  }

  const handleSettings = () => {
    setShowUserMenu(false)
    onOpenProfile?.()
  }

  // Generate avatar initials from user name or email
  const getAvatarInitials = () => {
    if (authUser?.full_name) {
      // For Korean names, take first 2 characters
      if (/[가-힣]/.test(authUser.full_name)) {
        return authUser.full_name.slice(0, 2)
      }
      // For English names, take first letter of first and last name
      const parts = authUser.full_name.split(' ')
      if (parts.length >= 2) {
        return parts[0][0] + parts[parts.length - 1][0]
      }
      return authUser.full_name.slice(0, 2).toUpperCase()
    }
    if (authUser?.email) {
      return authUser.email.slice(0, 2).toUpperCase()
    }
    return 'U'
  }

  const displayName = authUser?.full_name || authUser?.email || 'User'
  const displayEmail = authUser?.email || ''

  // Format system role for display
  const getRoleLabel = (role?: string) => {
    switch (role) {
      case 'superadmin':
        return '최고 관리자'
      case 'admin':
        return '관리자'
      case 'guest':
        return '게스트'
      default:
        return '게스트'
    }
  }

  return (
    <div className="w-64 h-screen bg-gray-900 text-white flex flex-col">
      {/* Project Title */}
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-purple-400 bg-clip-text text-transparent">
          Vision AI Platform
        </h1>
        <p className="text-xs text-gray-400 mt-1">Training Platform MVP</p>
      </div>

      {/* Recent Projects */}
      <div className="flex-1 overflow-hidden flex flex-col px-4 py-2">
        <div className="flex items-center justify-between mb-2 px-2">
          <h2 className="text-sm font-semibold text-gray-400">프로젝트</h2>
          <button
            onClick={onCreateProject}
            className={cn(
              'p-1.5 rounded-lg',
              'text-gray-400 hover:text-violet-400',
              'hover:bg-gray-800',
              'transition-colors'
            )}
            title="새 프로젝트"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1">
          {loadingProjects ? (
            <div className="text-center py-4 text-gray-500 text-sm">
              로딩 중...
            </div>
          ) : projects.length > 0 ? (
            projects.map((project) => (
              <button
                key={project.id}
                onClick={() => handleProjectClick(project.id)}
                className={cn(
                  'w-full px-3 py-2.5 rounded-lg',
                  'text-left transition-colors',
                  'flex items-start gap-2',
                  'group',
                  selectedProjectId === project.id
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-300 hover:bg-gray-800/50'
                )}
              >
                <FolderIcon className={cn(
                  'w-4 h-4 mt-0.5 flex-shrink-0',
                  selectedProjectId === project.id
                    ? 'text-violet-400'
                    : 'text-gray-500 group-hover:text-violet-400'
                )} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium truncate">{project.name}</p>
                    <span className="px-1.5 py-0.5 text-xs font-medium bg-emerald-500/20 text-emerald-400 rounded flex-shrink-0">
                      Owner
                    </span>
                  </div>
                  <p className="text-xs text-gray-500">
                    실험 {project.experiment_count}개
                  </p>
                </div>
              </button>
            ))
          ) : (
            <div className="text-center py-8 text-gray-500 text-sm">
              <p>프로젝트가 없습니다</p>
              <p className="text-xs mt-1 text-gray-600">
                채팅으로 프로젝트를 만들어보세요
              </p>
            </div>
          )}
        </div>
      </div>

      {/* User Info */}
      <div className="border-t border-gray-800 p-4">
        {isAuthenticated ? (
          <div className="relative" ref={userMenuRef}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="w-full flex items-center gap-3 rounded-lg p-2 hover:bg-gray-800 transition-colors"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-full flex items-center justify-center font-semibold">
                {getAvatarInitials()}
              </div>
              <div className="flex-1 min-w-0 text-left">
                <p className="text-sm font-semibold truncate">{displayName}</p>
                <p className="text-xs text-gray-400 truncate">{displayEmail}</p>
                <span className="px-2 py-0.5 text-xs font-medium bg-violet-500/20 text-violet-300 rounded inline-block mt-1">
                  {getRoleLabel(authUser?.system_role)}
                </span>
              </div>
            </button>

            {/* Dropdown Menu */}
            {showUserMenu && (
              <div className="absolute bottom-full left-0 right-0 mb-2 bg-gray-800 rounded-lg shadow-lg border border-gray-700 overflow-hidden">
                <button
                  onClick={handleSettings}
                  className="w-full px-4 py-3 text-left text-sm text-gray-300 hover:bg-gray-700 transition-colors flex items-center gap-3"
                >
                  <Settings className="w-4 h-4" />
                  <span>설정</span>
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full px-4 py-3 text-left text-sm text-red-400 hover:bg-gray-700 transition-colors flex items-center gap-3"
                >
                  <LogOut className="w-4 h-4" />
                  <span>로그아웃</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-gray-400 text-center">로그인이 필요합니다</p>
            <div className="space-y-2">
              <button
                onClick={onOpenLogin}
                className="w-full px-4 py-2 bg-violet-600 hover:bg-violet-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                로그인
              </button>
              <button
                onClick={onOpenRegister}
                className="w-full px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm font-medium rounded-lg transition-colors"
              >
                회원가입
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
