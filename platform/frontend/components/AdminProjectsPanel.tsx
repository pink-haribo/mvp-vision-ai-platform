'use client'

import { useState, useEffect } from 'react'
import { ArrowUpDown, ArrowUp, ArrowDown, Search, X, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Project {
  id: number
  name: string
  description: string | null
  task_type: string | null
  created_at: string
  updated_at: string
  experiment_count: number
  owner_id: number | null
  owner_name: string | null
  owner_email: string | null
}

type SortField = 'name' | 'owner_name' | 'task_type' | 'experiment_count' | 'created_at'
type SortDirection = 'asc' | 'desc' | null

export default function AdminProjectsPanel() {
  const [projects, setProjects] = useState<Project[]>([])
  const [filteredProjects, setFilteredProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)

  // Sorting state
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDirection, setSortDirection] = useState<SortDirection>(null)

  // Filter state - unified search
  const [searchQuery, setSearchQuery] = useState('')

  // Delete modal state
  const [deleteProject, setDeleteProject] = useState<Project | null>(null)

  useEffect(() => {
    fetchProjects()
  }, [])

  useEffect(() => {
    applyFiltersAndSort()
  }, [projects, searchQuery, sortField, sortDirection])

  const fetchProjects = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/projects`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setProjects(data)
      } else if (response.status === 403) {
        alert('관리자 권한이 필요합니다.')
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    } finally {
      setLoading(false)
    }
  }

  const applyFiltersAndSort = () => {
    let result = [...projects]

    // Apply unified search filter - search across all fields
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(p =>
        p.name.toLowerCase().includes(query) ||
        (p.description?.toLowerCase().includes(query)) ||
        (p.owner_name?.toLowerCase().includes(query)) ||
        (p.owner_email?.toLowerCase().includes(query)) ||
        (p.task_type?.toLowerCase().includes(query)) ||
        p.id.toString().includes(query)
      )
    }

    // Apply sorting
    if (sortField && sortDirection) {
      result.sort((a, b) => {
        let aVal: any = a[sortField]
        let bVal: any = b[sortField]

        // Handle null values
        if (aVal === null) aVal = ''
        if (bVal === null) bVal = ''

        // Compare
        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    }

    setFilteredProjects(result)
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      // Cycle through: asc -> desc -> null
      if (sortDirection === 'asc') {
        setSortDirection('desc')
      } else if (sortDirection === 'desc') {
        setSortDirection(null)
        setSortField(null)
      }
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 text-gray-400" />
    }
    if (sortDirection === 'asc') {
      return <ArrowUp className="w-4 h-4 text-violet-400" />
    }
    return <ArrowDown className="w-4 h-4 text-violet-400" />
  }

  const clearSearch = () => {
    setSearchQuery('')
  }

  const handleDeleteProject = async (projectId: number) => {
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/projects/${projectId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const result = await response.json()
        setDeleteProject(null)
        fetchProjects() // Refresh list
        alert(result.message || '프로젝트가 삭제되었습니다.')
      } else {
        const error = await response.json()
        alert(error.detail || '프로젝트 삭제에 실패했습니다.')
      }
    } catch (error) {
      console.error('Failed to delete project:', error)
      alert('프로젝트 삭제 중 오류가 발생했습니다.')
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    })
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-gray-500">로딩 중...</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h2 className="text-xl font-bold text-gray-900">프로젝트 관리</h2>
      </div>

      {/* Search */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="프로젝트명, 소유자, 작업 유형 등으로 검색..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>
          {searchQuery && (
            <button
              onClick={clearSearch}
              className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1"
            >
              <X className="w-4 h-4" />
              초기화
            </button>
          )}
          <div className="ml-auto text-sm text-gray-500">
            {searchQuery ? (
              <>표시 중: {filteredProjects.length}개 / 전체 {projects.length}개</>
            ) : (
              <>전체 {projects.length}개</>
            )}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="bg-gray-100 sticky top-0">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('name')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  프로젝트명
                  {getSortIcon('name')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('owner_name')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  소유자
                  {getSortIcon('owner_name')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('task_type')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  작업 유형
                  {getSortIcon('task_type')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('experiment_count')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  실험 수
                  {getSortIcon('experiment_count')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('created_at')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  생성일
                  {getSortIcon('created_at')}
                </button>
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                작업
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredProjects.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  {searchQuery
                    ? '검색 조건에 맞는 프로젝트가 없습니다.'
                    : '프로젝트가 없습니다.'}
                </td>
              </tr>
            ) : (
              filteredProjects.map((project) => (
                <tr key={project.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {project.id}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="font-medium text-gray-900">{project.name}</div>
                    {project.description && (
                      <div className="text-gray-500 text-xs mt-0.5 truncate max-w-xs">
                        {project.description}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {project.owner_name ? (
                      <div>
                        <div className="text-gray-900">{project.owner_name}</div>
                        <div className="text-gray-500 text-xs">{project.owner_email}</div>
                      </div>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {project.task_type || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {project.experiment_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {formatDate(project.created_at)}
                  </td>
                  <td className="px-4 py-3 text-sm text-center">
                    <button
                      onClick={() => setDeleteProject(project)}
                      className="p-1.5 text-red-600 hover:bg-red-50 rounded transition-colors"
                      title="삭제"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Delete Confirmation Dialog */}
      {deleteProject && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">프로젝트 삭제</h3>
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">{deleteProject.name}</span> 프로젝트를 정말 삭제하시겠습니까?
            </p>
            {deleteProject.experiment_count > 0 && (
              <p className="text-sm text-orange-600 mb-4">
                ⚠️ 이 프로젝트에는 {deleteProject.experiment_count}개의 실험이 포함되어 있습니다.
                프로젝트를 삭제하면 모든 실험도 함께 삭제됩니다.
              </p>
            )}
            <p className="text-sm text-gray-500 mb-6">
              이 작업은 되돌릴 수 없습니다.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setDeleteProject(null)}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
              >
                취소
              </button>
              <button
                onClick={() => handleDeleteProject(deleteProject.id)}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                삭제
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
