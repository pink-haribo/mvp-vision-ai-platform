'use client'

import { useState, useEffect } from 'react'
import { ArrowUpDown, ArrowUp, ArrowDown, Search, X, Edit2, Shield, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { useAuth } from '@/contexts/AuthContext'
import { getRoleLabel, getRoleBadgeColor } from '@/lib/utils/roleUtils'

interface User {
  id: number
  email: string
  full_name: string | null
  company: string | null
  company_custom: string | null
  division: string | null
  division_custom: string | null
  department: string | null
  phone_number: string | null
  system_role: string
  is_active: boolean
  created_at: string
  project_count: number
}

type SortField = 'email' | 'full_name' | 'company' | 'division' | 'department' | 'system_role' | 'project_count' | 'created_at'
type SortDirection = 'asc' | 'desc' | null

export default function AdminUsersPanel() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<User[]>([])
  const [filteredUsers, setFilteredUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)

  // Sorting state
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDirection, setSortDirection] = useState<SortDirection>(null)

  // Filter state - unified search
  const [searchQuery, setSearchQuery] = useState('')

  // Modal states
  const [roleEditUser, setRoleEditUser] = useState<User | null>(null)
  const [selectedRole, setSelectedRole] = useState<string>('')
  const [editUser, setEditUser] = useState<User | null>(null)
  const [deleteUser, setDeleteUser] = useState<User | null>(null)

  useEffect(() => {
    fetchUsers()
  }, [])

  useEffect(() => {
    applyFiltersAndSort()
  }, [users, searchQuery, sortField, sortDirection])

  const fetchUsers = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/users`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setUsers(data)
      } else if (response.status === 403) {
        alert('관리자 권한이 필요합니다.')
      }
    } catch (error) {
      console.error('Failed to fetch users:', error)
    } finally {
      setLoading(false)
    }
  }

  const applyFiltersAndSort = () => {
    let result = [...users]

    // Apply unified search filter - search across all fields
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(u =>
        u.full_name?.toLowerCase().includes(query) ||
        u.email.toLowerCase().includes(query) ||
        (u.company?.toLowerCase().includes(query)) ||
        (u.company_custom?.toLowerCase().includes(query)) ||
        (u.division?.toLowerCase().includes(query)) ||
        (u.division_custom?.toLowerCase().includes(query)) ||
        (u.department?.toLowerCase().includes(query)) ||
        u.system_role.toLowerCase().includes(query) ||
        u.id.toString().includes(query)
      )
    }

    // Apply sorting
    if (sortField && sortDirection) {
      result.sort((a, b) => {
        let aVal: any = a[sortField]
        let bVal: any = b[sortField]

        // Handle custom fields
        if (sortField === 'company') {
          aVal = a.company_custom || a.company || ''
          bVal = b.company_custom || b.company || ''
        } else if (sortField === 'division') {
          aVal = a.division_custom || a.division || ''
          bVal = b.division_custom || b.division || ''
        }

        // Handle null values
        if (aVal === null) aVal = ''
        if (bVal === null) bVal = ''

        // Compare
        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    }

    setFilteredUsers(result)
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

  const handleRoleChange = async (userId: number, newRole: string) => {
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/users/${userId}/role`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ system_role: newRole })
      })

      if (response.ok) {
        setRoleEditUser(null)
        fetchUsers() // Refresh list
        alert('권한이 변경되었습니다.')
      } else {
        const error = await response.json()
        alert(error.detail || '권한 변경에 실패했습니다.')
      }
    } catch (error) {
      console.error('Failed to update role:', error)
      alert('권한 변경 중 오류가 발생했습니다.')
    }
  }

  const handleUpdateUser = async (userId: number, updates: Partial<User>) => {
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/users/${userId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updates)
      })

      if (response.ok) {
        setEditUser(null)
        fetchUsers() // Refresh list
        alert('사용자 정보가 수정되었습니다.')
      } else {
        const error = await response.json()
        alert(error.detail || '사용자 정보 수정에 실패했습니다.')
      }
    } catch (error) {
      console.error('Failed to update user:', error)
      alert('사용자 정보 수정 중 오류가 발생했습니다.')
    }
  }

  const handleDeleteUser = async (userId: number) => {
    try {
      const token = localStorage.getItem('access_token')
      if (!token) return

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/admin/users/${userId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        setDeleteUser(null)
        fetchUsers() // Refresh list
        alert('사용자가 삭제되었습니다.')
      } else {
        const error = await response.json()
        alert(error.detail || '사용자 삭제에 실패했습니다.')
      }
    } catch (error) {
      console.error('Failed to delete user:', error)
      alert('사용자 삭제 중 오류가 발생했습니다.')
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
        <h2 className="text-xl font-bold text-gray-900">사용자 관리</h2>
      </div>

      {/* Search */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="이름, 이메일, 회사, 사업부, 부서, 권한 등으로 검색..."
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
              <>표시 중: {filteredUsers.length}명 / 전체 {users.length}명</>
            ) : (
              <>전체 {users.length}명</>
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
                  onClick={() => handleSort('full_name')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  이름
                  {getSortIcon('full_name')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('email')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  이메일
                  {getSortIcon('email')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('company')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  회사
                  {getSortIcon('company')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('division')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  사업부
                  {getSortIcon('division')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('department')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  부서
                  {getSortIcon('department')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('system_role')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  권한
                  {getSortIcon('system_role')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('project_count')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  프로젝트
                  {getSortIcon('project_count')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('created_at')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  가입일
                  {getSortIcon('created_at')}
                </button>
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                작업
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredUsers.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-4 py-8 text-center text-gray-500">
                  {searchQuery
                    ? '검색 조건에 맞는 사용자가 없습니다.'
                    : '사용자가 없습니다.'}
                </td>
              </tr>
            ) : (
              filteredUsers.map((user) => (
                <tr key={user.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.id}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.full_name || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.email}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.company_custom || user.company || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.division_custom || user.division || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.department || '-'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={cn(
                      'px-2 py-1 text-xs font-medium rounded',
                      getRoleBadgeColor(user.system_role)
                    )}>
                      {getRoleLabel(user.system_role)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900 text-center">
                    {user.project_count}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {formatDate(user.created_at)}
                  </td>
                  <td className="px-4 py-3 text-sm text-center">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => setEditUser(user)}
                        className="p-1.5 text-gray-600 hover:bg-gray-50 rounded transition-colors"
                        title="정보 수정"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setRoleEditUser(user)}
                        className="p-1.5 text-blue-600 hover:bg-blue-50 rounded transition-colors"
                        title="권한 수정"
                      >
                        <Shield className="w-4 h-4" />
                      </button>
                      {currentUser?.system_role === 'admin' && (
                        <button
                          onClick={() => setDeleteUser(user)}
                          className="p-1.5 text-red-600 hover:bg-red-50 rounded transition-colors"
                          title="삭제"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* User Info Edit Modal */}
      {editUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">사용자 정보 수정</h3>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                const formData = new FormData(e.currentTarget)
                const updates = {
                  full_name: formData.get('full_name') as string,
                  email: formData.get('email') as string,
                  company: formData.get('company') as string || null,
                  division: formData.get('division') as string || null,
                  department: formData.get('department') as string || null,
                  phone_number: formData.get('phone_number') as string || null,
                }
                handleUpdateUser(editUser.id, updates)
              }}
              className="space-y-4"
            >
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">이름</label>
                  <input
                    type="text"
                    name="full_name"
                    defaultValue={editUser.full_name || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">이메일</label>
                  <input
                    type="email"
                    name="email"
                    defaultValue={editUser.email}
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">회사</label>
                  <input
                    type="text"
                    name="company"
                    defaultValue={editUser.company_custom || editUser.company || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">사업부</label>
                  <input
                    type="text"
                    name="division"
                    defaultValue={editUser.division_custom || editUser.division || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">부서</label>
                  <input
                    type="text"
                    name="department"
                    defaultValue={editUser.department || ''}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">전화번호</label>
                  <input
                    type="text"
                    name="phone_number"
                    defaultValue={editUser.phone_number || ''}
                    placeholder="010-1234-5678"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setEditUser(null)}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
                >
                  취소
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700 transition-colors"
                >
                  저장
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Role Edit Modal */}
      {roleEditUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">권한 수정</h3>
            <p className="text-sm text-gray-600 mb-4">
              {roleEditUser.full_name || roleEditUser.email}의 권한을 변경합니다.
            </p>
            <div className="space-y-2 mb-6">
              {['guest', 'standard_engineer', 'advanced_engineer', 'manager', 'admin'].map((role) => {
                const isDisabled = currentUser?.system_role === 'manager' && (role === 'manager' || role === 'admin')
                return (
                  <button
                    key={role}
                    onClick={() => !isDisabled && setSelectedRole(role)}
                    disabled={isDisabled}
                    className={cn(
                      'w-full px-4 py-2 rounded-md text-left transition-colors',
                      (selectedRole || roleEditUser.system_role) === role
                        ? 'bg-violet-100 text-violet-700 font-medium'
                        : isDisabled
                        ? 'bg-gray-50 text-gray-400 cursor-not-allowed'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    )}
                  >
                    {getRoleLabel(role)}
                    {isDisabled && ' (권한 없음)'}
                  </button>
                )
              })}
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setRoleEditUser(null)
                  setSelectedRole('')
                }}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
              >
                취소
              </button>
              <button
                onClick={() => {
                  if (selectedRole && selectedRole !== roleEditUser.system_role) {
                    handleRoleChange(roleEditUser.id, selectedRole)
                  } else {
                    setRoleEditUser(null)
                  }
                  setSelectedRole('')
                }}
                className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700 transition-colors"
              >
                저장
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      {deleteUser && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">사용자 삭제</h3>
            <p className="text-sm text-gray-600 mb-2">
              <span className="font-medium">{deleteUser.full_name || deleteUser.email}</span>을(를) 정말 삭제하시겠습니까?
            </p>
            {deleteUser.project_count > 0 && (
              <p className="text-sm text-red-600 mb-4">
                ⚠️ 이 사용자는 {deleteUser.project_count}개의 프로젝트를 소유하고 있어 삭제할 수 없습니다.
                먼저 프로젝트를 삭제하거나 다른 사용자에게 이전해주세요.
              </p>
            )}
            <p className="text-sm text-gray-500 mb-6">
              이 작업은 되돌릴 수 없습니다.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setDeleteUser(null)}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
              >
                취소
              </button>
              <button
                onClick={() => handleDeleteUser(deleteUser.id)}
                disabled={deleteUser.project_count > 0}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
