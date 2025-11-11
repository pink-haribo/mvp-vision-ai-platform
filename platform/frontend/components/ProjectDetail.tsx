'use client'

import { useState, useEffect } from 'react'
import { ArrowLeftIcon, ArrowRightIcon, PlayIcon, CheckCircle2Icon, XCircleIcon, ClockIcon, EditIcon, SaveIcon, XIcon, PlusIcon, CopyIcon, Crown, UserPlus, X as XClose } from 'lucide-react'
import { cn } from '@/lib/utils'
import { getAvatarColorStyle, getAvatarRingColor } from '@/lib/utils/avatarColors'
import { getModelDisplayNameSync, getTaskDisplayName, formatTrainingJobTitle } from '@/lib/utils/modelUtils'
import { useAuth } from '@/contexts/AuthContext'

interface Experiment {
  id: number
  experiment_name: string | null
  model_name: string
  task_type: string
  framework: string
  status: string
  tags: string[] | null
  notes: string | null
  epochs: number
  batch_size: number
  learning_rate: number
  final_accuracy: number | null
  primary_metric: string | null
  primary_metric_mode: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

interface Project {
  id: number
  name: string
  description: string | null
  task_type: string | null
  created_at: string
  updated_at: string
  user_id: number | null
}

interface ProjectMember {
  user_id: number
  email: string
  full_name: string | null
  role: string
  joined_at: string
  is_owner: boolean
  badge_color?: string | null
}

interface ProjectDetailProps {
  projectId: number
  onBack?: () => void
  onStartNewTraining?: (projectId: number) => void
  onCloneExperiment?: (experimentId: number, projectId: number) => void
  onViewExperiment?: (experimentId: number) => void
}

export default function ProjectDetail({
  projectId,
  onBack,
  onStartNewTraining,
  onCloneExperiment,
  onViewExperiment
}: ProjectDetailProps) {
  const { user } = useAuth()
  const [project, setProject] = useState<Project | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedExpId, setExpandedExpId] = useState<number | null>(null)

  // Edit mode states
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editTaskType, setEditTaskType] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Member states
  const [members, setMembers] = useState<ProjectMember[]>([])
  const [showInviteModal, setShowInviteModal] = useState(false)
  const [inviteEmail, setInviteEmail] = useState('')
  const [inviteRole, setInviteRole] = useState('member')
  const [isInviting, setIsInviting] = useState(false)

  // Check if current user is the owner (both values must exist and match)
  const isOwner = !!(project?.user_id && user?.id && project.user_id === user.id)

  useEffect(() => {
    if (projectId) {
      fetchProjectDetails()
    }
  }, [projectId])

  const fetchProjectDetails = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('access_token')
      if (!token) {
        console.error('No access token found')
        setLoading(false)
        return
      }

      const headers = {
        'Authorization': `Bearer ${token}`
      }

      // Fetch project info
      const projectRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}`, {
        headers
      })
      if (projectRes.ok) {
        const projectData = await projectRes.json()
        setProject(projectData)
      } else {
        console.error('Failed to fetch project:', projectRes.status, projectRes.statusText)
      }

      // Fetch experiments
      const expRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}/experiments`, {
        headers
      })
      if (expRes.ok) {
        const expData = await expRes.json()
        setExperiments(expData)
      } else {
        console.error('Failed to fetch experiments:', expRes.status, expRes.statusText)
      }

      // Fetch members
      const membersRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}/members`, {
        headers
      })
      if (membersRes.ok) {
        const membersData = await membersRes.json()
        setMembers(membersData)
      } else {
        console.error('Failed to fetch members:', membersRes.status, membersRes.statusText)
      }
    } catch (error) {
      console.error('Failed to fetch project details:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2Icon className="w-5 h-5 text-emerald-600" />
      case 'running':
        return <PlayIcon className="w-5 h-5 text-blue-600" />
      case 'failed':
        return <XCircleIcon className="w-5 h-5 text-red-600" />
      default:
        return <ClockIcon className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusLabel = (status: string) => {
    const labels: Record<string, string> = {
      'pending': '대기 중',
      'running': '실행 중',
      'completed': '완료',
      'failed': '실패',
      'cancelled': '취소됨',
    }
    return labels[status] || status
  }

  const formatDate = (dateString: string | null) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const handleStartEdit = () => {
    if (!project) return
    setEditName(project.name)
    setEditDescription(project.description || '')
    setEditTaskType(project.task_type || 'image_classification')
    setIsEditing(true)
    setError(null)
  }

  const handleCancelEdit = () => {
    setIsEditing(false)
    setError(null)
  }

  const handleSaveEdit = async () => {
    if (!project || !editName.trim()) {
      setError('프로젝트 이름을 입력해주세요')
      return
    }

    setIsSaving(true)
    setError(null)

    try {
      const token = localStorage.getItem('access_token')

      if (!token) {
        setError('로그인이 필요합니다.')
        setIsSaving(false)
        return
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          name: editName.trim(),
          description: editDescription.trim() || null,
          task_type: editTaskType || null,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))

        // Handle duplicate project name
        if (response.status === 400 && errorData.detail?.includes('already exists')) {
          throw new Error('이미 존재하는 프로젝트 이름입니다. 다른 이름을 사용해주세요.')
        }

        throw new Error(errorData.detail || '프로젝트 수정에 실패했습니다')
      }

      const updatedProject = await response.json()
      setProject(updatedProject)
      setIsEditing(false)
    } catch (err) {
      console.error('Error updating project:', err)
      setError(err instanceof Error ? err.message : '프로젝트 수정 중 오류가 발생했습니다')
    } finally {
      setIsSaving(false)
    }
  }

  const handleInviteMember = async () => {
    if (!inviteEmail.trim()) {
      alert('이메일을 입력해주세요')
      return
    }

    setIsInviting(true)
    try {
      const token = localStorage.getItem('access_token')
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}/members`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: inviteEmail,
          role: inviteRole,
        }),
      })

      if (response.ok) {
        alert('멤버가 초대되었습니다')
        setShowInviteModal(false)
        setInviteEmail('')
        fetchProjectDetails() // Refresh members
      } else {
        const error = await response.json()
        alert(error.detail || '멤버 초대에 실패했습니다')
      }
    } catch (error) {
      console.error('Failed to invite member:', error)
      alert('멤버 초대 중 오류가 발생했습니다')
    } finally {
      setIsInviting(false)
    }
  }

  const handleRemoveMember = async (userId: number) => {
    if (!confirm('정말 이 멤버를 삭제하시겠습니까?')) return

    try {
      const token = localStorage.getItem('access_token')
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}/members/${userId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      if (response.ok) {
        alert('멤버가 삭제되었습니다')
        fetchProjectDetails() // Refresh members
      } else {
        const error = await response.json()
        alert(error.detail || '멤버 삭제에 실패했습니다')
      }
    } catch (error) {
      console.error('Failed to remove member:', error)
      alert('멤버 삭제 중 오류가 발생했습니다')
    }
  }

  const getAvatarInitials = (member: ProjectMember) => {
    if (member.full_name) {
      // For Korean names, take first 2 characters
      if (/[가-힣]/.test(member.full_name)) {
        return member.full_name.slice(0, 2)
      }
      // For English names, take first letter of first and last name
      const parts = member.full_name.split(' ')
      if (parts.length >= 2) {
        return parts[0][0] + parts[parts.length - 1][0]
      }
      return member.full_name.slice(0, 2).toUpperCase()
    }
    return member.email.slice(0, 2).toUpperCase()
  }

  const taskTypes = [
    { value: 'image_classification', label: '이미지 분류' },
    { value: 'object_detection', label: '객체 탐지' },
    { value: 'semantic_segmentation', label: '의미론적 분할' },
    { value: 'instance_segmentation', label: '인스턴스 분할' },
    { value: 'pose_estimation', label: '포즈 추정' },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">로딩 중...</div>
      </div>
    )
  }

  if (!project) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">프로젝트를 찾을 수 없습니다</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        {/* Title bar with back button and action buttons */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-4 flex-1">
            {onBack && (
              <button
                onClick={isEditing ? handleCancelEdit : onBack}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                title={isEditing ? '편집 취소' : '뒤로 가기'}
              >
                <ArrowLeftIcon className="w-5 h-5 text-gray-600" />
              </button>
            )}
            <h2 className="text-lg font-semibold text-gray-900">
              {isEditing ? '프로젝트 정보 수정' : project.name}
            </h2>

            {/* Project Members - Next to title (only in view mode) */}
            {!isEditing && (
              <div className="flex items-center gap-2 ml-4">
                {members.map((member) => {
                  const bgColorStyle = getAvatarColorStyle(member.badge_color)
                  const ringClass = getAvatarRingColor(member.badge_color)

                  return (
                    <div key={member.user_id} className="relative group">
                      <div
                        className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold text-white",
                          "hover:ring-2 hover:ring-offset-2 transition-all cursor-pointer",
                          ringClass
                        )}
                        style={bgColorStyle}
                        title={`${member.full_name || member.email} (${member.is_owner ? 'Owner' : member.role})`}
                      >
                        {member.is_owner && (
                          <Crown className="absolute -top-1 -right-1 w-4 h-4 text-yellow-400 fill-yellow-400" />
                        )}
                        {getAvatarInitials(member)}
                      </div>

                      {/* Hover tooltip */}
                      <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
                        <div className="font-medium">{member.full_name || member.email}</div>
                        <div className="text-gray-300">{member.is_owner ? 'Owner' : member.role}</div>
                        {!member.is_owner && isOwner && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleRemoveMember(member.user_id)
                            }}
                            className="mt-1 text-red-400 hover:text-red-300 text-xs pointer-events-auto"
                          >
                            삭제
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}

                {/* Add member button - Only for owner */}
                {isOwner && (
                  <button
                    onClick={() => setShowInviteModal(true)}
                    className={cn(
                      "w-10 h-10 rounded-full flex items-center justify-center",
                      "border-2 border-dashed border-gray-300 hover:border-violet-500",
                      "text-gray-400 hover:text-violet-500 transition-colors"
                    )}
                    title="멤버 초대"
                  >
                    <UserPlus className="w-5 h-5" />
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Edit/Save/Cancel Buttons */}
          <div className="flex gap-2">
            {!isEditing ? (
              <button
                onClick={handleStartEdit}
                className={cn(
                  'px-3 py-1.5 hover:bg-gray-100 rounded-lg transition-colors',
                  'text-gray-600 hover:text-violet-600',
                  'flex items-center gap-2 text-sm font-medium'
                )}
              >
                <EditIcon className="w-4 h-4" />
                <span>수정</span>
              </button>
            ) : (
              <>
                <button
                  onClick={handleCancelEdit}
                  disabled={isSaving}
                  className={cn(
                    'px-3 py-1.5 border border-gray-300 rounded-lg',
                    'text-gray-700 hover:bg-gray-50',
                    'transition-colors text-sm font-medium',
                    'disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                >
                  취소
                </button>
                <button
                  onClick={handleSaveEdit}
                  disabled={isSaving || !editName.trim()}
                  className={cn(
                    'px-3 py-1.5 bg-violet-600 text-white rounded-lg',
                    'hover:bg-violet-700 transition-colors text-sm font-medium',
                    'disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                >
                  {isSaving ? '저장 중...' : '저장'}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Content: View or Edit mode */}
        {!isEditing ? (
          <div className="mt-4">
            {/* Description */}
            {project.description && (
              <p className="text-sm text-gray-600 mb-4">{project.description}</p>
            )}

            {/* Task type and experiment count */}
            <div className="flex items-center gap-3">
              {project.task_type && (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-700">
                  {project.task_type}
                </span>
              )}
              <span className="text-sm text-gray-500">
                실험 {experiments.length}개
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Error Message */}
            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {/* Edit Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                프로젝트 이름 <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className={cn(
                  'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-sm'
                )}
                maxLength={100}
                disabled={isSaving}
                placeholder="예: ResNet 실험 프로젝트"
              />
              <p className="text-xs text-gray-500 mt-1">
                {editName.length}/100 자
              </p>
            </div>

            {/* Edit Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                설명 (선택)
              </label>
              <textarea
                value={editDescription}
                onChange={(e) => setEditDescription(e.target.value)}
                rows={3}
                className={cn(
                  'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-sm resize-none'
                )}
                maxLength={500}
                disabled={isSaving}
                placeholder="프로젝트에 대한 설명을 입력하세요"
              />
              <p className="text-xs text-gray-500 mt-1">
                {editDescription.length}/500 자
              </p>
            </div>

            {/* Edit Task Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                작업 유형 (선택)
              </label>
              <select
                value={editTaskType}
                onChange={(e) => setEditTaskType(e.target.value)}
                className={cn(
                  'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'text-sm bg-white'
                )}
                disabled={isSaving}
              >
                {taskTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Experiments List */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-700">실험 목록</h3>
          <button
            onClick={() => onStartNewTraining?.(projectId)}
            className={cn(
              'px-3 py-1.5 bg-violet-600 text-white rounded-lg',
              'hover:bg-violet-700 transition-colors',
              'flex items-center gap-2 text-sm font-medium'
            )}
          >
            <PlusIcon className="w-4 h-4" />
            <span>새 학습 시작</span>
          </button>
        </div>

        {experiments.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-500 text-sm">아직 실험이 없습니다</p>
            <p className="text-gray-400 text-xs mt-1">
              채팅에서 학습을 시작하여 첫 실험을 만드세요
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {experiments.map((exp) => (
              <div
                key={exp.id}
                onClick={() => setExpandedExpId(expandedExpId === exp.id ? null : exp.id)}
                className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(exp.status)}
                      <h4 className="font-semibold text-gray-900">
                        {exp.experiment_name || formatTrainingJobTitle(exp.framework, exp.model_name, exp.task_type)}
                      </h4>
                      <span className={cn(
                        'px-2 py-1 rounded text-xs font-medium',
                        exp.status === 'completed' && 'bg-emerald-100 text-emerald-700',
                        exp.status === 'running' && 'bg-blue-100 text-blue-700',
                        exp.status === 'failed' && 'bg-red-100 text-red-700',
                        exp.status === 'pending' && 'bg-gray-100 text-gray-700'
                      )}>
                        {getStatusLabel(exp.status)}
                      </span>
                    </div>

                    <div className="mt-2 flex items-center gap-4 text-sm text-gray-600">
                      <span>{getModelDisplayNameSync(exp.framework, exp.model_name)}</span>
                      <span>•</span>
                      <span>{exp.framework}</span>
                      <span>•</span>
                      <span>{exp.epochs} epochs</span>
                      {exp.final_accuracy && (
                        <>
                          <span>•</span>
                          <span className="text-emerald-600 font-medium">
                            {exp.primary_metric || 'accuracy'}: {(exp.final_accuracy * 100).toFixed(2)}%
                          </span>
                        </>
                      )}
                    </div>

                    {exp.tags && exp.tags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {exp.tags.map((tag, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-violet-100 text-violet-700"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {exp.notes && (
                      <p className="mt-2 text-sm text-gray-500 italic">
                        {exp.notes}
                      </p>
                    )}

                    <div className="mt-3 text-xs text-gray-400">
                      생성: {formatDate(exp.created_at)}
                      {exp.completed_at && (
                        <> • 완료: {formatDate(exp.completed_at)}</>
                      )}
                    </div>
                  </div>
                </div>

                {/* 확장된 상세 정보 */}
                {expandedExpId === exp.id && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h5 className="text-sm font-semibold text-gray-700 mb-3">학습 설정</h5>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-gray-500">프레임워크:</span>
                        <span className="ml-2 text-gray-900">{exp.framework}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">모델:</span>
                        <span className="ml-2 text-gray-900">
                          {getModelDisplayNameSync(exp.framework, exp.model_name)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">작업 유형:</span>
                        <span className="ml-2 text-gray-900">
                          {getTaskDisplayName(exp.task_type)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Epochs:</span>
                        <span className="ml-2 text-gray-900">{exp.epochs}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Batch Size:</span>
                        <span className="ml-2 text-gray-900">{exp.batch_size}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Learning Rate:</span>
                        <span className="ml-2 text-gray-900">{exp.learning_rate}</span>
                      </div>
                    </div>

                    {exp.status === 'running' && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                        <p className="text-sm text-blue-800">
                          🚀 학습이 진행 중입니다. 우측 패널에서 실시간 진행 상황을 확인하세요.
                        </p>
                      </div>
                    )}

                    {exp.status === 'completed' && exp.final_accuracy && (
                      <div className="mt-4 p-3 bg-emerald-50 rounded-lg">
                        <p className="text-sm text-emerald-800">
                          ✅ 학습 완료! 최종 {exp.primary_metric || 'accuracy'}: <strong>{(exp.final_accuracy * 100).toFixed(2)}%</strong>
                        </p>
                      </div>
                    )}

                    {exp.status === 'failed' && (
                      <div className="mt-4 p-3 bg-red-50 rounded-lg">
                        <p className="text-sm text-red-800">
                          ❌ 학습 실패. 로그를 확인해주세요.
                        </p>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="mt-4 flex gap-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onViewExperiment?.(exp.id)
                        }}
                        className={cn(
                          'px-3 py-1.5 bg-violet-600 text-white rounded-lg',
                          'hover:bg-violet-700',
                          'transition-colors text-sm font-medium',
                          'flex items-center gap-2'
                        )}
                      >
                        <ArrowRightIcon className="w-4 h-4" />
                        <span>
                          {exp.status === 'pending' && '학습 보기'}
                          {exp.status === 'running' && '학습 진행 보기'}
                          {(exp.status === 'completed' || exp.status === 'failed' || exp.status === 'cancelled') && '학습 상세 보기'}
                        </span>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onCloneExperiment?.(exp.id, projectId)
                        }}
                        className={cn(
                          'px-3 py-1.5 border border-violet-300 rounded-lg',
                          'text-violet-600 hover:bg-violet-50',
                          'transition-colors text-sm font-medium',
                          'flex items-center gap-2'
                        )}
                      >
                        <CopyIcon className="w-4 h-4" />
                        <span>복사하여 새 학습</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Invite Member Modal */}
      {showInviteModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">멤버 초대</h3>
              <button
                onClick={() => setShowInviteModal(false)}
                className="p-1 hover:bg-gray-100 rounded transition-colors"
              >
                <XClose className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  이메일 <span className="text-red-500">*</span>
                </label>
                <input
                  type="email"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  역할
                </label>
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-violet-500"
                >
                  <option value="member">Member (읽기/쓰기)</option>
                </select>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  onClick={() => setShowInviteModal(false)}
                  disabled={isInviting}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors disabled:opacity-50"
                >
                  취소
                </button>
                <button
                  onClick={handleInviteMember}
                  disabled={isInviting || !inviteEmail.trim()}
                  className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-md hover:bg-violet-700 transition-colors disabled:opacity-50"
                >
                  {isInviting ? '초대 중...' : '초대'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
