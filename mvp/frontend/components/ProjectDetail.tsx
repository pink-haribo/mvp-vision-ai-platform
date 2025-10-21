'use client'

import { useState, useEffect } from 'react'
import { ArrowLeftIcon, PlayIcon, CheckCircle2Icon, XCircleIcon, ClockIcon } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

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
}

interface ProjectDetailProps {
  projectId: number
  onBack?: () => void
}

export default function ProjectDetail({ projectId, onBack }: ProjectDetailProps) {
  const [project, setProject] = useState<Project | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (projectId) {
      fetchProjectDetails()
    }
  }, [projectId])

  const fetchProjectDetails = async () => {
    setLoading(true)
    try {
      // Fetch project info
      const projectRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}`)
      if (projectRes.ok) {
        const projectData = await projectRes.json()
        setProject(projectData)
      }

      // Fetch experiments
      const expRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects/${projectId}/experiments`)
      if (expRes.ok) {
        const expData = await expRes.json()
        setExperiments(expData)
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
        <div className="flex items-center space-x-4">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeftIcon className="w-5 h-5 text-gray-600" />
            </button>
          )}
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-gray-900">{project.name}</h2>
            {project.description && (
              <p className="text-sm text-gray-600 mt-1">{project.description}</p>
            )}
          </div>
        </div>

        <div className="mt-4 flex items-center gap-3">
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

      {/* Experiments List */}
      <div className="flex-1 overflow-y-auto p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">실험 목록</h3>

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
                className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(exp.status)}
                      <h4 className="font-semibold text-gray-900">
                        {exp.experiment_name || `Experiment ${exp.id}`}
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
                      <span>{exp.model_name}</span>
                      <span>•</span>
                      <span>{exp.framework}</span>
                      <span>•</span>
                      <span>{exp.epochs} epochs</span>
                      {exp.final_accuracy && (
                        <>
                          <span>•</span>
                          <span className="text-emerald-600 font-medium">
                            Accuracy: {(exp.final_accuracy).toFixed(2)}%
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
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
