'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Square, AlertCircle, ExternalLink, ArrowLeft, ChevronRight, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import MLflowMetricsCharts from './training/MLflowMetricsCharts'
import DatabaseMetricsTable from './training/DatabaseMetricsTable'
import ResumeDialog from './training/ResumeDialog'

interface TrainingJob {
  id: number
  project_id: number | null
  project_name: string | null
  framework: string
  model_name: string
  task_type: string
  num_classes: number | null
  dataset_format: string
  epochs: number
  batch_size: number
  learning_rate: number
  advanced_config: any | null
  status: string
  final_accuracy: number | null
}

interface TrainingMetric {
  id: number
  job_id: number
  epoch: number
  step?: number
  loss?: number
  accuracy?: number
  learning_rate?: number
  checkpoint_path?: string
  extra_metrics?: {
    train_loss?: number
    train_accuracy?: number
    val_loss?: number
    val_accuracy?: number
    [key: string]: any
  }
  created_at: string
}

interface TrainingLog {
  id: number
  log_type: string
  content: string
  created_at: string
}

interface TrainingPanelProps {
  trainingJobId: number | null
  onNavigateToExperiments?: () => void
}

export default function TrainingPanel({ trainingJobId, onNavigateToExperiments }: TrainingPanelProps) {
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [metrics, setMetrics] = useState<TrainingMetric[]>([])
  const [logs, setLogs] = useState<TrainingLog[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showConfigDetails, setShowConfigDetails] = useState(false)
  const [resumeDialogMode, setResumeDialogMode] = useState<'start' | 'restart' | null>(null)
  const [showLogs, setShowLogs] = useState(false)
  const logsContainerRef = useRef<HTMLDivElement>(null)

  // Fetch training job details
  useEffect(() => {
    if (!trainingJobId) return

    const fetchJob = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}`
        )
        if (response.ok) {
          const data = await response.json()
          setJob(data)
        }
      } catch (error) {
        console.error('Error fetching training job:', error)
      }
    }

    fetchJob()

    // Poll for updates every 2 seconds if training is running
    const interval = setInterval(fetchJob, 2000)
    return () => clearInterval(interval)
  }, [trainingJobId])

  // Fetch metrics
  useEffect(() => {
    if (!trainingJobId) return

    const fetchMetrics = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/metrics`
        )
        if (response.ok) {
          const data = await response.json()
          console.log('[DEBUG] Fetched metrics:', data)
          console.log('[DEBUG] Metrics count:', data.length)
          if (data.length > 0) {
            console.log('[DEBUG] First metric:', data[0])
          }
          setMetrics(data)
        } else {
          console.error('[DEBUG] Failed to fetch metrics, status:', response.status)
        }
      } catch (error) {
        console.error('Error fetching metrics:', error)
      }
    }

    fetchMetrics()

    // Poll for metrics every 2 seconds if training is running
    if (job?.status === 'running') {
      const interval = setInterval(fetchMetrics, 2000)
      return () => clearInterval(interval)
    }
  }, [trainingJobId, job?.status])

  // Fetch logs
  useEffect(() => {
    if (!trainingJobId) return

    const fetchLogs = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/logs?limit=500`
        )
        if (response.ok) {
          const data = await response.json()
          setLogs(data)
        }
      } catch (error) {
        console.error('Error fetching logs:', error)
      }
    }

    fetchLogs()

    // Poll for logs every 2 seconds if training is running
    if (job?.status === 'running') {
      const interval = setInterval(fetchLogs, 2000)
      return () => clearInterval(interval)
    }
  }, [trainingJobId, job?.status])

  // Auto-scroll logs to bottom when updated
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs])

  const startTraining = async () => {
    if (!trainingJobId) return

    // Check if there are any checkpoints
    const latestCheckpoint = metrics
      .slice()
      .reverse()
      .find((m) => m.checkpoint_path)

    if (latestCheckpoint) {
      // Show resume dialog if checkpoints exist
      setResumeDialogMode('start')
    } else {
      // Start from scratch if no checkpoints
      await startTrainingFromScratch()
    }
  }

  const startTrainingFromScratch = async () => {
    if (!trainingJobId) return

    setIsLoading(true)
    setResumeDialogMode(null)
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/start`,
        { method: 'POST' }
      )

      if (response.ok) {
        const data = await response.json()
        setJob(data)
      } else {
        alert('학습 시작 실패')
      }
    } catch (error) {
      console.error('Error starting training:', error)
      alert('학습 시작 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const startTrainingWithResume = async () => {
    if (!trainingJobId) return

    // Find the latest checkpoint
    const latestCheckpoint = metrics
      .slice()
      .reverse()
      .find((m) => m.checkpoint_path)

    if (!latestCheckpoint?.checkpoint_path) {
      alert('체크포인트를 찾을 수 없습니다')
      return
    }

    setIsLoading(true)
    setResumeDialogMode(null)
    try {
      const url = new URL(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/start`
      )
      url.searchParams.append('checkpoint_path', latestCheckpoint.checkpoint_path)
      url.searchParams.append('resume', 'true')

      const response = await fetch(url.toString(), { method: 'POST' })

      if (response.ok) {
        const data = await response.json()
        setJob(data)
      } else {
        alert('학습 재개 실패')
      }
    } catch (error) {
      console.error('Error resuming training:', error)
      alert('학습 재개 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const cancelTraining = async () => {
    if (!trainingJobId) return

    setIsLoading(true)
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/cancel`,
        { method: 'POST' }
      )

      if (response.ok) {
        const data = await response.json()
        setJob(data)
      } else {
        alert('학습 중단 실패')
      }
    } catch (error) {
      console.error('Error canceling training:', error)
      alert('학습 중단 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const restartTraining = async () => {
    if (!trainingJobId) return

    // Check if there are any checkpoints
    const latestCheckpoint = metrics
      .slice()
      .reverse()
      .find((m) => m.checkpoint_path)

    if (latestCheckpoint) {
      // Show resume dialog if checkpoints exist
      setResumeDialogMode('restart')
    } else {
      // No checkpoints - just restart from scratch
      await restartTrainingFromScratch()
    }
  }

  const restartTrainingFromScratch = async () => {
    if (!trainingJobId) return

    setIsLoading(true)
    setResumeDialogMode(null)
    try {
      // First reset the job
      const restartResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/restart`,
        { method: 'POST' }
      )

      if (!restartResponse.ok) {
        alert('학습 재시작 실패')
        return
      }

      const restartData = await restartResponse.json()
      setJob(restartData)
      setMetrics([])
      setLogs([])

      // Then start training
      const startResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/start`,
        { method: 'POST' }
      )

      if (startResponse.ok) {
        const startData = await startResponse.json()
        setJob(startData)
      } else {
        alert('학습 시작 실패')
      }
    } catch (error) {
      console.error('Error restarting training:', error)
      alert('학습 재시작 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const restartTrainingWithResume = async () => {
    if (!trainingJobId) return

    // Find the latest checkpoint
    const latestCheckpoint = metrics
      .slice()
      .reverse()
      .find((m) => m.checkpoint_path)

    if (!latestCheckpoint?.checkpoint_path) {
      alert('체크포인트를 찾을 수 없습니다')
      return
    }

    setIsLoading(true)
    setResumeDialogMode(null)
    try {
      // First reset the job to pending state
      const restartResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/restart`,
        { method: 'POST' }
      )

      if (!restartResponse.ok) {
        alert('학습 재시작 실패')
        return
      }

      const restartData = await restartResponse.json()
      setJob(restartData)

      // Don't clear metrics/logs when resuming - we need them!
      // setMetrics([])
      // setLogs([])

      // Then start training with checkpoint
      const url = new URL(
        `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/start`
      )
      url.searchParams.append('checkpoint_path', latestCheckpoint.checkpoint_path)
      url.searchParams.append('resume', 'true')

      const startResponse = await fetch(url.toString(), { method: 'POST' })

      if (startResponse.ok) {
        const startData = await startResponse.json()
        setJob(startData)
      } else {
        alert('학습 재개 실패')
      }
    } catch (error) {
      console.error('Error resuming training:', error)
      alert('학습 재개 실패')
    } finally {
      setIsLoading(false)
    }
  }

  // Calculate current epoch and iteration progress
  const getCurrentProgress = () => {
    if (!job || metrics.length === 0) {
      return { currentEpoch: 0, totalEpochs: job?.epochs || 0, currentIteration: 0, totalIterations: 0 }
    }

    // Get latest metric
    const latestMetric = metrics[metrics.length - 1]
    const currentEpoch = latestMetric?.epoch || 0
    const totalEpochs = job?.epochs || 0

    // Get iteration info from extra_metrics if available
    const currentIteration = latestMetric?.extra_metrics?.batch || 0
    const totalIterations = latestMetric?.extra_metrics?.total_batches || 0

    return { currentEpoch, totalEpochs, currentIteration, totalIterations }
  }

  const progress = getCurrentProgress()

  // Calculate progress percentage safely
  const epochProgressPercent = progress.totalEpochs > 0
    ? Math.round((progress.currentEpoch / progress.totalEpochs) * 100)
    : 0

  // Debug logging
  console.log('[DEBUG] Progress:', progress)
  console.log('[DEBUG] Job status:', job?.status)
  console.log('[DEBUG] Metrics length:', metrics.length)
  console.log('[DEBUG] Epoch progress percent:', epochProgressPercent)

  const getStatusBadge = (status: string) => {
    const styles = {
      pending: 'bg-gray-100 text-gray-800',
      running: 'bg-emerald-100 text-emerald-800',
      completed: 'bg-violet-100 text-violet-800',
      failed: 'bg-red-100 text-red-800',
      cancelled: 'bg-amber-100 text-amber-800',
    }

    return (
      <span
        className={cn(
          'px-2.5 py-1 rounded-md text-xs font-semibold',
          styles[status as keyof typeof styles] || styles.pending
        )}
      >
        {status.toUpperCase()}
      </span>
    )
  }

  if (!job) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-sm">학습 작업이 없습니다</p>
          <p className="text-xs mt-2">채팅에서 학습 설정을 완료하세요</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-6 bg-white border-b border-gray-200">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
          <button
            onClick={onNavigateToExperiments}
            className="hover:text-violet-600 transition-colors"
          >
            {job.project_name || '프로젝트'}
          </button>
          <ChevronRight className="w-4 h-4" />
          <button
            onClick={onNavigateToExperiments}
            className="hover:text-violet-600 transition-colors"
          >
            실험
          </button>
          <ChevronRight className="w-4 h-4" />
          <span className="text-gray-900 font-medium">학습 #{job.id}</span>
        </div>

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <button
              onClick={onNavigateToExperiments}
              className={cn(
                'p-2 rounded-lg',
                'text-gray-600 hover:text-violet-600',
                'hover:bg-gray-100',
                'transition-colors'
              )}
              title="실험 목록으로 돌아가기"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <h2 className="text-lg font-semibold text-gray-900">학습 진행 상황</h2>
          </div>
          {getStatusBadge(job.status)}
        </div>

        {/* Action Buttons and Progress */}
        <div className="space-y-3">
          {/* Buttons and Epoch Progress in one row */}
          <div className="flex items-center gap-4">
            {/* Action Buttons */}
            <div className="flex gap-2">
              {job.status === 'pending' && (
                <button
                  onClick={startTraining}
                  disabled={isLoading}
                  className={cn(
                    'px-4 py-2.5',
                    'bg-violet-600 hover:bg-violet-700',
                    'text-white font-semibold',
                    'rounded-lg shadow-md',
                    'transition-all duration-200',
                    'disabled:opacity-40',
                    'flex items-center gap-2'
                  )}
                >
                  <Play className="w-4 h-4" />
                  학습 시작
                </button>
              )}

              {job.status === 'running' && (
                <button
                  onClick={cancelTraining}
                  disabled={isLoading}
                  className={cn(
                    'px-4 py-2.5',
                    'bg-red-600 hover:bg-red-700',
                    'text-white font-semibold',
                    'rounded-lg',
                    'transition-all duration-200',
                    'disabled:opacity-40',
                    'flex items-center gap-2'
                  )}
                >
                  <Square className="w-4 h-4" />
                  중단
                </button>
              )}

              {(job.status === 'completed' || job.status === 'cancelled' || job.status === 'failed') && (
                <button
                  onClick={restartTraining}
                  disabled={isLoading}
                  className={cn(
                    'px-4 py-2.5',
                    'bg-violet-600 hover:bg-violet-700',
                    'text-white font-semibold',
                    'rounded-lg shadow-md',
                    'transition-all duration-200',
                    'disabled:opacity-40',
                    'flex items-center gap-2'
                  )}
                >
                  <Play className="w-4 h-4" />
                  재학습
                </button>
              )}
            </div>

            {/* Epoch Progress - Show when training has started */}
            {job.status !== 'pending' && metrics.length > 0 && (
              <div className="flex-1">
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span className="font-medium">Epoch {progress.currentEpoch} / {progress.totalEpochs}</span>
                  <span className="text-gray-500">
                    {epochProgressPercent}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-violet-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${epochProgressPercent}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Iteration Progress - Show only when training is running */}
          {job.status === 'running' && progress.totalIterations > 0 && (
            <div>
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>Iteration Progress (Current Epoch)</span>
                <span>{progress.currentIteration} / {progress.totalIterations}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-emerald-500 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${(progress.currentIteration / progress.totalIterations) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Config - Compact Summary */}
      <div className="p-6 bg-white border-b border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-900">학습 설정</h3>
          <button
            onClick={() => setShowConfigDetails(!showConfigDetails)}
            className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-violet-600 hover:bg-violet-50 rounded-lg transition-colors"
          >
            {showConfigDetails ? (
              <>
                <ChevronUp className="w-4 h-4" />
                간단히 보기
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                자세히 보기
              </>
            )}
          </button>
        </div>

        {/* Compact Summary (Always Visible) */}
        <div className="flex items-center gap-4 text-sm flex-wrap">
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 bg-violet-100 text-violet-700 rounded text-xs font-semibold">
              {job.model_name}
            </span>
          </div>
          <div className="text-gray-600">
            <span className="font-medium text-gray-900">{job.task_type}</span>
          </div>
          {job.num_classes && (
            <div className="text-gray-600">
              <span className="font-medium text-gray-900">{job.num_classes}</span>개 클래스
            </div>
          )}
          <div className="text-gray-600">
            <span className="font-medium text-gray-900">{job.epochs}</span> 에포크
          </div>
        </div>

        {/* Detailed Config (Expandable) */}
        {showConfigDetails && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="space-y-4">
              {/* Basic Config */}
              <div>
                <h4 className="text-xs font-semibold text-gray-700 mb-2">기본 설정</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600">프레임워크:</span>
                    <span className="ml-2 font-medium">{job.framework}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">작업 유형:</span>
                    <span className="ml-2 font-medium">{job.task_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">모델:</span>
                    <span className="ml-2 font-medium">{job.model_name}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">데이터셋 형식:</span>
                    <span className="ml-2 font-medium">{job.dataset_format}</span>
                  </div>
                  {job.num_classes && (
                    <div>
                      <span className="text-gray-600">클래스 수:</span>
                      <span className="ml-2 font-medium">{job.num_classes}</span>
                    </div>
                  )}
                  <div>
                    <span className="text-gray-600">에포크:</span>
                    <span className="ml-2 font-medium">{job.epochs}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">배치 크기:</span>
                    <span className="ml-2 font-medium">{job.batch_size}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">학습률:</span>
                    <span className="ml-2 font-medium">{job.learning_rate}</span>
                  </div>
                </div>
              </div>

              {/* Advanced Config */}
              {job.advanced_config && Object.keys(job.advanced_config).length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-gray-700 mb-2">고급 설정</h4>
                  <div className="space-y-3">
                    {/* Optimizer */}
                    {job.advanced_config.optimizer_type && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold text-gray-700">Optimizer</span>
                          <span className="px-2 py-0.5 bg-violet-100 text-violet-700 rounded text-xs font-medium">
                            {job.advanced_config.optimizer_type.toUpperCase()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                          {job.advanced_config.weight_decay !== undefined && (
                            <div>Weight Decay: <span className="font-medium">{job.advanced_config.weight_decay}</span></div>
                          )}
                          {job.advanced_config.momentum !== undefined && (
                            <div>Momentum: <span className="font-medium">{job.advanced_config.momentum}</span></div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Scheduler */}
                    {job.advanced_config.scheduler_type && job.advanced_config.scheduler_type !== 'none' && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold text-gray-700">LR Scheduler</span>
                          <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs font-medium">
                            {job.advanced_config.scheduler_type.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-xs text-gray-600">
                          {job.advanced_config.warmup_epochs !== undefined && (
                            <div>Warmup: {job.advanced_config.warmup_epochs} epochs</div>
                          )}
                        </div>
                      </div>
                    )}
                    {job.advanced_config.cos_lr !== undefined && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold text-gray-700">LR Scheduler</span>
                          <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs font-medium">
                            {job.advanced_config.cos_lr ? 'COSINE' : 'CONSTANT'}
                          </span>
                        </div>
                        {job.advanced_config.warmup_epochs !== undefined && (
                          <div className="text-xs text-gray-600">
                            Warmup: {job.advanced_config.warmup_epochs} epochs
                          </div>
                        )}
                      </div>
                    )}

                    {/* Augmentation - Classification (timm) */}
                    {(job.advanced_config.aug_enabled || job.advanced_config.random_flip || job.advanced_config.mixup || job.advanced_config.cutmix) && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold text-gray-700">Data Augmentation</span>
                          <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                            {job.advanced_config.aug_enabled ? '활성화' : '부분 활성화'}
                          </span>
                        </div>
                        <div className="flex gap-1 flex-wrap">
                          {job.advanced_config.random_flip && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">Flip</span>
                          )}
                          {job.advanced_config.random_rotation && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">Rotation</span>
                          )}
                          {job.advanced_config.color_jitter && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">Color Jitter</span>
                          )}
                          {job.advanced_config.mixup && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">Mixup</span>
                          )}
                          {job.advanced_config.cutmix && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">CutMix</span>
                          )}
                          {job.advanced_config.random_erasing && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">Random Erasing</span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Augmentation - Detection (YOLO) */}
                    {(job.advanced_config.mosaic !== undefined || job.advanced_config.fliplr !== undefined) && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-xs font-semibold text-gray-700">Data Augmentation (YOLO)</span>
                          <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                            활성화
                          </span>
                        </div>
                        <div className="flex gap-1 flex-wrap">
                          {job.advanced_config.mosaic > 0 && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                              Mosaic ({(job.advanced_config.mosaic * 100).toFixed(0)}%)
                            </span>
                          )}
                          {job.advanced_config.mixup > 0 && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                              Mixup ({(job.advanced_config.mixup * 100).toFixed(0)}%)
                            </span>
                          )}
                          {job.advanced_config.copy_paste > 0 && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                              Copy-Paste ({(job.advanced_config.copy_paste * 100).toFixed(0)}%)
                            </span>
                          )}
                          {job.advanced_config.fliplr > 0 && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                              Flip ({(job.advanced_config.fliplr * 100).toFixed(0)}%)
                            </span>
                          )}
                          {job.advanced_config.degrees > 0 && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">
                              Rotation (±{job.advanced_config.degrees}°)
                            </span>
                          )}
                          {(job.advanced_config.hsv_h > 0 || job.advanced_config.hsv_s > 0 || job.advanced_config.hsv_v > 0) && (
                            <span className="px-2 py-0.5 bg-gray-200 text-gray-700 rounded text-xs">HSV Aug</span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Other Settings */}
                    {(job.advanced_config.mixed_precision || job.advanced_config.gradient_clip_value || job.advanced_config.amp !== undefined) && (
                      <div className="bg-gray-50 rounded-lg p-3">
                        <span className="text-xs font-semibold text-gray-700 block mb-2">기타</span>
                        <div className="flex gap-2 flex-wrap text-xs">
                          {(job.advanced_config.mixed_precision || job.advanced_config.amp) && (
                            <span className="px-2 py-0.5 bg-amber-100 text-amber-700 rounded">Mixed Precision</span>
                          )}
                          {job.advanced_config.gradient_clip_value && (
                            <span className="px-2 py-0.5 bg-amber-100 text-amber-700 rounded">
                              Gradient Clipping ({job.advanced_config.gradient_clip_value})
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Show message if not started yet */}
        {job.status === 'pending' && (
          <div className="p-6 bg-white rounded-lg border border-gray-200 text-center">
            <p className="text-sm text-gray-500">학습을 시작하면 메트릭과 실험 정보가 표시됩니다</p>
          </div>
        )}

        {/* Metrics Section - Show after training starts */}
        {job.status !== 'pending' && (
          <>
            {/* Metrics Charts */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-900">
                  학습 메트릭 차트
                </h3>
                <div className="flex items-center gap-3">
                  <a
                    href="http://localhost:3001"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-orange-600 hover:text-orange-700 hover:underline flex items-center gap-1"
                  >
                    <ExternalLink className="w-3 h-3" />
                    Grafana
                  </a>
                  <a
                    href={`http://localhost:5000/#/experiments/1/runs?searchFilter=tags.mlflow.runName%20%3D%20%22job-${job.id}%22`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-600 hover:text-blue-700 hover:underline flex items-center gap-1"
                  >
                    <ExternalLink className="w-3 h-3" />
                    MLflow
                  </a>
                </div>
              </div>
              <MLflowMetricsCharts jobId={job.id} />
            </div>

            {/* Metrics Table */}
            <div>
              <DatabaseMetricsTable
                metrics={metrics}
                onCheckpointSelect={(checkpointPath, epoch) => {
                  console.log('Selected checkpoint:', checkpointPath, 'epoch:', epoch)
                  // Will implement resume dialog in next step
                }}
              />
            </div>

            {/* Final Accuracy */}
            {job.final_accuracy !== null && (
              <div className="p-4 bg-violet-50 rounded-lg border border-violet-200">
                <p className="text-sm font-semibold text-violet-900">
                  최종 정확도: {(job.final_accuracy * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </>
        )}

        {/* Logs Section - Collapsible */}
        <div className="bg-white rounded-lg border border-gray-200">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-gray-900">학습 로그</h3>
              {logs.length > 0 && (
                <span className="text-xs text-gray-500">
                  ({logs.length}개)
                </span>
              )}
            </div>
            {showLogs ? (
              <ChevronUp className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            )}
          </button>

          {showLogs && (
            <div className="border-t border-gray-200 p-4">
              {logs.length === 0 ? (
                <p className="text-sm text-gray-500">학습을 시작하면 로그가 표시됩니다</p>
              ) : (
                <div
                  ref={logsContainerRef}
                  className="bg-gray-900 rounded-lg p-4 font-mono text-xs overflow-auto"
                  style={{ maxHeight: '600px' }}
                >
                  {logs.map((log) => (
                    <div
                      key={log.id}
                      className={cn(
                        'mb-1 whitespace-pre-wrap break-words',
                        log.log_type === 'stderr' ? 'text-red-400' : 'text-green-400'
                      )}
                    >
                      {log.content}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Resume Dialog */}
      <ResumeDialog
        isOpen={resumeDialogMode !== null}
        onClose={() => setResumeDialogMode(null)}
        onStartFromScratch={
          resumeDialogMode === 'restart'
            ? restartTrainingFromScratch
            : startTrainingFromScratch
        }
        onResume={
          resumeDialogMode === 'restart'
            ? restartTrainingWithResume
            : startTrainingWithResume
        }
        latestCheckpointEpoch={
          metrics
            .slice()
            .reverse()
            .find((m) => m.checkpoint_path)?.epoch
        }
      />
    </div>
  )
}
