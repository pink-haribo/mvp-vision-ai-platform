'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Square, AlertCircle, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import MLflowMetricsCharts from './training/MLflowMetricsCharts'
import MLflowMetricsTable from './training/MLflowMetricsTable'
import MLflowBestModel from './training/MLflowBestModel'

interface TrainingJob {
  id: number
  framework: string
  model_name: string
  task_type: string
  num_classes: number | null
  dataset_format: string
  epochs: number
  batch_size: number
  learning_rate: number
  status: string
  final_accuracy: number | null
}

interface TrainingMetric {
  epoch: number
  loss: number
  accuracy: number
  extra_metrics: {
    train_loss: number
    train_accuracy: number
  }
}

interface TrainingLog {
  id: number
  log_type: string
  content: string
  created_at: string
}

interface TrainingPanelProps {
  trainingJobId: number | null
}

export default function TrainingPanel({ trainingJobId }: TrainingPanelProps) {
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [metrics, setMetrics] = useState<TrainingMetric[]>([])
  const [logs, setLogs] = useState<TrainingLog[]>([])
  const [isLoading, setIsLoading] = useState(false)
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
          setMetrics(data)
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
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${trainingJobId}/logs?limit=200`
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

    setIsLoading(true)
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

  // Calculate current epoch and iteration progress
  const getCurrentProgress = () => {
    if (!job || metrics.length === 0) {
      return { currentEpoch: 0, totalEpochs: job?.epochs || 0, currentIteration: 0, totalIterations: 0 }
    }

    // Get latest metric
    const latestMetric = metrics[metrics.length - 1]
    const currentEpoch = latestMetric.epoch || 0
    const totalEpochs = job.epochs

    // Get iteration info from extra_metrics if available
    const currentIteration = latestMetric.extra_metrics?.batch || 0
    const totalIterations = latestMetric.extra_metrics?.total_batches || 0

    return { currentEpoch, totalEpochs, currentIteration, totalIterations }
  }

  const progress = getCurrentProgress()

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
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">학습 진행 상황</h2>
          {getStatusBadge(job.status)}
        </div>

        {/* Action Buttons and Progress */}
        <div className="space-y-3">
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
          </div>

          {/* Progress Bars - Show only when training is running */}
          {job.status === 'running' && metrics.length > 0 && (
            <div className="space-y-2">
              {/* Epoch Progress */}
              <div>
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span>Epoch Progress</span>
                  <span>{progress.currentEpoch} / {progress.totalEpochs}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-violet-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${(progress.currentEpoch / progress.totalEpochs) * 100}%` }}
                  />
                </div>
              </div>

              {/* Iteration Progress */}
              {progress.totalIterations > 0 && (
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
          )}
        </div>
      </div>

      {/* Training Config */}
      <div className="p-6 bg-white border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">학습 설정</h3>
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
              <span className="text-gray-600">클래스:</span>
              <span className="ml-2 font-medium">{job.num_classes}</span>
            </div>
          )}
          <div>
            <span className="text-gray-600">에포크:</span>
            <span className="ml-2 font-medium">{job.epochs}</span>
          </div>
          <div>
            <span className="text-gray-600">배치:</span>
            <span className="ml-2 font-medium">{job.batch_size}</span>
          </div>
          <div>
            <span className="text-gray-600">학습률:</span>
            <span className="ml-2 font-medium">{job.learning_rate}</span>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* External Monitoring Links */}
        {(job.status === 'running' || job.status === 'completed') && (
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">모니터링 대시보드</h3>
            <div className="flex gap-3">
              <a
                href="http://localhost:3001"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-orange-500 to-red-500 text-white font-medium rounded-lg hover:from-orange-600 hover:to-red-600 transition-all"
              >
                <ExternalLink className="w-4 h-4" />
                Grafana 대시보드
              </a>
              <a
                href={`http://localhost:5000/#/experiments/1/runs?searchFilter=tags.mlflow.runName%20%3D%20%22job-${job.id}%22`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-medium rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all"
              >
                <ExternalLink className="w-4 h-4" />
                MLflow 실험 추적
              </a>
            </div>
          </div>
        )}

        {/* MLflow Metrics and Best Model - Show when training is running or completed */}
        {(job.status === 'running' || job.status === 'completed') && (
          <>
            {/* Best Model Info */}
            <MLflowBestModel jobId={job.id} />

            {/* Metrics Charts */}
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                학습 메트릭 차트
              </h3>
              <MLflowMetricsCharts jobId={job.id} />
            </div>

            {/* Metrics Table */}
            <div>
              <MLflowMetricsTable jobId={job.id} />
            </div>
          </>
        )}

        {/* Show message if not started yet */}
        {job.status === 'pending' && (
          <div className="p-6 bg-white rounded-lg border border-gray-200 text-center">
            <p className="text-sm text-gray-500">학습을 시작하면 메트릭과 실험 정보가 표시됩니다</p>
          </div>
        )}

        {/* Final Accuracy */}
        {job.final_accuracy !== null && (
          <div className="p-4 bg-violet-50 rounded-lg border border-violet-200">
            <p className="text-sm font-semibold text-violet-900">
              최종 정확도: {job.final_accuracy.toFixed(2)}%
            </p>
          </div>
        )}

        {/* Logs */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 mb-3">학습 로그</h3>

          {logs.length === 0 ? (
            <p className="text-sm text-gray-500">학습을 시작하면 로그가 표시됩니다</p>
          ) : (
            <div
              ref={logsContainerRef}
              className="bg-gray-900 rounded-lg p-4 font-mono text-xs overflow-auto max-h-96"
            >
              {logs.map((log) => (
                <div
                  key={log.id}
                  className={cn(
                    'mb-1',
                    log.log_type === 'stderr' ? 'text-red-400' : 'text-green-400'
                  )}
                >
                  {log.content}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
