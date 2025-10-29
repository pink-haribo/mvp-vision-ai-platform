'use client'

/**
 * TestInferencePanel Component
 *
 * Displays test runs and inference jobs for a trained model.
 * Allows users to run new tests on labeled datasets or inference on unlabeled data.
 */

import { useState, useEffect } from 'react'
import { Play, FileText, Image, CheckCircle, XCircle, Clock, ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface TestRun {
  id: number
  training_job_id: number
  checkpoint_path: string
  dataset_path: string
  dataset_split: string
  status: string
  error_message: string | null
  task_type: string
  primary_metric_name: string | null
  primary_metric_value: number | null
  overall_loss: number | null
  metrics: any
  total_images: number
  inference_time_ms: number | null
  created_at: string
  completed_at: string | null
}

interface InferenceJob {
  id: number
  training_job_id: number
  checkpoint_path: string
  inference_type: string
  status: string
  task_type: string
  total_images: number
  avg_inference_time_ms: number | null
  created_at: string
  completed_at: string | null
}

interface TestInferencePanelProps {
  jobId: number
}

export default function TestInferencePanel({ jobId }: TestInferencePanelProps) {
  const [testRuns, setTestRuns] = useState<TestRun[]>([])
  const [inferenceJobs, setInferenceJobs] = useState<InferenceJob[]>([])
  const [loading, setLoading] = useState(true)
  const [activeView, setActiveView] = useState<'test' | 'inference'>('test')
  const [expandedTestId, setExpandedTestId] = useState<number | null>(null)
  const [expandedInferenceId, setExpandedInferenceId] = useState<number | null>(null)

  // Fetch test runs
  useEffect(() => {
    fetchTestRuns()
  }, [jobId])

  // Fetch inference jobs
  useEffect(() => {
    fetchInferenceJobs()
  }, [jobId])

  const fetchTestRuns = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/test_inference/test/jobs/${jobId}/runs`
      )
      if (response.ok) {
        const data = await response.json()
        setTestRuns(data.test_runs || [])
      }
    } catch (error) {
      console.error('Error fetching test runs:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchInferenceJobs = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/test_inference/inference/jobs/${jobId}`
      )
      if (response.ok) {
        const data = await response.json()
        setInferenceJobs(data.inference_jobs || [])
      }
    } catch (error) {
      console.error('Error fetching inference jobs:', error)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'running':
        return <Clock className="h-5 w-5 text-blue-500 animate-pulse" />
      default:
        return <Clock className="h-5 w-5 text-gray-400" />
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleString('ko-KR', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with tabs */}
      <div className="flex items-center justify-between">
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveView('test')}
            className={cn(
              'px-4 py-2 rounded-lg font-medium transition-colors',
              activeView === 'test'
                ? 'bg-indigo-100 text-indigo-700'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <FileText className="inline h-4 w-4 mr-2" />
            Test Runs ({testRuns.length})
          </button>
          <button
            onClick={() => setActiveView('inference')}
            className={cn(
              'px-4 py-2 rounded-lg font-medium transition-colors',
              activeView === 'inference'
                ? 'bg-indigo-100 text-indigo-700'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <Image className="inline h-4 w-4 mr-2" />
            Inference Jobs ({inferenceJobs.length})
          </button>
        </div>

        <div className="flex space-x-2">
          <button
            onClick={() => {
              // TODO: Open test run dialog
              alert('Test run dialog coming soon!')
            }}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center"
          >
            <Play className="h-4 w-4 mr-2" />
            Run Test
          </button>
          <button
            onClick={() => {
              // TODO: Open inference dialog
              alert('Inference dialog coming soon!')
            }}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center"
          >
            <Play className="h-4 w-4 mr-2" />
            Run Inference
          </button>
        </div>
      </div>

      {/* Test Runs View */}
      {activeView === 'test' && (
        <div className="space-y-3">
          {testRuns.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
              <p>No test runs yet</p>
              <p className="text-sm mt-1">Run a test on your trained model to evaluate its performance</p>
            </div>
          ) : (
            testRuns.map((testRun) => (
              <div
                key={testRun.id}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => setExpandedTestId(expandedTestId === testRun.id ? null : testRun.id)}
                >
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(testRun.status)}
                    <div>
                      <div className="font-medium">
                        Test Run #{testRun.id}
                      </div>
                      <div className="text-sm text-gray-500">
                        {formatDate(testRun.created_at)} • {testRun.total_images} images
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    {testRun.primary_metric_value !== null && (
                      <div className="text-right">
                        <div className="text-sm text-gray-500">{testRun.primary_metric_name}</div>
                        <div className="text-lg font-semibold text-indigo-600">
                          {(testRun.primary_metric_value * 100).toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {expandedTestId === testRun.id ? (
                      <ChevronDown className="h-5 w-5 text-gray-400" />
                    ) : (
                      <ChevronRight className="h-5 w-5 text-gray-400" />
                    )}
                  </div>
                </div>

                {expandedTestId === testRun.id && (
                  <div className="mt-4 pt-4 border-t border-gray-200 space-y-2">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Dataset:</span>
                        <span className="ml-2 font-mono text-xs">{testRun.dataset_path}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Split:</span>
                        <span className="ml-2">{testRun.dataset_split}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Task:</span>
                        <span className="ml-2">{testRun.task_type}</span>
                      </div>
                      {testRun.inference_time_ms && (
                        <div>
                          <span className="text-gray-500">Inference Time:</span>
                          <span className="ml-2">{(testRun.inference_time_ms / 1000).toFixed(2)}s</span>
                        </div>
                      )}
                    </div>

                    {testRun.metrics && (
                      <div className="mt-3">
                        <div className="text-sm font-medium text-gray-700 mb-2">Metrics:</div>
                        <div className="bg-gray-50 p-3 rounded text-xs font-mono">
                          <pre>{JSON.stringify(testRun.metrics, null, 2)}</pre>
                        </div>
                      </div>
                    )}

                    {testRun.error_message && (
                      <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                        {testRun.error_message}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Inference Jobs View */}
      {activeView === 'inference' && (
        <div className="space-y-3">
          {inferenceJobs.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <Image className="h-12 w-12 mx-auto mb-3 text-gray-300" />
              <p>No inference jobs yet</p>
              <p className="text-sm mt-1">Run inference on new images using your trained model</p>
            </div>
          ) : (
            inferenceJobs.map((job) => (
              <div
                key={job.id}
                className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
              >
                <div
                  className="flex items-center justify-between cursor-pointer"
                  onClick={() => setExpandedInferenceId(expandedInferenceId === job.id ? null : job.id)}
                >
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <div className="font-medium">
                        Inference Job #{job.id}
                      </div>
                      <div className="text-sm text-gray-500">
                        {formatDate(job.created_at)} • {job.total_images} images • {job.inference_type}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    {job.avg_inference_time_ms !== null && (
                      <div className="text-right">
                        <div className="text-sm text-gray-500">Avg Time/Image</div>
                        <div className="text-lg font-semibold text-green-600">
                          {job.avg_inference_time_ms.toFixed(1)}ms
                        </div>
                      </div>
                    )}
                    {expandedInferenceId === job.id ? (
                      <ChevronDown className="h-5 w-5 text-gray-400" />
                    ) : (
                      <ChevronRight className="h-5 w-5 text-gray-400" />
                    )}
                  </div>
                </div>

                {expandedInferenceId === job.id && (
                  <div className="mt-4 pt-4 border-t border-gray-200 space-y-2">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Checkpoint:</span>
                        <span className="ml-2 font-mono text-xs">{job.checkpoint_path}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Task:</span>
                        <span className="ml-2">{job.task_type}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Type:</span>
                        <span className="ml-2">{job.inference_type}</span>
                      </div>
                      {job.completed_at && (
                        <div>
                          <span className="text-gray-500">Completed:</span>
                          <span className="ml-2">{formatDate(job.completed_at)}</span>
                        </div>
                      )}
                    </div>

                    <button
                      onClick={() => {
                        // TODO: Navigate to results page
                        alert(`View results for inference job ${job.id}`)
                      }}
                      className="mt-3 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded text-sm font-medium transition-colors"
                    >
                      View Predictions
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
