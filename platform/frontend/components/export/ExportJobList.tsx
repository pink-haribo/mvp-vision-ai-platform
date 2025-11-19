'use client'

import { useState, useEffect } from 'react'
import ExportJobCard, { ExportJob } from './ExportJobCard'
import { Loader2, AlertCircle } from 'lucide-react'

interface ExportJobListProps {
  trainingJobId: number
  onCreateExport?: () => void
  onDeploy?: (exportJobId: number) => void
  refreshKey?: number // Incremented by parent when WebSocket receives export updates
}

export default function ExportJobList({ trainingJobId, onCreateExport, onDeploy, refreshKey }: ExportJobListProps) {
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchExportJobs = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const token = localStorage.getItem('access_token')
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/training/${trainingJobId}/exports`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch export jobs')
      }

      const data = await response.json()
      setExportJobs(data.export_jobs || [])
    } catch (err) {
      console.error('Error fetching export jobs:', err)
      setError(err instanceof Error ? err.message : 'Failed to load export jobs')
    } finally {
      setIsLoading(false)
    }
  }

  // Initial fetch and WebSocket-triggered refresh (no polling)
  useEffect(() => {
    fetchExportJobs()
  }, [trainingJobId, refreshKey])

  const handleDownload = async (exportJobId: number) => {
    try {
      const token = localStorage.getItem('access_token')
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/${exportJobId}/download`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to generate download URL')
      }

      const data = await response.json()

      // Open presigned URL in new tab
      window.open(data.download_url, '_blank')
    } catch (err) {
      console.error('Error downloading export:', err)
      alert('Failed to download export. Please try again.')
    }
  }

  const handleDelete = async (exportJobId: number) => {
    if (!confirm('Are you sure you want to delete this export job?')) {
      return
    }

    try {
      const token = localStorage.getItem('access_token')
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/jobs/${exportJobId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to delete export job')
      }

      // Refresh list
      await fetchExportJobs()
    } catch (err) {
      console.error('Error deleting export job:', err)
      alert('Failed to delete export job. Please try again.')
    }
  }

  if (isLoading && exportJobs.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
        <p className="text-gray-600">Loading export jobs...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg border border-red-200 p-8 text-center">
        <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-4" />
        <p className="text-red-600 mb-2">Failed to load export jobs</p>
        <p className="text-sm text-gray-500">{error}</p>
        <button
          onClick={fetchExportJobs}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
        >
          Retry
        </button>
      </div>
    )
  }

  if (exportJobs.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <div className="text-gray-400 text-4xl mb-4">📦</div>
        <p className="text-gray-600 mb-2">No export jobs yet</p>
        <p className="text-sm text-gray-500 mb-4">
          Export your trained model to production formats (ONNX, TensorRT, CoreML, etc.)
        </p>
        <button
          onClick={onCreateExport}
          className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium"
        >
          + Create Export
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Export Jobs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {exportJobs.map((job) => (
          <ExportJobCard
            key={job.id}
            job={job}
            onDownload={() => handleDownload(job.id)}
            onDeploy={() => onDeploy?.(job.id)}
            onDelete={() => handleDelete(job.id)}
          />
        ))}
      </div>

      {/* Loading indicator for ongoing refresh */}
      {isLoading && exportJobs.length > 0 && (
        <div className="text-center py-2">
          <span className="text-sm text-gray-500">Refreshing...</span>
        </div>
      )}
    </div>
  )
}
