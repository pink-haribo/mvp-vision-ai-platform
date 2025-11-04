'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import DatasetImageGallery from '@/components/datasets/DatasetImageGallery'
import DatasetImageUpload from '@/components/datasets/DatasetImageUpload'

interface Dataset {
  id: string
  name: string
  description: string
  format: string
  task_type: string
  num_items: number
  source: string
}

export default function DatasetDetailPage() {
  const params = useParams()
  const router = useRouter()
  const datasetId = params.id as string

  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        setLoading(true)
        setError(null)

        const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
        const response = await fetch(`${baseUrl}/datasets/available`)

        if (!response.ok) {
          throw new Error(`Failed to fetch datasets: ${response.statusText}`)
        }

        const datasets = await response.json()
        const found = datasets.find((ds: Dataset) => ds.id === datasetId)

        if (!found) {
          throw new Error('Dataset not found')
        }

        setDataset(found)
      } catch (err) {
        console.error('Error fetching dataset:', err)
        setError(err instanceof Error ? err.message : 'Failed to load dataset')
      } finally {
        setLoading(false)
      }
    }

    if (datasetId) {
      fetchDataset()
    }
  }, [datasetId])

  const handleUploadSuccess = () => {
    // Refresh the gallery by updating the key
    setRefreshKey((prev) => prev + 1)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    )
  }

  if (error || !dataset) {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <Link href="/datasets" className="text-blue-600 hover:text-blue-700 text-sm font-medium">
              ← Back to Datasets
            </Link>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <p className="text-red-800">Error: {error || 'Dataset not found'}</p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link href="/datasets" className="text-blue-600 hover:text-blue-700 text-sm font-medium mb-4 inline-block">
            ← Back to Datasets
          </Link>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{dataset.name}</h1>
              <p className="text-sm text-gray-600 mt-1">{dataset.description}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                {dataset.format.toUpperCase()}
              </span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                {dataset.task_type.replace('_', ' ')}
              </span>
            </div>
          </div>

          {/* Dataset Info */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">ID:</span>
              <p className="font-medium text-gray-900 truncate" title={dataset.id}>{dataset.id}</p>
            </div>
            <div>
              <span className="text-gray-600">Total Images:</span>
              <p className="font-medium text-gray-900">{dataset.num_items.toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-600">Source:</span>
              <p className="font-medium text-gray-900">{dataset.source}</p>
            </div>
            <div>
              <span className="text-gray-600">Format:</span>
              <p className="font-medium text-gray-900">{dataset.format}</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left column: Image Upload */}
          <div className="lg:col-span-1">
            <DatasetImageUpload datasetId={datasetId} onUploadSuccess={handleUploadSuccess} />
          </div>

          {/* Right column: Image Gallery */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Dataset Images</h3>
              <DatasetImageGallery key={refreshKey} datasetId={datasetId} />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
