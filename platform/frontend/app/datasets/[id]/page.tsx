'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import Sidebar from '@/components/Sidebar'
import DatasetImageGallery from '@/components/datasets/DatasetImageGallery'
import DatasetImageUpload from '@/components/datasets/DatasetImageUpload'
import { ArrowLeft } from 'lucide-react'

interface Dataset {
  id: string
  name: string
  description: string
  format: string
  labeled: boolean
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
        const token = localStorage.getItem('access_token')

        if (!token) {
          setError('로그인이 필요합니다.')
          setLoading(false)
          return
        }

        const response = await fetch(`${baseUrl}/datasets/available`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!response.ok) {
          if (response.status === 401) {
            throw new Error('로그인이 필요합니다.')
          }
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
      <div className="h-screen flex">
        <Sidebar onProjectSelect={() => {}} onCreateProject={() => {}} />
        <div className="flex-1 flex items-center justify-center bg-gray-50">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-600" />
        </div>
      </div>
    )
  }

  if (error || !dataset) {
    return (
      <div className="h-screen flex">
        <Sidebar onProjectSelect={() => {}} onCreateProject={() => {}} />
        <div className="flex-1 flex flex-col bg-gray-50">
          <header className="bg-white border-b border-gray-200 px-6 py-4">
            <button
              onClick={() => router.push('/')}
              className="text-violet-600 hover:text-violet-700 text-sm font-medium flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              홈으로 돌아가기
            </button>
          </header>
          <main className="flex-1 p-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <p className="text-red-800">Error: {error || 'Dataset not found'}</p>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex">
      <Sidebar onProjectSelect={() => {}} onCreateProject={() => {}} />

      <div className="flex-1 flex flex-col overflow-hidden bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex-shrink-0">
          <button
            onClick={() => router.push('/')}
            className="text-violet-600 hover:text-violet-700 text-sm font-medium mb-4 inline-flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            홈으로 돌아가기
          </button>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{dataset.name}</h1>
              <p className="text-sm text-gray-600 mt-1">{dataset.description}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                {dataset.format.toUpperCase()}
              </span>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                dataset.labeled
                  ? 'bg-green-100 text-green-800'
                  : 'bg-gray-100 text-gray-600'
              }`}>
                {dataset.labeled ? 'Labeled' : 'Unlabeled'}
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
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-auto p-6">
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
    </div>
  )
}
