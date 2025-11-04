'use client'

import { useState, useEffect } from 'react'
import { Search, X, FolderPlus, Eye, ArrowUpDown, ArrowUp, ArrowDown, Database, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import CreateDatasetModal from './datasets/CreateDatasetModal'
import DatasetImageGallery from './datasets/DatasetImageGallery'
import DatasetImageUpload from './datasets/DatasetImageUpload'

interface Dataset {
  id: string
  name: string
  description: string
  format: string
  labeled: boolean
  num_items: number
  source: string
  size_mb?: number
  tags?: string[]
  created_at?: string
}

type SortField = 'name' | 'format' | 'labeled' | 'num_items' | 'source'
type SortDirection = 'asc' | 'desc' | null

export default function DatasetPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [filteredDatasets, setFilteredDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [deleteDatasetId, setDeleteDatasetId] = useState<string | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleting, setDeleting] = useState(false)

  // Sorting state
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDirection, setSortDirection] = useState<SortDirection>(null)

  useEffect(() => {
    fetchDatasets()
  }, [refreshKey])

  useEffect(() => {
    applyFiltersAndSort()
  }, [datasets, searchQuery, sortField, sortDirection])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const token = localStorage.getItem('access_token')

      if (!token) {
        console.error('No access token found')
        setDatasets([])
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
          console.error('Unauthorized: Please login')
        }
        throw new Error(`Failed to fetch datasets: ${response.statusText}`)
      }

      const data = await response.json()
      setDatasets(data)
    } catch (error) {
      console.error('Error fetching datasets:', error)
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  const applyFiltersAndSort = () => {
    let result = [...datasets]

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      result = result.filter(d =>
        d.name.toLowerCase().includes(query) ||
        d.description.toLowerCase().includes(query) ||
        d.id.toLowerCase().includes(query) ||
        d.format.toLowerCase().includes(query) ||
        d.source.toLowerCase().includes(query)
      )
    }

    // Apply sorting
    if (sortField && sortDirection) {
      result.sort((a, b) => {
        let aVal: any = a[sortField]
        let bVal: any = b[sortField]

        if (aVal === null || aVal === undefined) aVal = ''
        if (bVal === null || bVal === undefined) bVal = ''

        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    }

    setFilteredDatasets(result)
  }

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      if (sortDirection === 'asc') {
        setSortDirection('desc')
      } else if (sortDirection === 'desc') {
        setSortField(null)
        setSortDirection(null)
      } else {
        setSortDirection('asc')
      }
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) return <ArrowUpDown className="w-3 h-3" />
    if (sortDirection === 'asc') return <ArrowUp className="w-3 h-3" />
    if (sortDirection === 'desc') return <ArrowDown className="w-3 h-3" />
    return <ArrowUpDown className="w-3 h-3" />
  }

  const clearSearch = () => {
    setSearchQuery('')
  }

  const handleCreateSuccess = async (datasetId: string) => {
    setShowCreateModal(false)
    setRefreshKey(prev => prev + 1)
  }

  const handleViewDataset = (datasetId: string) => {
    setSelectedDatasetId(selectedDatasetId === datasetId ? null : datasetId)
  }

  const handleImageUploadSuccess = () => {
    // Refresh the image gallery
    setRefreshKey(prev => prev + 1)
  }

  const handleDeleteClick = (datasetId: string) => {
    setDeleteDatasetId(datasetId)
    setShowDeleteConfirm(true)
  }

  const handleDeleteConfirm = async () => {
    if (!deleteDatasetId) return

    setDeleting(true)
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const token = localStorage.getItem('access_token')

      if (!token) {
        alert('로그인이 필요합니다.')
        setDeleting(false)
        setShowDeleteConfirm(false)
        return
      }

      const response = await fetch(`${baseUrl}/datasets/${deleteDatasetId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      const data = await response.json()

      if (data.status === 'success') {
        // Refresh dataset list
        setRefreshKey(prev => prev + 1)
        // Close the expanded view if the deleted dataset was selected
        if (selectedDatasetId === deleteDatasetId) {
          setSelectedDatasetId(null)
        }
      } else {
        alert(`삭제 실패: ${data.message}`)
      }
    } catch (error) {
      console.error('Error deleting dataset:', error)
      alert('데이터셋 삭제 중 오류가 발생했습니다.')
    } finally {
      setDeleting(false)
      setShowDeleteConfirm(false)
      setDeleteDatasetId(null)
    }
  }

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false)
    setDeleteDatasetId(null)
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-gray-500 text-sm">로딩 중...</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-violet-600" />
            <h2 className="text-xl font-bold text-gray-900">데이터셋 관리</h2>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-3 py-2 bg-violet-600 text-white text-sm font-medium rounded-lg hover:bg-violet-700 transition-colors flex items-center gap-2"
          >
            <FolderPlus className="w-4 h-4" />
            데이터셋 생성
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="bg-white border-b border-gray-200 px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="데이터셋명, 포맷, 작업 유형 등으로 검색..."
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
              <>표시 중: {filteredDatasets.length}개 / 전체 {datasets.length}개</>
            ) : (
              <>전체 {datasets.length}개</>
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
                <button
                  onClick={() => handleSort('name')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  데이터셋명
                  {getSortIcon('name')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('labeled')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  상태
                  {getSortIcon('labeled')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('format')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  포맷
                  {getSortIcon('format')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('source')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  소스
                  {getSortIcon('source')}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('num_items')}
                  className="flex items-center gap-1 hover:text-violet-600"
                >
                  이미지 수
                  {getSortIcon('num_items')}
                </button>
              </th>
              <th className="px-4 py-3 text-center text-xs font-semibold text-gray-700 uppercase tracking-wider">
                작업
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredDatasets.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-500 text-sm">
                  {searchQuery
                    ? '검색 조건에 맞는 데이터셋이 없습니다.'
                    : '데이터셋이 없습니다. 데이터셋을 업로드하세요.'}
                </td>
              </tr>
            ) : (
              filteredDatasets.map((dataset) => (
                <>
                  <tr
                    key={dataset.id}
                    onClick={() => handleViewDataset(dataset.id)}
                    className="hover:bg-gray-50 cursor-pointer"
                  >
                    <td className="px-4 py-3 text-sm">
                      <div className="font-medium text-gray-900">{dataset.name}</div>
                      <div className="text-gray-500 text-xs mt-0.5 truncate max-w-xs">
                        {dataset.description}
                      </div>
                      <div className="text-gray-400 text-xs mt-0.5 font-mono">
                        {dataset.id}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={cn(
                        "px-2 py-1 rounded-full text-xs font-medium",
                        dataset.labeled
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-600'
                      )}>
                        {dataset.labeled ? 'Labeled' : 'Unlabeled'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 uppercase">
                      {dataset.format}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {dataset.source}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {dataset.num_items.toLocaleString()}
                      {dataset.size_mb && (
                        <span className="text-xs text-gray-500 ml-2">
                          ({dataset.size_mb.toFixed(1)} MB)
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm text-center">
                      <div className="flex items-center justify-center gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleViewDataset(dataset.id)
                          }}
                          className={cn(
                            "p-2 rounded-lg transition-colors",
                            selectedDatasetId === dataset.id
                              ? "bg-violet-100 text-violet-700"
                              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                          )}
                          title={selectedDatasetId === dataset.id ? '닫기' : '이미지 보기'}
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteClick(dataset.id)
                          }}
                          className="p-2 rounded-lg transition-colors bg-red-50 text-red-700 hover:bg-red-100"
                          title="데이터셋 삭제"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>

                  {/* Expanded row for image gallery */}
                  {selectedDatasetId === dataset.id && (
                    <tr>
                      <td colSpan={6} className="px-4 py-4 bg-gray-50">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                          {/* Upload Section */}
                          <div className="lg:col-span-1">
                            <DatasetImageUpload
                              datasetId={dataset.id}
                              onUploadSuccess={handleImageUploadSuccess}
                            />
                          </div>

                          {/* Gallery Section */}
                          <div className="lg:col-span-2">
                            <div className="bg-white rounded-lg border border-gray-200 p-4">
                              <h4 className="text-sm font-semibold text-gray-900 mb-3">데이터셋 이미지</h4>
                              <DatasetImageGallery key={refreshKey} datasetId={dataset.id} />
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Create Dataset Modal */}
      <CreateDatasetModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={handleCreateSuccess}
      />

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                데이터셋 삭제 확인
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                이 데이터셋을 삭제하시겠습니까? 이 작업은 되돌릴 수 없으며, 모든 이미지와 메타데이터가 영구적으로 삭제됩니다.
              </p>
              {deleteDatasetId && (
                <div className="bg-gray-50 rounded p-3 mb-4">
                  <p className="text-xs text-gray-600 mb-1">삭제할 데이터셋:</p>
                  <p className="text-sm font-medium text-gray-900">
                    {datasets.find(d => d.id === deleteDatasetId)?.name}
                  </p>
                  <p className="text-xs text-gray-500 font-mono mt-1">
                    {deleteDatasetId}
                  </p>
                </div>
              )}
            </div>
            <div className="bg-gray-50 px-6 py-4 flex justify-end gap-3 rounded-b-lg">
              <button
                onClick={handleDeleteCancel}
                disabled={deleting}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                취소
              </button>
              <button
                onClick={handleDeleteConfirm}
                disabled={deleting}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {deleting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    삭제 중...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    삭제
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
