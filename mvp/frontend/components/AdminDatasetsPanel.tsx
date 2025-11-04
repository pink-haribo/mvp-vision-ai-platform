'use client'

import { useState, useEffect } from 'react'
import { ArrowUpDown, ArrowUp, ArrowDown, Search, Database, Tag } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { Dataset } from '@/types/dataset'

type SortField = 'name' | 'format' | 'task_type' | 'num_items' | 'source'
type SortDirection = 'asc' | 'desc' | null

const taskTypeColors: Record<string, string> = {
  image_classification: 'bg-blue-100 text-blue-800',
  object_detection: 'bg-green-100 text-green-800',
  instance_segmentation: 'bg-purple-100 text-purple-800',
  semantic_segmentation: 'bg-pink-100 text-pink-800',
  pose_estimation: 'bg-yellow-100 text-yellow-800',
}

const formatNames: Record<string, string> = {
  imagefolder: 'ImageFolder',
  yolo: 'YOLO',
  coco: 'COCO',
  pascal_voc: 'Pascal VOC',
}

export default function AdminDatasetsPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [filteredDatasets, setFilteredDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)

  // Sorting state
  const [sortField, setSortField] = useState<SortField | null>(null)
  const [sortDirection, setSortDirection] = useState<SortDirection>(null)

  // Filter state
  const [searchQuery, setSearchQuery] = useState('')
  const [taskTypeFilter, setTaskTypeFilter] = useState<string>('all')

  useEffect(() => {
    fetchDatasets()
  }, [])

  useEffect(() => {
    applyFiltersAndSort()
  }, [datasets, searchQuery, taskTypeFilter, sortField, sortDirection])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const response = await fetch(`${baseUrl}/datasets/available`)

      if (response.ok) {
        const data = await response.json()
        setDatasets(data)
      } else {
        console.error('Failed to fetch datasets')
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
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
        d.id.toLowerCase().includes(query) ||
        d.description.toLowerCase().includes(query) ||
        d.format.toLowerCase().includes(query) ||
        d.task_type.toLowerCase().includes(query)
      )
    }

    // Apply task type filter
    if (taskTypeFilter !== 'all') {
      result = result.filter(d => d.task_type === taskTypeFilter)
    }

    // Apply sorting
    if (sortField && sortDirection) {
      result.sort((a, b) => {
        let aVal = a[sortField]
        let bVal = b[sortField]

        // Handle nulls
        if (aVal === null || aVal === undefined) return 1
        if (bVal === null || bVal === undefined) return -1

        if (typeof aVal === 'string' && typeof bVal === 'string') {
          return sortDirection === 'asc'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal)
        }

        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
        }

        return 0
      })
    }

    setFilteredDatasets(result)
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

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 text-gray-400" />
    }
    if (sortDirection === 'asc') {
      return <ArrowUp className="w-4 h-4 text-indigo-600" />
    }
    return <ArrowDown className="w-4 h-4 text-indigo-600" />
  }

  const taskTypes = [
    { value: 'all', label: 'All Task Types' },
    { value: 'image_classification', label: 'Classification' },
    { value: 'object_detection', label: 'Detection' },
    { value: 'instance_segmentation', label: 'Instance Seg.' },
    { value: 'semantic_segmentation', label: 'Semantic Seg.' },
    { value: 'pose_estimation', label: 'Pose' },
  ]

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <Database className="w-6 h-6 text-indigo-600" />
              데이터셋 관리
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              플랫폼에서 제공하는 공용 데이터셋 목록
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-indigo-600">{filteredDatasets.length}</div>
            <div className="text-xs text-gray-500">Total Datasets</div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>

          {/* Task Type Filter */}
          <select
            value={taskTypeFilter}
            onChange={(e) => setTaskTypeFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
          >
            {taskTypes.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="bg-gray-100 sticky top-0 z-10">
            <tr>
              <th className="px-6 py-3 text-left">
                <button
                  onClick={() => handleSort('name')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Name
                  <SortIcon field="name" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <button
                  onClick={() => handleSort('task_type')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Task Type
                  <SortIcon field="task_type" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <button
                  onClick={() => handleSort('format')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Format
                  <SortIcon field="format" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <button
                  onClick={() => handleSort('num_items')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Images
                  <SortIcon field="num_items" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <button
                  onClick={() => handleSort('source')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Source
                  <SortIcon field="source" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  Description
                </span>
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredDatasets.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-12 text-center">
                  <Database className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-500">No datasets found</p>
                  <p className="text-sm text-gray-400 mt-1">Try adjusting your filters</p>
                </td>
              </tr>
            ) : (
              filteredDatasets.map((dataset) => {
                const taskTypeColor = taskTypeColors[dataset.task_type] || 'bg-gray-100 text-gray-800'
                const formatName = formatNames[dataset.format] || dataset.format

                return (
                  <tr key={dataset.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-medium text-gray-900">{dataset.name}</div>
                        <div className="text-xs text-gray-500 mt-0.5 font-mono">{dataset.id}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={cn('px-2 py-1 rounded-full text-xs font-medium', taskTypeColor)}>
                        {dataset.task_type.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        {formatName}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm font-medium text-gray-900">
                        {dataset.num_items.toLocaleString()}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm text-gray-600 capitalize">{dataset.source}</span>
                    </td>
                    <td className="px-6 py-4">
                      <p className="text-sm text-gray-600 line-clamp-2">{dataset.description}</p>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Footer Stats */}
      <div className="bg-white border-t border-gray-200 px-6 py-3">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div>
            Showing <span className="font-medium text-gray-900">{filteredDatasets.length}</span> of{' '}
            <span className="font-medium text-gray-900">{datasets.length}</span> datasets
          </div>
          <div className="flex gap-4">
            <div>
              Total Images:{' '}
              <span className="font-medium text-gray-900">
                {filteredDatasets.reduce((sum, d) => sum + d.num_items, 0).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
