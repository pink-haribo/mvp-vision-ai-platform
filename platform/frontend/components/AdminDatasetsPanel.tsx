'use client'

import { useState, useEffect } from 'react'
import { ArrowUpDown, ArrowUp, ArrowDown, Search, Database, Tag, Globe, Lock } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { Dataset } from '@/types/dataset'
import { getAvatarColorStyle } from '@/lib/utils/avatarColors'

type SortField = 'name' | 'labeled' | 'num_items' | 'source' | 'visibility'
type SortDirection = 'asc' | 'desc' | null

const formatNames: Record<string, string> = {
  imagefolder: 'ImageFolder',
  yolo: 'YOLO',
  coco: 'COCO',
  pascal_voc: 'Pascal VOC',
  dice: 'DICE Format',
}

// Avatar helper function
const getAvatarInitials = (owner_name: string | null | undefined, owner_email: string | null | undefined): string => {
  if (owner_name) {
    // Korean name: take first 2 characters
    if (/[가-힣]/.test(owner_name)) {
      return owner_name.slice(0, 2)
    }
    // English name: take first letter of first and last name
    const parts = owner_name.split(' ')
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
    }
    return owner_name.slice(0, 2).toUpperCase()
  }
  if (owner_email) {
    return owner_email.slice(0, 2).toUpperCase()
  }
  return '?'
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
  const [labeledFilter, setLabeledFilter] = useState<string>('all')

  useEffect(() => {
    fetchDatasets()
  }, [])

  useEffect(() => {
    applyFiltersAndSort()
  }, [datasets, searchQuery, labeledFilter, sortField, sortDirection])

  const fetchDatasets = async () => {
    setLoading(true)
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const token = localStorage.getItem('access_token')

      if (!token) {
        console.error('No access token found')
        setLoading(false)
        return
      }

      const response = await fetch(`${baseUrl}/datasets/available`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setDatasets(data)
      } else {
        console.error('Failed to fetch datasets:', response.status, response.statusText)
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
        (d.owner_name && d.owner_name.toLowerCase().includes(query)) ||
        (d.owner_email && d.owner_email.toLowerCase().includes(query))
      )
    }

    // Apply labeled filter
    if (labeledFilter === 'labeled') {
      result = result.filter(d => d.labeled === true)
    } else if (labeledFilter === 'unlabeled') {
      result = result.filter(d => d.labeled === false)
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

  const labelFilters = [
    { value: 'all', label: 'All Datasets' },
    { value: 'labeled', label: 'Labeled' },
    { value: 'unlabeled', label: 'Unlabeled' },
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

          {/* Label Status Filter */}
          <select
            value={labeledFilter}
            onChange={(e) => setLabeledFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
          >
            {labelFilters.map((filter) => (
              <option key={filter.value} value={filter.value}>
                {filter.label}
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
                  onClick={() => handleSort('labeled')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Status
                  <SortIcon field="labeled" />
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
                <button
                  onClick={() => handleSort('visibility')}
                  className="flex items-center gap-2 text-xs font-semibold text-gray-700 uppercase tracking-wider hover:text-indigo-600"
                >
                  Visibility
                  <SortIcon field="visibility" />
                </button>
              </th>
              <th className="px-6 py-3 text-left">
                <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">
                  Owner
                </span>
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
                <td colSpan={7} className="px-6 py-12 text-center">
                  <Database className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-500">No datasets found</p>
                  <p className="text-sm text-gray-400 mt-1">Try adjusting your filters</p>
                </td>
              </tr>
            ) : (
              filteredDatasets.map((dataset) => {
                const labeledBadge = dataset.labeled
                  ? { color: 'bg-green-100 text-green-800', text: 'Labeled' }
                  : { color: 'bg-gray-100 text-gray-600', text: 'Unlabeled' }

                const avatarInitials = getAvatarInitials(dataset.owner_name, dataset.owner_email)
                const avatarColorStyle = getAvatarColorStyle(dataset.owner_badge_color)
                const ownerTooltip = dataset.owner_name
                  ? `${dataset.owner_name} (${dataset.owner_email})`
                  : dataset.owner_email || 'Unknown'

                return (
                  <tr key={dataset.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4">
                      <div>
                        <div className="font-medium text-gray-900">{dataset.name}</div>
                        <div className="text-xs text-gray-500 mt-0.5 font-mono">{dataset.id}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={cn('px-2 py-1 rounded-full text-xs font-medium', labeledBadge.color)}>
                        {labeledBadge.text}
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
                      <div className="flex items-center gap-2">
                        {dataset.visibility === 'public' ? (
                          <Globe className="w-4 h-4 text-green-600" />
                        ) : (
                          <Lock className="w-4 h-4 text-gray-600" />
                        )}
                        <span className="text-sm capitalize">{dataset.visibility || 'private'}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {dataset.owner_name || dataset.owner_email ? (
                        <div
                          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold text-white cursor-pointer hover:ring-2 hover:ring-offset-2 transition-all"
                          style={avatarColorStyle}
                          title={ownerTooltip}
                        >
                          {avatarInitials}
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
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
