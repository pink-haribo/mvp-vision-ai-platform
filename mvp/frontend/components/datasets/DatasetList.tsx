import React, { useState, useEffect } from 'react';
import { Dataset } from '@/types/dataset';
import DatasetCard from './DatasetCard';
import CreateDatasetModal from './CreateDatasetModal';

interface DatasetListProps {
  onSelectDataset?: (dataset: Dataset) => void;
  selectedDatasetId?: string | null;
  labeledFilter?: string | null;  // 'all', 'labeled', 'unlabeled'
}

export default function DatasetList({
  onSelectDataset,
  selectedDatasetId,
  labeledFilter,
}: DatasetListProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLabeledFilter, setSelectedLabeledFilter] = useState<string>(labeledFilter || 'all');
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Fetch datasets from API
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setLoading(true);
        setError(null);

        const params = new URLSearchParams();
        if (selectedLabeledFilter && selectedLabeledFilter !== 'all') {
          params.append('labeled', selectedLabeledFilter === 'labeled' ? 'true' : 'false');
        }

        const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
        const url = `${baseUrl}/datasets/available${params.toString() ? `?${params}` : ''}`;
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`Failed to fetch datasets: ${response.statusText}`);
        }

        const data = await response.json();
        setDatasets(data);
      } catch (err) {
        console.error('Error fetching datasets:', err);
        setError(err instanceof Error ? err.message : 'Failed to load datasets');
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [selectedLabeledFilter]);

  // Update selected labeled filter from prop
  useEffect(() => {
    if (labeledFilter) {
      setSelectedLabeledFilter(labeledFilter);
    }
  }, [labeledFilter]);

  // Filter datasets by search query
  const filteredDatasets = datasets.filter((dataset) => {
    const matchesSearch =
      dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dataset.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dataset.id.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesSearch;
  });

  // Label status filters
  const labelFilters = [
    { value: 'all', label: 'All Datasets' },
    { value: 'labeled', label: 'Labeled (With Annotations)' },
    { value: 'unlabeled', label: 'Unlabeled (No Annotations)' },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center">
          <svg className="w-5 h-5 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    );
  }

  const handleCreateSuccess = async (datasetId: string) => {
    console.log('Dataset created successfully, ID:', datasetId);
    // Refresh dataset list
    setLoading(true);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
      const response = await fetch(`${baseUrl}/datasets/available`);
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
      }
    } catch (err) {
      console.error('Error refreshing datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Datasets</h2>
          <p className="text-gray-600 mt-1">
            Select a dataset to start training
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Create Dataset
        </button>
      </div>

      {/* Filters */}
      <div className="space-y-3">
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search datasets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
          <svg
            className="absolute left-3 top-3 w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        {/* Label Status Filter */}
        <div className="flex flex-wrap gap-2">
          {labelFilters.map((filter) => (
            <button
              key={filter.value}
              onClick={() => setSelectedLabeledFilter(filter.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedLabeledFilter === filter.value
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      {/* Dataset Grid */}
      {filteredDatasets.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
            />
          </svg>
          <p className="mt-4 text-gray-600">No datasets found</p>
          <p className="mt-1 text-sm text-gray-500">
            Try adjusting your search or filters
          </p>
        </div>
      ) : (
        <>
          <div className="text-sm text-gray-600">
            Found {filteredDatasets.length} dataset{filteredDatasets.length !== 1 ? 's' : ''}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredDatasets.map((dataset) => (
              <DatasetCard
                key={dataset.id}
                dataset={dataset}
                onSelect={onSelectDataset}
                selected={dataset.id === selectedDatasetId}
              />
            ))}
          </div>
        </>
      )}

      {/* Create Dataset Modal */}
      <CreateDatasetModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSuccess={handleCreateSuccess}
      />
    </div>
  );
}
