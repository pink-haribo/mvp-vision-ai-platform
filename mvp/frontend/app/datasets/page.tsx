'use client'

import { useState } from 'react'
import { DatasetPanel } from '@/components/datasets'
import { Dataset } from '@/types/dataset'

export default function DatasetsPage() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    console.log('Selected dataset:', dataset)
    // You can navigate to training configuration or auto-fill chat here
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-sm text-gray-600 mt-1">
            Browse and select datasets for training your models
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <DatasetPanel
          onSelectDataset={handleSelectDataset}
          selectedDatasetId={selectedDataset?.id}
        />

        {/* Selected dataset info */}
        {selectedDataset && (
          <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Selected Dataset Details
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">ID:</span>
                <p className="font-medium text-gray-900">{selectedDataset.id}</p>
              </div>
              <div>
                <span className="text-gray-600">Name:</span>
                <p className="font-medium text-gray-900">{selectedDataset.name}</p>
              </div>
              <div>
                <span className="text-gray-600">Format:</span>
                <p className="font-medium text-gray-900">{selectedDataset.format}</p>
              </div>
              <div>
                <span className="text-gray-600">Task Type:</span>
                <p className="font-medium text-gray-900">{selectedDataset.task_type}</p>
              </div>
              <div>
                <span className="text-gray-600">Images:</span>
                <p className="font-medium text-gray-900">{selectedDataset.num_items.toLocaleString()}</p>
              </div>
              <div>
                <span className="text-gray-600">Source:</span>
                <p className="font-medium text-gray-900">{selectedDataset.source}</p>
              </div>
              <div className="col-span-2">
                <span className="text-gray-600">Description:</span>
                <p className="font-medium text-gray-900">{selectedDataset.description}</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
