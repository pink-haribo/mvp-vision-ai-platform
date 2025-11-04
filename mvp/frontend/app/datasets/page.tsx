'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/Sidebar'
import { DatasetPanel } from '@/components/datasets'
import { Dataset } from '@/types/dataset'
import { ArrowLeft } from 'lucide-react'

export default function DatasetsPage() {
  const router = useRouter()
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    console.log('Selected dataset:', dataset)
    // You can navigate to training configuration or auto-fill chat here
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
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-sm text-gray-600 mt-1">
            Browse and select datasets for training your models
          </p>
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-auto p-6">
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
                  <span className="text-gray-600">Status:</span>
                  <p className="font-medium text-gray-900">
                    {selectedDataset.labeled ? 'Labeled' : 'Unlabeled'}
                  </p>
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
    </div>
  )
}
