import React, { useState } from 'react';
import { Dataset } from '@/types/dataset';
import DatasetList from './DatasetList';

interface DatasetPanelProps {
  onSelectDataset?: (dataset: Dataset) => void;
  onClose?: () => void;
  selectedDatasetId?: string | null;
}

export default function DatasetPanel({
  onSelectDataset,
  onClose,
  selectedDatasetId,
}: DatasetPanelProps) {
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    if (onSelectDataset) {
      onSelectDataset(dataset);
    }
  };

  const handleConfirm = () => {
    if (selectedDataset && onSelectDataset) {
      onSelectDataset(selectedDataset);
    }
    if (onClose) {
      onClose();
    }
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Select Dataset</h2>
          <p className="text-sm text-gray-600 mt-1">
            Choose a dataset to configure your training
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <DatasetList
          onSelectDataset={handleSelectDataset}
          selectedDatasetId={selectedDataset?.id || selectedDatasetId}
        />
      </div>

      {/* Footer with actions */}
      {selectedDataset && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">
                Selected: {selectedDataset.name}
              </p>
              <p className="text-xs text-gray-600 mt-1">
                {selectedDataset.num_items.toLocaleString()} images • {selectedDataset.format} format
              </p>
            </div>
            <div className="flex gap-2">
              {onClose && (
                <button
                  onClick={onClose}
                  className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
              )}
              <button
                onClick={handleConfirm}
                className="px-4 py-2 text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Use This Dataset
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
