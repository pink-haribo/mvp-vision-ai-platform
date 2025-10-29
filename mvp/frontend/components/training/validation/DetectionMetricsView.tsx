/**
 * DetectionMetricsView Component
 *
 * Displays object detection validation metrics in a clean, compact layout.
 * Shows mAP, precision, recall, and loss metrics.
 */

import React, { useState } from 'react';
import { SlidePanel } from '@/components/SlidePanel';
import { DetectionImageViewer } from './DetectionImageViewer';

interface ValidationResult {
  id: number;
  job_id: number;
  epoch: number;
  task_type: string;
  primary_metric_name: string | null;
  primary_metric_value: number | null;
  overall_loss: number | null;
  metrics: any;
  per_class_metrics: any;
  confusion_matrix: number[][] | null;
  pr_curves: any;
  class_names: string[] | null;
  visualization_data: any;
  created_at: string;
}

interface DetectionMetricsViewProps {
  validationResult: ValidationResult;
  jobId: number;
}

export const DetectionMetricsView: React.FC<DetectionMetricsViewProps> = ({
  validationResult,
  jobId
}) => {
  const metrics = validationResult.metrics || {};
  const perClassMetrics = validationResult.per_class_metrics || {};
  const classNames = validationResult.class_names || {};

  // Extract detection metrics
  const mAP50 = metrics['mAP@0.5'] || 0;
  const mAP50_95 = metrics['mAP@0.5:0.95'] || 0;
  const precision = metrics.precision || 0;
  const recall = metrics.recall || 0;

  // State for per-class table sorting
  const [sortKey, setSortKey] = useState<'class' | 'ap_50' | 'ap_50_95' | 'precision' | 'recall' | 'support'>('ap_50_95');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // State for slide panel
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [selectedClass, setSelectedClass] = useState<{ id: number; name: string } | null>(null);

  // State for bbox visibility toggles
  const [showTrueBoxes, setShowTrueBoxes] = useState(true);
  const [showPredictedBoxes, setShowPredictedBoxes] = useState(true);

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Prepare per-class data for table
  const perClassData = Object.entries(perClassMetrics).map(([className, data]: [string, any]) => ({
    class: className,
    ap_50: data.ap_50 || 0,
    ap_50_95: data.ap_50_95 || 0,
    precision: data.precision || 0,
    recall: data.recall || 0,
    support: data.support || 0,
  }));

  // Get class_id from class_name (index in class_names array)
  const getClassId = (className: string): number => {
    // class_names should be an array, but handle both array and object cases
    if (Array.isArray(classNames)) {
      return classNames.indexOf(className);
    } else if (typeof classNames === 'object') {
      // If it's stored as object with indices as keys
      const entries = Object.entries(classNames);
      const entry = entries.find(([_, name]) => name === className);
      return entry ? parseInt(entry[0]) : -1;
    }
    return -1;
  };

  const handleRowClick = (className: string) => {
    const classId = getClassId(className);
    if (classId >= 0) {
      setSelectedClass({ id: classId, name: className });
      setIsPanelOpen(true);
    }
  };

  // Sort per-class data
  const sortedPerClassData = [...perClassData].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDirection === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }

    return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
  });

  const handleSort = (key: 'class' | 'ap_50' | 'ap_50_95' | 'precision' | 'recall' | 'support') => {
    if (sortKey === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  const SortIcon = ({ column }: { column: 'class' | 'ap_50' | 'ap_50_95' | 'precision' | 'recall' | 'support' }) => {
    if (sortKey !== column) return null;
    return (
      <span className="ml-1">
        {sortDirection === 'asc' ? '↑' : '↓'}
      </span>
    );
  };

  return (
    <div className="space-y-3">
      {/* Overall Metrics Card */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-2 border-b border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900">Overall Metrics</h4>
        </div>
        <div className="p-4">
          {/* Single Row - All Metrics */}
          <div className="grid grid-cols-5 gap-4">
            <div>
              <div className="text-xs text-gray-600 mb-1">mAP@0.5</div>
              <div className="text-base font-semibold text-gray-900">
                {formatPercent(mAP50)}
              </div>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <div className="text-xs text-gray-600 mb-1">mAP@0.5:0.95</div>
              <div className="text-base font-semibold text-gray-900">
                {formatPercent(mAP50_95)}
              </div>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <div className="text-xs text-gray-600 mb-1">Precision</div>
              <div className="text-base font-semibold text-gray-900">
                {formatPercent(precision)}
              </div>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <div className="text-xs text-gray-600 mb-1">Recall</div>
              <div className="text-base font-semibold text-gray-900">
                {formatPercent(recall)}
              </div>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <div className="text-xs text-gray-600 mb-1">Val Loss</div>
              <div className="text-base font-semibold text-gray-900">
                {validationResult.overall_loss?.toFixed(4) || 'N/A'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Per-Class Metrics Table */}
      {sortedPerClassData.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-4 py-2 border-b border-gray-200">
            <h4 className="text-sm font-semibold text-gray-900">Per-Class AP</h4>
          </div>
          <div className="overflow-x-auto">
            <div className="max-h-80 overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th
                      className="px-3 py-2 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('class')}
                    >
                      Class <SortIcon column="class" />
                    </th>
                    <th
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('ap_50')}
                    >
                      AP@0.5 <SortIcon column="ap_50" />
                    </th>
                    <th
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('ap_50_95')}
                    >
                      AP@0.5:0.95 <SortIcon column="ap_50_95" />
                    </th>
                    <th
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('precision')}
                    >
                      Precision <SortIcon column="precision" />
                    </th>
                    <th
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('recall')}
                    >
                      Recall <SortIcon column="recall" />
                    </th>
                    <th
                      className="px-3 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('support')}
                    >
                      N <SortIcon column="support" />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPerClassData.map((row, idx) => (
                    <tr
                      key={row.class}
                      onClick={() => handleRowClick(row.class)}
                      className={`cursor-pointer transition-colors ${
                        idx % 2 === 0 ? 'bg-white hover:bg-gray-100' : 'bg-gray-50 hover:bg-gray-100'
                      }`}
                    >
                      <td className="px-3 py-2 text-gray-900 font-medium">
                        {row.class}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-900">
                        {formatPercent(row.ap_50)}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-900">
                        {formatPercent(row.ap_50_95)}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-900">
                        {formatPercent(row.precision)}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-900">
                        {formatPercent(row.recall)}
                      </td>
                      <td className="px-3 py-2 text-right text-gray-600">
                        {row.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Info Note */}
      <div className="bg-blue-50 rounded border border-blue-200 p-3">
        <div className="flex items-start gap-2">
          <svg className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <div className="text-xs text-blue-800">
            <div className="font-semibold mb-1">Detection Metrics</div>
            <div>
              <strong>mAP@0.5</strong>: Mean Average Precision at IoU threshold 0.5<br/>
              <strong>mAP@0.5:0.95</strong>: Mean AP averaged across IoU thresholds from 0.5 to 0.95
            </div>
          </div>
        </div>
      </div>

      {/* Slide Panel for Image Viewer */}
      {selectedClass && (
        <SlidePanel
          isOpen={isPanelOpen}
          onClose={() => setIsPanelOpen(false)}
          title={`클래스: ${selectedClass.name}`}
          width="lg"
        >
          {/* Bbox Toggle Controls */}
          <div className="border-b border-gray-200 bg-gray-50 px-6 py-3">
            <div className="flex items-center gap-6">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTrueBoxes}
                  onChange={(e) => setShowTrueBoxes(e.target.checked)}
                  className="w-4 h-4 text-green-600 border-gray-300 rounded focus:ring-green-500"
                />
                <span className="text-sm text-gray-700">True Boxes</span>
                <div className="w-4 h-4 border-2 border-green-500"></div>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showPredictedBoxes}
                  onChange={(e) => setShowPredictedBoxes(e.target.checked)}
                  className="w-4 h-4 text-red-600 border-gray-300 rounded focus:ring-red-500"
                />
                <span className="text-sm text-gray-700">Predicted Boxes</span>
                <div className="w-4 h-4 border-2 border-red-500"></div>
              </label>
            </div>
          </div>

          {/* Detection Image Viewer */}
          <DetectionImageViewer
            jobId={jobId}
            epoch={validationResult.epoch}
            classId={selectedClass.id}
            className={selectedClass.name}
            showTrueBoxes={showTrueBoxes}
            showPredictedBoxes={showPredictedBoxes}
            onClose={() => setIsPanelOpen(false)}
          />
        </SlidePanel>
      )}
    </div>
  );
};
