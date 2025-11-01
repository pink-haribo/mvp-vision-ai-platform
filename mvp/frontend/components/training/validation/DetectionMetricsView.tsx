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

  // Debug logging
  console.log('[DetectionMetricsView] validationResult:', validationResult);
  console.log('[DetectionMetricsView] task_type:', validationResult.task_type);
  console.log('[DetectionMetricsView] metrics keys:', Object.keys(metrics));
  console.log('[DetectionMetricsView] metrics:', metrics);
  console.log('[DetectionMetricsView] perClassMetrics:', perClassMetrics);

  // Detect if this is a segmentation task
  const isSegmentation = validationResult.task_type === 'instance_segmentation' ||
                         validationResult.task_type === 'semantic_segmentation';

  // Check if we have separate box and mask metrics (sanitized format: precision_B_, precision_M_)
  const hasBoxMaskMetrics = isSegmentation && (
    'precision_B_' in metrics || 'precision_M_' in metrics ||
    'precision(B)' in metrics || 'precision(M)' in metrics
  );

  console.log('[DetectionMetricsView] isSegmentation:', isSegmentation);
  console.log('[DetectionMetricsView] hasBoxMaskMetrics:', hasBoxMaskMetrics);

  // Extract detection metrics - handle both box and mask versions
  const mAP50 = metrics['mAP@0.5'] || metrics['mAP50_B_'] || metrics['mAP50(B)'] || 0;
  const mAP50_95 = metrics['mAP@0.5:0.95'] || metrics['mAP50-95_B_'] || metrics['mAP50-95(B)'] || 0;
  const precision = metrics.precision || metrics['precision_B_'] || metrics['precision(B)'] || 0;
  const recall = metrics.recall || metrics['recall_B_'] || metrics['recall(B)'] || 0;

  // Mask metrics (for segmentation)
  const mAP50_mask = metrics['mAP50_M_'] || metrics['mAP50(M)'] || 0;
  const mAP50_95_mask = metrics['mAP50-95_M_'] || metrics['mAP50-95(M)'] || 0;
  const precision_mask = metrics['precision_M_'] || metrics['precision(M)'] || 0;
  const recall_mask = metrics['recall_M_'] || metrics['recall(M)'] || 0;

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
    // Box metrics (default or explicit box metrics)
    ap_50: data.ap_50 || data['ap_50_B_'] || data['ap_50(B)'] || 0,
    ap_50_95: data.ap_50_95 || data['ap_50_95_B_'] || data['ap_50_95(B)'] || 0,
    precision: data.precision || data['precision_B_'] || data['precision(B)'] || 0,
    recall: data.recall || data['recall_B_'] || data['recall(B)'] || 0,
    // Mask metrics (for segmentation)
    ap_50_mask: data['ap_50_M_'] || data['ap_50(M)'] || 0,
    ap_50_95_mask: data['ap_50_95_M_'] || data['ap_50_95(M)'] || 0,
    precision_mask: data['precision_M_'] || data['precision(M)'] || 0,
    recall_mask: data['recall_M_'] || data['recall(M)'] || 0,
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
      {/* Task Type Badge (for segmentation) */}
      {isSegmentation && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200 p-3">
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 bg-purple-500 text-white text-xs font-semibold rounded-full">
              🎭 Instance Segmentation
            </span>
            <span className="text-xs text-gray-700">
              {hasBoxMaskMetrics
                ? 'Box와 Mask 메트릭을 모두 표시합니다'
                : 'Segmentation 메트릭'}
            </span>
          </div>
        </div>
      )}

      {/* Overall Metrics Card */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-2 border-b border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900">Overall Metrics</h4>
        </div>
        <div className="p-4">
          {hasBoxMaskMetrics ? (
            // Segmentation: Show both Box and Mask metrics
            <div className="space-y-4">
              {/* Box Metrics Row */}
              <div>
                <div className="text-xs font-semibold text-blue-700 mb-2 flex items-center gap-1">
                  <span>📦</span> Bounding Box Metrics
                </div>
                <div className="grid grid-cols-4 gap-4">
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
                </div>
              </div>

              {/* Mask Metrics Row */}
              <div className="pt-3 border-t border-gray-200">
                <div className="text-xs font-semibold text-purple-700 mb-2 flex items-center gap-1">
                  <span>🎭</span> Segmentation Mask Metrics
                </div>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <div className="text-xs text-gray-600 mb-1">mAP@0.5</div>
                    <div className="text-base font-semibold text-gray-900">
                      {formatPercent(mAP50_mask)}
                    </div>
                  </div>
                  <div className="border-l border-gray-200 pl-4">
                    <div className="text-xs text-gray-600 mb-1">mAP@0.5:0.95</div>
                    <div className="text-base font-semibold text-gray-900">
                      {formatPercent(mAP50_95_mask)}
                    </div>
                  </div>
                  <div className="border-l border-gray-200 pl-4">
                    <div className="text-xs text-gray-600 mb-1">Precision</div>
                    <div className="text-base font-semibold text-gray-900">
                      {formatPercent(precision_mask)}
                    </div>
                  </div>
                  <div className="border-l border-gray-200 pl-4">
                    <div className="text-xs text-gray-600 mb-1">Recall</div>
                    <div className="text-base font-semibold text-gray-900">
                      {formatPercent(recall_mask)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Val Loss */}
              <div className="pt-3 border-t border-gray-200">
                <div className="text-xs text-gray-600 mb-1">Validation Loss</div>
                <div className="text-base font-semibold text-gray-900">
                  {validationResult.overall_loss?.toFixed(4) || 'N/A'}
                </div>
              </div>
            </div>
          ) : (
            // Detection or Classification: Single row
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
          )}
        </div>
      </div>

      {/* Per-Class Metrics Table */}
      {sortedPerClassData.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-4 py-2 border-b border-gray-200">
            <h4 className="text-sm font-semibold text-gray-900">
              {hasBoxMaskMetrics ? 'Per-Class Metrics (Box & Mask)' : 'Per-Class AP'}
            </h4>
          </div>
          <div className="overflow-x-auto">
            <div className="max-h-80 overflow-y-auto">
              <table className="w-full text-xs">
                {hasBoxMaskMetrics ? (
                  // Segmentation: Show separate Box and Mask columns
                  <>
                    <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
                      <tr>
                        <th
                          className="px-2 py-2 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('class')}
                          rowSpan={2}
                        >
                          Class <SortIcon column="class" />
                        </th>
                        <th
                          className="px-2 py-1 text-center font-semibold text-blue-700 border-l border-gray-300"
                          colSpan={2}
                        >
                          📦 Box
                        </th>
                        <th
                          className="px-2 py-1 text-center font-semibold text-purple-700 border-l border-gray-300"
                          colSpan={2}
                        >
                          🎭 Mask
                        </th>
                        <th
                          className="px-2 py-2 text-right font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 border-l border-gray-300"
                          onClick={() => handleSort('support')}
                          rowSpan={2}
                        >
                          N <SortIcon column="support" />
                        </th>
                      </tr>
                      <tr>
                        <th
                          className="px-2 py-1 text-right text-[10px] font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 border-l border-gray-300"
                          onClick={() => handleSort('precision')}
                        >
                          Prec <SortIcon column="precision" />
                        </th>
                        <th
                          className="px-2 py-1 text-right text-[10px] font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('recall')}
                        >
                          Rec <SortIcon column="recall" />
                        </th>
                        <th className="px-2 py-1 text-right text-[10px] font-semibold text-gray-600 border-l border-gray-300">
                          Prec
                        </th>
                        <th className="px-2 py-1 text-right text-[10px] font-semibold text-gray-600">
                          Rec
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
                          <td className="px-2 py-2 text-gray-900 font-medium">
                            {row.class}
                          </td>
                          <td className="px-2 py-2 text-right text-gray-900 font-mono text-[10px] border-l border-gray-200">
                            {formatPercent(row.precision)}
                          </td>
                          <td className="px-2 py-2 text-right text-gray-900 font-mono text-[10px]">
                            {formatPercent(row.recall)}
                          </td>
                          <td className="px-2 py-2 text-right text-purple-900 font-mono text-[10px] border-l border-gray-200">
                            {formatPercent(row.precision_mask)}
                          </td>
                          <td className="px-2 py-2 text-right text-purple-900 font-mono text-[10px]">
                            {formatPercent(row.recall_mask)}
                          </td>
                          <td className="px-2 py-2 text-right text-gray-600 font-mono text-[10px] border-l border-gray-200">
                            {row.support}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </>
                ) : (
                  // Detection: Original single-column layout
                  <>
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
                  </>
                )}
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Info Note */}
      <div className={`rounded border p-3 ${
        hasBoxMaskMetrics ? 'bg-purple-50 border-purple-200' : 'bg-blue-50 border-blue-200'
      }`}>
        <div className="flex items-start gap-2">
          <svg className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
            hasBoxMaskMetrics ? 'text-purple-600' : 'text-blue-600'
          }`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <div className={`text-xs ${hasBoxMaskMetrics ? 'text-purple-800' : 'text-blue-800'}`}>
            <div className="font-semibold mb-1">
              {hasBoxMaskMetrics ? 'Segmentation Metrics' : 'Detection Metrics'}
            </div>
            <div>
              {hasBoxMaskMetrics ? (
                <>
                  <strong>Box Metrics</strong>: Bounding box의 정확도를 측정합니다 (IoU 기반)<br/>
                  <strong>Mask Metrics</strong>: Segmentation mask의 픽셀 단위 정확도를 측정합니다<br/>
                  <strong>mAP</strong>: Mean Average Precision (IoU threshold별 평균)
                </>
              ) : (
                <>
                  <strong>mAP@0.5</strong>: Mean Average Precision at IoU threshold 0.5<br/>
                  <strong>mAP@0.5:0.95</strong>: Mean AP averaged across IoU thresholds from 0.5 to 0.95
                </>
              )}
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
