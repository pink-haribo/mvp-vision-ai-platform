/**
 * ValidationDashboard Component
 *
 * Main dashboard for displaying validation results and metrics.
 * Automatically routes to task-specific visualization based on task type.
 */

import React, { useEffect, useState } from 'react';
import { TaskSpecificVisualization } from './TaskSpecificVisualization';
import { SlidePanel } from '../../SlidePanel';
import { ImageViewer } from './ImageViewer';

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

interface ValidationSummary {
  job_id: number;
  task_type: string;
  total_epochs: number;
  best_epoch: number | null;
  best_metric_value: number | null;
  best_metric_name: string | null;
  epoch_metrics: any[];
}

interface ValidationDashboardProps {
  jobId: number;
  currentEpoch?: number;
  jobStatus?: string;
}

export const ValidationDashboard: React.FC<ValidationDashboardProps> = ({
  jobId,
  currentEpoch,
  jobStatus
}) => {
  const [summary, setSummary] = useState<ValidationSummary | null>(null);
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Slide panel state for image viewer
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{
    trueLabelId: number;
    predictedLabelId: number;
    trueLabel: string;
    predictedLabel: string;
  } | null>(null);

  // Fetch validation summary on mount
  useEffect(() => {
    // Don't fetch if job is pending
    if (jobStatus === 'pending') {
      setLoading(false);
      return;
    }

    fetchValidationSummary();
  }, [jobId, jobStatus]);

  // Fetch specific epoch when selected or currentEpoch changes
  useEffect(() => {
    if (!summary) return; // Wait for summary to load

    // Determine which epoch to fetch (priority: selectedEpoch > currentEpoch > best_epoch)
    let epochToFetch = selectedEpoch !== null ? selectedEpoch : currentEpoch;
    if (epochToFetch === null || epochToFetch === undefined) {
      epochToFetch = summary.best_epoch ?? undefined;
    }

    // Check if the epoch exists in the summary
    if (epochToFetch !== null && epochToFetch !== undefined) {
      const epochExists = summary.epoch_metrics.some((m) => m.epoch === epochToFetch);

      if (epochExists) {
        // Set selected epoch if not already set
        if (selectedEpoch === null) {
          setSelectedEpoch(epochToFetch);
        }
        fetchValidationResult(epochToFetch);
      } else {
        // Epoch doesn't exist, fallback to best_epoch
        console.warn(`Epoch ${epochToFetch} not found in validation results, using best epoch ${summary.best_epoch}`);
        if (summary.best_epoch !== null && summary.best_epoch !== epochToFetch) {
          setSelectedEpoch(summary.best_epoch);
        }
      }
    }
  }, [selectedEpoch, currentEpoch, summary]);

  const fetchValidationSummary = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/validation/jobs/${jobId}/summary`);

      if (!response.ok) {
        if (response.status === 404) {
          setError('No validation results available yet');
          return;
        }
        throw new Error(`Failed to fetch validation summary: ${response.status}`);
      }

      const data = await response.json();
      setSummary(data);

      // Don't auto-select here - let the other useEffect handle it
      // This prevents conflicts with currentEpoch prop
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch validation summary');
      console.error('Error fetching validation summary:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchValidationResult = async (epoch: number) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/validation/jobs/${jobId}/results/${epoch}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch validation result: ${response.status}`);
      }

      const data = await response.json();
      setValidationResult(data);
    } catch (err) {
      console.error('Error fetching validation result:', err);
    }
  };

  // Show message for pending status (no spinner)
  if (jobStatus === 'pending') {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <p className="text-sm text-gray-500">
            학습을 시작하면 검증 메트릭이 표시됩니다
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading validation results...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">{error}</div>
      </div>
    );
  }

  if (!summary || !validationResult) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">No validation results available</div>
      </div>
    );
  }

  const formatMetricName = (name: string) => {
    return name.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  // Handle confusion matrix cell click
  const handleCellClick = (trueLabelId: number, predictedLabelId: number, trueLabel: string, predictedLabel: string) => {
    setSelectedCell({ trueLabelId, predictedLabelId, trueLabel, predictedLabel });
    setIsPanelOpen(true);
  };

  // Handle panel close
  const handlePanelClose = () => {
    setIsPanelOpen(false);
  };

  return (
    <div className="space-y-4">
      {/* Header with Epoch Selector - Compact */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-2 border-b border-gray-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-900">검증 결과</h3>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-600">Epoch:</span>
            <select
              value={selectedEpoch || ''}
              onChange={(e) => setSelectedEpoch(Number(e.target.value))}
              className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-violet-500 focus:border-violet-500"
            >
              {summary.epoch_metrics.map((epochData) => (
                <option key={epochData.epoch} value={epochData.epoch}>
                  Epoch {epochData.epoch}{epochData.epoch === summary.best_epoch ? ' ⭐' : ''} - {summary.best_metric_name}: {epochData.primary_metric?.toFixed(4)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Summary Metrics - Inline Bar */}
        <div className="px-4 py-2 bg-gray-50 flex items-center gap-6 text-xs overflow-x-auto">
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Task:</span>
            <span className="font-medium text-gray-900">{formatMetricName(summary.task_type)}</span>
          </div>
          <div className="w-px h-4 bg-gray-300" />
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Total Epochs:</span>
            <span className="font-medium text-gray-900">{summary.total_epochs}</span>
          </div>
          <div className="w-px h-4 bg-gray-300" />
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Best Epoch:</span>
            <span className="font-medium text-violet-600">{summary.best_epoch ?? 'N/A'}</span>
          </div>
          <div className="w-px h-4 bg-gray-300" />
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Best {formatMetricName(summary.best_metric_name || '')}:</span>
            <span className="font-semibold text-violet-600">
              {summary.best_metric_value ? `${(summary.best_metric_value * 100).toFixed(2)}%` : 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Task-Specific Visualization */}
      <TaskSpecificVisualization
        taskType={validationResult.task_type}
        validationResult={validationResult}
        jobId={jobId}
        onConfusionMatrixCellClick={handleCellClick}
      />

      {/* Slide Panel for Image Viewer */}
      {selectedCell && selectedEpoch !== null && (
        <SlidePanel
          isOpen={isPanelOpen}
          onClose={handlePanelClose}
          title="Validation Images"
          width="md"
        >
          <ImageViewer
            jobId={jobId}
            epoch={selectedEpoch}
            trueLabelId={selectedCell.trueLabelId}
            predictedLabelId={selectedCell.predictedLabelId}
            trueLabel={selectedCell.trueLabel}
            predictedLabel={selectedCell.predictedLabel}
          />
        </SlidePanel>
      )}
    </div>
  );
};
