/**
 * ClassificationMetricsView Component
 *
 * Displays classification validation metrics including:
 * - Overall metrics (accuracy, precision, recall, f1)
 * - Confusion matrix heatmap
 * - Per-class metrics table
 */

import React from 'react';
import { ConfusionMatrixView } from './ConfusionMatrixView';
import { PerClassMetricsTable } from './PerClassMetricsTable';

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

interface ClassificationMetricsViewProps {
  validationResult: ValidationResult;
  jobId: number;
  onConfusionMatrixCellClick?: (trueLabelId: number, predictedLabelId: number, trueLabel: string, predictedLabel: string) => void;
}

export const ClassificationMetricsView: React.FC<ClassificationMetricsViewProps> = ({
  validationResult,
  jobId,
  onConfusionMatrixCellClick
}) => {
  const { metrics, per_class_metrics, confusion_matrix, class_names, overall_loss } = validationResult;

  // Format percentage
  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format float
  const formatFloat = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(4);
  };

  return (
    <div className="space-y-4">
      {/* Overall Metrics - 2-Row Format */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
          <h4 className="text-xs font-semibold text-gray-900">Overall Metrics</h4>
        </div>
        <div className="px-4 py-3 flex items-start gap-6 text-xs overflow-x-auto">
          <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-gray-600">Accuracy</span>
            <span className="font-semibold text-violet-600 text-sm">{formatPercent(metrics?.accuracy)}</span>
          </div>
          <div className="w-px h-10 bg-gray-300 self-center" />
          <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-gray-600">Precision</span>
            <span className="font-semibold text-gray-900 text-sm">{formatPercent(metrics?.precision)}</span>
          </div>
          <div className="w-px h-10 bg-gray-300 self-center" />
          <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-gray-600">Recall</span>
            <span className="font-semibold text-gray-900 text-sm">{formatPercent(metrics?.recall)}</span>
          </div>
          <div className="w-px h-10 bg-gray-300 self-center" />
          <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-gray-600">F1 Score</span>
            <span className="font-semibold text-gray-900 text-sm">{formatPercent(metrics?.f1_score)}</span>
          </div>
          <div className="w-px h-10 bg-gray-300 self-center" />
          <div className="flex flex-col items-center gap-1 min-w-[60px]">
            <span className="text-gray-600">Val Loss</span>
            <span className="font-semibold text-gray-900 text-sm">{formatFloat(overall_loss)}</span>
          </div>
          {metrics?.top5_accuracy && (
            <>
              <div className="w-px h-10 bg-gray-300 self-center" />
              <div className="flex flex-col items-center gap-1 min-w-[60px]">
                <span className="text-gray-600">Top-5 Acc</span>
                <span className="font-semibold text-gray-900 text-sm">{formatPercent(metrics.top5_accuracy)}</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Confusion Matrix and Per-Class Metrics - Stacked (2 rows, 1 column) */}
      <div className="space-y-4">
        {/* Confusion Matrix */}
        {confusion_matrix && class_names && (
          <ConfusionMatrixView
            confusionMatrix={confusion_matrix}
            classNames={class_names}
            onCellClick={onConfusionMatrixCellClick}
          />
        )}

        {/* Per-Class Metrics */}
        {per_class_metrics && (
          <PerClassMetricsTable
            perClassMetrics={per_class_metrics}
          />
        )}
      </div>
    </div>
  );
};
