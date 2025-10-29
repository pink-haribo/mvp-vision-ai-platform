/**
 * TaskSpecificVisualization Component
 *
 * Auto-router that renders appropriate visualization based on task type.
 * This enables task-agnostic validation system.
 */

import React from 'react';
import { ClassificationMetricsView } from './ClassificationMetricsView';
import { DetectionMetricsView } from './DetectionMetricsView';

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

interface TaskSpecificVisualizationProps {
  taskType: string;
  validationResult: ValidationResult;
  jobId: number;
  onConfusionMatrixCellClick?: (trueLabelId: number, predictedLabelId: number, trueLabel: string, predictedLabel: string) => void;
}

export const TaskSpecificVisualization: React.FC<TaskSpecificVisualizationProps> = ({
  taskType,
  validationResult,
  jobId,
  onConfusionMatrixCellClick
}) => {
  // Route to appropriate visualization based on task type
  switch (taskType) {
    case 'image_classification':
      return (
        <ClassificationMetricsView
          validationResult={validationResult}
          jobId={jobId}
          onConfusionMatrixCellClick={onConfusionMatrixCellClick}
        />
      );

    case 'object_detection':
      return (
        <DetectionMetricsView
          validationResult={validationResult}
          jobId={jobId}
        />
      );

    case 'instance_segmentation':
    case 'semantic_segmentation':
      return (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Segmentation Metrics
          </h3>
          <div className="text-gray-400">
            Segmentation metrics visualization coming soon.
          </div>
          {/* TODO: Implement SegmentationMetricsView */}
          <div className="mt-4 space-y-2">
            <div className="text-sm">
              <span className="text-gray-400">Mean IoU:</span>{' '}
              <span className="text-white">{validationResult.metrics?.mean_iou || 'N/A'}</span>
            </div>
            <div className="text-sm">
              <span className="text-gray-400">Pixel Accuracy:</span>{' '}
              <span className="text-white">{validationResult.metrics?.pixel_accuracy || 'N/A'}</span>
            </div>
          </div>
        </div>
      );

    case 'pose_estimation':
      return (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Pose Estimation Metrics
          </h3>
          <div className="text-gray-400">
            Pose estimation metrics visualization coming soon.
          </div>
          {/* TODO: Implement PoseMetricsView */}
          <div className="mt-4 space-y-2">
            <div className="text-sm">
              <span className="text-gray-400">OKS:</span>{' '}
              <span className="text-white">{validationResult.metrics?.OKS || 'N/A'}</span>
            </div>
            <div className="text-sm">
              <span className="text-gray-400">PCK:</span>{' '}
              <span className="text-white">{validationResult.metrics?.PCK || 'N/A'}</span>
            </div>
          </div>
        </div>
      );

    default:
      return (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Generic Metrics
          </h3>
          <div className="text-gray-400">
            Unsupported task type: {taskType}
          </div>
          <div className="mt-4">
            <pre className="text-xs text-gray-300 bg-gray-700 p-4 rounded overflow-auto">
              {JSON.stringify(validationResult.metrics, null, 2)}
            </pre>
          </div>
        </div>
      );
  }
};
