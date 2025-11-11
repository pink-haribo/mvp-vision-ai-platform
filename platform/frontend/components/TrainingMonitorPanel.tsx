/**
 * Training Monitor Panel Component
 *
 * Displays real-time training status, metrics, and logs using WebSocket
 */

'use client';

import { useState, useEffect } from 'react';
import { useTrainingMonitor, TrainingMetrics, TrainingLog } from '@/hooks/useTrainingMonitor';

export interface TrainingMonitorPanelProps {
  jobId: number;
  sessionId?: number;
}

export function TrainingMonitorPanel({ jobId, sessionId }: TrainingMonitorPanelProps) {
  const [status, setStatus] = useState<string>('pending');
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [logs, setLogs] = useState<TrainingLog[]>([]);

  const { isConnected, lastMessage } = useTrainingMonitor({
    jobId,
    sessionId,
    autoConnect: true,
    onStatusChange: (jobId, oldStatus, newStatus) => {
      console.log(`[Training ${jobId}] Status: ${oldStatus} → ${newStatus}`);
      setStatus(newStatus);
    },
    onMetrics: (jobId, newMetrics) => {
      console.log(`[Training ${jobId}] Metrics:`, newMetrics);
      setMetrics((prev) => [...prev, newMetrics].slice(-50)); // Keep last 50
    },
    onLog: (jobId, log) => {
      console.log(`[Training ${jobId}] Log:`, log);
      setLogs((prev) => [...prev, log].slice(-100)); // Keep last 100
    },
  });

  // Get latest metrics
  const latestMetrics = metrics[metrics.length - 1];

  return (
    <div className="training-monitor-panel space-y-4">
      {/* Connection Status */}
      <div className="flex items-center justify-between p-4 bg-gray-100 rounded-lg">
        <div className="flex items-center space-x-2">
          <div
            className={`w-3 h-3 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span className="text-sm font-medium">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="text-sm text-gray-600">Job ID: {jobId}</div>
      </div>

      {/* Status Card */}
      <div className="p-4 bg-white border rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Training Status</h3>
        <div className="flex items-center space-x-4">
          <StatusBadge status={status} />
          {latestMetrics && (
            <div className="text-sm text-gray-600">
              Epoch {latestMetrics.epoch} / {latestMetrics.total_epochs || '?'}
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {latestMetrics && latestMetrics.total_epochs && (
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{
                  width: `${(latestMetrics.epoch / latestMetrics.total_epochs) * 100}%`,
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Metrics Cards */}
      {latestMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Loss" value={latestMetrics.loss.toFixed(4)} />
          {latestMetrics.accuracy !== undefined && (
            <MetricCard
              label="Accuracy"
              value={`${(latestMetrics.accuracy * 100).toFixed(2)}%`}
            />
          )}
          {latestMetrics.val_loss !== undefined && (
            <MetricCard label="Val Loss" value={latestMetrics.val_loss.toFixed(4)} />
          )}
          {latestMetrics.val_accuracy !== undefined && (
            <MetricCard
              label="Val Accuracy"
              value={`${(latestMetrics.val_accuracy * 100).toFixed(2)}%`}
            />
          )}
        </div>
      )}

      {/* Metrics Chart */}
      {metrics.length > 0 && (
        <div className="p-4 bg-white border rounded-lg shadow-sm">
          <h3 className="text-lg font-semibold mb-2">Training Metrics</h3>
          <MetricsChart metrics={metrics} />
        </div>
      )}

      {/* Logs Panel */}
      <div className="p-4 bg-white border rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Training Logs</h3>
        <div className="h-64 overflow-y-auto bg-gray-900 text-gray-100 p-3 rounded font-mono text-sm">
          {logs.length === 0 ? (
            <div className="text-gray-500">No logs yet...</div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="mb-1">
                <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
                <span
                  className={
                    log.level === 'ERROR'
                      ? 'text-red-400'
                      : log.level === 'WARNING'
                      ? 'text-yellow-400'
                      : 'text-gray-100'
                  }
                >
                  [{log.level}]
                </span>{' '}
                {log.message}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// Status Badge Component
function StatusBadge({ status }: { status: string }) {
  const statusColors = {
    pending: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    cancelled: 'bg-gray-100 text-gray-800',
  };

  const colorClass = statusColors[status as keyof typeof statusColors] || 'bg-gray-100 text-gray-800';

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${colorClass}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

// Metric Card Component
function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-4 bg-white border rounded-lg shadow-sm">
      <div className="text-sm text-gray-600 mb-1">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

// Metrics Chart Component (simplified)
function MetricsChart({ metrics }: { metrics: TrainingMetrics[] }) {
  const maxLoss = Math.max(...metrics.map((m) => m.loss));
  const minLoss = Math.min(...metrics.map((m) => m.loss));

  return (
    <div className="h-48 flex items-end space-x-1">
      {metrics.map((metric, index) => {
        const heightPercent = ((metric.loss - minLoss) / (maxLoss - minLoss || 1)) * 100;
        return (
          <div
            key={index}
            className="flex-1 bg-blue-500 rounded-t"
            style={{ height: `${100 - heightPercent}%` }}
            title={`Epoch ${metric.epoch}: Loss ${metric.loss.toFixed(4)}`}
          />
        );
      })}
    </div>
  );
}
