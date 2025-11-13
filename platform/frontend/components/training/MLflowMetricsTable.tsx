"use client";

import React, { useEffect, useState } from "react";

interface MetricDataPoint {
  step: number;
  value: number;
  timestamp: number;
}

interface MLflowMetricsData {
  found: boolean;
  run_id: string | null;
  metrics: {
    train_loss?: MetricDataPoint[];
    val_loss?: MetricDataPoint[];
    train_accuracy?: MetricDataPoint[];
    val_accuracy?: MetricDataPoint[];
  };
}

interface MLflowMetricsTableProps {
  jobId: number | string;
}

export default function MLflowMetricsTable({
  jobId,
}: MLflowMetricsTableProps) {
  const [metricsData, setMetricsData] = useState<MLflowMetricsData | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${jobId}/mlflow/metrics`
        );
        if (response.ok) {
          const data = await response.json();
          setMetricsData(data);
        }
      } catch (error) {
        console.error("Error fetching MLflow metrics:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchMetrics();

    // Poll every 5 seconds for updates
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [jobId]);

  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="h-64 bg-gray-200 rounded-lg"></div>
      </div>
    );
  }

  if (!metricsData || !metricsData.found) {
    return (
      <div className="p-6 bg-amber-50 rounded-lg border border-amber-200 text-center">
        <p className="text-sm text-amber-800">
          학습 데이터를 찾을 수 없습니다.
        </p>
      </div>
    );
  }

  const { metrics } = metricsData;

  // Combine metrics by epoch
  const epochs: {
    epoch: number;
    train_loss?: number;
    train_accuracy?: number;
    val_loss?: number;
    val_accuracy?: number;
  }[] = [];

  const maxEpoch = Math.max(
    ...(metrics.train_loss?.map((m) => m.step) || [0]),
    ...(metrics.val_loss?.map((m) => m.step) || [0]),
    ...(metrics.train_accuracy?.map((m) => m.step) || [0]),
    ...(metrics.val_accuracy?.map((m) => m.step) || [0])
  );

  for (let epoch = 1; epoch <= maxEpoch; epoch++) {
    epochs.push({
      epoch,
      train_loss: metrics.train_loss?.find((m) => m.step === epoch)?.value,
      train_accuracy: metrics.train_accuracy?.find((m) => m.step === epoch)
        ?.value,
      val_loss: metrics.val_loss?.find((m) => m.step === epoch)?.value,
      val_accuracy: metrics.val_accuracy?.find((m) => m.step === epoch)?.value,
    });
  }

  // Show last 10 epochs
  const recentEpochs = epochs.slice(-10).reverse();

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900">
          학습 메트릭 (최근 {recentEpochs.length} Epochs)
        </h4>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-2 text-left font-semibold text-gray-700">
                Epoch
              </th>
              <th className="px-4 py-2 text-right font-semibold text-gray-700">
                Train Loss
              </th>
              <th className="px-4 py-2 text-right font-semibold text-gray-700">
                Train Acc
              </th>
              <th className="px-4 py-2 text-right font-semibold text-gray-700">
                Val Loss
              </th>
              <th className="px-4 py-2 text-right font-semibold text-gray-700">
                Val Acc
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {recentEpochs.map((row, index) => (
              <tr
                key={row.epoch}
                className={index === 0 ? "bg-violet-50" : "hover:bg-gray-50"}
              >
                <td className="px-4 py-2 font-medium text-gray-900">
                  {row.epoch}
                  {index === 0 && (
                    <span className="ml-2 text-xs text-violet-600 font-semibold">
                      Latest
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-right text-gray-700 font-mono">
                  {row.train_loss !== undefined
                    ? row.train_loss.toFixed(4)
                    : "-"}
                </td>
                <td className="px-4 py-2 text-right text-gray-700 font-mono">
                  {row.train_accuracy !== undefined
                    ? `${row.train_accuracy.toFixed(2)}%`
                    : "-"}
                </td>
                <td className="px-4 py-2 text-right text-gray-700 font-mono">
                  {row.val_loss !== undefined ? row.val_loss.toFixed(4) : "-"}
                </td>
                <td className="px-4 py-2 text-right text-gray-700 font-mono">
                  {row.val_accuracy !== undefined
                    ? `${row.val_accuracy.toFixed(2)}%`
                    : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {epochs.length > 10 && (
        <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-center">
          <p className="text-xs text-gray-500">
            총 {epochs.length} epochs (최근 10개만 표시)
          </p>
        </div>
      )}
    </div>
  );
}
