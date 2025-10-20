"use client";

import React, { useEffect, useState } from "react";
import { TrendingUp, TrendingDown } from "lucide-react";

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

interface MLflowMetricsChartsProps {
  jobId: number | string;
}

export default function MLflowMetricsCharts({
  jobId,
}: MLflowMetricsChartsProps) {
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
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-48 bg-gray-200 rounded-lg"></div>
        </div>
        <div className="animate-pulse">
          <div className="h-48 bg-gray-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (!metricsData || !metricsData.found) {
    return (
      <div className="p-6 bg-amber-50 rounded-lg border border-amber-200 text-center">
        <p className="text-sm text-amber-800">
          MLflow 데이터를 찾을 수 없습니다. 학습이 시작되면 표시됩니다.
        </p>
      </div>
    );
  }

  const { metrics } = metricsData;
  const hasLoss = metrics.train_loss && metrics.train_loss.length > 0;
  const hasAccuracy = metrics.train_accuracy && metrics.train_accuracy.length > 0;

  return (
    <div className="space-y-6">
      {/* Loss Chart */}
      {hasLoss && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-semibold text-gray-900">Loss</h4>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
                <span className="text-gray-600">Train</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-purple-500 rounded-sm"></div>
                <span className="text-gray-600">Val</span>
              </div>
            </div>
          </div>
          <SimpleLineChart
            trainData={metrics.train_loss || []}
            valData={metrics.val_loss || []}
            color1="rgb(59, 130, 246)"
            color2="rgb(168, 85, 247)"
          />
          {metrics.val_loss && metrics.val_loss.length > 0 && (
            <MetricSummary
              label="Latest Val Loss"
              value={metrics.val_loss[metrics.val_loss.length - 1].value}
              trend={
                metrics.val_loss.length > 1
                  ? metrics.val_loss[metrics.val_loss.length - 1].value <
                    metrics.val_loss[metrics.val_loss.length - 2].value
                  : null
              }
            />
          )}
        </div>
      )}

      {/* Accuracy Chart */}
      {hasAccuracy && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-sm font-semibold text-gray-900">Accuracy</h4>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-emerald-500 rounded-sm"></div>
                <span className="text-gray-600">Train</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-cyan-500 rounded-sm"></div>
                <span className="text-gray-600">Val</span>
              </div>
            </div>
          </div>
          <SimpleLineChart
            trainData={metrics.train_accuracy || []}
            valData={metrics.val_accuracy || []}
            color1="rgb(16, 185, 129)"
            color2="rgb(6, 182, 212)"
          />
          {metrics.val_accuracy && metrics.val_accuracy.length > 0 && (
            <MetricSummary
              label="Latest Val Accuracy"
              value={metrics.val_accuracy[metrics.val_accuracy.length - 1].value}
              trend={
                metrics.val_accuracy.length > 1
                  ? metrics.val_accuracy[metrics.val_accuracy.length - 1].value >
                    metrics.val_accuracy[metrics.val_accuracy.length - 2].value
                  : null
              }
              isAccuracy
            />
          )}
        </div>
      )}
    </div>
  );
}

// Simple SVG line chart component
function SimpleLineChart({
  trainData,
  valData,
  color1,
  color2,
}: {
  trainData: MetricDataPoint[];
  valData: MetricDataPoint[];
  color1: string;
  color2: string;
}) {
  const width = 600;
  const height = 200;
  const padding = { top: 10, right: 10, bottom: 30, left: 50 };

  // Find min/max for scaling
  const allValues = [
    ...trainData.map((d) => d.value),
    ...valData.map((d) => d.value),
  ];
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const maxStep = Math.max(
    ...trainData.map((d) => d.step),
    ...valData.map((d) => d.step)
  );

  // Scaling functions
  const xScale = (step: number) =>
    padding.left + ((step - 1) / maxStep) * (width - padding.left - padding.right);
  const yScale = (value: number) =>
    height -
    padding.bottom -
    ((value - minValue) / (maxValue - minValue)) *
      (height - padding.top - padding.bottom);

  // Create path string
  const createPath = (data: MetricDataPoint[]) => {
    if (data.length === 0) return "";
    return data
      .map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(d.step)} ${yScale(d.value)}`)
      .join(" ");
  };

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      style={{ maxHeight: "200px" }}
    >
      {/* Grid lines */}
      <g className="grid" stroke="#e5e7eb" strokeWidth="1">
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
          <line
            key={ratio}
            x1={padding.left}
            y1={
              height -
              padding.bottom -
              ratio * (height - padding.top - padding.bottom)
            }
            x2={width - padding.right}
            y2={
              height -
              padding.bottom -
              ratio * (height - padding.top - padding.bottom)
            }
          />
        ))}
      </g>

      {/* Y-axis labels */}
      <g className="y-axis" fontSize="10" fill="#6b7280">
        {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
          const value = minValue + ratio * (maxValue - minValue);
          return (
            <text
              key={ratio}
              x={padding.left - 5}
              y={
                height -
                padding.bottom -
                ratio * (height - padding.top - padding.bottom)
              }
              textAnchor="end"
              dominantBaseline="middle"
            >
              {value.toFixed(3)}
            </text>
          );
        })}
      </g>

      {/* X-axis labels */}
      <g className="x-axis" fontSize="10" fill="#6b7280">
        {trainData
          .filter((_, i) => i % Math.ceil(trainData.length / 5) === 0)
          .map((d) => (
            <text
              key={d.step}
              x={xScale(d.step)}
              y={height - padding.bottom + 15}
              textAnchor="middle"
            >
              Epoch {d.step}
            </text>
          ))}
      </g>

      {/* Lines */}
      {trainData.length > 0 && (
        <path
          d={createPath(trainData)}
          fill="none"
          stroke={color1}
          strokeWidth="2"
        />
      )}
      {valData.length > 0 && (
        <path
          d={createPath(valData)}
          fill="none"
          stroke={color2}
          strokeWidth="2"
        />
      )}

      {/* Points */}
      {trainData.map((d) => (
        <circle
          key={`train-${d.step}`}
          cx={xScale(d.step)}
          cy={yScale(d.value)}
          r="3"
          fill={color1}
        />
      ))}
      {valData.map((d) => (
        <circle
          key={`val-${d.step}`}
          cx={xScale(d.step)}
          cy={yScale(d.value)}
          r="3"
          fill={color2}
        />
      ))}
    </svg>
  );
}

function MetricSummary({
  label,
  value,
  trend,
  isAccuracy = false,
}: {
  label: string;
  value: number;
  trend: boolean | null;
  isAccuracy?: boolean;
}) {
  const displayValue = isAccuracy
    ? `${(value * 100).toFixed(2)}%`
    : value.toFixed(4);

  // For accuracy, up is good. For loss, down is good.
  const isGoodTrend = isAccuracy ? trend === true : trend === false;

  return (
    <div className="mt-4 flex items-center justify-between text-sm border-t border-gray-200 pt-3">
      <span className="text-gray-600">{label}:</span>
      <div className="flex items-center gap-2">
        <span className="font-semibold text-gray-900">{displayValue}</span>
        {trend !== null && (
          <span
            className={
              isGoodTrend ? "text-emerald-600" : "text-red-600"
            }
          >
            {trend ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
          </span>
        )}
      </div>
    </div>
  );
}
