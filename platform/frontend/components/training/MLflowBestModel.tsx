"use client";

import React, { useEffect, useState } from "react";
import { Award, TrendingUp, Database, Clock } from "lucide-react";

interface MLflowSummary {
  found: boolean;
  run_id: string | null;
  status: string | null;
  latest_metrics: {
    [key: string]: number;
  };
  params: {
    [key: string]: string;
  };
  best_val_accuracy?: number;
  best_val_loss?: number;
  start_time?: number;
  end_time?: number;
  artifact_uri?: string;
}

interface MLflowBestModelProps {
  jobId: number | string;
}

export default function MLflowBestModel({ jobId }: MLflowBestModelProps) {
  const [summary, setSummary] = useState<MLflowSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${jobId}/mlflow/summary`
        );
        if (response.ok) {
          const data = await response.json();
          setSummary(data);
        }
      } catch (error) {
        console.error("Error fetching MLflow summary:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSummary();

    // Poll every 5 seconds for updates
    const interval = setInterval(fetchSummary, 5000);
    return () => clearInterval(interval);
  }, [jobId]);

  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="h-32 bg-gray-200 rounded-lg"></div>
      </div>
    );
  }

  if (!summary || !summary.found) {
    return (
      <div className="p-6 bg-amber-50 rounded-lg border border-amber-200 text-center">
        <p className="text-sm text-amber-800">
          모델 정보를 찾을 수 없습니다.
        </p>
      </div>
    );
  }

  const hasBestMetrics =
    summary.best_val_accuracy != null &&
    summary.best_val_loss != null &&
    typeof summary.best_val_accuracy === 'number' &&
    typeof summary.best_val_loss === 'number';

  // Calculate duration if available
  const duration =
    summary.start_time && summary.end_time
      ? formatDuration((summary.end_time - summary.start_time) / 1000)
      : summary.start_time && summary.status === "RUNNING"
      ? formatDuration((Date.now() - summary.start_time) / 1000)
      : null;

  return (
    <div className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-lg border border-violet-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <Award className="w-5 h-5 text-violet-600" />
        <h4 className="text-sm font-semibold text-gray-900">
          Best Model Performance
        </h4>
      </div>

      {hasBestMetrics ? (
        <div className="space-y-4">
          {/* Best Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <MetricCard
              icon={<TrendingUp className="w-4 h-4" />}
              label="Best Val Accuracy"
              value={`${(summary.best_val_accuracy! * 100).toFixed(2)}%`}
              color="text-emerald-600"
              bgColor="bg-emerald-100"
            />
            <MetricCard
              icon={<Database className="w-4 h-4" />}
              label="Best Val Loss"
              value={summary.best_val_loss!.toFixed(4)}
              color="text-blue-600"
              bgColor="bg-blue-100"
            />
          </div>

          {/* Additional Info */}
          <div className="pt-4 border-t border-violet-200 space-y-2">
            <InfoRow
              label="Run ID"
              value={summary.run_id?.substring(0, 8) || "N/A"}
              mono
            />
            <InfoRow
              label="Status"
              value={
                <span
                  className={`px-2 py-0.5 rounded text-xs font-semibold ${
                    summary.status === "FINISHED"
                      ? "bg-green-100 text-green-800"
                      : summary.status === "RUNNING"
                      ? "bg-blue-100 text-blue-800"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {summary.status}
                </span>
              }
            />
            {duration && (
              <InfoRow
                label="Duration"
                value={
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-gray-500" />
                    <span>{duration}</span>
                  </div>
                }
              />
            )}
          </div>

          {/* Key Parameters */}
          {Object.keys(summary.params).length > 0 && (
            <div className="pt-4 border-t border-violet-200">
              <p className="text-xs font-semibold text-gray-700 mb-2">
                Key Parameters
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(summary.params)
                  .slice(0, 6)
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-600">
                        {key.replace("_", " ")}:
                      </span>
                      <span className="font-medium text-gray-900">{value}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-gray-600">
          Best model metrics will appear once training completes.
        </p>
      )}
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  color,
  bgColor,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
  bgColor: string;
}) {
  return (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <div className={`inline-flex p-2 rounded-lg ${bgColor} mb-2`}>
        <div className={color}>{icon}</div>
      </div>
      <p className="text-xs text-gray-600 mb-1">{label}</p>
      <p className={`text-lg font-bold ${color}`}>{value}</p>
    </div>
  );
}

function InfoRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex justify-between items-center text-sm">
      <span className="text-gray-600">{label}:</span>
      <span className={`font-medium text-gray-900 ${mono ? "font-mono" : ""}`}>
        {value}
      </span>
    </div>
  );
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}
