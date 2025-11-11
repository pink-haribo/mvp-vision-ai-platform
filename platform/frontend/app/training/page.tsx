"use client";

import { useEffect, useState } from "react";
import { TrainingJobCard } from "@/components/training/training-job-card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { trainingApi, TrainingJob } from "@/lib/api/training";
import { AlertCircle, Plus } from "lucide-react";

export default function TrainingPage() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadJobs();

    // Poll every 5 seconds for updates
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadJobs = async () => {
    try {
      const data = await trainingApi.listJobs();
      setJobs(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load training jobs");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (id: string) => {
    try {
      await trainingApi.cancelJob(id);
      await loadJobs(); // Reload jobs
    } catch (err: any) {
      alert(`Failed to cancel job: ${err.message}`);
    }
  };

  return (
    <main className="container mx-auto p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">학습 작업</h1>
            <p className="text-muted-foreground mt-1">
              진행 중인 모델 학습 작업을 관리하세요
            </p>
          </div>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            새 학습 작업
          </Button>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>오류</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-48 bg-muted animate-pulse rounded-lg" />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && jobs.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground mb-4">
              아직 학습 작업이 없습니다
            </p>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              첫 학습 작업 시작하기
            </Button>
          </div>
        )}

        {/* Job Cards */}
        {!loading && jobs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {jobs.map((job) => (
              <TrainingJobCard
                key={job.id}
                job={job}
                onView={(id) => (window.location.href = `/training/${id}`)}
                onCancel={handleCancel}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
