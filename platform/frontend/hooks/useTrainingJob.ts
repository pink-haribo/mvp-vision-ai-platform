/**
 * React Hook for Training Job Data (REST API)
 *
 * Fetches job configuration and metadata (not real-time updates)
 * For real-time monitoring, use useTrainingMonitor hook
 */

import { useState, useEffect } from 'react';

export interface TrainingJob {
  job_id: number;
  session_id?: number;
  model_name: string;
  framework: 'timm' | 'ultralytics' | 'huggingface' | 'mmdet' | 'mmpretrain' | 'mmseg' | 'mmyolo';
  task_type: string;
  dataset_path: string;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size?: number;
  num_classes?: number;
  executor_type: 'subprocess' | 'kubernetes';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  execution_id?: string;
  mlflow_run_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  hyperparameters?: Record<string, any>;
}

export interface UseTrainingJobOptions {
  refetchInterval?: number;
}

export function useTrainingJob(jobId: number, options: UseTrainingJobOptions = {}) {
  const { refetchInterval } = options;

  const [job, setJob] = useState<TrainingJob | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchJob = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
        const response = await fetch(`${apiUrl}/training/jobs/${jobId}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch job: ${response.statusText}`);
        }

        const data = await response.json();
        setJob(data);
        setError(null);
      } catch (err) {
        console.error('[useTrainingJob] Error:', err);
        setError(err as Error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchJob();

    // Optional polling for job metadata updates
    if (refetchInterval) {
      const interval = setInterval(fetchJob, refetchInterval);
      return () => clearInterval(interval);
    }
  }, [jobId, refetchInterval]);

  return {
    job,
    isLoading,
    error,
  };
}
