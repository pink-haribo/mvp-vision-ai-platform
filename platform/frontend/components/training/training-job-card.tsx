"use client";

import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { TrainingJob } from "@/lib/api/training";

interface TrainingJobCardProps {
  job: TrainingJob;
  onView?: (id: string) => void;
  onCancel?: (id: string) => void;
}

const statusColors = {
  pending: "secondary" as const,
  running: "info" as const,
  completed: "success" as const,
  failed: "destructive" as const,
  cancelled: "secondary" as const,
};

const statusLabels = {
  pending: "대기 중",
  running: "학습 중",
  completed: "완료",
  failed: "실패",
  cancelled: "취소됨",
};

export function TrainingJobCard({ job, onView, onCancel }: TrainingJobCardProps) {
  const progress = Math.round(job.progress * 100);

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <CardTitle className="text-base font-semibold">{job.model_name}</CardTitle>
        <Badge variant={statusColors[job.status]}>{statusLabels[job.status]}</Badge>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">프레임워크</span>
            <span className="font-medium">{job.framework}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">진행률</span>
            <span className="font-medium">{progress}%</span>
          </div>
        </div>

        {job.status === "running" && (
          <Progress value={progress} className="w-full" />
        )}

        {job.message && (
          <p className="text-sm text-muted-foreground line-clamp-2">{job.message}</p>
        )}

        {job.error && (
          <p className="text-sm text-destructive line-clamp-2">{job.error}</p>
        )}
      </CardContent>
      <CardFooter className="flex gap-2">
        {onView && (
          <Button
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={() => onView(job.id)}
          >
            상세보기
          </Button>
        )}
        {onCancel && job.status === "running" && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => onCancel(job.id)}
          >
            중지
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}
