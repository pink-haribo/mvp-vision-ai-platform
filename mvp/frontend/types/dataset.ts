// Dataset types for frontend

export interface Dataset {
  id: string;
  name: string;
  description: string;
  format: string;
  task_type: string;
  num_items: number;
  size_mb?: number | null;
  source: string;
  path: string;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
}

export interface DatasetAnalysis {
  status: string;
  dataset_info: {
    format: string;
    confidence: number;
    task_type: string;
    structure: {
      num_classes: number;
      num_images: number;
      classes: string[];
    };
    statistics: {
      total_images: number;
      source: string;
      validated: boolean;
    };
  };
}
