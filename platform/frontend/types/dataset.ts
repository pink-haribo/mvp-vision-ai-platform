// Dataset types for frontend

export interface Dataset {
  id: string;
  name: string;
  description: string;
  format: string;
  labeled: boolean;  // Whether dataset has annotations (annotation.json)
  num_images: number;
  size_mb?: number | null;
  source: string;
  path: string;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
  visibility?: string;  // 'public', 'private', 'organization'
  owner_id?: number | null;
  owner_name?: string | null;
  owner_email?: string | null;
  owner_badge_color?: string | null;
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
