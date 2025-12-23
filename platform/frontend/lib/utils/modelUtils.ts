/**
 * Model and Task Display Name Utilities
 * Phase 1: Consistent model/task naming across UI
 */

// Task display name mapping (Korean)
const TASK_DISPLAY_NAMES: Record<string, string> = {
  image_classification: "이미지 분류",
  object_detection: "객체 탐지",
  instance_segmentation: "인스턴스 분할",
  semantic_segmentation: "시맨틱 분할",
  pose_estimation: "자세 추정",
  super_resolution: "초해상화",
  segmentation: "분할", // Unified segmentation
};

/**
 * Get user-friendly display name for a task type
 *
 * @param taskType - Task type (e.g., "image_classification")
 * @returns Korean display name (e.g., "이미지 분류")
 */
export function getTaskDisplayName(taskType: string): string {
  return TASK_DISPLAY_NAMES[taskType] || taskType;
}

/**
 * Get model display name from model registry
 * This fetches from backend API /models/get
 *
 * @param framework - Framework name (timm, ultralytics, huggingface, mmdet, mmpretrain, mmseg, mmyolo)
 * @param modelId - Model ID (e.g., "resnet50", "google/vit-base-patch16-224")
 * @returns Display name or fallback to modelId
 */
export async function getModelDisplayName(
  framework: string,
  modelId: string
): Promise<string> {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    const response = await fetch(
      `${apiUrl}/models/get?framework=${framework}&model_name=${encodeURIComponent(
        modelId
      )}`
    );

    if (!response.ok) {
      console.warn(
        `Failed to fetch model info for ${framework}/${modelId}, using fallback`
      );
      return modelId;
    }

    const data = await response.json();
    return data.display_name || modelId;
  } catch (error) {
    console.error("Error fetching model display name:", error);
    return modelId; // Fallback
  }
}

/**
 * Synchronous version: Get model display name from client-side cache/mapping
 * Use this when you can't use async (e.g., in component render)
 *
 * Note: This returns a simplified display name based on model ID
 * For accurate names, use getModelDisplayName (async) instead
 *
 * @param framework - Framework name
 * @param modelId - Model ID
 * @returns Simplified display name
 */
export function getModelDisplayNameSync(
  framework: string,
  modelId: string
): string {
  // Simple mapping for common models
  const commonModels: Record<string, string> = {
    // timm
    resnet50: "ResNet-50",
    resnet18: "ResNet-18",
    "tf_efficientnetv2_s.in1k": "EfficientNetV2-Small",

    // ultralytics - YOLO11
    yolo11n: "YOLOv11 Nano",
    yolo11m: "YOLOv11 Medium",
    yolo11l: "YOLO11-Large",

    // ultralytics - YOLOv8
    yolov8n: "YOLOv8 Nano",
    yolov8s: "YOLOv8 Small",
    yolov8m: "YOLOv8 Medium",
    yolov8l: "YOLOv8l",

    // ultralytics - YOLOv8 specialized
    "yolov8n-seg": "YOLOv8n-Seg",
    "yolov8s-seg": "YOLOv8s-Seg",
    "yolov8m-seg": "YOLOv8m-seg",
    "yolov8n-pose": "YOLOv8n-Pose",
    "yolov8x-pose": "YOLOv8x-pose",

    // ultralytics - YOLO-World
    yolo_world_v2_s: "YOLO-World v2 Small",
    yolo_world_v2_m: "YOLO-World v2 Medium",

    // huggingface (use last part of path)
    "google/vit-base-patch16-224": "Vision Transformer (ViT) Base",
    "nvidia/segformer-b0-finetuned-ade-512-512": "SegFormer-B0 (ADE20K)",
    "caidas/swin2SR-classical-sr-x2-64": "Swin2SR 2x",
    "caidas/swin2SR-classical-sr-x4-64": "Swin2SR 4x",
    "ustc-community/dfine-x-coco": "D-FINE",

    // mmdet (MMDetection)
    "faster-rcnn_r50_fpn": "Faster R-CNN (ResNet-50)",
    "faster-rcnn_r101_fpn": "Faster R-CNN (ResNet-101)",
    "detr_r50": "DETR (ResNet-50)",
    "dino-4scale_r50": "DINO (ResNet-50)",
    "dino-5scale_swin-l": "DINO (Swin-L)",
    "rtmdet_tiny": "RTMDet Tiny",
    "rtmdet_s": "RTMDet Small",
    "rtmdet_m": "RTMDet Medium",
    "rtmdet_l": "RTMDet Large",
    "co_dino_5scale_r50": "Co-DETR (ResNet-50)",
    "co_dino_5scale_swin_l": "Co-DETR (Swin-L)",

    // mmpretrain (MMPreTrain)
    "resnet50_8xb32": "ResNet-50",
    "resnet101_8xb32": "ResNet-101",
    "swin-tiny_16xb64": "Swin Transformer Tiny",
    "swin-small_16xb64": "Swin Transformer Small",
    "swin-base_16xb64": "Swin Transformer Base",
    "convnext-tiny_32xb128": "ConvNeXt Tiny",
    "convnext-small_32xb128": "ConvNeXt Small",
    "convnext-base_32xb128": "ConvNeXt Base",
    "vit-base-p16_64xb64": "ViT Base (Patch16)",
    "efficientnet-b0_8xb32": "EfficientNet-B0",
    "efficientnet-b4_8xb32": "EfficientNet-B4",

    // mmseg (MMSegmentation)
    "deeplabv3plus_r50-d8": "DeepLabV3+ (ResNet-50)",
    "deeplabv3plus_r101-d8": "DeepLabV3+ (ResNet-101)",
    "segformer_mit-b0": "SegFormer (MiT-B0)",
    "segformer_mit-b2": "SegFormer (MiT-B2)",
    "segformer_mit-b5": "SegFormer (MiT-B5)",
    "mask2former_swin-s": "Mask2Former (Swin-S)",
    "mask2former_swin-b": "Mask2Former (Swin-B)",
    "upernet_swin-t": "UperNet (Swin-T)",
    "upernet_swin-b": "UperNet (Swin-B)",

    // mmyolo (MMYOLO)
    "yolov5_n": "YOLOv5 Nano",
    "yolov5_s": "YOLOv5 Small",
    "yolov5_m": "YOLOv5 Medium",
    "yolov5_l": "YOLOv5 Large",
    "yolov6_n": "YOLOv6 Nano",
    "yolov6_s": "YOLOv6 Small",
    "yolov7_tiny": "YOLOv7 Tiny",
    "yolov7_l": "YOLOv7 Large",
    "yolov8_n": "YOLOv8 Nano",
    "yolov8_s": "YOLOv8 Small",
    "rtmdet_tiny_syncbn": "RTMDet Tiny",
    "rtmdet_s_syncbn": "RTMDet Small",
    ppyoloe_plus_s: "PPYOLOE+ Small",
    ppyoloe_plus_m: "PPYOLOE+ Medium",
  };

  // Check common models first
  if (commonModels[modelId]) {
    return commonModels[modelId];
  }

  // For HuggingFace models, try to extract a readable name
  if (framework === "huggingface" && modelId.includes("/")) {
    const parts = modelId.split("/");
    const modelName = parts[parts.length - 1];

    // Convert to title case and clean up
    return modelName
      .split("-")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  // Fallback: return model ID
  return modelId;
}

/**
 * Format training job title with model and task name
 *
 * @param framework - Framework name
 * @param modelId - Model ID
 * @param taskType - Task type
 * @returns Formatted title (e.g., "ResNet-50 - 이미지 분류")
 */
export function formatTrainingJobTitle(
  framework: string,
  modelId: string,
  taskType: string
): string {
  const modelName = getModelDisplayNameSync(framework, modelId);
  const taskName = getTaskDisplayName(taskType);

  return `${modelName} - ${taskName}`;
}
