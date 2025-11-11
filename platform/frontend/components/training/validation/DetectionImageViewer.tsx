/**
 * DetectionImageViewer Component
 *
 * Displays validation images with bounding box overlays for object detection.
 * Supports toggling true/predicted bbox visibility.
 */

import React, { useState, useEffect, useRef } from 'react';

interface BoundingBox {
  class_id: number;
  bbox: number[];  // [x, y, w, h] - format can vary
  confidence?: number;
  format?: 'yolo' | 'coco' | 'xyxy';  // yolo: normalized, coco: absolute, xyxy: [x1, y1, x2, y2]
}

interface DetectionImage {
  id: number;
  image_path: string;
  image_name: string;
  true_label: string;
  true_label_id: number;
  predicted_label: string | null;
  predicted_label_id: number | null;
  confidence: number | null;
  true_boxes: BoundingBox[];
  predicted_boxes: BoundingBox[];
  is_correct: boolean;
}

interface DetectionImageViewerProps {
  jobId: number;
  epoch: number;
  classId: number;
  className: string;
  showTrueBoxes: boolean;
  showPredictedBoxes: boolean;
  onClose: () => void;
}

export const DetectionImageViewer: React.FC<DetectionImageViewerProps> = ({
  jobId,
  epoch,
  classId,
  className,
  showTrueBoxes,
  showPredictedBoxes,
  onClose,
}) => {
  const [images, setImages] = useState<DetectionImage[]>([]);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showAllClasses, setShowAllClasses] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Fetch images filtered by class_id
  useEffect(() => {
    fetchImages();
  }, [jobId, epoch, classId]);

  // Redraw canvas when boxes visibility changes or image changes
  useEffect(() => {
    if (images.length > 0 && imageRef.current?.complete) {
      drawBoundingBoxes();
    }
  }, [images, currentIndex, showTrueBoxes, showPredictedBoxes, showAllClasses, showLabels]);

  const fetchImages = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/validation/jobs/${jobId}/results/${epoch}/images?true_label_id=${classId}&limit=1000`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`);
      }

      const data = await response.json();
      setImages(data.images || []);
      setClassNames(data.class_names || []);
      setCurrentIndex(0);
    } catch (err) {
      console.error('Failed to fetch detection images:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch images');
    } finally {
      setLoading(false);
    }
  };

  const getClassName = (classId: number): string => {
    if (classNames && classNames.length > classId) {
      return classNames[classId];
    }
    return `Class ${classId}`;
  };

  const drawBoundingBoxes = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !image.complete) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match image
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const currentImage = images[currentIndex];
    if (!currentImage) return;

    // Filter boxes by class if needed
    const filterBoxes = (boxes: BoundingBox[]) => {
      if (showAllClasses) return boxes;
      return boxes.filter(box => box.class_id === classId);
    };

    // Get colors based on whether box belongs to selected class
    const getTrueBoxColor = (box: BoundingBox) => {
      return box.class_id === classId ? '#10B981' : '#86EFAC'; // green : light green
    };

    const getPredBoxColor = (box: BoundingBox) => {
      return box.class_id === classId ? '#EF4444' : '#FCA5A5'; // red : light red
    };

    // Draw true boxes (green)
    if (showTrueBoxes && currentImage.true_boxes) {
      const filteredTrueBoxes = filterBoxes(currentImage.true_boxes);
      ctx.lineWidth = 3;
      filteredTrueBoxes.forEach((box) => {
        const color = getTrueBoxColor(box);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        drawBox(ctx, box, canvas.width, canvas.height);

        // Draw label if enabled
        if (showLabels) {
          const [x, y] = getBboxCoords(box, canvas.width, canvas.height);
          const label = getClassName(box.class_id);
          drawLabel(ctx, label, x, y, color, '#FFFFFF');
        }
      });
    }

    // Draw predicted boxes (red)
    if (showPredictedBoxes && currentImage.predicted_boxes) {
      const filteredPredBoxes = filterBoxes(currentImage.predicted_boxes);
      ctx.lineWidth = 3;
      filteredPredBoxes.forEach((box) => {
        const color = getPredBoxColor(box);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        drawBox(ctx, box, canvas.width, canvas.height);

        // Draw label if enabled
        if (showLabels && box.confidence !== undefined) {
          const [x, y] = getBboxCoords(box, canvas.width, canvas.height);
          const label = `${getClassName(box.class_id)} ${(box.confidence * 100).toFixed(1)}%`;
          drawLabel(ctx, label, x, y, color, '#FFFFFF');
        }
      });
    }
  };

  const drawBox = (
    ctx: CanvasRenderingContext2D,
    box: BoundingBox,
    imageWidth: number,
    imageHeight: number
  ) => {
    const [x, y, w, h] = getBboxCoords(box, imageWidth, imageHeight);
    ctx.strokeRect(x, y, w, h);
  };

  const drawLabel = (
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    bgColor: string,
    textColor: string
  ) => {
    ctx.font = 'bold 14px sans-serif';
    const metrics = ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = 16;
    const padding = 4;

    // Draw background rectangle
    ctx.fillStyle = bgColor;
    ctx.fillRect(
      x,
      y - textHeight - padding * 2,
      textWidth + padding * 2,
      textHeight + padding * 2
    );

    // Draw text
    ctx.fillStyle = textColor;
    ctx.fillText(text, x + padding, y - padding);
  };

  const getBboxCoords = (
    box: BoundingBox,
    imageWidth: number,
    imageHeight: number
  ): [number, number, number, number] => {
    const [b0, b1, b2, b3] = box.bbox;

    if (box.format === 'yolo') {
      // YOLO format: normalized [x_center, y_center, width, height]
      const x = (b0 - b2 / 2) * imageWidth;
      const y = (b1 - b3 / 2) * imageHeight;
      const w = b2 * imageWidth;
      const h = b3 * imageHeight;
      return [x, y, w, h];
    } else if (box.format === 'xyxy') {
      // XYXY format: [x1, y1, x2, y2]
      return [b0, b1, b2 - b0, b3 - b1];
    } else {
      // COCO format (default): [x, y, width, height] in absolute coordinates
      return [b0, b1, b2, b3];
    }
  };

  const handleImageLoad = () => {
    drawBoundingBoxes();
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(images.length - 1, prev + 1));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">로딩 중...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-red-500">에러: {error}</div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">이미지가 없습니다</div>
      </div>
    );
  }

  const currentImage = images[currentIndex];

  return (
    <div className="flex flex-col h-full">
      {/* Image Display */}
      <div className="flex-1 relative bg-gray-100 flex items-center justify-center overflow-hidden">
        <div className="relative">
          <img
            ref={imageRef}
            src={`${process.env.NEXT_PUBLIC_API_URL}/validation/images/${currentImage.id}`}
            alt={currentImage.image_name}
            className="max-w-full max-h-[70vh] object-contain"
            onLoad={handleImageLoad}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
            style={{ imageRendering: 'crisp-edges' }}
          />
        </div>
      </div>

      {/* Navigation Controls */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="flex items-center justify-center gap-3 mb-3">
          <button
            onClick={handlePrevious}
            disabled={currentIndex === 0}
            className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 rounded text-sm font-medium transition-colors"
          >
            &lt; 이전
          </button>
          <div className="text-sm text-gray-600 font-medium">
            {currentIndex + 1} / {images.length}
          </div>
          <button
            onClick={handleNext}
            disabled={currentIndex === images.length - 1}
            className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 disabled:text-gray-400 rounded text-sm font-medium transition-colors"
          >
            다음 &gt;
          </button>
        </div>

        {/* Display Options */}
        <div className="mb-3 pb-3 border-b border-gray-200">
          <div className="text-xs font-semibold text-gray-700 mb-2">Display Options</div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showAllClasses}
                onChange={(e) => setShowAllClasses(e.target.checked)}
                className="w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
              />
              <span className="text-xs text-gray-700">Show all classes</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                className="w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
              />
              <span className="text-xs text-gray-700">Show labels</span>
            </label>
          </div>
        </div>

        {/* Image Info */}
        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-gray-600">이미지:</span>
            <span className="font-medium text-gray-900">{currentImage.image_name}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-600">True Class:</span>
            <span className="font-medium text-gray-900">{currentImage.true_label}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Predicted Class:</span>
            <span className="font-medium text-gray-900">
              {currentImage.predicted_label || 'N/A'}
              {currentImage.confidence && (
                <span className="text-gray-600 ml-1">
                  ({(currentImage.confidence * 100).toFixed(1)}%)
                </span>
              )}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-600">True Boxes:</span>
            <span className="font-medium text-green-600">
              {showAllClasses
                ? (currentImage.true_boxes?.length || 0)
                : (currentImage.true_boxes?.filter(box => box.class_id === classId).length || 0)
              }
              {!showAllClasses && (
                <span className="text-gray-500 text-xs ml-1">
                  / {currentImage.true_boxes?.length || 0} ({className} only)
                </span>
              )}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Predicted Boxes:</span>
            <span className="font-medium text-red-600">
              {showAllClasses
                ? (currentImage.predicted_boxes?.length || 0)
                : (currentImage.predicted_boxes?.filter(box => box.class_id === classId).length || 0)
              }
              {!showAllClasses && (
                <span className="text-gray-500 text-xs ml-1">
                  / {currentImage.predicted_boxes?.length || 0} ({className} only)
                </span>
              )}
            </span>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="border-t border-gray-200 bg-gray-50 p-3">
        <div className="text-xs">
          <div className="font-semibold text-gray-700 mb-2">Legend</div>
          <div className="space-y-1">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2" style={{ borderColor: '#10B981' }}></div>
                <span className="text-gray-700">True ({className})</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2" style={{ borderColor: '#EF4444' }}></div>
                <span className="text-gray-700">Predicted ({className})</span>
              </div>
            </div>
            {showAllClasses && (
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2" style={{ borderColor: '#86EFAC' }}></div>
                  <span className="text-gray-700">True (other classes)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2" style={{ borderColor: '#FCA5A5' }}></div>
                  <span className="text-gray-700">Predicted (other classes)</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
