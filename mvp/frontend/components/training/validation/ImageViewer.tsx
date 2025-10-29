/**
 * ImageViewer Component
 *
 * Displays validation images filtered by confusion matrix cell selection.
 * Shows image thumbnails with prediction details and confidence scores.
 */

import React, { useEffect, useState } from 'react';

interface ValidationImage {
  id: number;
  validation_result_id: number;
  job_id: number;
  epoch: number;
  image_path: string;
  image_name: string;
  image_index: number;
  true_label: string;
  true_label_id: number;
  predicted_label: string;
  predicted_label_id: number;
  confidence: number;
  top5_predictions: any;
  is_correct: boolean;
}

interface ImageViewerProps {
  jobId: number;
  epoch: number;
  trueLabelId: number;
  predictedLabelId: number;
  trueLabel: string;
  predictedLabel: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  jobId,
  epoch,
  trueLabelId,
  predictedLabelId,
  trueLabel,
  predictedLabel
}) => {
  const [images, setImages] = useState<ValidationImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 20;

  useEffect(() => {
    fetchImages();
  }, [jobId, epoch, trueLabelId, predictedLabelId, currentPage]);

  const fetchImages = async () => {
    try {
      setLoading(true);
      setError(null);

      const skip = (currentPage - 1) * pageSize;
      const url = `${process.env.NEXT_PUBLIC_API_URL}/validation/jobs/${jobId}/results/${epoch}/images?true_label_id=${trueLabelId}&predicted_label_id=${predictedLabelId}&skip=${skip}&limit=${pageSize}`;

      console.log('[ImageViewer] Fetching images from:', url);
      console.log('[ImageViewer] Params:', { jobId, epoch, trueLabelId, predictedLabelId, skip, limit: pageSize });

      const response = await fetch(url);

      console.log('[ImageViewer] Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[ImageViewer] Error response:', errorText);
        throw new Error(`Failed to fetch images: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('[ImageViewer] Received data:', data);
      console.log('[ImageViewer] Total count:', data.total_count);
      console.log('[ImageViewer] Images count:', data.images?.length);

      setImages(data.images);
      setTotalCount(data.total_count);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch images');
      console.error('Error fetching validation images:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const totalPages = Math.ceil(totalCount / pageSize);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center space-y-2">
        <p className="text-sm font-semibold text-red-600">Error Loading Images</p>
        <p className="text-xs text-gray-600">{error}</p>
        <button
          onClick={fetchImages}
          className="mt-3 px-4 py-2 text-xs bg-violet-600 text-white rounded hover:bg-violet-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const isCorrect = trueLabelId === predictedLabelId;

  return (
    <div>
      {/* Header Info - Sticky */}
      <div className="sticky top-0 z-10 bg-white px-6 pt-6 pb-4">
        <div className="p-3 bg-gray-50 rounded border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-gray-600">True Label:</div>
            <div className="text-sm font-semibold text-gray-900">{trueLabel}</div>
          </div>
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs text-gray-600">Predicted Label:</div>
            <div className="text-sm font-semibold text-gray-900">{predictedLabel}</div>
          </div>
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-600">Total Images:</div>
            <div className="text-sm font-semibold text-gray-900">{totalCount}</div>
          </div>
        </div>
      </div>

      {/* Image Grid */}
      <div className="px-6 pb-6">
      {images.length === 0 ? (
        <div className="text-center py-8 text-sm text-gray-400">
          No images found for this combination
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3">
            {images.map((image) => (
              <div
                key={image.id}
                className={`border rounded overflow-hidden ${
                  image.is_correct
                    ? 'border-green-300 bg-green-50'
                    : 'border-red-300 bg-red-50'
                }`}
              >
                {/* Image */}
                <div className="aspect-square bg-gray-200 flex items-center justify-center overflow-hidden">
                  {image.image_path ? (
                    <img
                      src={`${process.env.NEXT_PUBLIC_API_URL}/validation/images/${image.id}`}
                      alt={image.image_name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        // Fallback to placeholder on error
                        e.currentTarget.style.display = 'none';
                        e.currentTarget.parentElement!.innerHTML = `<div class="text-xs text-gray-500 text-center px-2">${image.image_name}</div>`;
                      }}
                    />
                  ) : (
                    <div className="text-xs text-gray-500 text-center px-2">
                      {image.image_name}
                    </div>
                  )}
                </div>

                {/* Image Info */}
                <div className="p-2 bg-white">
                  <div className="text-[10px] text-gray-600 mb-1 truncate" title={image.image_name}>
                    {image.image_name}
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-gray-600">Confidence:</span>
                    <span className={`font-semibold ${
                      image.confidence > 0.8 ? 'text-green-600' :
                      image.confidence > 0.5 ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {formatConfidence(image.confidence)}
                    </span>
                  </div>
                  {!image.is_correct && (
                    <div className="mt-1 text-[10px] text-red-600 font-medium">
                      Incorrect Prediction
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-4 flex items-center justify-between text-xs">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Previous
              </button>
              <span className="text-gray-600">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 border border-gray-300 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
      </div>
    </div>
  );
};
