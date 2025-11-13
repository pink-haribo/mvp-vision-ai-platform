import React, { useState, useRef, useCallback } from 'react';

interface DatasetUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (datasetId: string) => void;
}

interface UploadResponse {
  status: string;
  dataset_id: string | null;
  message: string;
  metadata?: {
    name: string;
    format_version: string;
    task_type: string;
    num_images: number;
    num_classes: number;
    visibility: string;
    storage_path: string;
  };
}

export default function DatasetUploadModal({
  isOpen,
  onClose,
  onUploadSuccess,
}: DatasetUploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [visibility, setVisibility] = useState<'public' | 'private' | 'organization'>('private');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetState = () => {
    setSelectedFile(null);
    setVisibility('private');
    setUploading(false);
    setUploadProgress(0);
    setUploadStatus('idle');
    setStatusMessage('');
    setIsDragging(false);
  };

  const handleClose = () => {
    if (!uploading) {
      resetState();
      onClose();
    }
  };

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.endsWith('.zip')) {
        setSelectedFile(file);
        setUploadStatus('idle');
        setStatusMessage('');
      } else {
        setUploadStatus('error');
        setStatusMessage('Only .zip files are supported');
      }
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.endsWith('.zip')) {
        setSelectedFile(file);
        setUploadStatus('idle');
        setStatusMessage('');
      } else {
        setUploadStatus('error');
        setStatusMessage('Only .zip files are supported');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('error');
      setStatusMessage('Please select a file first');
      return;
    }

    try {
      setUploading(true);
      setUploadStatus('uploading');
      setUploadProgress(0);
      setStatusMessage('Uploading dataset...');

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('visibility', visibility);
      formData.append('user_id', 'platform'); // TODO: Get from auth

      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
      const response = await fetch(`${baseUrl}/datasets/upload`, {
        method: 'POST',
        body: formData,
      });

      const data: UploadResponse = await response.json();

      if (data.status === 'success' && data.dataset_id) {
        setUploadStatus('success');
        setStatusMessage(`Dataset uploaded successfully! ID: ${data.dataset_id}`);
        setUploadProgress(100);

        // Call success callback
        if (onUploadSuccess) {
          onUploadSuccess(data.dataset_id);
        }

        // Auto-close after 2 seconds
        setTimeout(() => {
          handleClose();
        }, 2000);
      } else {
        setUploadStatus('error');
        setStatusMessage(data.message || 'Upload failed');
        setUploadProgress(0);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setStatusMessage(error instanceof Error ? error.message : 'Upload failed');
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Upload Dataset</h2>
            <button
              onClick={handleClose}
              disabled={uploading}
              className="text-gray-400 hover:text-gray-600 disabled:opacity-50"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Drag & Drop Area */}
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragging
                ? 'border-indigo-600 bg-indigo-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>

            {selectedFile ? (
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <>
                <p className="mt-4 text-sm text-gray-600">
                  Drag and drop your DICE Format dataset (.zip) here, or
                </p>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Browse Files
                </button>
              </>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {/* Visibility Selection */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Visibility
            </label>
            <div className="flex gap-2">
              {(['private', 'organization', 'public'] as const).map((vis) => (
                <button
                  key={vis}
                  onClick={() => setVisibility(vis)}
                  disabled={uploading}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                    visibility === vis
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {vis.charAt(0).toUpperCase() + vis.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Upload Progress */}
          {uploadStatus === 'uploading' && (
            <div className="mt-6">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-gray-600">Uploading...</span>
                <span className="text-gray-900 font-medium">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Status Message */}
          {statusMessage && (
            <div
              className={`mt-4 p-4 rounded-lg ${
                uploadStatus === 'success'
                  ? 'bg-green-50 text-green-800 border border-green-200'
                  : uploadStatus === 'error'
                  ? 'bg-red-50 text-red-800 border border-red-200'
                  : 'bg-blue-50 text-blue-800 border border-blue-200'
              }`}
            >
              <p className="text-sm">{statusMessage}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-6 flex justify-end gap-3">
            <button
              onClick={handleClose}
              disabled={uploading}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading || uploadStatus === 'success'}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
