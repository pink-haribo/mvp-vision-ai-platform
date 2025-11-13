"use client";

import React from "react";
import { X, PlayCircle, RotateCcw } from "lucide-react";

interface ResumeDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onStartFromScratch: () => void;
  onResume: () => void;
  latestCheckpointEpoch?: number;
}

export default function ResumeDialog({
  isOpen,
  onClose,
  onStartFromScratch,
  onResume,
  latestCheckpointEpoch,
}: ResumeDialogProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      ></div>

      {/* Dialog */}
      <div className="relative bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            학습 시작 방식 선택
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          <p className="text-sm text-gray-600 mb-4">
            {latestCheckpointEpoch
              ? `저장된 체크포인트가 있습니다 (Epoch ${latestCheckpointEpoch}). 어떻게 학습을 시작하시겠습니까?`
              : "학습을 어떻게 시작하시겠습니까?"}
          </p>

          {/* Options */}
          <div className="space-y-3">
            {/* Start from scratch */}
            <button
              onClick={onStartFromScratch}
              className="w-full flex items-center gap-3 px-4 py-3 border-2 border-gray-200 rounded-lg hover:border-violet-500 hover:bg-violet-50 transition-all group"
            >
              <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center group-hover:bg-violet-100 transition-colors">
                <RotateCcw className="w-5 h-5 text-gray-600 group-hover:text-violet-600" />
              </div>
              <div className="flex-1 text-left">
                <div className="font-medium text-gray-900">처음부터 시작</div>
                <div className="text-xs text-gray-500">
                  모든 학습 데이터를 초기화하고 새로 시작합니다
                </div>
              </div>
            </button>

            {/* Resume from checkpoint */}
            {latestCheckpointEpoch && (
              <button
                onClick={onResume}
                className="w-full flex items-center gap-3 px-4 py-3 border-2 border-violet-200 bg-violet-50 rounded-lg hover:border-violet-500 hover:bg-violet-100 transition-all group"
              >
                <div className="flex-shrink-0 w-10 h-10 bg-violet-100 rounded-full flex items-center justify-center group-hover:bg-violet-200 transition-colors">
                  <PlayCircle className="w-5 h-5 text-violet-600" />
                </div>
                <div className="flex-1 text-left">
                  <div className="font-medium text-gray-900">
                    이어서 시작 (권장)
                  </div>
                  <div className="text-xs text-gray-600">
                    Epoch {latestCheckpointEpoch}부터 학습을 이어갑니다
                  </div>
                </div>
              </button>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 rounded-b-lg">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
          >
            취소
          </button>
        </div>
      </div>
    </div>
  );
}
