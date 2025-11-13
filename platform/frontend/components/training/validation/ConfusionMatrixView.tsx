/**
 * ConfusionMatrixView Component
 *
 * Displays confusion matrix as a heatmap with color intensity
 * representing the number of predictions.
 */

import React from 'react';

interface ConfusionMatrixViewProps {
  confusionMatrix: number[][];
  classNames: string[];
  onCellClick?: (trueLabelId: number, predictedLabelId: number, trueLabel: string, predictedLabel: string) => void;
}

export const ConfusionMatrixView: React.FC<ConfusionMatrixViewProps> = ({
  confusionMatrix,
  classNames,
  onCellClick
}) => {
  // No color intensity needed - using fixed colors

  // Calculate cell size based on number of classes - Compact
  const numClasses = classNames.length;
  const cellSize = numClasses <= 5 ? 'w-10 h-10' : numClasses <= 10 ? 'w-8 h-8' : 'w-6 h-6';
  const fontSize = numClasses <= 5 ? 'text-xs' : numClasses <= 10 ? 'text-[10px]' : 'text-[9px]';

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
        <h4 className="text-xs font-semibold text-gray-900">Confusion Matrix</h4>
      </div>

      <div className="p-3 overflow-auto">
        <div className="inline-block min-w-full">
          {/* Matrix Table */}
          <table className="border-collapse">
            <thead>
              <tr>
                <th className={`${cellSize} border border-gray-200`}></th>
                <th className={`${cellSize} border border-gray-200`}></th>
                <th
                  colSpan={numClasses}
                  className="text-center text-[10px] text-gray-600 py-1 border border-gray-200"
                >
                  Predicted
                </th>
              </tr>
              <tr>
                <th className={`${cellSize} border border-gray-200`}></th>
                <th className={`${cellSize} border border-gray-200`}></th>
                {classNames.map((className, idx) => (
                  <th
                    key={idx}
                    className={`${cellSize} ${fontSize} text-gray-700 border border-gray-200 p-1 font-medium`}
                  >
                    <div className="truncate">{className}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  {rowIdx === 0 && (
                    <th
                      rowSpan={numClasses}
                      className="text-center text-[10px] text-gray-600 px-1 border border-gray-200 align-middle"
                      style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                    >
                      Actual
                    </th>
                  )}
                  <th
                    className={`${cellSize} ${fontSize} text-gray-700 border border-gray-200 p-1 font-medium`}
                  >
                    <div className="truncate">{classNames[rowIdx]}</div>
                  </th>
                  {row.map((value, colIdx) => {
                    const isCorrect = rowIdx === colIdx;
                    const isZero = value === 0;

                    // Determine text color
                    let textColor = 'text-gray-900';
                    if (isZero) {
                      textColor = 'text-gray-300'; // Very light gray for zero values
                    } else if (!isCorrect) {
                      textColor = 'text-red-600'; // Red for incorrect predictions
                    }

                    return (
                      <td
                        key={colIdx}
                        className={`${cellSize} ${fontSize} border border-gray-200 text-center font-semibold relative ${
                          isCorrect ? 'bg-green-100' : 'bg-white'
                        } ${onCellClick ? 'cursor-pointer hover:ring-2 hover:ring-violet-400' : ''}`}
                        onClick={() => {
                          if (onCellClick && value > 0) {
                            onCellClick(rowIdx, colIdx, classNames[rowIdx], classNames[colIdx]);
                          }
                        }}
                      >
                        <div className={textColor}>{value}</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Legend - Compact */}
          <div className="mt-2 flex items-center justify-center gap-4 text-[10px] text-gray-600">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-100 border border-gray-300"></div>
              <span>Correct predictions</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border border-gray-300 flex items-center justify-center">
                <span className="text-red-600 text-[8px] font-bold">0</span>
              </div>
              <span>Incorrect predictions</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
