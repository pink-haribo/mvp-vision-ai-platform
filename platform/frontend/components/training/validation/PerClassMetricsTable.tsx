/**
 * PerClassMetricsTable Component
 *
 * Displays per-class metrics (precision, recall, f1-score, support)
 * in a sortable table format.
 */

import React, { useState } from 'react';

interface PerClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

interface PerClassMetricsTableProps {
  perClassMetrics: Record<string, PerClassMetric>;
}

type SortField = 'class' | 'precision' | 'recall' | 'f1_score' | 'support';
type SortOrder = 'asc' | 'desc';

export const PerClassMetricsTable: React.FC<PerClassMetricsTableProps> = ({
  perClassMetrics
}) => {
  const [sortField, setSortField] = useState<SortField>('class');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  // Format percentage
  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Convert object to array for sorting
  const metricsArray = Object.entries(perClassMetrics).map(([className, metrics]) => ({
    class: className,
    ...metrics
  }));

  // Sort data
  const sortedData = [...metricsArray].sort((a, b) => {
    let aVal = a[sortField];
    let bVal = b[sortField];

    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = (bVal as string).toLowerCase();
    }

    if (sortOrder === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  // Handle column header click
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  // Render sort icon
  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <span className="text-gray-400 ml-0.5 text-[9px]">⇅</span>;
    }
    return <span className="ml-0.5 text-violet-600">{sortOrder === 'asc' ? '↑' : '↓'}</span>;
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
        <h4 className="text-xs font-semibold text-gray-900">Per-Class Metrics</h4>
      </div>

      <div className="overflow-auto max-h-[400px]">
        <table className="w-full text-[10px]">
          <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
            <tr>
              <th
                className="text-left px-2 py-1 text-gray-700 font-semibold cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('class')}
              >
                Class <SortIcon field="class" />
              </th>
              <th
                className="text-right px-2 py-1 text-gray-700 font-semibold cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('precision')}
              >
                Prec <SortIcon field="precision" />
              </th>
              <th
                className="text-right px-2 py-1 text-gray-700 font-semibold cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('recall')}
              >
                Rec <SortIcon field="recall" />
              </th>
              <th
                className="text-right px-2 py-1 text-gray-700 font-semibold cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('f1_score')}
              >
                F1 <SortIcon field="f1_score" />
              </th>
              <th
                className="text-right px-2 py-1 text-gray-700 font-semibold cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('support')}
              >
                N <SortIcon field="support" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sortedData.map((row, idx) => (
              <tr
                key={row.class}
                className="hover:bg-gray-50"
              >
                <td className="px-2 py-1 text-gray-900 font-medium">{row.class}</td>
                <td className="px-2 py-1 text-right text-gray-700 font-mono">
                  {formatPercent(row.precision)}
                </td>
                <td className="px-2 py-1 text-right text-gray-700 font-mono">
                  {formatPercent(row.recall)}
                </td>
                <td className="px-2 py-1 text-right text-gray-700 font-mono">
                  {formatPercent(row.f1_score)}
                </td>
                <td className="px-2 py-1 text-right text-gray-700 font-mono">
                  {row.support}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Stats - Compact Footer */}
      <div className="px-3 py-2 bg-gray-50 border-t border-gray-200 flex items-center gap-6 text-[10px] overflow-x-auto">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Avg Prec:</span>
          <span className="font-semibold text-gray-900">
            {formatPercent(
              metricsArray.reduce((sum, row) => sum + row.precision, 0) / metricsArray.length
            )}
          </span>
        </div>
        <div className="w-px h-3 bg-gray-300" />
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Avg Rec:</span>
          <span className="font-semibold text-gray-900">
            {formatPercent(
              metricsArray.reduce((sum, row) => sum + row.recall, 0) / metricsArray.length
            )}
          </span>
        </div>
        <div className="w-px h-3 bg-gray-300" />
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Avg F1:</span>
          <span className="font-semibold text-gray-900">
            {formatPercent(
              metricsArray.reduce((sum, row) => sum + row.f1_score, 0) / metricsArray.length
            )}
          </span>
        </div>
        <div className="w-px h-3 bg-gray-300" />
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">Total:</span>
          <span className="font-semibold text-gray-900">
            {metricsArray.reduce((sum, row) => sum + row.support, 0)}
          </span>
        </div>
      </div>
    </div>
  );
};
