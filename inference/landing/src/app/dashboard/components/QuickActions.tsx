import React from 'react';
import { BaseDashboardCard } from './BaseDashboardCard';

interface QuickActionsProps {
  onRefresh: () => void;
}

export function QuickActions({ onRefresh }: QuickActionsProps) {
  return (
    <BaseDashboardCard>
      <div className="flex flex-wrap gap-4">
        <a
          href="/"
          className="inline-flex items-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
        >
          Back to Home
        </a>
        <button
          onClick={onRefresh}
          className="inline-flex items-center px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 transition-colors"
        >
          Refresh Data
        </button>
      </div>
    </BaseDashboardCard>
  );
}