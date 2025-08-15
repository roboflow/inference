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
          className="inline-flex items-center px-4 py-2 rounded-md bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 hover:border-gray-400 transition-colors shadow-sm"
        >
          Home
        </a>
        <button
          onClick={onRefresh}
          className="inline-flex items-center px-4 py-2 rounded-md bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 hover:border-gray-400 transition-colors shadow-sm"
        >
          Refresh Data
        </button>
      </div>
    </BaseDashboardCard>
  );
}