import React from 'react';
import { ModelInfo } from '../types';
import { BaseDashboardCard } from './BaseDashboardCard';

interface ModelsCardProps {
  models: ModelInfo[];
  loading?: boolean;
  error?: string | null;
}

export function ModelsCard({ models, loading = false, error }: ModelsCardProps) {
  const modelsBadge = (
    <div className="bg-primary-100 text-primary-800 px-3 py-1 rounded-full text-sm font-medium">
      {models.length}
    </div>
  );

  return (
    <BaseDashboardCard 
      title="Loaded Models" 
      badge={modelsBadge}
      hover={true}
    >
      <div className="space-y-2">
        {loading ? (
          <p className="text-gray-500 text-sm">Loading models...</p>
        ) : error ? (
          <p className="text-red-500 text-sm">Error: {error}</p>
        ) : models.length === 0 ? (
          <p className="text-gray-500 text-sm">No models loaded</p>
        ) : (
          <div className="max-h-96 overflow-y-auto space-y-2">
            {models.map((model, index) => (
              <div
                key={index}
                className="border border-gray-200 rounded p-4 text-sm"
              >
                <div className="font-medium text-gray-900 mb-1">
                  {model.model_id}
                </div>
                <div className="text-gray-600">
                  Task: {model.task_type} | Batch Size: {model.batch_size}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </BaseDashboardCard>
  );
}