import React from 'react';
import { ModelInfo } from '../types';

interface ModelsCardProps {
  models: ModelInfo[];
  loading?: boolean;
  error?: string | null;
}

export function ModelsCard({ models, loading = false, error }: ModelsCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">
          Loaded Models
        </h2>
        <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
          {models.length}
        </div>
      </div>
      
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
    </div>
  );
}