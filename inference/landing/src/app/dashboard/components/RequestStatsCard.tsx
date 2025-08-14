import React from 'react';
import { roboto_mono } from '../../fonts';
import classNames from 'classnames';
import { RequestStats } from '../types';

interface RequestStatsCardProps {
  requestStats: RequestStats;
  loading?: boolean;
  error?: string | null;
}

export function RequestStatsCard({ requestStats, loading = false, error }: RequestStatsCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">
        Inference API Requests
      </h2>
      
      <div className="space-y-3">
        {loading ? (
          <p className="text-gray-500 text-sm">Loading request statistics...</p>
        ) : error ? (
          <p className="text-red-500 text-sm">Error: {error}</p>
        ) : (
          <>
            <div>
              <span className="font-medium text-gray-700">Total: </span>
              <span className={roboto_mono.className}>
                {requestStats.total.toLocaleString()}
              </span>
            </div>
            
            <div>
              <span className="font-medium text-gray-700">Success Rate: </span>
              <span className={classNames(
                roboto_mono.className,
                requestStats.successRate >= 95 ? "text-green-600" : 
                requestStats.successRate >= 90 ? "text-yellow-600" : "text-red-600"
              )}>
                {requestStats.successRate.toFixed(1)}%
              </span>
            </div>
            
            {requestStats.topEndpoints.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="text-sm font-medium text-gray-700 mb-3">Top Endpoints:</div>
                <div className="space-y-2">
                  {requestStats.topEndpoints.map((ep, i) => (
                    <div key={i} className="text-sm">
                      <div className="flex items-center justify-between">
                        <span className={roboto_mono.className}>{ep.endpoint}</span>
                        <span className="text-gray-500">({ep.count})</span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs text-gray-600">
                          Success: {ep.successCount} | Errors: {ep.errorCount}
                        </span>
                        <span className={classNames(
                          "text-xs font-medium px-2 py-1 rounded",
                          ep.successRate >= 95 ? "bg-green-100 text-green-700" :
                          ep.successRate >= 90 ? "bg-yellow-100 text-yellow-700" : 
                          "bg-red-100 text-red-700"
                        )}>
                          {ep.successRate.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}