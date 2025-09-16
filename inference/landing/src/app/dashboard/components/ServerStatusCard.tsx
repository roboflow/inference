import React from 'react';
import { roboto_mono } from '../../fonts';
import classNames from 'classnames';
import { ServerInfo, HealthStatus, Metrics } from '../types';
import { BaseDashboardCard } from './BaseDashboardCard';

interface ServerStatusCardProps {
  serverInfo: ServerInfo | null;
  healthStatus: HealthStatus;
  metrics: Metrics;
  uptimeSeconds?: number;
}

function formatBytes(bytes?: number): string {
  if (bytes === undefined) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"] as const;
  let i = 0;
  let value = bytes;
  while (value >= 1024 && i < units.length - 1) {
    value = value / 1024;
    i++;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

function formatDuration(seconds?: number): string {
  if (seconds === undefined) return "-";
  const d = Math.floor(seconds / (3600 * 24));
  const h = Math.floor((seconds % (3600 * 24)) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const parts = [] as string[];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}

export function ServerStatusCard({ 
  serverInfo, 
  healthStatus, 
  metrics, 
  uptimeSeconds 
}: ServerStatusCardProps) {
  const statusBadge = (
    <div
      className={classNames(
        "px-3 py-1 rounded-full text-sm font-medium",
        {
          "bg-green-100 text-green-800": healthStatus === "healthy",
          "bg-red-100 text-red-800": healthStatus === "error",
          "bg-gray-100 text-gray-800": healthStatus === "loading",
        }
      )}
    >
      {healthStatus}
    </div>
  );

  return (
    <BaseDashboardCard 
      title="Server Status & Metrics" 
      badge={statusBadge}
      className="p-8 mb-8"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-3">
          <h3 className="font-medium text-gray-900">Server Info</h3>
          {serverInfo && (
            <>
              <div>
                <span className="font-medium text-gray-700">Version: </span>
                <span className={roboto_mono.className}>{serverInfo.version}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Server ID: </span>
                <span className={classNames(roboto_mono.className, "text-sm")}>
                  {serverInfo.uuid}
                </span>
              </div>
            </>
          )}
        </div>
        
        <div className="space-y-3">
          <h3 className="font-medium text-gray-900">System Metrics</h3>
          <div>
            <span className="font-medium text-gray-700">Memory: </span>
            <span className={roboto_mono.className}>
              {formatBytes(metrics.residentMemoryBytes)}
            </span>
          </div>
          <div>
            <span className="font-medium text-gray-700">Uptime: </span>
            <span className={roboto_mono.className}>
              {formatDuration(uptimeSeconds)}
            </span>
          </div>
        </div>
      </div>
    </BaseDashboardCard>
  );
}