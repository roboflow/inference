"use client";

import React, { useCallback } from "react";

// Import our new hooks and components
import { useServerData } from './hooks/useServerData';
import { useModelsData } from './hooks/useModelsData';
import { useMetricsData } from './hooks/useMetricsData';
import { useLogsData } from './hooks/useLogsData';
import { useAutoRefresh } from './hooks/useAutoRefresh';

import { ServerStatusCard } from './components/ServerStatusCard';
import { ModelsCard } from './components/ModelsCard';
import { RequestStatsCard } from './components/RequestStatsCard';
import { LogViewerCard } from './components/LogViewerCard';
import { QuickActions } from './components/QuickActions';
import { ErrorBanner } from './components/ErrorBanner';

function DashboardHeader() {
  return (
    <div className="mb-8">
      <h1 className="text-4xl font-bold text-gray-900 mb-2">
      <span className="text-primary-700">Inference</span> Dashboard
      </h1>
      <p className="text-gray-600">
        Here are your Roboflow Inference server status and metrics
      </p>
    </div>
  );
}

export default function Dashboard() {
  // Use our new custom hooks
  const serverData = useServerData();
  const modelsData = useModelsData(); 
  const metricsData = useMetricsData();
  const logsData = useLogsData();

  // Auto-refresh all data sources
  const refreshAll = useCallback(() => {
    serverData.refetch();
    modelsData.refetch();
    metricsData.refetch();
    if (logsData.logsAvailable) {
      logsData.refetch();
    }
  }, [serverData.refetch, modelsData.refetch, metricsData.refetch, logsData.refetch, logsData.logsAvailable]);
  
  // Set up auto-refresh every 5 seconds
  useAutoRefresh(refreshAll, { interval: 5000 });
  
  // Collect all errors from different sources
  const allErrors = [
    ...(serverData.error ? [serverData.error] : []),
    ...(modelsData.error ? [modelsData.error] : []),
    ...(metricsData.error ? [metricsData.error] : []),
    ...(logsData.error ? [logsData.error] : [])
  ];
  
  const isLoading = serverData.loading || modelsData.loading || metricsData.loading;

  // Calculate uptime from metrics
  const uptimeSeconds = metricsData.metrics.startTimeSeconds
    ? Math.max(0, Date.now() / 1000 - metricsData.metrics.startTimeSeconds)
    : undefined;

  if (isLoading) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center p-8">
        <div className="text-xl">Loading dashboard...</div>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col p-8 bg-gradient-to-br from-gray-50 to-primary-50">
      <div className="max-w-7xl mx-auto w-full">
        <DashboardHeader />
        <ErrorBanner errors={allErrors} />
        <QuickActions onRefresh={refreshAll} />
        
        <ServerStatusCard 
          serverInfo={serverData.serverInfo}
          healthStatus={serverData.healthStatus}
          metrics={metricsData.metrics}
          uptimeSeconds={uptimeSeconds}
        />
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ModelsCard 
            models={modelsData.models}
            loading={modelsData.loading}
            error={modelsData.error}
          />
          
          <RequestStatsCard 
            requestStats={metricsData.requestStats}
            loading={metricsData.loading}
            error={metricsData.error}
          />
        </div>

        {/* Show log viewer only when logs are available */}
        {logsData.logsAvailable && (
          <div className="mt-8">
            <LogViewerCard
              logs={logsData.logs}
              loading={logsData.loading}
              error={logsData.error}
            />
          </div>
        )}
      </div>
    </main>
  );
}