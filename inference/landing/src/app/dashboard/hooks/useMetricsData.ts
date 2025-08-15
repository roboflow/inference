import { useState, useEffect, useCallback } from 'react';
import { Metrics, RequestStats, MetricsDataState } from '../types';

// Whitelist of endpoints to include in API statistics
const WHITELISTED_ENDPOINTS = [
  '/workflows/run',
  '/inference_pipelines/',
  '/infer/',
];

function parsePrometheusMetrics(text: string): { metrics: Metrics; requests: RequestStats } {
  const lines = text.split(/\n/);
  const metrics: Metrics = {};
  const requestCounts: { [key: string]: number } = {};
  const successCounts: { [key: string]: number } = {};
  const errorCounts: { [key: string]: number } = {};
  let totalRequests = 0;
  let successRequests = 0;
  
  for (const line of lines) {
    if (!line || line.startsWith("#")) continue;
    
    // Parse system metrics
    if (line.startsWith("process_resident_memory_bytes")) {
      const v = Number(line.split(/\s+/).pop());
      if (!Number.isNaN(v)) metrics.residentMemoryBytes = v;
    } else if (line.startsWith("process_start_time_seconds")) {
      const v = Number(line.split(/\s+/).pop());
      if (!Number.isNaN(v)) metrics.startTimeSeconds = v;
    } 
    // Parse request metrics (whitelist only inference/workflow endpoints)
    else if (line.startsWith("http_requests_total")) {
      // Extract handler, status, and count
      const match = line.match(/handler="([^"]+)".*status="(\d)xx".*\s+(\d+(?:\.\d+)?)/);
      if (match) {
        const [, handler, statusCode, countStr] = match;
        const count = parseFloat(countStr);
        
        // Only include whitelisted endpoints
        const isWhitelisted = WHITELISTED_ENDPOINTS.some(endpoint => 
          handler.startsWith(endpoint) || handler.includes(endpoint)
        );
        
        if (isWhitelisted && handler !== "none") {
          // Track total requests per endpoint
          requestCounts[handler] = (requestCounts[handler] || 0) + count;
          
          // Track success/error counts per endpoint
          if (statusCode === "2") {
            successCounts[handler] = (successCounts[handler] || 0) + count;
            successRequests += count;
          } else if (statusCode === "4" || statusCode === "5") {
            errorCounts[handler] = (errorCounts[handler] || 0) + count;
          }
          
          totalRequests += count;
        }
      }
    }
  }
  
  // Calculate top endpoints with success rates
  const topEndpoints = Object.entries(requestCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([endpoint, count]) => {
      const successCount = successCounts[endpoint] || 0;
      const errorCount = errorCounts[endpoint] || 0;
      const successRate = count > 0 ? (successCount / count) * 100 : 100;
      
      return {
        endpoint,
        count,
        successRate,
        successCount,
        errorCount
      };
    });
  
  const successRate = totalRequests > 0 ? (successRequests / totalRequests) * 100 : 100;
  
  return {
    metrics,
    requests: {
      total: totalRequests,
      successRate,
      topEndpoints
    }
  };
}

export function useMetricsData(): MetricsDataState {
  const [metrics, setMetrics] = useState<Metrics>({});
  const [requestStats, setRequestStats] = useState<RequestStats>({ 
    total: 0, 
    successRate: 100, 
    topEndpoints: [] 
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/metrics');
      if (response.ok) {
        const text = await response.text();
        const parsed = parsePrometheusMetrics(text);
        setMetrics(parsed.metrics);
        setRequestStats(parsed.requests);
      } else {
        setError(`Failed to fetch metrics (${response.status})`);
        // Keep previous metrics on error
      }
    } catch (err) {
      setError('Failed to connect to metrics endpoint');
      console.error('Failed to fetch metrics:', err);
      // Keep previous metrics on error, don't treat as critical
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    metrics,
    requestStats,
    loading,
    error,
    refetch: fetchData
  };
}