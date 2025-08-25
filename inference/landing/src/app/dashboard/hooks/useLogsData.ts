import { useState, useEffect, useCallback } from 'react';
import { LogsDataState, LogsResponse } from '../types';

export function useLogsData(): LogsDataState {
  const [logs, setLogs] = useState<LogsResponse['logs']>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [logsAvailable, setLogsAvailable] = useState(false);

  const fetchLogs = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/logs?limit=200');
      
      if (response.status === 404) {
        // Logs endpoint not available in this environment
        setLogsAvailable(false);
        setLogs([]);
      } else if (response.ok) {
        const data: LogsResponse = await response.json();
        setLogs(data.logs);
        setLogsAvailable(true);
      } else {
        setError(`Failed to fetch logs (${response.status})`);
        setLogsAvailable(false);
      }
    } catch (err) {
      // Network error or logs not available
      setLogsAvailable(false);
      setLogs([]);
      console.error('Failed to fetch logs:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  return {
    logs,
    loading,
    error,
    logsAvailable,
    refetch: fetchLogs
  };
}