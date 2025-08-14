import { useState, useEffect, useCallback } from 'react';
import { ServerInfo, ServerDataState } from '../types';

export function useServerData(): ServerDataState {
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/info');
      if (response.ok) {
        const info = await response.json();
        setServerInfo(info);
      } else {
        setError(`Failed to fetch server info (${response.status})`);
      }
    } catch (err) {
      setError('Server connection failed');
      console.error('Failed to fetch server info:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const healthStatus = error ? 'error' : (serverInfo ? 'healthy' : 'loading');

  return {
    serverInfo,
    healthStatus,
    loading,
    error,
    refetch: fetchData
  };
}