import { useState, useEffect, useCallback } from 'react';
import { ModelInfo, ModelsResponse, ModelsDataState } from '../types';

export function useModelsData(): ModelsDataState {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      
      const response = await fetch('/model/registry');
      if (response.ok) {
        const data: ModelsResponse = await response.json();
        setModels(data.models || []);
      } else {
        setError(`Failed to fetch models (${response.status})`);
        // Keep previous models on error, don't reset to empty array
      }
    } catch (err) {
      setError('Failed to connect to models endpoint');
      console.error('Failed to fetch models:', err);
      // Keep previous models on error
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    models,
    loading,
    error,
    refetch: fetchData
  };
}