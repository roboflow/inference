import { useEffect, useRef } from 'react';

interface UseAutoRefreshOptions {
  interval?: number;
  enabled?: boolean;
}

export function useAutoRefresh(
  callback: () => void,
  { interval = 5000, enabled = true }: UseAutoRefreshOptions = {}
) {
  const savedCallback = useRef(callback);

  // Update saved callback when it changes
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  // Handle auto-refresh interval
  useEffect(() => {
    if (!enabled) return;

    const intervalId = setInterval(() => {
      savedCallback.current();
    }, interval);

    return () => clearInterval(intervalId);
  }, [interval, enabled]);
}