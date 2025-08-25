import React, { useState, useRef, useEffect } from 'react';
import { BaseDashboardCard } from './BaseDashboardCard';
import { LogEntry } from '../types';

interface LogViewerCardProps {
  logs: LogEntry[];
  loading: boolean;
  error: string | null;
}

const LOG_LEVEL_COLORS = {
  DEBUG: 'text-gray-600',
  INFO: 'text-blue-600', 
  WARNING: 'text-yellow-600',
  ERROR: 'text-red-600',
  CRITICAL: 'text-red-800 font-bold'
};

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

export function LogViewerCard({ logs, loading, error }: LogViewerCardProps) {
  const [selectedLevel, setSelectedLevel] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const previousLogsLength = useRef(logs.length);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logs.length > previousLogsLength.current && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
    previousLogsLength.current = logs.length;
  }, [logs, autoScroll]);

  // Filter logs based on level and search term
  const filteredLogs = logs.filter(log => {
    const levelMatch = !selectedLevel || log.level === selectedLevel;
    const searchMatch = !searchTerm || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.logger.toLowerCase().includes(searchTerm.toLowerCase());
    return levelMatch && searchMatch;
  });

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  if (loading) {
    return (
      <BaseDashboardCard title="Application Logs">
        <div className="text-gray-600">Loading logs...</div>
      </BaseDashboardCard>
    );
  }

  if (error) {
    return (
      <BaseDashboardCard title="Application Logs">
        <div className="text-red-600">Error loading logs: {error}</div>
      </BaseDashboardCard>
    );
  }

  return (
    <BaseDashboardCard title="Application Logs">
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center space-x-2">
            <label htmlFor="log-level" className="text-sm font-medium text-gray-700">
              Level:
            </label>
            <select
              id="log-level"
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
              className="border border-gray-300 rounded px-2 py-1 text-sm"
            >
              <option value="">All Levels</option>
              {LOG_LEVELS.map(level => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center space-x-2 flex-1 max-w-xs">
            <label htmlFor="log-search" className="text-sm font-medium text-gray-700">
              Search:
            </label>
            <input
              id="log-search"
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter logs..."
              className="border border-gray-300 rounded px-2 py-1 text-sm flex-1"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              id="auto-scroll"
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="auto-scroll" className="text-sm font-medium text-gray-700">
              Auto-scroll
            </label>
          </div>
        </div>

        {/* Log entries */}
        <div 
          ref={logContainerRef}
          className="bg-black text-green-400 p-4 rounded-lg font-mono text-xs overflow-y-auto h-96"
          style={{ fontFamily: 'Monaco, "Lucida Console", monospace' }}
        >
          {filteredLogs.length === 0 ? (
            <div className="text-gray-500">
              {logs.length === 0 ? 'No logs available' : 'No logs match the current filters'}
            </div>
          ) : (
            filteredLogs.map((log, index) => (
              <div key={index} className="mb-1 break-words">
                <span className="text-gray-400">
                  [{formatTimestamp(log.timestamp)}]
                </span>{' '}
                <span className={`font-semibold ${LOG_LEVEL_COLORS[log.level as keyof typeof LOG_LEVEL_COLORS] || 'text-gray-400'}`}>
                  {log.level}
                </span>{' '}
                <span className="text-cyan-400">
                  {log.logger}:
                </span>{' '}
                <span className="text-green-400">
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Log stats */}
        <div className="text-sm text-gray-600 flex justify-between">
          <span>
            Showing {filteredLogs.length} of {logs.length} log entries
          </span>
          {logs.length > 0 && (
            <span>
              Latest: {formatTimestamp(logs[logs.length - 1]?.timestamp)}
            </span>
          )}
        </div>
      </div>
    </BaseDashboardCard>
  );
}