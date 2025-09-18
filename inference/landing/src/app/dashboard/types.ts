// Dashboard-specific TypeScript interfaces and types

export interface ServerInfo {
  name: string;
  version: string;
  uuid: string;
}

export interface ModelInfo {
  model_id: string;
  task_type: string;
  batch_size: number;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface Metrics {
  residentMemoryBytes?: number;
  startTimeSeconds?: number;
}

export interface RequestStats {
  total: number;
  successRate: number;
  topEndpoints: Array<{ 
    endpoint: string; 
    count: number; 
    successRate: number;
    successCount: number;
    errorCount: number;
  }>;
}

export type HealthStatus = "healthy" | "error" | "loading";

// Hook return types
export interface ServerDataState {
  serverInfo: ServerInfo | null;
  healthStatus: HealthStatus;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export interface ModelsDataState {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export interface MetricsDataState {
  metrics: Metrics;
  requestStats: RequestStats;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  module: string;
  line: number;
}

export interface LogsResponse {
  logs: LogEntry[];
  total_count: number;
}

export interface LogsDataState {
  logs: LogEntry[];
  loading: boolean;
  error: string | null;
  logsAvailable: boolean;
  refetch: () => void;
}