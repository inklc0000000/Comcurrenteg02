// src/api/types.ts

// ---- Tipos del backend (copiados 1:1 para evitar mapeos raros) ----

// GET /users/{user_id}
export interface User {
  user_id: number;
  name: string;
  gender: string;
  age: number;
  occupation: number;
  zip_code: string;
}

// GET /recommend/{user_id}
export interface MovieWithDetails {
  movie_id: number;
  title: string;
  genres: string[] | null;
  score: number;
  nodes?: string[];
}

export interface RecommendHTTPResponse {
  user_id: number;
  items: MovieWithDetails[];
}

// GET /history/{user_id}
export interface Item {
  movie_id: number;
  score: number;
  nodes?: string[]; // por si queremos usarlo en el futuro
}

export interface RecommendationRecord {
  user_id: number;
  top_n: number;
  items: Item[];
  node_addr: string;
  created_at: string; // viene como ISO string
}

// GET /cluster/status
export interface ClusterNodeStatus {
  addr: string;
  up: boolean;
  last_latency_ms?: number | null;
}

export interface ClusterStatusResponse {
  nodes: ClusterNodeStatus[];
  checked_at: string;
}

// GET /metrics/system
export interface SystemMetricsResponse {
  cpu_percent: number;
  num_goroutines: number;
  num_nodes: number;
  timestamp: string;
}

// POST /metrics/benchmark
export interface MetricsBenchmarkRequest {
  requests: number;
  concurrency: number;
  user_id: number;
  top_n: number;
}

export interface MetricsBenchmarkResponse {
  total_requests: number;
  ok: number;
  errors: number;
  total_time_sec: number;
  latency_avg_ms: number;
  latency_min_ms: number;
  latency_max_ms: number;
  rps: number;
}

export type LoginResponse = {
  token: string;
  user_id: number;
  role: 'user' | 'admin';
};

// GET /metrics/redis
export interface RedisMetricsResponse {
  db_size: number;
  used_memory_bytes: number;
  connected_clients: number;
  keyspace_hits: number;
  keyspace_misses: number;
}
// Eventos de auditoría (GET /audit/latest)
export interface AuditEvent {
  id: string;
  timestamp: string;   // ISO string
  type: string;        // "login", "recommendation", etc.
  user_id?: number;    // puede venir null/omitido
  detail: string;
}
