// src/api/client.ts
import axios from 'axios';
import type {
  User,
  RecommendHTTPResponse,
  RecommendationRecord,
  ClusterStatusResponse,
  SystemMetricsResponse,
  MetricsBenchmarkRequest,
  MetricsBenchmarkResponse,
  RedisMetricsResponse,
  LoginResponse,
  AuditEvent,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Permite que el contexto configure el header Authorization globalmente
export function setAuthToken(token: string | null) {
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common['Authorization'];
  }
}

// ---------- Login (JWT) ----------

export async function loginRequest(
  userId: number | null,
  adminKey?: string,
): Promise<LoginResponse> {
  const payload: any = {};
  if (userId !== null) payload.user_id = userId;
  if (adminKey) payload.admin_key = adminKey;

  const res = await api.post<LoginResponse>('/auth/login', payload);
  return res.data;
}

// ---------- User / recomendaciones ----------

export async function getUser(userId: number): Promise<User> {
  const res = await api.get<User>(`/users/${userId}`);
  return res.data;
}

export async function getRecommendations(
  userId: number,
  topN = 20,
): Promise<RecommendHTTPResponse> {
  const res = await api.get<RecommendHTTPResponse>(
    `/recommend/${userId}?top_n=${topN}`,
  );
  return res.data;
}

export async function getHistory(userId: number): Promise<RecommendationRecord[]> {
  // El backend puede devolver [] o null; aquí normalizamos a []
  const res = await api.get<RecommendationRecord[] | null>(`/history/${userId}`);
  return res.data ?? [];
}

// ---------- Admin / métricas ----------

export async function getClusterStatus(): Promise<ClusterStatusResponse> {
  const res = await api.get<ClusterStatusResponse>('/cluster/status');
  return res.data;
}

export async function getSystemMetrics(): Promise<SystemMetricsResponse> {
  const res = await api.get<SystemMetricsResponse>('/metrics/system');
  return res.data;
}

export async function getRedisMetrics(): Promise<RedisMetricsResponse> {
  const res = await api.get<RedisMetricsResponse>('/metrics/redis');
  return res.data;
}

export async function runBenchmark(
  payload: MetricsBenchmarkRequest,
): Promise<MetricsBenchmarkResponse> {
  const res = await api.post<MetricsBenchmarkResponse>(
    '/metrics/benchmark',
    payload,
  );
  return res.data;
}

export async function getLatestAuditEvents(
  limit = 50,
): Promise<AuditEvent[]> {
  const res = await api.get<AuditEvent[]>(`/audit/latest?limit=${limit}`);
  return res.data;
}
