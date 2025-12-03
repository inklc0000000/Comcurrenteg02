// src/pages/AdminPage.tsx
import React, { useEffect, useState } from 'react';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import type {
  ClusterStatusResponse,
  SystemMetricsResponse,
  MetricsBenchmarkRequest,
  MetricsBenchmarkResponse,
  RedisMetricsResponse,
  AuditEvent,
} from '../api/types';
import {
  getClusterStatus,
  getSystemMetrics,
  getRedisMetrics,
  runBenchmark,
  getLatestAuditEvents,
} from '../api/client';
import ClusterStatusCard from '../components/admin/ClusterStatusCard';
import MetricsForm from '../components/admin/MetricsForm';
import MetricsResultCard from '../components/admin/MetricsResultCard';
import MetricsChart from '../components/admin/MetricsChart';
import RedisStatusCard from '../components/admin/RedisStatusCard';
import AuditLog from '../components/admin/AuditLog';
import { useUser } from '../context/UserContext';

const AdminPage: React.FC = () => {
  const { token, role } = useUser();

  const [cluster, setCluster] = useState<ClusterStatusResponse | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetricsResponse | null>(null);
  const [redisMetrics, setRedisMetrics] = useState<RedisMetricsResponse | null>(null);
  const [benchmark, setBenchmark] = useState<MetricsBenchmarkResponse | null>(null);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [loadingInit, setLoadingInit] = useState(false);
  const [loadingBenchmark, setLoadingBenchmark] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Carga inicial + refresco periódico de estado de cluster y métricas de Redis
  useEffect(() => {
    if (!token || role !== 'admin') return;

    let cancelled = false;

    const loadInitial = async () => {
      try {
        setLoadingInit(true);
        setError(null);

        const [clusterData, systemData, redisData, auditData] = await Promise.all([
          getClusterStatus(),
          getSystemMetrics(),
          getRedisMetrics(),
          getLatestAuditEvents(),       // <- NUEVO
        ]);

        if (!cancelled) {
          setCluster(clusterData);
          setSystemMetrics(systemData);
          setRedisMetrics(redisData);
          setAuditEvents(auditData);    // <- NUEVO
        }
      } catch (e: any) {
        console.error(e);
        if (!cancelled) {
          setError('Error cargando métricas iniciales. Revisa API, nodos ML y Redis.');
        }
      } finally {
        if (!cancelled) {
          setLoadingInit(false);
        }
      }
    };

    const refreshCluster = async () => {
      try {
        const data = await getClusterStatus();
        if (!cancelled) {
          setCluster(data);
        }
      } catch (e) {
        console.error('Error refrescando estado del cluster', e);
      }
    };

    const refreshRedis = async () => {
      try {
        const data = await getRedisMetrics();
        if (!cancelled) {
          setRedisMetrics(data);
        }
      } catch (e) {
        console.error('Error refrescando métricas de Redis', e);
      }
    };


    const refreshAudit = async () => {
      try {
        const data = await getLatestAuditEvents();
        if (!cancelled) {
          setAuditEvents(data);
        }
      } catch (e) {
        console.error('Error refrescando eventos de auditoría', e);
      }
    };

    // Carga inicial
    loadInitial();

    // Polling suave: cluster cada 5s, Redis y auditoría cada 10s
    const clusterInterval = window.setInterval(refreshCluster, 5000);
    const redisInterval = window.setInterval(refreshRedis, 10000);
    const auditInterval = window.setInterval(refreshAudit, 10000);

    return () => {
      cancelled = true;
      window.clearInterval(clusterInterval);
      window.clearInterval(redisInterval);
      window.clearInterval(auditInterval);
    };
  }, [token, role]);


  // WebSocket para métricas de sistema en “tiempo real”
  useEffect(() => {
    if (!token || role !== 'admin') return;

    const ws = new WebSocket(
      `ws://localhost:8000/ws/metrics?token=${encodeURIComponent(token)}`
    );

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as SystemMetricsResponse;
        setSystemMetrics(data);
      } catch (err) {
        console.error('Error parseando mensaje WS metrics', err);
      }
    };

    ws.onerror = (event) => {
      console.error('Error en WebSocket /ws/metrics', event);
    };

    ws.onclose = () => {
      console.log('WebSocket /ws/metrics cerrado');
    };

    return () => {
      ws.close();
    };
  }, [token, role]);

  const handleRunBenchmark = async (payload: MetricsBenchmarkRequest) => {
    try {
      setLoadingBenchmark(true);
      setError(null);

      const res = await runBenchmark(payload);
      setBenchmark(res);

      // Después del benchmark, refrescamos métricas de sistema y Redis por HTTP
      const [sys, redis] = await Promise.all([
        getSystemMetrics(),
        getRedisMetrics(),
      ]);
      setSystemMetrics(sys);
      setRedisMetrics(redis);
    } catch (e: any) {
      console.error(e);
      setError('Error ejecutando benchmark.');
    } finally {
      setLoadingBenchmark(false);
    }
  };

  if (loadingInit && !cluster && !systemMetrics && !redisMetrics) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3} sx={{ width: '100%' }}>
      {error && (
        <Grid size={12}>
          <Alert severity="error">{error}</Alert>
        </Grid>
      )}

      <Grid size={{ xs: 12, md: 6 }}>
        <ClusterStatusCard data={cluster} />
      </Grid>

      <Grid size={{ xs: 12, md: 6 }}>
        <RedisStatusCard metrics={redisMetrics} />
      </Grid>

      <Grid size={12}>
        <MetricsForm onRun={handleRunBenchmark} loading={loadingBenchmark} />
      </Grid>

      <Grid size={12}>
        <MetricsResultCard benchmark={benchmark} system={systemMetrics} />
        <MetricsChart benchmark={benchmark} />
      </Grid>
      <Grid size={12}>
        <AuditLog events={auditEvents} />
      </Grid>
    </Grid>
  );
};

export default AdminPage;
