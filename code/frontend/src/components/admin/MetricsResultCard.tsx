// src/components/admin/MetricsResultCard.tsx
import React from 'react';
import { Card, CardContent, Typography, Grid } from '@mui/material';
import type {
  MetricsBenchmarkResponse,
  SystemMetricsResponse,
} from '../../api/types';

interface Props {
  benchmark: MetricsBenchmarkResponse | null;
  system: SystemMetricsResponse | null;
}

const MetricsResultCard: React.FC<Props> = ({ benchmark, system }) => {
  if (!benchmark && !system) return null;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Resultados de métricas
        </Typography>

        <Grid container spacing={2}>
          {system && (
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1">Sistema /metrics/system</Typography>
              <Typography>CPU: {system.cpu_percent.toFixed(1)} %</Typography>
              <Typography>
                Goroutines: {system.num_goroutines}
              </Typography>
              <Typography>
                Nodos configurados: {system.num_nodes}
              </Typography>
              <Typography variant="body2">
                Timestamp: {new Date(system.timestamp).toLocaleString()}
              </Typography>
            </Grid>
          )}

          {benchmark && (
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1">
                Benchmark /metrics/benchmark
              </Typography>
              <Typography>
                Requests OK: {benchmark.ok} / {benchmark.total_requests}
              </Typography>
              <Typography>Errores: {benchmark.errors}</Typography>
              <Typography>
                Tiempo total: {benchmark.total_time_sec.toFixed(3)} s
              </Typography>
              <Typography>
                Latencia promedio: {benchmark.latency_avg_ms.toFixed(2)} ms
              </Typography>
              <Typography>
                Latencia min/max:{' '}
                {benchmark.latency_min_ms.toFixed(2)} /{' '}
                {benchmark.latency_max_ms.toFixed(2)} ms
              </Typography>
              <Typography>
                Throughput: {benchmark.rps.toFixed(2)} req/s
              </Typography>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MetricsResultCard;
