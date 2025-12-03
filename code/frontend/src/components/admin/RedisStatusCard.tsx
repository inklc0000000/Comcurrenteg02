// src/components/admin/RedisStatusCard.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import LinearProgress from '@mui/material/LinearProgress';
import type { RedisMetricsResponse } from '../../api/types';

type Props = {
  metrics: RedisMetricsResponse | null;
};

const RedisStatusCard: React.FC<Props> = ({ metrics }) => {
  if (!metrics) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Redis – caché de recomendaciones
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No hay métricas disponibles todavía. Verifica que Redis esté levantado y que la API
            pueda conectarse.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const totalLookups =
    metrics.keyspace_hits + metrics.keyspace_misses;
  const hitRatio =
    totalLookups > 0 ? (metrics.keyspace_hits / totalLookups) * 100 : 0;
  const usedMB = metrics.used_memory_bytes / (1024 * 1024);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Redis – caché de recomendaciones
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Segundo nivel de almacenamiento para recomendaciones precalculadas.
        </Typography>

        <Box sx={{ mt: 1, mb: 1 }}>
          <Typography variant="body2">
            <strong>Claves en DB:</strong> {metrics.db_size}
          </Typography>
          <Typography variant="body2">
            <strong>Memoria usada:</strong> {usedMB.toFixed(1)} MB
          </Typography>
          <Typography variant="body2">
            <strong>Clientes conectados:</strong> {metrics.connected_clients}
          </Typography>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            Hit ratio de caché ({hitRatio.toFixed(1)}%)
          </Typography>
          <LinearProgress
            variant="determinate"
            value={Math.max(0, Math.min(100, hitRatio))}
          />
          <Typography variant="caption" color="text.secondary">
            Hits: {metrics.keyspace_hits} · Misses: {metrics.keyspace_misses}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RedisStatusCard;
