// src/components/admin/ClusterStatusCard.tsx
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from '@mui/material';
import type { ClusterStatusResponse } from '../../api/types';

interface Props {
  data: ClusterStatusResponse | null;
}

const ClusterStatusCard: React.FC<Props> = ({ data }) => {
  if (!data) {
    return null;
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Estado del cluster de nodos ML
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          Última verificación: {new Date(data.checked_at).toLocaleString()}
        </Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dirección</TableCell>
              <TableCell>Estado</TableCell>
              <TableCell align="right">Latencia (ms)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.nodes.map((n) => (
              <TableRow key={n.addr}>
                <TableCell>{n.addr}</TableCell>
                <TableCell>{n.up ? 'UP' : 'DOWN'}</TableCell>
                <TableCell align="right">
                  {n.last_latency_ms != null
                    ? n.last_latency_ms.toFixed(2)
                    : '-'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};

export default ClusterStatusCard;
