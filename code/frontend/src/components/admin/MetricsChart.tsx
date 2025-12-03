// src/components/admin/MetricsChart.tsx
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import type { MetricsBenchmarkResponse } from '../../api/types';

interface Props {
  benchmark: MetricsBenchmarkResponse | null;
}

const MetricsChart: React.FC<Props> = ({ benchmark }) => {
  if (!benchmark) return null;

  const data = [
    { name: 'Promedio', ms: benchmark.latency_avg_ms },
    { name: 'Mínima', ms: benchmark.latency_min_ms },
    { name: 'Máxima', ms: benchmark.latency_max_ms },
  ];

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Latencias (ms)
        </Typography>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="ms" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

export default MetricsChart;
