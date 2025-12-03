// src/components/admin/MetricsForm.tsx
import React, { useState } from 'react';
import { Card, CardContent, TextField, Button, Stack, Typography } from '@mui/material';
import type { MetricsBenchmarkRequest } from '../../api/types';

interface Props {
  onRun: (payload: MetricsBenchmarkRequest) => void;
  loading: boolean;
}

const MetricsForm: React.FC<Props> = ({ onRun, loading }) => {
  const [requests, setRequests] = useState('100');
  const [concurrency, setConcurrency] = useState('10');
  const [userId, setUserId] = useState('1');
  const [topN, setTopN] = useState('20');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const payload: MetricsBenchmarkRequest = {
      requests: Number(requests) || 100,
      concurrency: Number(concurrency) || 10,
      user_id: Number(userId) || 1,
      top_n: Number(topN) || 20,
    };
    onRun(payload);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Benchmark /metrics/benchmark
        </Typography>
        <form onSubmit={handleSubmit}>
          <Stack spacing={2}>
            <TextField
              label="Requests"
              type="number"
              value={requests}
              onChange={(e) => setRequests(e.target.value)}
            />
            <TextField
              label="Concurrency"
              type="number"
              value={concurrency}
              onChange={(e) => setConcurrency(e.target.value)}
            />
            <TextField
              label="User ID"
              type="number"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
            <TextField
              label="Top-N"
              type="number"
              value={topN}
              onChange={(e) => setTopN(e.target.value)}
            />
            <Button type="submit" variant="contained" disabled={loading}>
              {loading ? 'Ejecutando...' : 'Ejecutar benchmark'}
            </Button>
          </Stack>
        </form>
      </CardContent>
    </Card>
  );
};

export default MetricsForm;
