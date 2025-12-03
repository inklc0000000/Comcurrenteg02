// src/components/user/RecommendationsChart.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import type { MovieWithDetails } from '../../api/types';

type Props = {
  items: MovieWithDetails[];
};

const RecommendationsChart: React.FC<Props> = ({ items }) => {
  if (!items || items.length === 0) {
    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6">
            Recomendaciones (score)
          </Typography>
          <Typography variant="body2" color="text.secondary">
            No hay recomendaciones para graficar.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const data = items.map((it, idx) => ({
    idx: idx + 1,
    title: it.title || `Movie ${it.movie_id}`,
    score: it.score,
  }));

  return (
    <Card sx={{ mt: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recomendaciones (score)
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Cada barra corresponde a una de las recomendaciones Top-N actuales.
        </Typography>
        <Box sx={{ width: '100%', height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="idx" />
              <YAxis />
              <Tooltip
                formatter={(value: any) => [`${value}`, 'score']}
                labelFormatter={(label: any) => {
                  const index = Number(label) - 1;
                  const item = data[index];
                  return item ? `${label}. ${item.title}` : label;
                }}
              />
              <Bar dataKey="score" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default RecommendationsChart;
