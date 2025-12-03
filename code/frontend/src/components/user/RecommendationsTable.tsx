// src/components/user/RecommendationsTable.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import TableContainer from '@mui/material/TableContainer';
import Paper from '@mui/material/Paper';
import type { MovieWithDetails } from '../../api/types';

type Props = {
  items: MovieWithDetails[];
};

function prettyNode(addr: string): string {
  if (addr.endsWith(':9001')) return 'Nodo 1';
  if (addr.endsWith(':9002')) return 'Nodo 2';
  if (addr.endsWith(':9003')) return 'Nodo 3';
  return addr;
}

const RecommendationsTable: React.FC<Props> = ({ items }) => {
  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recomendaciones Top-N
        </Typography>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Movie ID</TableCell>
                <TableCell>Título</TableCell>
                <TableCell>Géneros</TableCell>
                <TableCell align="right">Score</TableCell>
                <TableCell>Nodos</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((it) => {
                const genres = it.genres?.join(', ') ?? '-';
                const nodes = (it.nodes ?? []).map(prettyNode).join(', ');
                return (
                  <TableRow key={it.movie_id}>
                    <TableCell>{it.movie_id}</TableCell>
                    <TableCell>{it.title}</TableCell>
                    <TableCell>{genres}</TableCell>
                    <TableCell align="right">
                      {it.score.toFixed(3)}
                    </TableCell>
                    <TableCell>{nodes || '-'}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default RecommendationsTable;
