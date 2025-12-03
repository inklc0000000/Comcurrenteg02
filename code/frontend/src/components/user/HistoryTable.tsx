// src/components/user/HistoryTable.tsx
import React from 'react';
import {
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TableContainer,
  Typography,
} from '@mui/material';
import type { RecommendationRecord } from '../../api/types';

interface Props {
  history: RecommendationRecord[];
}

const HistoryTable: React.FC<Props> = ({ history }) => {
  if (!history || history.length === 0) {
    return (
      <Typography variant="body2">
        No hay historial de recomendaciones para este usuario.
      </Typography>
    );
  }

  return (
    <>
      <Typography variant="h6" sx={{ mt: 3 }} gutterBottom>
        Historial de recomendaciones
      </Typography>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Fecha</TableCell>
              <TableCell>Top-N</TableCell>
              <TableCell>N° ítems</TableCell>
              <TableCell>Nodo / cluster</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {history.map((rec, idx) => (
              <TableRow key={idx}>
                <TableCell>
                  {new Date(rec.created_at).toLocaleString()}
                </TableCell>
                <TableCell>{rec.top_n}</TableCell>
                <TableCell>{rec.items.length}</TableCell>
                <TableCell>{rec.node_addr}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
};

export default HistoryTable;
