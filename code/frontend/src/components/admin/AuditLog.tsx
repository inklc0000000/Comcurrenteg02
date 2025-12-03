// src/components/admin/AuditLog.tsx
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
import type { AuditEvent } from '../../api/types';

interface Props {
  events: AuditEvent[];
}

const AuditLog: React.FC<Props> = ({ events }) => {
  const hasEvents = events && events.length > 0;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Flujo end-to-end (bitácora)
        </Typography>

        {!hasEvents ? (
          <Typography variant="body2">
            No hay eventos de auditoría todavía.
          </Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Fecha / hora</TableCell>
                <TableCell>Tipo</TableCell>
                <TableCell>User ID</TableCell>
                <TableCell>Detalle</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {events.map((evt) => (
                <TableRow key={evt.id}>
                  <TableCell>
                    {new Date(evt.timestamp).toLocaleString()}
                  </TableCell>
                  <TableCell>{evt.type}</TableCell>
                  <TableCell>{evt.user_id ?? '-'}</TableCell>
                  <TableCell>{evt.detail}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
};

export default AuditLog;
