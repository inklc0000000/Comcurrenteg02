// src/pages/UserPage.tsx
import React, { useEffect, useState } from 'react';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';

import type {
  User as UserType,
  MovieWithDetails,
  RecommendationRecord,
} from '../api/types';
import { getUser, getRecommendations, getHistory } from '../api/client';
import { useUser } from '../context/UserContext';

import UserInfoCard from '../components/user/UserInfoCard';
import RecommendationsTable from '../components/user/RecommendationsTable';
import RecommendationsChart from '../components/user/RecommendationsChart';
import HistoryTable from '../components/user/HistoryTable';

const UserPage: React.FC = () => {
  const { userId } = useUser();

  const [user, setUser] = useState<UserType | null>(null);
  const [recs, setRecs] = useState<MovieWithDetails[]>([]);
  const [history, setHistory] = useState<RecommendationRecord[]>([]);
  const [topN, setTopN] = useState<number>(20); // valor por defecto
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Carga de datos del usuario + recomendaciones + historial
  useEffect(() => {
    if (userId == null) return;

    const fetchAll = async () => {
      try {
        setLoading(true);
        setError(null);

        const [userData, recData, historyData] = await Promise.all([
          getUser(userId),
          getRecommendations(userId, topN),
          getHistory(userId),
        ]);

        setUser(userData);
        setRecs(recData.items);
        setHistory(historyData);
      } catch (e: any) {
        console.error(e);
        setError('Error cargando datos del usuario. Revisa la API y Mongo.');
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, [userId, topN]);

  const handleTopNChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(e.target.value);
    if (!Number.isFinite(value)) return;
    const n = Math.floor(value);
    if (n <= 0) return;
    if (n > 500) {
      setTopN(500); // límite superior razonable
    } else {
      setTopN(n);
    }
  };

  if (userId == null) {
    return <Alert severity="warning">Debes iniciar sesión para ver este panel.</Alert>;
  }

  if (loading && !user && recs.length === 0 && history.length === 0) {
    return <CircularProgress />;
  }

  return (
    <Grid container spacing={3} sx={{ width: '100%' }}>
      {error && (
        <Grid size={12}>
          <Alert severity="error">{error}</Alert>
        </Grid>
      )}

      <Grid size={{ xs: 12, md: 4 }}>
        {user && <UserInfoCard user={user} />}
      </Grid>

      <Grid size={{ xs: 12, md: 8 }}>
        <Box
          sx={{
            mb: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <Typography variant="subtitle1">
            Recomendaciones Top N
          </Typography>
          <TextField
            label="Top N"
            type="number"
            size="small"
            value={topN}
            onChange={handleTopNChange}
            inputProps={{ min: 1, max: 500 }}
          />
          <Typography variant="body2" color="text.secondary">
            (por defecto 20, puedes cambiarlo)
          </Typography>
        </Box>

        <RecommendationsTable items={recs} />
        <RecommendationsChart items={recs} />
        <Box sx={{ mt: 3 }}>
          <HistoryTable history={history} />
        </Box>
      </Grid>
    </Grid>
  );
};

export default UserPage;
