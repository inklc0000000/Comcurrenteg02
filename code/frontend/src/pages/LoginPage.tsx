import React, { useState } from 'react';
import { Box, Button, Card, CardContent, TextField, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { loginRequest } from '../api/client';

const LoginPage: React.FC = () => {
  const [userIdInput, setUserIdInput] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [loading, setLoading] = useState(false);
  const { setUserId, setAuth } = useUser();
  const navigate = useNavigate();

  const parseUserId = (): number | null => {
    if (!userIdInput.trim()) return null;
    const n = Number(userIdInput);
    if (!Number.isInteger(n) || n <= 0) {
      alert('El User ID debe ser un entero positivo');
      return null;
    }
    return n;
  };

  const handleUserPanel = async () => {
    const id = parseUserId();
    if (id === null) return;

    try {
      setLoading(true);
      const resp = await loginRequest(id, undefined);
      setUserId(resp.user_id);
      setAuth(resp.token, resp.role);
      navigate('/user');
    } catch (err) {
      console.error(err);
      alert('Error en login de usuario (revisa que el user_id exista en Mongo).');
    } finally {
      setLoading(false);
    }
  };

  const handleAdminPanel = async () => {
    if (!adminKey.trim()) {
      alert('Ingresa la Admin Key para acceder al panel admin');
      return;
    }

    try {
      setLoading(true);
      // Para admin no necesitamos user_id; puede ir null
      const resp = await loginRequest(null, adminKey);
      if (resp.role !== 'admin') {
        alert('La Admin Key no es válida');
        return;
      }
      // Para mostrar algo en el navbar ponemos user_id = 0
      setUserId(resp.user_id);
      setAuth(resp.token, resp.role);
      navigate('/admin');
    } catch (err) {
      console.error(err);
      alert('Error en login admin (revisa ADMIN_KEY en la API).');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: '#121212',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-start',
        pl: 4,
      }}
    >
      <Card sx={{ maxWidth: 380 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recomendador distribuido – Login simple
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Ingresa un <strong>user_id</strong> del dataset para acceder al panel de usuario
            o una Admin Key para el panel de administrador.
          </Typography>

          <TextField
            label="User ID"
            fullWidth
            margin="normal"
            value={userIdInput}
            onChange={(e) => setUserIdInput(e.target.value)}
            disabled={loading}
          />

          <TextField
            label="Admin Key (solo admin)"
            type="password"
            fullWidth
            margin="normal"
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            disabled={loading}
          />

          <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
            <Button
              variant="contained"
              onClick={handleUserPanel}
              disabled={loading}
            >
              Ir al panel de usuario
            </Button>
            <Button
              variant="outlined"
              onClick={handleAdminPanel}
              disabled={loading}
            >
              Ir al panel admin
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default LoginPage;
