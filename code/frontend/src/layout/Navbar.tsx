// src/layout/Navbar.tsx
import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import { useLocation, useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';

const Navbar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { userId, role, setUserId, setAuth } = useUser();

  const isUserRoute = location.pathname.startsWith('/user');
  const isAdminRoute = location.pathname.startsWith('/admin');

  const canGoUser = role === 'user' && userId !== null && userId > 0;
  const canGoAdmin = role === 'admin';

  const labelRight =
    role === 'admin'
      ? 'ADMIN'
      : userId !== null && userId > 0
      ? `User ID: ${userId}`
      : '';

  const handleLogout = () => {
    setUserId(null);
    setAuth(null, null);
    navigate('/');
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          Recomendador Distribuido
        </Typography>

        <Box sx={{ mr: 2 }}>
          <Button
            color={isUserRoute ? 'secondary' : 'inherit'}
            onClick={() => navigate('/user')}
            disabled={!canGoUser}
          >
            Usuario
          </Button>
          <Button
            color={isAdminRoute ? 'secondary' : 'inherit'}
            onClick={() => navigate('/admin')}
            disabled={!canGoAdmin}
          >
            Admin
          </Button>
        </Box>

        {labelRight && (
          <Typography variant="body2" sx={{ mr: 2 }}>
            {labelRight}
          </Typography>
        )}

        <Button color="inherit" onClick={handleLogout}>
          Salir
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
