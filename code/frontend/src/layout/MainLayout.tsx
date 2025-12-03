// src/layout/MainLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Navbar from './Navbar';

const MainLayout: React.FC = () => {
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5' }}>
      <Navbar />
      <Container sx={{ mt: 3, mb: 4 }}>
        <Outlet />
      </Container>
    </Box>
  );
};

export default MainLayout;
