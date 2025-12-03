// src/router.tsx
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import LoginPage from './pages/LoginPage';
import UserPage from './pages/UserPage';
import AdminPage from './pages/AdminPage';
import { useUser } from './context/UserContext';

export const AppRouter: React.FC = () => {
  const { userId, role } = useUser();

  const hasValidUser = role === 'user' && userId !== null && userId > 0;
  const isAdmin = role === 'admin';

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginPage />} />

        <Route element={<MainLayout />}>
          <Route
            path="/user"
            element={hasValidUser ? <UserPage /> : <Navigate to="/" replace />}
          />
          <Route
            path="/admin"
            element={isAdmin ? <AdminPage /> : <Navigate to="/" replace />}
          />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
};
