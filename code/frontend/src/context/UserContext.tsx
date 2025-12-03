import React, { createContext, useContext, useState } from 'react';
import { setAuthToken } from '../api/client';

type Role = 'user' | 'admin';

type UserContextType = {
  userId: number | null;
  setUserId: (id: number | null) => void;
  token: string | null;
  role: Role | null;
  setAuth: (token: string | null, role: Role | null) => void;
};

const UserContext = createContext<UserContextType | undefined>(undefined);

export const UserProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [userId, setUserId] = useState<number | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [role, setRole] = useState<Role | null>(null);

  const setAuth = (newToken: string | null, newRole: Role | null) => {
    setToken(newToken);
    setRole(newRole);
    setAuthToken(newToken);
  };

  const value: UserContextType = {
    userId,
    setUserId,
    token,
    role,
    setAuth,
  };

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
};

export function useUser(): UserContextType {
  const ctx = useContext(UserContext);
  if (!ctx) {
    throw new Error('useUser must be used dentro de UserProvider');
  }
  return ctx;
}
