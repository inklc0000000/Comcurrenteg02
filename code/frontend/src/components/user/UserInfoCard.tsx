// src/components/user/UserInfoCard.tsx
import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import type { User } from '../../api/types';

interface Props {
  user: User;
}

const UserInfoCard: React.FC<Props> = ({ user }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Información del usuario
        </Typography>
        <Typography>Nombre: {user.name}</Typography>
        <Typography>ID: {user.user_id}</Typography>
        <Typography>Género: {user.gender}</Typography>
        <Typography>Edad: {user.age}</Typography>
        <Typography>Ocupación: {user.occupation}</Typography>
        <Typography>Código postal: {user.zip_code}</Typography>
      </CardContent>
    </Card>
  );
};

export default UserInfoCard;
