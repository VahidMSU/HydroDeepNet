import React, { useState, useEffect } from 'react';
import { Box, Typography, Avatar } from '@mui/material';
import { Person as PersonIcon } from '@mui/icons-material';
import api from '../services/api';

const UserInfo = () => {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const response = await api.get('/validate-session');
        if (response.data && response.data.username) {
          setUsername(response.data.username);
        }
      } catch (error) {
        console.error('Error fetching user info:', error);
      } finally {
        setLoading(false);
      }
    };

    if (localStorage.getItem('authToken')) {
      fetchUserInfo();
    }
  }, []);

  if (!localStorage.getItem('authToken') || loading) {
    return null;
  }

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        p: 2,
        borderBottom: '1px solid #687891',
      }}
    >
      <Avatar
        sx={{
          bgcolor: '#ff8500',
          width: 40,
          height: 40,
          mr: 2,
        }}
      >
        <PersonIcon />
      </Avatar>
      <Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: 'white' }}>
          {username}
        </Typography>
        <Typography variant="caption" sx={{ color: '#adb5bd', display: 'block' }}>
          Logged In
        </Typography>
      </Box>
    </Box>
  );
};

export default UserInfo;
