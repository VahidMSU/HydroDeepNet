import React, { useEffect, useState } from 'react';
import { Box, Avatar, Typography } from '@mui/material';

const UserInfo = ({ userName }) => {
  const [displayName, setDisplayName] = useState(userName || 'User');

  useEffect(() => {
    // Update displayName when userName prop changes
    if (userName && userName !== 'undefined') {
      setDisplayName(userName);
    } else {
      // If userName is not provided, try to get it from localStorage
      const storedName = localStorage.getItem('userName') || localStorage.getItem('username');
      if (storedName && storedName !== 'undefined') {
        setDisplayName(storedName);
      } else {
        // Last resort: try to parse the JWT token to get user info
        try {
          const token = localStorage.getItem('authToken');
          if (token) {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(
              atob(base64)
                .split('')
                .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join(''),
            );
            const payload = JSON.parse(jsonPayload);
            if (payload.name || payload.email || payload.sub) {
              const tokenName = payload.name || payload.email || payload.sub;
              setDisplayName(tokenName);
              localStorage.setItem('userName', tokenName);
            }
          }
        } catch (e) {
          console.error('Failed to extract user info from token', e);
        }
      }
    }
  }, [userName]);

  // Generate a color based on the username for the avatar
  const stringToColor = (string) => {
    let hash = 0;
    for (let i = 0; i < string.length; i++) {
      hash = string.charCodeAt(i) + ((hash << 5) - hash);
    }
    let color = '#';
    for (let i = 0; i < 3; i++) {
      const value = (hash >> (i * 8)) & 0xff;
      color += `00${value.toString(16)}`.slice(-2);
    }
    return color;
  };

  // Get initials from the username
  const getInitials = (name) => {
    if (!name) return '?';
    return name
      .split(' ')
      .map((part) => part[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', p: 2, mb: 2 }}>
      <Avatar
        sx={{
          bgcolor: displayName ? stringToColor(displayName) : '#687891',
          color: 'white',
          fontWeight: 'bold',
          width: 40,
          height: 40,
        }}
      >
        {getInitials(displayName)}
      </Avatar>
      <Box sx={{ ml: 2 }}>
        <Typography variant="body1" sx={{ fontWeight: 'bold', color: 'white' }}>
          {displayName}
        </Typography>
        <Typography variant="body2" sx={{ color: '#687891' }}>
          Online
        </Typography>
      </Box>
    </Box>
  );
};

export default UserInfo;
