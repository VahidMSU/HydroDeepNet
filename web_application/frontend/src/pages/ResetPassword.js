import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Alert,
  CircularProgress,
  Link,
} from '@mui/material';
import { resetPassword } from '../services/api';

const ResetPassword = () => {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [token, setToken] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  // Extract token from URL when component mounts
  useEffect(() => {
    const query = new URLSearchParams(location.search);
    const tokenParam = query.get('token');

    if (tokenParam) {
      setToken(tokenParam);
    } else {
      setMessage({
        type: 'error',
        text: 'Missing password reset token. Please check your email link and try again.',
      });
    }
  }, [location]);

  const validatePassword = () => {
    // Password must be at least 8 characters with letters and numbers
    if (password.length < 8) {
      setMessage({ type: 'error', text: 'Password must be at least 8 characters' });
      return false;
    }

    // Check if passwords match
    if (password !== confirmPassword) {
      setMessage({ type: 'error', text: 'Passwords do not match' });
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!token) {
      setMessage({ type: 'error', text: 'Missing reset token' });
      return;
    }

    if (!validatePassword()) {
      return;
    }

    setLoading(true);
    setMessage({ type: '', text: '' });

    try {
      const response = await resetPassword({ token, password });

      if (response.data.success) {
        setMessage({
          type: 'success',
          text:
            response.data.message || 'Password has been reset successfully. You can now log in.',
        });

        // Redirect to login page after a brief delay
        setTimeout(() => {
          navigate('/login');
        }, 3000);
      } else {
        setMessage({
          type: 'error',
          text: response.data.message || 'Failed to reset password. Please try again.',
        });
      }
    } catch (error) {
      console.error('Password reset error:', error);
      if (error.response) {
        setMessage({
          type: 'error',
          text: error.response.data.message || 'Error resetting password. Please try again.',
        });
      } else {
        setMessage({
          type: 'error',
          text: 'An unexpected error occurred. Please try again later.',
        });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        bgcolor: '#2a2a2a', // Dark gray background
        height: '100vh',
        width: '100vw',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        px: 3,
        overflow: 'hidden',
        position: 'fixed',
        top: 0,
        left: 0,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          bgcolor: '#f5f5f5', // Light gray paper background
          p: 3.5,
          borderRadius: 2,
          maxWidth: 420,
          width: '100%',
          boxShadow: '0 6px 18px rgba(0, 0, 0, 0.2)', // Darker shadow
        }}
      >
        <Typography 
          component="h1" 
          variant="h4" 
          align="center" 
          gutterBottom
          sx={{ 
            color: '#ffffff', 
            fontWeight: 'bold',
            mb: 3 
          }}
        >
          Reset Your Password
        </Typography>

        {message.text && (
          <Alert
            severity={message.type}
            sx={{ 
              mb: 3,
              ...(message.type === 'error' && {
                bgcolor: 'rgba(244, 67, 54, 0.1)',
                color: '#d32f2f',
                '& .MuiAlert-icon': {
                  color: '#d32f2f'
                }
              }),
              ...(message.type === 'success' && {
                bgcolor: 'rgba(76, 175, 80, 0.1)',
                color: '#388e3c',
                '& .MuiAlert-icon': {
                  color: '#388e3c'
                }
              })
            }}
            onClose={() => setMessage({ type: '', text: '' })}
          >
            {message.text}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit} noValidate>
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="New Password"
            type="password"
            id="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                bgcolor: '#ffffff',
                '& fieldset': {
                  borderColor: '#cccccc',
                },
                '&:hover fieldset': {
                  borderColor: '#ff6b00',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#ff6b00',
                },
              },
              '& .MuiInputLabel-root': {
                color: '#555555',
              },
              '& .MuiInputBase-input': {
                color: '#333333',
              },
            }}
          />

          <TextField
            margin="normal"
            required
            fullWidth
            name="confirmPassword"
            label="Confirm New Password"
            type="password"
            id="confirmPassword"
            autoComplete="new-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            disabled={loading}
            sx={{
              mb: 2,
              '& .MuiOutlinedInput-root': {
                bgcolor: '#ffffff',
                '& fieldset': {
                  borderColor: '#cccccc',
                },
                '&:hover fieldset': {
                  borderColor: '#ff6b00',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#ff6b00',
                },
              },
              '& .MuiInputLabel-root': {
                color: '#555555',
              },
              '& .MuiInputBase-input': {
                color: '#333333',
              },
            }}
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ 
              bgcolor: '#ff6b00', // Orange button
              color: '#ffffff',
              p: '10px',
              fontWeight: 'bold',
              '&:hover': {
                bgcolor: '#e06000', // Darker orange on hover
              },
              mb: 2,
            }}
            disabled={loading || !token}
          >
            {loading ? <CircularProgress size={24} /> : 'Reset Password'}
          </Button>

          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Link 
              href="#" 
              variant="body2" 
              onClick={() => navigate('/login')}
              sx={{
                color: '#ff6b00',
                textDecoration: 'none',
                fontWeight: 500,
                '&:hover': {
                  textDecoration: 'underline',
                }
              }}
            >
              Remember your password? Sign in
            </Link>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default ResetPassword;
