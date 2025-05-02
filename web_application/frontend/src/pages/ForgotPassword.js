import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { requestPasswordReset } from '../services/api';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email) {
      setMessage({ type: 'error', text: 'Please enter your email address' });
      return;
    }

    setLoading(true);
    setMessage({ type: '', text: '' });

    try {
      const response = await requestPasswordReset(email);

      if (response.data.success) {
        setMessage({
          type: 'success',
          text: 'Password reset instructions have been sent to your email',
        });
      } else {
        setMessage({
          type: 'error',
          text: response.data.message || 'Error sending reset instructions',
        });
      }
    } catch (error) {
      console.error('Reset password error:', error);
      if (error.response && error.response.status === 404) {
        setMessage({ type: 'error', text: 'Email address not found' });
      } else {
        setMessage({
          type: 'error',
          text: 'An error occurred. Please try again later.',
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
          Forgot Password
        </Typography>

        <Typography 
          variant="body2" 
          align="center" 
          gutterBottom 
          sx={{ 
            mb: 3,
            color: '#555555' 
          }}
        >
          Enter your email address below, and we'll send you instructions to reset your password.
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
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
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
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Send Reset Instructions'}
          </Button>

          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
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

            <Link 
              href="#" 
              variant="body2" 
              onClick={() => navigate('/signup')}
              sx={{
                color: '#ff6b00',
                textDecoration: 'none',
                fontWeight: 500,
                '&:hover': {
                  textDecoration: 'underline',
                }
              }}
            >
              Don't have an account? Sign up
            </Link>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default ForgotPassword;
