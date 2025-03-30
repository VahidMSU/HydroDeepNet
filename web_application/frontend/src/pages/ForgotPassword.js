import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
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

const ForgetPassword = () => {
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
    <Container maxWidth="sm">
      <Box
        sx={{
          mt: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            width: '100%',
            borderRadius: 2,
          }}
        >
          <Typography component="h1" variant="h5" align="center" gutterBottom>
            Forgot Password
          </Typography>

          <Typography variant="body2" align="center" gutterBottom sx={{ mb: 3 }}>
            Enter your email address below, and we'll send you instructions to reset your password.
          </Typography>

          {message.text && (
            <Alert
              severity={message.type}
              sx={{ mb: 3 }}
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
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Send Reset Instructions'}
            </Button>

            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
              <Link href="#" variant="body2" onClick={() => navigate('/login')}>
                Remember your password? Sign in
              </Link>

              <Link href="#" variant="body2" onClick={() => navigate('/register')}>
                Don't have an account? Sign up
              </Link>
            </Box>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default ForgetPassword;
