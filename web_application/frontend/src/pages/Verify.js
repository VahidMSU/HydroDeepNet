import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Alert,
  CircularProgress,
} from '@mui/material';

const Verify = () => {
  const [verificationCode, setVerificationCode] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const [email, setEmail] = useState('');

  // Add useEffect to check for stored email from login redirect
  useEffect(() => {
    const storedEmail = localStorage.getItem('verificationEmail');
    if (storedEmail) {
      setEmail(storedEmail);
      // Clear it from localStorage after using it
      localStorage.removeItem('verificationEmail');
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');
    setLoading(true);

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, verification_code: verificationCode }),
      });

      const result = await response.json();

      if (response.ok) {
        setSuccessMessage(result.message || 'Verification successful!');
        setTimeout(() => {
          navigate('/login');
        }, 1000);
      } else {
        setErrorMessage(result.message || 'Verification failed. Please try again.');
      }
    } catch (error) {
      setErrorMessage('An error occurred. Please try again.');
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
          Email Verification
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
          Please enter the verification code we sent to your email.
        </Typography>

        {errorMessage && (
          <Alert
            severity="error"
            sx={{ 
              mb: 3,
              bgcolor: 'rgba(244, 67, 54, 0.1)',
              color: '#d32f2f',
              '& .MuiAlert-icon': {
                color: '#d32f2f'
              }
            }}
          >
            {errorMessage}
          </Alert>
        )}
        
        {successMessage && (
          <Alert
            severity="success"
            sx={{ 
              mb: 3,
              bgcolor: 'rgba(76, 175, 80, 0.1)',
              color: '#388e3c',
              '& .MuiAlert-icon': {
                color: '#388e3c'
              }
            }}
          >
            {successMessage}
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
          
          <TextField
            margin="normal"
            required
            fullWidth
            id="verification_code"
            label="Verification Code"
            name="verification_code"
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value)}
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
            {loading ? <CircularProgress size={24} /> : 'Verify Email'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default Verify;
