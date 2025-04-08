import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Alert,
  Divider,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CancelOutlinedIcon from '@mui/icons-material/CancelOutlined';
import GoogleIcon from '@mui/icons-material/Google';

const SignUp = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [errors, setErrors] = useState({});
  const [flashMessages, setFlashMessages] = useState([]);
  const [showPasswordRequirements, setShowPasswordRequirements] = useState(false);
  const [showUsernameRequirements, setShowUsernameRequirements] = useState(false);

  // Password requirements validation
  const passwordRequirements = [
    {
      label: 'At least 8 characters long',
      valid: formData.password.length >= 8,
    },
    {
      label: 'Contains at least one uppercase letter',
      valid: /[A-Z]/.test(formData.password),
    },
    {
      label: 'Contains at least one lowercase letter',
      valid: /[a-z]/.test(formData.password),
    },
    {
      label: 'Contains at least one number',
      valid: /\d/.test(formData.password),
    },
    {
      label: 'Contains at least one special character',
      valid: /[@#$^&*()_+={}[\]|\\:;"'<>,.?/~`-]/.test(formData.password),
    },
    {
      label: 'Passwords match',
      valid: formData.password === formData.confirmPassword && formData.password !== '',
    },
  ];

  // Username requirements validation
  const usernameRequirements = [
    {
      label: 'Contains only letters and numbers',
      valid: /^[a-zA-Z0-9]*$/.test(formData.username),
    },
    {
      label: 'Username is not empty',
      valid: formData.username.length > 0,
    },
  ];

  const handleChange = ({ target: { name, value } }) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Show password requirements when user starts typing in password field
    if (name === 'password' && !showPasswordRequirements) {
      setShowPasswordRequirements(true);
    }

    // Show username requirements when user starts typing in username field
    if (name === 'username' && !showUsernameRequirements) {
      setShowUsernameRequirements(true);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors({});

    const newErrors = {};

    // Validate username
    if (!formData.username) {
      newErrors.username = 'Username is required';
    } else if (!/^[a-zA-Z0-9]+$/.test(formData.username)) {
      newErrors.username = 'Username must contain only letters and numbers';
    }

    if (!formData.email) {
      newErrors.email = 'Email is required';
    }

    // Check all password requirements
    const invalidRequirements = passwordRequirements.filter((req) => !req.valid);
    if (invalidRequirements.length > 0) {
      newErrors.password = 'Password does not meet requirements';
    }

    if (Object.keys(newErrors).length) {
      setErrors(newErrors);
      return;
    }

    try {
      const response = await fetch('/api/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();
      if (response.ok) {
        setFlashMessages([{ category: 'info', text: result.message }]);
        // Store email in localStorage for verification page
        localStorage.setItem('verificationEmail', formData.email);
        navigate('/verify'); // Redirect to verification page after successful signup
      } else if (result.errors) {
        // Handle field-specific errors from the server
        setErrors(result.errors);
        setFlashMessages([{ category: 'error', text: result.message }]);
      } else {
        setFlashMessages([{ category: 'error', text: result.message }]);
      }
    } catch (error) {
      setFlashMessages([
        { category: 'error', text: 'An error occurred during signup. Please try again.' },
      ]);
    }
  };

  const handleGoogleSignUp = () => {
    // Record that we're initiating Google OAuth flow from signup page
    sessionStorage.setItem('google_oauth_initiated', 'true');
    sessionStorage.setItem('oauth_from_signup', 'true');
    // Redirect to Google OAuth endpoint
    window.location.href = '/api/login/google';
  };

  return (
    <Box
      sx={{
        bgcolor: '#444e5e',
        height: '100vh',
        width: '100vw',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        px: 3,
        overflow: 'auto',
        position: 'fixed',
        top: 0,
        left: 0,
      }}
    >
      <Card sx={{ bgcolor: '#2b2b2c', p: 3, borderRadius: 2, maxWidth: 500, width: '100%', my: 4 }}>
        <CardContent>
          <Typography
            variant="h4"
            sx={{ color: 'white', textAlign: 'center', fontWeight: 'bold', mb: 3 }}
          >
            Sign Up
          </Typography>

          {flashMessages.length > 0 && (
            <Box mb={2}>
              {flashMessages.map((msg, idx) => (
                <Alert key={idx} severity={msg.category === 'error' ? 'error' : 'info'}>
                  {msg.text}
                </Alert>
              ))}
            </Box>
          )}

          {/* Google Sign Up Button */}
          <Button
            fullWidth
            variant="contained"
            startIcon={<GoogleIcon />}
            onClick={handleGoogleSignUp}
            sx={{
              bgcolor: '#ffffff',
              color: '#757575',
              mb: 2,
              '&:hover': { bgcolor: '#f5f5f5' },
              fontWeight: 'bold',
            }}
          >
            Sign up with Google
          </Button>

          <Divider sx={{ my: 2, color: 'white' }}>
            <Typography variant="body2" sx={{ color: 'white', px: 1 }}>
              OR
            </Typography>
          </Divider>

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Username"
              variant="outlined"
              margin="dense"
              name="username"
              value={formData.username}
              onChange={handleChange}
              error={Boolean(errors.username)}
              helperText={errors.username || 'Username can only contain letters and numbers'}
              onFocus={() => setShowUsernameRequirements(true)}
              sx={{
                bgcolor: 'white',
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              }}
            />

            {showUsernameRequirements && (
              <Paper
                elevation={0}
                sx={{
                  p: 1,
                  mt: 1,
                  mb: 2,
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1,
                }}
              >
                <Typography variant="subtitle2" sx={{ color: 'white', mb: 1 }}>
                  Username requirements:
                </Typography>
                <List dense disablePadding>
                  {usernameRequirements.map((requirement, index) => (
                    <ListItem key={index} dense disablePadding sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        {requirement.valid ? (
                          <CheckCircleOutlineIcon color="success" fontSize="small" />
                        ) : (
                          <CancelOutlinedIcon color="error" fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            sx={{
                              color: requirement.valid ? '#4caf50' : '#f44336',
                            }}
                          >
                            {requirement.label}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            )}

            <TextField
              fullWidth
              label="Email"
              type="email"
              variant="outlined"
              margin="dense"
              name="email"
              value={formData.email}
              onChange={handleChange}
              error={Boolean(errors.email)}
              helperText={errors.email}
              sx={{
                bgcolor: 'white',
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              }}
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              variant="outlined"
              margin="dense"
              name="password"
              value={formData.password}
              onChange={handleChange}
              error={Boolean(errors.password)}
              helperText={errors.password}
              onFocus={() => setShowPasswordRequirements(true)}
              sx={{
                bgcolor: 'white',
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              }}
            />

            {showPasswordRequirements && (
              <Paper
                elevation={0}
                sx={{
                  p: 1,
                  mt: 1,
                  mb: 2,
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: 1,
                }}
              >
                <Typography variant="subtitle2" sx={{ color: 'white', mb: 1 }}>
                  Password must:
                </Typography>
                <List dense disablePadding>
                  {passwordRequirements.map((requirement, index) => (
                    <ListItem key={index} dense disablePadding sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        {requirement.valid ? (
                          <CheckCircleOutlineIcon color="success" fontSize="small" />
                        ) : (
                          <CancelOutlinedIcon color="error" fontSize="small" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            sx={{
                              color: requirement.valid ? '#4caf50' : '#f44336',
                            }}
                          >
                            {requirement.label}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            )}

            <TextField
              fullWidth
              label="Confirm Password"
              type="password"
              variant="outlined"
              margin="dense"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              error={Boolean(errors.confirmPassword)}
              helperText={errors.confirmPassword}
              sx={{
                bgcolor: 'white',
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1,
                },
              }}
            />
            <Button
              fullWidth
              type="submit"
              variant="contained"
              sx={{ bgcolor: '#ff8500', color: 'white', mt: 2, '&:hover': { bgcolor: '#e67e00' } }}
            >
              Sign Up
            </Button>
          </form>

          <Typography variant="body2" sx={{ color: 'white', textAlign: 'center', mt: 3 }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#ff8500' }}>
              Log in
            </Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SignUp;
