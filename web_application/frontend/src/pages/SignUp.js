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
        bgcolor: '#2a2a2a',
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
      <Card sx={{ 
        bgcolor: '#f5f5f5',
        p: 3.5, 
        borderRadius: 2, 
        maxWidth: 450, 
        width: '100%', 
        my: 4,
        boxShadow: '0 6px 18px rgba(0, 0, 0, 0.2)'
      }}>
        <CardContent>
          <Typography
            variant="h4"
            sx={{ 
              color: '#ffffff', 
              textAlign: 'center', 
              fontWeight: 'bold', 
              mb: 3 
            }}
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
              color: '#000000',
              border: '1px solid #dddddd',
              mb: 2,
              '&:hover': { bgcolor: '#eeeeee' },
              fontWeight: 'bold',
            }}
          >
            Sign up with Google
          </Button>

          <Divider sx={{ my: 2, color: '#777777' }}>
            <Typography variant="body2" sx={{ color: '#777777', px: 1 }}>
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
                '& .MuiFormHelperText-root': {
                  color: errors.username ? '#f44336' : '#666666',
                }
              }}
            />

            {showUsernameRequirements && (
              <Paper
                elevation={0}
                sx={{
                  p: 1,
                  mt: 1,
                  mb: 2,
                  bgcolor: 'rgba(255, 107, 0, 0.1)',
                  border: '1px solid rgba(255, 107, 0, 0.2)',
                  borderRadius: 1,
                }}
              >
                <List dense disablePadding>
                  {usernameRequirements.map((req, idx) => (
                    <ListItem key={idx} disablePadding disableGutters sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        {req.valid ? (
                          <CheckCircleOutlineIcon fontSize="small" color="success" />
                        ) : (
                          <CancelOutlinedIcon fontSize="small" color="error" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={req.label}
                        primaryTypographyProps={{
                          fontSize: '0.85rem',
                          color: '#555555',
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            )}

            <TextField
              fullWidth
              label="Email"
              variant="outlined"
              margin="dense"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              error={Boolean(errors.email)}
              helperText={errors.email}
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
                '& .MuiFormHelperText-root': {
                  color: errors.email ? '#f44336' : '#666666',
                }
              }}
            />

            <TextField
              fullWidth
              label="Password"
              variant="outlined"
              margin="dense"
              name="password"
              type="password"
              value={formData.password}
              onChange={handleChange}
              error={Boolean(errors.password)}
              helperText={errors.password}
              onFocus={() => setShowPasswordRequirements(true)}
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
                '& .MuiFormHelperText-root': {
                  color: errors.password ? '#f44336' : '#666666',
                }
              }}
            />

            <TextField
              fullWidth
              label="Confirm Password"
              variant="outlined"
              margin="dense"
              name="confirmPassword"
              type="password"
              value={formData.confirmPassword}
              onChange={handleChange}
              error={Boolean(errors.confirmPassword)}
              helperText={errors.confirmPassword}
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
                '& .MuiFormHelperText-root': {
                  color: errors.confirmPassword ? '#f44336' : '#666666',
                }
              }}
            />

            {showPasswordRequirements && (
              <Paper
                elevation={0}
                sx={{
                  p: 1,
                  mt: 1,
                  mb: 2,
                  bgcolor: 'rgba(255, 107, 0, 0.1)',
                  border: '1px solid rgba(255, 107, 0, 0.2)',
                  borderRadius: 1,
                }}
              >
                <List dense disablePadding>
                  {passwordRequirements.map((req, idx) => (
                    <ListItem key={idx} disablePadding disableGutters sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 30 }}>
                        {req.valid ? (
                          <CheckCircleOutlineIcon fontSize="small" color="success" />
                        ) : (
                          <CancelOutlinedIcon fontSize="small" color="error" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={req.label}
                        primaryTypographyProps={{
                          fontSize: '0.85rem',
                          color: '#555555',
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            )}

            <Button
              fullWidth
              type="submit"
              variant="contained"
              sx={{
                bgcolor: '#ff6b00',
                color: '#ffffff',
                p: '10px',
                fontWeight: 'bold',
                '&:hover': {
                  bgcolor: '#e06000',
                },
                mt: 1,
              }}
            >
              Create Account
            </Button>

            <Box sx={{ textAlign: 'center', mt: 3 }}>
              <Typography sx={{ color: '#555555' }}>
                Already have an account?{' '}
                <Link to="/login" style={{ color: '#ff6b00', fontWeight: 500, textDecoration: 'none' }}>
                  Login
                </Link>
              </Typography>
            </Box>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SignUp;
