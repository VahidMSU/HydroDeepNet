import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Checkbox,
  FormControlLabel,
  Divider,
} from '@mui/material';
import { Link } from 'react-router-dom';
import GoogleIcon from '@mui/icons-material/Google';

const Login = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    remember_me: false,
  });
  const [errors, setErrors] = useState({});
  const [errorMessage, setErrorMessage] = useState('');
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    // First check for existing auth token
    const checkAuth = async () => {
      setIsCheckingAuth(true);
      const token = localStorage.getItem('authToken');
      
      if (token) {
        console.log('Auth token found, redirecting to home');
        navigate('/');
        return;
      }
      
      // Check for error parameters in URL (for OAuth failures)
      const queryParams = new URLSearchParams(location.search);
      const error = queryParams.get('error');
      if (error === 'google_auth_failed') {
        setErrorMessage('Google authentication failed. Please try again.');
      } else if (error === 'user_creation_failed') {
        setErrorMessage('Failed to create user account. Please try another method.');
      }
      
      setIsCheckingAuth(false);
    };
    
    checkAuth();
  }, [navigate, location.search]);

  const handleChange = ({ target: { name, value, type, checked } }) => {
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isCheckingAuth) return; // Prevent submission during auth check
    
    setErrors({});

    const newErrors = {};
    if (!formData.username) {
      newErrors.username = 'Username is required';
    }
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    if (Object.keys(newErrors).length) {
      setErrors(newErrors);
      return;
    }

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
        credentials: 'include', // Include cookies in the request
      });

      const data = await response.json();
      console.log('Login response:', data);

      if (data.status === 'success') {
        // User is logged in successfully
        localStorage.setItem('username', data.user.username);
        localStorage.setItem('userInfo', JSON.stringify(data.user));
        // Store auth token to indicate logged-in state
        localStorage.setItem('authToken', 'true');
        navigate('/');
      } else {
        setErrors({ login: data.message || 'Login failed. Please try again.' });
        alert(data.message || 'Login failed. Please try again.');
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrors({ login: 'Login failed. Try again.' });
    }
  };

  const handleGoogleLogin = () => {
    // Record that we're initiating Google OAuth flow
    sessionStorage.setItem('google_oauth_initiated', 'true');
    // Redirect to Google OAuth endpoint
    window.location.href = '/api/login/google';
  };

  // If still checking auth, show a loading indicator
  if (isCheckingAuth) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div className="spinner"></div>
      </Box>
    );
  }

  // Normal render with login form
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
        overflow: 'hidden',
        position: 'fixed',
        top: 0,
        left: 0,
      }}
    >
      <Card sx={{ bgcolor: '#2b2b2c', p: 3, borderRadius: 2, maxWidth: 400, width: '100%' }}>
        <CardContent>
          <Typography
            variant="h4"
            sx={{ color: 'white', textAlign: 'center', fontWeight: 'bold', mb: 3 }}
          >
            Login
          </Typography>

          {/* Display error message if present */}
          {errorMessage && (
            <Typography variant="body2" sx={{ color: '#f44336', textAlign: 'center', mb: 2 }}>
              {errorMessage}
            </Typography>
          )}

          {/* OAuth Login Buttons */}
          <Button
            fullWidth
            variant="contained"
            startIcon={<GoogleIcon />}
            onClick={handleGoogleLogin}
            sx={{
              bgcolor: '#ffffff',
              color: '#757575',
              mb: 2,
              '&:hover': { bgcolor: '#f5f5f5' },
              fontWeight: 'bold',
            }}
          >
            Sign in with Google
          </Button>

          <Divider sx={{ my: 2, color: 'white' }}>
            <Typography variant="body2" sx={{ color: 'white', px: 1 }}>
              OR
            </Typography>
          </Divider>

          {/* Standard Login Form */}
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
              helperText={errors.username}
              inputProps={{
                autoComplete: 'username',
              }}
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
              inputProps={{
                autoComplete: 'current-password',
              }}
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
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={formData.remember_me}
                    onChange={handleChange}
                    name="remember_me"
                    sx={{ color: 'white' }}
                  />
                }
                label={<Typography sx={{ color: 'white' }}>Remember Me</Typography>}
              />
              <Link to="/forgot-password" style={{ color: '#ff8500', textDecoration: 'none' }}>
                <Typography variant="body2">Forgot Password?</Typography>
              </Link>
            </Box>
            <Button
              fullWidth
              type="submit"
              variant="contained"
              sx={{ bgcolor: '#ff8500', color: 'white', mt: 2, '&:hover': { bgcolor: '#e67e00' } }}
            >
              Login
            </Button>
          </form>

          <Typography variant="body1" sx={{ color: 'white', textAlign: 'center', my: 2 }}>
            - or -
          </Typography>


          {/* Sign Up Link */}
          <Typography variant="body2" sx={{ color: 'white', textAlign: 'center', mt: 3 }}>
            Don&apos;t have an account?{' '}
            <Link to="/signup" style={{ color: '#ff8500' }}>
              Sign up
            </Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Login;
