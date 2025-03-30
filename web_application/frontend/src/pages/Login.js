import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import { Link } from 'react-router-dom';

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    remember_me: false,
  });
  const [errors, setErrors] = useState({});

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    if (token) {
      navigate('/');
    }
  }, [navigate]);

  const handleChange = ({ target: { name, value, type, checked } }) => {
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
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

          {/* MSU NetID Login Button */}
          <Button
            fullWidth
            variant="contained"
            sx={{ bgcolor: '#ff8500', color: 'white', '&:hover': { bgcolor: '#e67e00' } }}
            href="/login?msu_oauth=True"
          >
            Login with MSU NetID
          </Button>

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
