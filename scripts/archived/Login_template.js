import React from 'react';
import { Box, Card, CardContent, Typography, Button, TextField, Checkbox, FormControlLabel } from '@mui/material';
import { Link } from 'react-router-dom';

const LoginTemplate = ({ formData, handleChange, handleSubmit, errors }) => {
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
        left: 0
      }}
    >
      <Card sx={{ bgcolor: '#2b2b2c', p: 3, borderRadius: 2, maxWidth: 400, width: '100%' }}>
        <CardContent>
          <Typography variant="h4" sx={{ color: 'white', textAlign: 'center', fontWeight: 'bold', mb: 3 }}>
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
              sx={{ 
                bgcolor: 'white', 
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1
                }
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
              sx={{ 
                bgcolor: 'white', 
                borderRadius: 1,
                '& .MuiInputLabel-root': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1
                },
                '& .MuiInputLabel-shrink': {
                  bgcolor: 'white',
                  px: 1,
                  borderRadius: 1
                }
              }}
            />
            <FormControlLabel
              control={<Checkbox checked={formData.remember_me} onChange={handleChange} name="remember_me" sx={{ color: 'white' }} />}
              label={<Typography sx={{ color: 'white' }}>Remember Me</Typography>}
            />
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
            Don&apos;t have an account? <Link to="/signup" style={{ color: '#ff8500' }}>Sign up</Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default LoginTemplate;
