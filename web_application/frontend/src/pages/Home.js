// pages/Home.js
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import HomeTemplate from '../components/templates/Home'; // Import the new HomeTemplate component

const Home = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Check for Google OAuth redirect parameters
    const queryParams = new URLSearchParams(location.search);
    const googleLogin = queryParams.get('google_login');
    const username = queryParams.get('username');

    if (googleLogin === 'success' && username) {
      console.log('Google login successful, setting auth state');
      
      // Set auth token to mark user as logged in
      localStorage.setItem('authToken', 'true');
      localStorage.setItem('username', username);
      
      // Remove query parameters from URL for cleanliness
      navigate('/', { replace: true });
    }
  }, [location.search, navigate]);

  return <HomeTemplate />;
};

export default Home;
