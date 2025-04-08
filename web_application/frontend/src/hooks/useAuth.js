import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import SessionService from '../services/SessionService';

/**
 * Custom hook to handle authentication state
 * @param {boolean} redirectToLogin - Whether to redirect to login if not authenticated
 * @returns {Object} Authentication state and helper functions
 */
const useAuth = (redirectToLogin = true) => {
  const [isAuthenticated, setIsAuthenticated] = useState(
    Boolean(localStorage.getItem('authToken'))
  );
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const checkAuthentication = async () => {
      setIsLoading(true);
      
      // First, check if there's a Google OAuth redirect in progress
      const urlParams = new URLSearchParams(location.search);
      const isOAuthRedirect = urlParams.has('google_login') && urlParams.has('username');
      
      if (isOAuthRedirect) {
        const googleLogin = urlParams.get('google_login');
        const username = urlParams.get('username');
        
        if (googleLogin === 'success' && username) {
          console.log('OAuth redirect detected, setting authentication state');
          localStorage.setItem('authToken', 'true');
          localStorage.setItem('username', username);
          localStorage.setItem('userName', username);
          
          setIsAuthenticated(true);
          setUser({ username });
          setIsLoading(false);
          
          // Clear the Google OAuth initiated flag
          sessionStorage.removeItem('google_oauth_initiated');
          return;
        }
      }
      
      // Check if we have an auth token
      const token = localStorage.getItem('authToken');
      if (token) {
        // Try to get user info from localStorage
        try {
          const username = localStorage.getItem('username');
          const userInfoStr = localStorage.getItem('userInfo');
          let userInfo = null;
          
          if (userInfoStr) {
            userInfo = JSON.parse(userInfoStr);
          }
          
          setUser(userInfo || { username });
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Error parsing user info:', error);
          setIsAuthenticated(false);
          if (redirectToLogin) {
            navigate('/login');
          }
        }
      } else {
        setIsAuthenticated(false);
        if (redirectToLogin) {
          navigate('/login');
        }
      }
      
      setIsLoading(false);
    };
    
    checkAuthentication();
  }, [navigate, location.search, redirectToLogin]);

  // Function to handle logout
  const logout = async () => {
    try {
      await fetch('/api/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } catch (error) {
      console.error('Error during logout:', error);
    } finally {
      // Clear local storage regardless of API success
      localStorage.removeItem('authToken');
      localStorage.removeItem('username');
      localStorage.removeItem('userName');
      localStorage.removeItem('userInfo');
      setIsAuthenticated(false);
      navigate('/login');
    }
  };

  // Function to check authentication status with backend
  const checkAuthStatus = async () => {
    try {
      const result = await SessionService.checkSession();
      setIsAuthenticated(result);
      return result;
    } catch (error) {
      console.error('Error checking authentication status:', error);
      setIsAuthenticated(false);
      return false;
    }
  };

  return {
    isAuthenticated,
    isLoading,
    user,
    logout,
    checkAuthStatus,
  };
};

export default useAuth;
