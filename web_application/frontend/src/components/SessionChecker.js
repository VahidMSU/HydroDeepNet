import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import SessionService from '../services/SessionService';

/**
 * Component to initialize and maintain session checks
 * This component doesn't render anything
 */
const SessionChecker = () => {
  const location = useLocation();

  useEffect(() => {
    // Check if this is a Google OAuth redirect
    const checkOAuthRedirect = () => {
      const urlParams = new URLSearchParams(location.search);
      const isOAuthRedirect = urlParams.has('google_login') && urlParams.has('username');
      
      if (isOAuthRedirect) {
        console.log('SessionChecker detected OAuth redirect, processing...');
        SessionService.checkGoogleOAuthLogin(window.location.href);
      }
    };
    
    // First check for OAuth redirect
    checkOAuthRedirect();
    
    // Then start session monitoring
    SessionService.startSessionMonitor();

    // Perform an immediate session check
    SessionService.checkSession();

    // Clean up when component unmounts
    return () => {
      SessionService.stopSessionMonitor();
    };
  }, [location.search]);

  // This component doesn't render anything
  return null;
};

export default SessionChecker;
