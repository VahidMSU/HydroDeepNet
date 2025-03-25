import { useEffect } from 'react';
import SessionService from '../services/SessionService';

/**
 * Component to initialize and maintain session checks
 * This component doesn't render anything
 */
const SessionChecker = () => {
  useEffect(() => {
    // Start session monitoring when component mounts
    SessionService.startSessionMonitor();

    // Perform an immediate check
    SessionService.checkSession();

    // Clean up when component unmounts
    return () => {
      SessionService.stopSessionMonitor();
    };
  }, []);

  // This component doesn't render anything
  return null;
};

export default SessionChecker;
