import api from './api';

class SessionService {
  constructor() {
    this.checkInterval = null;
    this.checkIntervalTime = 60000; // Check every minute by default
  }

  /**
   * Starts the session monitoring
   * @param {number} checkIntervalMs - Milliseconds between session checks
   */
  startSessionMonitor(checkIntervalMs = 60000) {
    // Clear any existing interval
    this.stopSessionMonitor();

    // Set the new interval time
    this.checkIntervalTime = checkIntervalMs;

    // Start the new interval
    this.checkInterval = setInterval(() => {
      this.checkSession();
    }, this.checkIntervalTime);

    console.log(`Session monitoring started with ${checkIntervalMs}ms interval`);
  }

  /**
   * Stops the session monitoring
   */
  stopSessionMonitor() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
      console.log('Session monitoring stopped');
    }
  }

  /**
   * Checks if the current session is valid
   * @returns {Promise<boolean>} - True if session is valid
   */
  async checkSession() {
    try {
      // Call the session validation endpoint with the correct path
      await api.get('/api/validate-session');
      return true;
    } catch (error) {
      console.log('Session check failed:', error);

      // If we get a 401 or 403 error, the session is invalid
      if (error.response && (error.response.status === 401 || error.response.status === 403)) {
        this.handleInvalidSession();
        return false;
      }

      // For other errors, we assume the session is still valid
      return true;
    }
  }

  /**
   * Handles the case where the session is invalid
   */
  handleInvalidSession() {
    console.warn('Session is invalid or expired, redirecting to login');

    // Clear any auth data
    localStorage.removeItem('authToken');
    localStorage.removeItem('username');
    localStorage.removeItem('userInfo');

    // Redirect to login page
    window.location.href = '/login';
  }
}

export default new SessionService();
