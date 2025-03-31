import api from './api';

class SessionService {
  constructor() {
    this.checkInterval = null;
    this.checkIntervalTime = 120000; // Increase check interval to 2 minutes to reduce load
    this.consecutiveFailures = 0;
    this.maxConsecutiveFailures = 3; // Stop checking after 3 consecutive failures
    this.inProgressCheck = false; // Flag to prevent concurrent checks
    this.isMapInteractionActive = false; // Flag to pause checks during map interactions
    this.isSessionValid = true; // Assume session is valid initially
    this.lastCheckTime = Date.now();
    this.redirectDetected = false; // Flag to track if redirects were detected
    this.productionHost = 'ciwre-bae.campusad.msu.edu'; // Your production host
    this.serverUnavailable = false; // Flag to track server availability
    this.serverUnavailableRetryTime = 300000; // 5 minutes before retrying after server unavailable
  }

  /**
   * Starts the session monitoring
   * @param {number} checkIntervalMs - Milliseconds between session checks
   */
  startSessionMonitor(checkIntervalMs = 120000) {
    // Clear any existing interval
    this.stopSessionMonitor();

    // Reset state
    this.consecutiveFailures = 0;
    this.inProgressCheck = false;
    this.lastCheckTime = Date.now();
    this.redirectDetected = false;
    this.serverUnavailable = false;

    // Set the new interval time (minimum 60 seconds)
    this.checkIntervalTime = Math.max(60000, checkIntervalMs);

    // In production, use even longer intervals to reduce server load
    if (window.location.host === this.productionHost) {
      this.checkIntervalTime = Math.max(180000, this.checkIntervalTime); // 3+ minutes in production
    }

    // Start the new interval
    this.checkInterval = setInterval(() => {
      // Skip if:
      // 1. A check is already in progress
      // 2. During map interactions
      // 3. Less than 30 seconds since last check
      // 4. Redirect loop was detected previously
      // 5. Server is known to be unavailable
      const timeSinceLastCheck = Date.now() - this.lastCheckTime;
      if (
        this.inProgressCheck ||
        this.isMapInteractionActive ||
        timeSinceLastCheck < 30000 ||
        this.redirectDetected ||
        (this.serverUnavailable && timeSinceLastCheck < this.serverUnavailableRetryTime)
      ) {
        return;
      }

      this.checkSession().catch((err) => {
        if (err.code !== 'ERR_SESSION_CHECK_CANCELED') {
          console.error('Unhandled error in session check:', err);
        }
      });
    }, this.checkIntervalTime);

    console.log(`Session monitoring started with ${this.checkIntervalTime}ms interval`);

    // Add window event listeners for visibility and performance optimization
    this.addVisibilityListeners();
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

    // Remove visibility listeners
    this.removeVisibilityListeners();
  }

  /**
   * Add event listeners to handle page visibility changes
   */
  addVisibilityListeners() {
    // Handle page visibility change (pause checks when tab is hidden)
    document.addEventListener('visibilitychange', this.handleVisibilityChange);

    // Handle page unload
    window.addEventListener('beforeunload', this.handleBeforeUnload);
  }

  /**
   * Remove event listeners
   */
  removeVisibilityListeners() {
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    window.removeEventListener('beforeunload', this.handleBeforeUnload);
  }

  /**
   * Handler for visibility change
   */
  handleVisibilityChange = () => {
    if (document.visibilityState === 'visible') {
      // When tab becomes visible again, check if enough time has passed
      const timeSinceLastCheck = Date.now() - this.lastCheckTime;
      if (timeSinceLastCheck > 60000 && !this.redirectDetected) {
        // Only if 60+ seconds have passed
        this.checkSession().catch((err) => {
          if (err.code !== 'ERR_SESSION_CHECK_CANCELED') {
            console.error('Visibility change session check error:', err);
          }
        });
      }
    }
  };

  /**
   * Handler for page unload
   */
  handleBeforeUnload = () => {
    this.stopSessionMonitor();
  };

  /**
   * Flag to pause session checks during map interactions
   * @param {boolean} isActive - Whether map interaction is happening
   */
  setMapInteractionState(isActive) {
    // If state is not changing, do nothing
    if (this.isMapInteractionActive === isActive) {
      return;
    }

    this.isMapInteractionActive = isActive;

    // If interaction ended and it's been a while since last check
    if (!isActive && !this.redirectDetected) {
      const timeSinceLastCheck = Date.now() - this.lastCheckTime;
      if (timeSinceLastCheck > 180000) {
        // 3+ minutes
        setTimeout(() => {
          this.checkSession().catch((err) => {
            if (err.code !== 'ERR_SESSION_CHECK_CANCELED') {
              console.error('Post-map interaction session check error:', err);
            }
          });
        }, 2000);
      }
    }
  }

  /**
   * Checks if the current session is valid
   * @returns {Promise<boolean>} - True if session is valid
   */
  async checkSession() {
    // Prevent concurrent checks
    if (this.inProgressCheck) {
      return this.isSessionValid;
    }

    // Skip checks on redirect detection
    if (this.redirectDetected) {
      return this.isSessionValid;
    }

    this.inProgressCheck = true;
    this.lastCheckTime = Date.now();

    try {
      // Call the session validation endpoint with the correct path
      // Use a timeout parameter to help prevent caching
      await api.get(`/validate-session?_=${Date.now()}`);

      // On success, reset failure states
      this.consecutiveFailures = 0;
      this.serverUnavailable = false;
      this.inProgressCheck = false;
      this.isSessionValid = true;

      return true;
    } catch (error) {
      // If check was canceled, propagate the error
      if (error.code === 'ERR_SESSION_CHECK_CANCELED') {
        this.inProgressCheck = false;
        throw error;
      }

      console.log('Session check failed:', error);

      // Increment failures counter
      this.consecutiveFailures++;

      // Handle 503 Service Unavailable and other server errors
      if (
        error.response &&
        (error.response.status === 503 ||
          (error.response.status >= 500 && error.response.status < 600))
      ) {
        console.warn(
          `Server unavailable (${error.response.status}). Pausing session checks temporarily.`,
        );
        this.serverUnavailable = true;

        // Use exponential backoff for retries (max 30 minutes)
        const backoffMinutes = Math.min(30, Math.pow(2, this.consecutiveFailures - 1) * 5);
        this.serverUnavailableRetryTime = backoffMinutes * 60 * 1000;

        console.log(`Will retry in approximately ${backoffMinutes} minutes`);

        // Reset the in-progress flag but keep the session valid assumption
        this.inProgressCheck = false;
        return this.isSessionValid;
      }

      // Stop checking after too many consecutive failures
      if (this.consecutiveFailures >= this.maxConsecutiveFailures) {
        console.warn(
          `Stopping session monitoring after ${this.maxConsecutiveFailures} consecutive failures`,
        );
        this.stopSessionMonitor();
      }

      // Handle specific errors
      if (error.code === 'ERR_TOO_MANY_REDIRECTS') {
        console.warn('Session validation redirect loop detected - disabling checks');
        // Set the redirect detected flag to prevent future attempts
        this.redirectDetected = true;

        // Clear any problematic cookies that might be causing the redirect loop
        this.clearAllCookies();

        // Temporarily suspend session checks on redirect loops
        setTimeout(() => {
          this.redirectDetected = false; // Reset after timeout
          this.consecutiveFailures = 0;
          this.startSessionMonitor(this.checkIntervalTime * 2); // Restart with longer interval
        }, 600000); // 10 minutes

        this.stopSessionMonitor();
      }

      // Special handling for Network errors
      if (error.code === 'ERR_NETWORK') {
        console.warn('Network error during session check - will retry later');
        // For network errors, wait longer between retries
        setTimeout(() => {
          this.inProgressCheck = false;
        }, 5000);
        return this.isSessionValid;
      }

      // If we get a 401 or 403 error, the session is invalid
      if (error.response && (error.response.status === 401 || error.response.status === 403)) {
        this.handleInvalidSession();
        this.inProgressCheck = false;
        this.isSessionValid = false;
        return false;
      }

      this.inProgressCheck = false;
      // For other errors, maintain the current session validity state
      return this.isSessionValid;
    }
  }

  /**
   * Clear all cookies to help resolve redirect issues
   */
  clearAllCookies() {
    try {
      document.cookie.split(';').forEach(function (c) {
        document.cookie = c
          .replace(/^ +/, '')
          .replace(/=.*/, '=;expires=' + new Date().toUTCString() + ';path=/');
      });

      // Also try to clear cookies with specific paths
      const paths = ['/', '/api', '/static'];
      paths.forEach((path) => {
        document.cookie.split(';').forEach(function (c) {
          document.cookie = c
            .replace(/^ +/, '')
            .replace(/=.*/, '=;expires=' + new Date().toUTCString() + `;path=${path}`);
        });
      });

      console.log('Cleared cookies to help resolve redirect issues');
    } catch (e) {
      console.error('Error clearing cookies:', e);
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

    // Redirect to login page if not already there
    if (!window.location.pathname.includes('/login')) {
      window.location.href = '/login';
    }
  }

  /**
   * Force an immediate session check
   * Useful after certain key actions
   */
  forceSessionCheck() {
    // Only if not currently checking and not during map interaction
    if (!this.inProgressCheck && !this.isMapInteractionActive && !this.redirectDetected) {
      return this.checkSession();
    }
    return Promise.resolve(this.isSessionValid);
  }

  /**
   * Get the current session validity state without checking
   */
  get isValid() {
    return this.isSessionValid;
  }
}

export default new SessionService();
