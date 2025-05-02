import axios from 'axios';

// Create a base axios instance with common configuration
const api = axios.create({
  // Never use a base URL with /api in it to prevent duplications
  baseURL: '',
  headers: {
    'Content-Type': 'application/json',
  },
  // Add maximum redirects to prevent redirect loops
  maxRedirects: 5,
  // Add timeout to prevent long-hanging requests
  timeout: 30000,
  // Add retry logic for network errors
  retry: 3,
  retryDelay: 1000,
});

// Export API_URL with correct path
export const API_URL = '/api';

// Add request interceptor to ensure API path is correct and add retry logic
let isRetryingSession = false;

api.interceptors.request.use(
  (config) => {
    // Clone the config to avoid mutation issues
    const newConfig = { ...config };

    // Add retry counter if not exists
    if (newConfig.retry === undefined) {
      newConfig.retry = api.defaults.retry;
    }

    // Special handling for session validation to prevent redirect loops
    if (newConfig.url?.includes('validate-session') && isRetryingSession) {
      return Promise.reject({
        message: 'Session validation canceled to prevent loop',
        code: 'ERR_SESSION_CHECK_CANCELED',
      });
    }

    // Fix URL by ensuring it has exactly one /api prefix
    // First, remove any baseURL that might cause duplications
    const url = newConfig.url || '';

    // Always use absolute paths from the domain root to prevent nesting
    if (url.includes('/api/')) {
      // URL already has /api in it, use as is
      newConfig.url = url;
    } else if (url.startsWith('/api')) {
      // URL starts with /api, use as is
      newConfig.url = url;
    } else if (url.startsWith('http')) {
      // External URL, use as is
      newConfig.url = url;
    } else {
      // Add /api prefix for internal API calls
      newConfig.url = `/api${url.startsWith('/') ? '' : '/'}${url}`;
    }

    return newConfig;
  },
  (error) => Promise.reject(error),
);

// Add response interceptor for error handling and retries
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Get config and create a new one for retry
    const config = error.config || {};

    // Set retry count
    config.retry = config.retry ?? api.defaults.retry;
    config.retryCount = config.retryCount ?? 0;

    // Log all network errors for debugging
    if (error.code === 'ERR_NETWORK') {
      console.warn(`Network error for ${config.url}: ${error.message}`);
    }

    // Handle redirect errors - prevent cascading redirect failures
    if (error.code === 'ERR_TOO_MANY_REDIRECTS') {
      console.error('Too many redirects:', error);

      // Clear any problematic cookies that might be causing the loop
      try {
        document.cookie.split(';').forEach(function (c) {
          document.cookie = c
            .replace(/^ +/, '')
            .replace(/=.*/, '=;expires=' + new Date().toUTCString() + ';path=/');
        });
      } catch (e) {
        console.error('Error clearing cookies:', e);
      }

      // If this is a session validation request, flag it to prevent future attempts
      if (config.url?.includes('validate-session')) {
        isRetryingSession = true;
        // Reset after 5 minutes
        setTimeout(() => {
          isRetryingSession = false;
        }, 300000);
      }

      // Don't retry on redirect errors
      return Promise.reject(error);
    }

    // Handle network errors with retry logic
    if (error.code === 'ERR_NETWORK' && config.retryCount < config.retry) {
      // Increase the retry count
      config.retryCount += 1;

      // Create a new promise to handle retry
      return new Promise((resolve) => {
        console.log(`Retrying API request (${config.retryCount}/${config.retry})`, config.url);

        // Set retry flag for session validation
        if (config.url?.includes('validate-session')) {
          isRetryingSession = true;

          // Reset the flag after a timeout
          setTimeout(() => {
            isRetryingSession = false;
          }, 10000);
        }

        // Retry after delay with exponential backoff
        setTimeout(
          () => {
            resolve(api(config));
          },
          (config.retryDelay || 1000) * Math.pow(2, config.retryCount - 1),
        );
      });
    }

    // Handle authentication errors
    if (error.response && (error.response.status === 401 || error.response.status === 403)) {
      console.error('Authentication error:', error);
      // Clear authentication data
      localStorage.removeItem('authToken');
      localStorage.removeItem('username');
      localStorage.removeItem('userInfo');
      // Redirect to login page only if not already there
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  },
);

// API service methods
// Auth
export const signUp = (userData) => api.post('/signup', userData);
export const login = (credentials) => api.post('/login', credentials);
export const logout = () => api.post('/logout');
export const verifyEmail = (data) => api.post('/verify', data);
export const requestPasswordReset = (email) => api.post('/reset-password-request', { email });
export const resetPassword = (resetData) => api.post('/reset-password', resetData);

// User information - fixed path for validate-session (always start with /)
export const validateSession = () => api.get('/validate-session');

// User data
export const getUserDashboard = () => api.get('/user_dashboard');
export const getUserFiles = (subdir = '') => api.get('/user_files', { params: { subdir } });

// Visualizations
export const getOptions = () => api.get('/get_options');
export const getVisualizations = (name, ver, variable) =>
  api.get('/visualizations', {
    params: { NAME: name, ver, variable },
  });

// Station search
export const searchStation = (searchTerm) =>
  api.get('/search_site', {
    params: { search_term: searchTerm },
  });

// Station details
export const getStationCharacteristics = (stationNo) =>
  api.get('/get_station_characteristics', {
    params: { station: stationNo },
  });

// Model creation
export const createModel = (modelSettings) => api.post('/model-settings', modelSettings);

// Report generation
export const generateReport = (reportParams) => api.post('/generate_report', reportParams);
export const getReports = () => api.get('/get_reports');
export const getReportStatus = (reportId) => api.get(`/reports/${reportId}/status`);
export const downloadReport = (reportId) => api.get(`/reports/${reportId}/download`);
export const viewReport = (reportId, subpath = '') =>
  api.get(`/reports/${reportId}/view/${subpath}`);

// Hydro Geo Dataset
export const getHydroGeoVariables = () => api.get('/hydro_geo_dataset');
export const getHydroGeoSubvariables = (variable) =>
  api.get('/hydro_geo_dataset', {
    params: { variable },
  });
export const fetchHydroGeoData = (data) => api.post('/hydro_geo_dataset', data);

// Chatbot
export const initializeChatbot = (context = 'general') =>
  api.post('/chatbot/initialize', { context });
export const sendChatbotMessage = (message) => api.post('/chatbot', { message });

export default api;
