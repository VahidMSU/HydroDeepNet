import axios from 'axios';

// Create a base axios instance with common configuration
const api = axios.create({
  baseURL: '/api', // Ensure all requests are prefixed with /api
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle authentication errors
    if (error.response && error.response.status === 401) {
      console.error('Authentication error:', error);
      // Redirect to login page or show authentication modal
      window.location.href = '/login';
    }

    // Log other errors
    console.error('API request error:', error);

    return Promise.reject(error);
  },
);

// API service methods

// Auth
export const signUp = (userData) => api.post('/signup', userData);
export const login = (credentials) => api.post('/login', credentials);
export const logout = () => api.post('/logout');
export const verifyEmail = (data) => api.post('/verify', data);

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
export const createModel = (modelSettings) => api.post('/api/model-settings', modelSettings);

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
