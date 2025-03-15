import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { library } from '@fortawesome/fontawesome-svg-core';
import { fas } from '@fortawesome/free-solid-svg-icons';
import { createApiMonitor } from './utils/debugUtils';
import { QueryClient, QueryClientProvider } from 'react-query';

// Add all solid icons to the library for use throughout the app
library.add(fas);

// Enable API monitoring in development mode
if (process.env.NODE_ENV === 'development') {
  console.log('API monitoring enabled for development');
  createApiMonitor();

  // Add a global error handler for promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled Promise Rejection:', event.reason);
  });

  // Log application version
  console.log(`Application Version: ${process.env.REACT_APP_VERSION || '1.0.0'}`);
}

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);

reportWebVitals();
