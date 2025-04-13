import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import './styles/GlobalBackground.css';
import './styles/MapOverrides.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { library } from '@fortawesome/fontawesome-svg-core';
import { fas } from '@fortawesome/free-solid-svg-icons';
import { createApiMonitor } from './utils/debugUtils';
import { QueryClient, QueryClientProvider } from 'react-query';

// Fix for passive wheel event listener warnings
// This patch makes wheel events passive by default to improve scrolling performance
// Particularly needed for ArcGIS JavaScript API
(function() {
  if (typeof window !== 'undefined') {
    // Store reference to the original method
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    
    // Override the method with our custom implementation
    EventTarget.prototype.addEventListener = function(type, listener, options) {
      // For wheel, touchstart and touchmove events, make them passive by default
      if (type === 'wheel' || type === 'mousewheel' || type === 'touchstart' || type === 'touchmove') {
        let newOptions = options;
        
        // If options is a boolean, convert it to an object
        if (typeof options === 'boolean') {
          newOptions = {
            capture: options,
            passive: true
          };
        } else if (typeof options === 'object') {
          // If options is already an object, ensure passive is true unless explicitly set to false
          newOptions = {
            ...options,
            passive: options.passive === false ? false : true
          };
        } else {
          // If no options provided, make it passive
          newOptions = {
            passive: true
          };
        }
        
        // Call the original method with our modified options
        return originalAddEventListener.call(this, type, listener, newOptions);
      }
      
      // For all other event types, use the original behavior
      return originalAddEventListener.call(this, type, listener, options);
    };
    
    console.log('Applied passive wheel event listener fix for performance improvement');
    
    // Improved requestAnimationFrame optimization for ArcGIS
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const originalCancelAnimationFrame = window.cancelAnimationFrame; // Store the original cancelAnimationFrame
    const callbacks = new Map(); // Use Map to store callbacks with their IDs
    let nextCallbackId = 1;
    let isProcessing = false;
    let lastFrameTime = 0;
    
    // Detect if a callback is from the ArcGIS API based on the stack trace
    const isArcGISCallback = () => {
      try {
        const stackError = new Error();
        const stack = stackError.stack || '';
        return stack.includes('esri') || 
               stack.includes('arcgis') || 
               stack.includes('MapView') || 
               stack.includes('SceneView');
      } catch (e) {
        return false;
      }
    };
    
    // Process callbacks with a frame budget
    const processCallbacks = (timestamp) => {
      isProcessing = true;
      
      // Calculate available time for this frame (target 16ms for 60fps)
      const frameBudget = 12; // ms - leave some headroom
      const startTime = performance.now();
      let timeElapsed = 0;
      
      // Get all current callbacks
      const currentCallbacks = Array.from(callbacks.entries());
      callbacks.clear();
      
      // Create high and low priority queues
      const highPriorityCallbacks = [];
      const lowPriorityCallbacks = [];
      
      // Split callbacks by priority
      currentCallbacks.forEach(([id, callback]) => {
        if (callback.isArcGIS) {
          lowPriorityCallbacks.push({ id, callback: callback.fn });
        } else {
          highPriorityCallbacks.push({ id, callback: callback.fn });
        }
      });
      
      // Process high priority callbacks first
      for (let i = 0; i < highPriorityCallbacks.length; i++) {
        const { callback } = highPriorityCallbacks[i];
        try {
          callback(timestamp);
        } catch (e) {
          console.error('Error in animation frame callback:', e);
        }
        
        // Check time budget
        timeElapsed = performance.now() - startTime;
        if (timeElapsed >= frameBudget) break;
      }
      
      // Process as many low priority callbacks as we can within the time budget
      let processed = 0;
      for (let i = 0; i < lowPriorityCallbacks.length; i++) {
        const { id, callback } = lowPriorityCallbacks[i];
        
        // Check if we're out of time budget
        timeElapsed = performance.now() - startTime;
        if (timeElapsed >= frameBudget) {
          // Reschedule remaining callbacks for next frame
          lowPriorityCallbacks.slice(i).forEach(({ id, callback }) => {
            callbacks.set(id, { fn: callback, isArcGIS: true });
          });
          processed = i;
          break;
        }
        
        try {
          callback(timestamp);
          processed++;
        } catch (e) {
          console.error('Error in animation frame callback:', e);
        }
      }
      
      // If we have more callbacks to process, schedule the next frame
      if (callbacks.size > 0) {
        originalRequestAnimationFrame(processCallbacks);
      } else {
        isProcessing = false;
      }
      
      // Log performance metrics in dev mode
      if (process.env.NODE_ENV === 'development' && lowPriorityCallbacks.length > 5) {
        console.debug(`RAF processed ${processed}/${lowPriorityCallbacks.length} low priority callbacks in ${timeElapsed.toFixed(2)}ms`);
      }
    };
    
    // Override requestAnimationFrame
    window.requestAnimationFrame = function(callback) {
      const id = nextCallbackId++;
      
      // Detect if this is an ArcGIS API callback
      const isArcGIS = isArcGISCallback();
      
      // Store callback with metadata
      callbacks.set(id, {
        fn: callback,
        isArcGIS
      });
      
      // Start processing if not already running
      if (!isProcessing) {
        isProcessing = true;
        originalRequestAnimationFrame(processCallbacks);
      }
      
      return id;
    };
    
    // Override cancelAnimationFrame to properly remove callbacks
    window.cancelAnimationFrame = function(id) {
      if (callbacks.has(id)) {
        callbacks.delete(id);
        return true;
      }
      return originalCancelAnimationFrame(id);
    };
    
    console.log('Applied advanced requestAnimationFrame optimization for better performance');
  }
})();

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
