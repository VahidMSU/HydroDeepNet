// Debugging component to test station geometry loading
import React, { useState, useEffect } from 'react';

const MapLoadingDebugger = () => {
  const [debugLog, setDebugLog] = useState([]);
  const [stationCount, setStationCount] = useState(0);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [lastLoadSource, setLastLoadSource] = useState('Not loaded');

  // Add logging function that can be used by other components
  const addLog = (message) => {
    const timestamp = new Date().toISOString().split('T')[1].substring(0, 8);
    setDebugLog((prev) => [`[${timestamp}] ${message}`, ...prev.slice(0, 19)]);
  };

  // Expose the logging function globally for console access
  useEffect(() => {
    window.mapDebug = {
      log: addLog,
      clearLogs: () => setDebugLog([]),

      // Check the current state of station geometries in storage
      checkStoredGeometries: () => {
        try {
          // Check localStorage first (persistent)
          const localStorageData = localStorage.getItem('stationGeometriesPermanentCache');
          if (localStorageData) {
            const parsed = JSON.parse(localStorageData);
            const timestamp = new Date(parsed.timestamp).toLocaleString();
            const features = parsed.data || [];
            addLog(`LocalStorage: ${features.length} stations from ${timestamp}`);
            setStationCount(features.length);
            setLastLoadSource('localStorage');
            setLastRefresh(timestamp);
            return features;
          }

          // Then check sessionStorage (current session only)
          const sessionStorageData = sessionStorage.getItem('stationGeometriesPermanentCache');
          if (sessionStorageData) {
            const parsed = JSON.parse(sessionStorageData);
            const timestamp = new Date(parsed.timestamp).toLocaleString();
            const features = parsed.data || [];
            addLog(`SessionStorage: ${features.length} stations from ${timestamp}`);
            setStationCount(features.length);
            setLastLoadSource('sessionStorage');
            setLastRefresh(timestamp);
            return features;
          }

          // Check if we have data in the window object (global variable)
          if (window.stationGeometries && window.stationGeometries.length) {
            addLog(`Window object: ${window.stationGeometries.length} stations (in-memory)`);
            setStationCount(window.stationGeometries.length);
            setLastLoadSource('window.stationGeometries');
            return window.stationGeometries;
          }

          addLog('No stored station geometries found');
          return null;
        } catch (error) {
          addLog(`Error checking stored geometries: ${error.message}`);
          console.error('Error checking stored geometries:', error);
          return null;
        }
      },

      // Fetch stations from static file instead of API
      refreshGeometries: async () => {
        addLog('Manually fetching station geometries from static file...');

        try {
          const timestamp = new Date().getTime();
          // Use static file path instead of API endpoint
          const response = await fetch(`/static/stations.geojson?_=${timestamp}`, {
            headers: {
              'Cache-Control': 'no-cache, no-store, must-revalidate',
              Pragma: 'no-cache',
              Expires: '0',
            },
          });

          if (!response.ok) {
            throw new Error(`Failed to fetch geometries: ${response.status}`);
          }

          const data = await response.json();
          const features = data.features || [];
          setStationCount(features.length);
          setLastRefresh(new Date().toISOString());
          setLastLoadSource('static file (fresh load)');
          addLog(`Successfully loaded ${features.length} stations from static file`);

          // Store in localStorage for persistence
          try {
            const cacheData = {
              data: features,
              timestamp: Date.now(),
            };
            localStorage.setItem('stationGeometriesPermanentCache', JSON.stringify(cacheData));
            sessionStorage.setItem('stationGeometriesPermanentCache', JSON.stringify(cacheData));
            addLog('Saved stations to persistent storage');
          } catch (e) {
            addLog(`Error saving to storage: ${e.message}`);
          }

          // Store in window for inspection
          window.stationGeometries = features;

          return features;
        } catch (error) {
          addLog(`Error loading geometries: ${error.message}`);
          throw error;
        }
      },

      // Clear geometry caches
      clearGeometries: () => {
        try {
          localStorage.removeItem('stationGeometriesPermanentCache');
          sessionStorage.removeItem('stationGeometriesPermanentCache');
          delete window.stationGeometries;
          setStationCount(0);
          setLastRefresh(null);
          setLastLoadSource('cleared');
          addLog('All station geometry caches cleared');
        } catch (error) {
          addLog(`Error clearing geometries: ${error.message}`);
        }
      },
    };

    // Check for stored geometries rather than fetching automatically
    window.mapDebug.checkStoredGeometries();

    return () => {
      // Clean up global reference when component unmounts
      delete window.mapDebug;
    };
  }, []);

  // Simple styles for the debugger panel
  const styles = {
    container: {
      position: 'fixed',
      bottom: '10px',
      right: '10px',
      width: '300px',
      maxHeight: '300px',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      color: '#00ff00',
      fontFamily: 'monospace',
      fontSize: '12px',
      padding: '10px',
      borderRadius: '5px',
      zIndex: 10000,
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    },
    header: {
      display: 'flex',
      justifyContent: 'space-between',
      marginBottom: '5px',
    },
    title: {
      fontWeight: 'bold',
      color: '#ffffff',
    },
    stats: {
      marginBottom: '5px',
      fontSize: '11px',
    },
    log: {
      margin: '2px 0',
    },
    buttons: {
      display: 'flex',
      gap: '5px',
      marginBottom: '5px',
    },
    button: {
      backgroundColor: '#444',
      color: 'white',
      border: 'none',
      padding: '3px 8px',
      borderRadius: '3px',
      cursor: 'pointer',
      fontSize: '11px',
    },
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>Map Debugger</div>
        <div>{stationCount} stations</div>
      </div>

      <div style={styles.stats}>
        <div>Last refresh: {lastRefresh || 'Never'}</div>
        <div>Source: {lastLoadSource}</div>
      </div>

      <div style={styles.buttons}>
        <button style={styles.button} onClick={() => window.mapDebug.checkStoredGeometries()}>
          Check Cache
        </button>
        <button style={styles.button} onClick={() => window.mapDebug.refreshGeometries()}>
          Load Static File
        </button>
        <button style={styles.button} onClick={() => window.mapDebug.clearGeometries()}>
          Clear Cache
        </button>
        <button style={styles.button} onClick={() => setDebugLog([])}>
          Clear Log
        </button>
      </div>

      <div>
        {debugLog.map((log, i) => (
          <div key={i} style={styles.log}>
            {log}
          </div>
        ))}
      </div>
    </div>
  );
};

export default MapLoadingDebugger;
