// Debugging component to test station geometry loading
import React, { useState, useEffect } from 'react';

const MapLoadingDebugger = () => {
  const [debugLog, setDebugLog] = useState([]);
  const [stationCount, setStationCount] = useState(0);
  const [lastRefresh, setLastRefresh] = useState(null);

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
      refreshGeometries: async () => {
        addLog('Manually refreshing station geometries...');

        try {
          setLastRefresh(new Date().toISOString());
          const timestamp = new Date().getTime();

          // Make a direct fetch call with cache-busting headers
          const response = await fetch(`/api/get_station_geometries?_=${timestamp}`, {
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
          addLog(`Successfully fetched ${features.length} stations`);

          // Store in window for inspection
          window.stationGeometries = features;

          return features;
        } catch (error) {
          addLog(`Error refreshing geometries: ${error.message}`);
          throw error;
        }
      },
    };

    // Initial fetch to test connection
    window.mapDebug
      .refreshGeometries()
      .catch((err) => console.error('Initial geometry fetch error:', err));

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

      <div style={styles.stats}>Last refresh: {lastRefresh || 'Never'}</div>

      <div style={styles.buttons}>
        <button style={styles.button} onClick={() => window.mapDebug.refreshGeometries()}>
          Refresh Geometries
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
