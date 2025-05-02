/**
 * Utility functions for debugging API issues
 */

/**
 * Logs a detailed object to the console with formatting
 * @param {String} label - Log label
 * @param {Object} data - Data to log
 */
export const debugLog = (label, data) => {
  const styles = 'color: #2196f3; font-weight: bold; font-size: 12px;';
  console.group(`%c${label}`, styles);
  console.log(data);
  console.groupEnd();
};

/**
 * Validates polygon coordinates before submission
 * @param {Object|String} polygonCoords - Polygon coordinates (object or string)
 * @returns {Object} - Validation result
 */
export const validatePolygonCoordinates = (polygonCoords) => {
  if (!polygonCoords) {
    return { valid: false, message: 'No polygon coordinates provided' };
  }

  try {
    // Parse if it's a string
    const coords = typeof polygonCoords === 'string' ? JSON.parse(polygonCoords) : polygonCoords;

    // Check if it's an array
    if (!Array.isArray(coords)) {
      return { valid: false, message: 'Polygon coordinates must be an array' };
    }

    // Check if it has at least 3 points to form a polygon
    if (coords.length < 3) {
      return { valid: false, message: 'Polygon must have at least 3 vertices' };
    }

    // Check if each coordinate has latitude and longitude
    for (const coord of coords) {
      if (!coord.latitude || !coord.longitude) {
        return { valid: false, message: 'Each coordinate must have latitude and longitude' };
      }
    }

    return { valid: true, coords };
  } catch (e) {
    return { valid: false, message: `Invalid polygon format: ${e.message}` };
  }
};

/**
 * Monitors the progress of API calls for debugging
 */
export const createApiMonitor = () => {
  const originalFetch = window.fetch;

  window.fetch = async function (...args) {
    const url = args[0];
    const options = args[1] || {};

    // Only monitor API calls
    if (typeof url === 'string' && (url.startsWith('/api/') || url === '/hydro_geo_dataset')) {
      // Sanitize request body to remove sensitive information before logging
      let sanitizedBody = null;
      if (options.body) {
        try {
          const bodyObj = JSON.parse(options.body);
          // Create sanitized copy without password
          const sanitized = { ...bodyObj };

          // Replace password with asterisks if it exists
          if (sanitized.password) {
            sanitized.password = '********';
          }

          sanitizedBody = sanitized;
        } catch (e) {
          sanitizedBody = 'Non-JSON body';
        }
      }

      debugLog(`API Request to ${url}`, {
        method: options.method || 'GET',
        headers: options.headers,
        body: sanitizedBody,
      });

      const startTime = Date.now();
      try {
        const response = await originalFetch.apply(this, args);

        // Clone the response to read it twice
        const clonedResponse = response.clone();

        // Try to parse as JSON, fall back to text
        let responseData;
        try {
          responseData = await clonedResponse.json();

          // Sanitize response if it contains sensitive information
          if (responseData && responseData.user && responseData.user.password) {
            responseData.user.password = '********';
          }
        } catch (e) {
          responseData = await clonedResponse.text();
        }

        debugLog(`API Response from ${url} (${Date.now() - startTime}ms)`, {
          status: response.status,
          statusText: response.statusText,
          data: responseData,
        });

        return response;
      } catch (e) {
        debugLog(`API Error for ${url}`, e);
        throw e;
      }
    }

    // Pass through for non-monitored calls
    return originalFetch.apply(this, args);
  };

  return () => {
    // Restore original fetch
    window.fetch = originalFetch;
  };
};
