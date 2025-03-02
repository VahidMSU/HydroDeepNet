import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';

/**
 * Converts a geometry to a GeoJSON-compatible format
 * @param {Object} geometry - ArcGIS geometry object
 * @returns {Object} Converted GeoJSON-like geometry
 */
export const convertToGeoJSON = (geometry) => {
  if (!geometry) return null;

  try {
    // Convert to geographic coordinates
    const geoGeom = webMercatorUtils.webMercatorToGeographic(geometry);
    if (!geoGeom) return null;

    let result = {
      type: geometry.type,
    };

    if (geometry.type === 'polygon' && geoGeom.rings) {
      result.rings = geoGeom.rings.map((ring) =>
        ring.map((coord) => [parseFloat(coord[0]).toFixed(6), parseFloat(coord[1]).toFixed(6)]),
      );
    } else if (geometry.type === 'extent') {
      result = {
        type: 'extent',
        xmin: parseFloat(geoGeom.xmin).toFixed(6),
        ymin: parseFloat(geoGeom.ymin).toFixed(6),
        xmax: parseFloat(geoGeom.xmax).toFixed(6),
        ymax: parseFloat(geoGeom.ymax).toFixed(6),
        spatialReference: { wkid: 4326 },
      };
    } else if (geometry.type === 'point') {
      result = {
        type: 'point',
        x: parseFloat(geoGeom.x).toFixed(6),
        y: parseFloat(geoGeom.y).toFixed(6),
        spatialReference: { wkid: 4326 },
      };
    }

    return result;
  } catch (error) {
    console.error('Error converting geometry to GeoJSON', error);
    return null;
  }
};

/**
 * Extracts a bounding box from a geometry
 * @param {Object} geometry - ArcGIS or GeoJSON geometry object
 * @returns {Object} Bounds object with min/max lat/lon values
 */
export const extractBounds = (geometry) => {
  if (!geometry) return null;

  try {
    if (geometry.type === 'extent') {
      return {
        min_latitude: geometry.ymin,
        max_latitude: geometry.ymax,
        min_longitude: geometry.xmin,
        max_longitude: geometry.xmax,
      };
    } else if (geometry.type === 'polygon' && geometry.rings && geometry.rings.length > 0) {
      // Flatten the coordinates
      const coordinates = geometry.rings[0];
      if (!coordinates || !coordinates.length) return null;

      const latitudes = coordinates.map((coord) => parseFloat(coord[1]));
      const longitudes = coordinates.map((coord) => parseFloat(coord[0]));

      return {
        min_latitude: Math.min(...latitudes).toFixed(6),
        max_latitude: Math.max(...latitudes).toFixed(6),
        min_longitude: Math.min(...longitudes).toFixed(6),
        max_longitude: Math.max(...longitudes).toFixed(6),
      };
    } else if (geometry.type === 'point') {
      // For points, create a small bounding box around the point
      const lat = parseFloat(geometry.y);
      const lon = parseFloat(geometry.x);
      const buffer = 0.001; // ~100m buffer

      return {
        min_latitude: (lat - buffer).toFixed(6),
        max_latitude: (lat + buffer).toFixed(6),
        min_longitude: (lon - buffer).toFixed(6),
        max_longitude: (lon + buffer).toFixed(6),
      };
    }

    return null;
  } catch (error) {
    console.error('Error extracting bounds from geometry', error);
    return null;
  }
};

/**
 * Safely copies a geometry to prevent reference issues
 * @param {Object} geometry - The geometry to clone
 * @returns {Object} A deep copy of the geometry
 */
export const cloneGeometry = (geometry) => {
  if (!geometry) return null;

  try {
    return JSON.parse(JSON.stringify(geometry));
  } catch (error) {
    console.error('Error cloning geometry', error);
    return null;
  }
};

/**
 * Checks if the geometry is valid for use in the application
 * @param {Object} geometry - The geometry to validate
 * @returns {Boolean} Whether the geometry is valid
 */
export const isValidGeometry = (geometry) => {
  if (!geometry || !geometry.type) return false;

  try {
    if (geometry.type === 'polygon') {
      return geometry.rings && geometry.rings.length > 0;
    } else if (geometry.type === 'extent') {
      return (
        typeof geometry.xmin !== 'undefined' &&
        typeof geometry.ymin !== 'undefined' &&
        typeof geometry.xmax !== 'undefined' &&
        typeof geometry.ymax !== 'undefined'
      );
    } else if (geometry.type === 'point') {
      return typeof geometry.x !== 'undefined' && typeof geometry.y !== 'undefined';
    }
    return false;
  } catch (error) {
    console.error('Error validating geometry', error);
    return false;
  }
};

export default {
  convertToGeoJSON,
  extractBounds,
  cloneGeometry,
  isValidGeometry,
};
