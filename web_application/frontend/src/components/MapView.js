///data/SWATGenXApp/codes/web_application/frontend/src/components/MapView.js
import React, { useEffect, useRef } from 'react';
import MapView from '@arcgis/core/views/MapView';
import Map from '@arcgis/core/Map';
import Graphic from '@arcgis/core/Graphic';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import esriConfig from '@arcgis/core/config';

// Initialize ArcGIS configuration once, outside of component
// This ensures it's only done once for the entire application
(function configureEsriPassiveEvents() {
  if (typeof window !== 'undefined') {
    // Set global dojoConfig before ArcGIS JS API loads
    window.dojoConfig = {
      ...window.dojoConfig,
      passiveEvents: true,
      has: {
        ...(window.dojoConfig?.has || {}),
        'esri-passive-events': true,
      },
    };

    // Configure esriConfig for passive events
    esriConfig.options = {
      ...esriConfig.options,
      events: {
        add: { passive: true },
        remove: { passive: true },
      },
    };
  }
})();

function EsriMap({ stationData }) {
  const mapRef = useRef(null);
  const viewRef = useRef(null);
  const graphicsLayerRef = useRef(null);
  const originalAddEventListenerRef = useRef(null);

  useEffect(() => {
    // Keep original addEventListener for restoration during cleanup
    if (!originalAddEventListenerRef.current) {
      originalAddEventListenerRef.current = EventTarget.prototype.addEventListener;

      // Override addEventListener to make touch/wheel events passive
      EventTarget.prototype.addEventListener = function (type, listener, options) {
        if (
          type === 'touchstart' ||
          type === 'touchmove' ||
          type === 'wheel' ||
          type === 'mousewheel'
        ) {
          // If options is a boolean (useCapture), convert to object
          if (typeof options === 'boolean') {
            options = {
              capture: options,
              passive: true,
            };
          } else if (typeof options === 'object' && options !== null) {
            options.passive = true;
          } else {
            options = { passive: true };
          }
        }
        return originalAddEventListenerRef.current.call(this, type, listener, options);
      };
    }

    // Set ArcGIS specific configurations
    esriConfig.request.timeout = 60000; // Set timeout to 60 seconds
    esriConfig.request.maxUrlLength = 2000; // Limit URL length

    // Enable ArcGIS API to use passive events
    if (!esriConfig.has) esriConfig.has = {};
    esriConfig.has['esri-passive-events'] = true;

    // Create a map with efficient basemap
    const map = new Map({
      basemap: 'topo-vector', // Use vector basemap for better performance
    });

    // Create a graphics layer for the station marker
    const graphicsLayer = new GraphicsLayer({
      elevationInfo: { mode: 'on-the-ground' }, // Keep markers on ground level for better performance
    });
    map.add(graphicsLayer);
    graphicsLayerRef.current = graphicsLayer;

    // Configure view for performance
    const view = new MapView({
      container: mapRef.current,
      map: map,
      zoom: 4,
      center: [-90, 38],
      constraints: {
        snapToZoom: false, // Improves performance by disabling snap-to-zoom
      },
      // Disable unnecessary UI components
      ui: {
        components: ['zoom'],
      },
      // Optimize rendering
      qualityProfile: 'high',
    });

    // Store view reference for cleanup
    viewRef.current = view;

    // Additional options to improve performance
    view.popup.defaultPopupTemplateEnabled = false; // Disable default popups

    // If navigation options are available, set passive event options
    if (view.navigation) {
      if (typeof view.navigation.mouseWheelEventOptions === 'object') {
        view.navigation.mouseWheelEventOptions.passive = true;
      }
      if (typeof view.navigation.browserTouchPanEventOptions === 'object') {
        view.navigation.browserTouchPanEventOptions.passive = true;
      }
    }

    // If station data is available, add it to the map
    if (stationData?.Latitude && stationData?.Longitude) {
      // Clear previous graphics first
      graphicsLayer.removeAll();

      // Create station marker
      const point = {
        type: 'point',
        longitude: stationData.Longitude,
        latitude: stationData.Latitude,
      };

      const marker = new Graphic({
        geometry: point,
        symbol: {
          type: 'simple-marker',
          color: [226, 119, 40],
          size: '12px',
          outline: {
            color: [255, 255, 255],
            width: 1,
          },
        },
        attributes: {
          name: stationData.StationName || 'Station',
          id: stationData.StationID || '',
        },
      });

      graphicsLayer.add(marker);

      // Wait for the view to be ready before attempting navigation
      view.when(() => {
        // Wrap goTo in a try/catch to prevent uncaught exceptions
        try {
          view
            .goTo(
              {
                center: [stationData.Longitude, stationData.Latitude],
                zoom: 10,
              },
              {
                duration: 1000, // Smooth animation
                easing: 'ease-in-out', // Smoother animation
              },
            )
            .catch((err) => {
              // Silently handle navigation errors
              console.warn('Navigation error:', err);
            });
        } catch (err) {
          console.warn('Error navigating to station:', err);
        }
      });
    }

    // Clean up function
    return () => {
      // Safely handle cleanup
      if (graphicsLayerRef.current) {
        try {
          graphicsLayerRef.current.removeAll();
        } catch (e) {
          console.warn('Error clearing graphics:', e);
        }
      }

      if (viewRef.current) {
        // Stop all animations before destroying
        try {
          viewRef.current.goTo && viewRef.current.goTo.cancel && viewRef.current.goTo.cancel();
        } catch (e) {
          // Ignore cancellation errors
        }

        // Use a timeout to ensure proper cleanup
        const viewToDestroy = viewRef.current;
        viewRef.current = null; // Clear reference first to prevent multiple destroy attempts

        setTimeout(() => {
          try {
            viewToDestroy && !viewToDestroy.destroyed && viewToDestroy.destroy();
          } catch (e) {
            console.warn('Error during view cleanup:', e);
          }
        }, 0);
      }

      // Restore original addEventListener if we have a reference to it
      if (originalAddEventListenerRef.current) {
        EventTarget.prototype.addEventListener = originalAddEventListenerRef.current;
      }
    };
  }, [stationData]);

  return <div ref={mapRef} style={{ height: '500px', border: '1px solid #ccc' }} />;
}

export default EsriMap;
