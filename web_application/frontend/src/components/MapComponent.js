//MapComponent.js
import React, {
  useEffect,
  useCallback,
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
} from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Sketch from '@arcgis/core/widgets/Sketch';
import Legend from '@arcgis/core/widgets/Legend';
import BasemapToggle from '@arcgis/core/widgets/BasemapToggle';
import Measurement from '@arcgis/core/widgets/Measurement';
import ScaleBar from '@arcgis/core/widgets/ScaleBar';
import Search from '@arcgis/core/widgets/Search';
import SnappingOptions from '@arcgis/core/views/interactive/snapping/SnappingOptions';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import '@arcgis/core/assets/esri/themes/light/main.css';
import Graphic from '@arcgis/core/Graphic';
import Polygon from '@arcgis/core/geometry/Polygon';
import Extent from '@arcgis/core/geometry/Extent';
import Point from '@arcgis/core/geometry/Point';
import { SimpleFillSymbol } from '@arcgis/core/symbols';
//import '../styles/map-widgets.css';

// Use forwardRef to fix the React ref warning
const MapComponent = forwardRef(
  (
    {
      setFormData,
      onGeometryChange,
      centerCoordinates,
      containerId = 'viewDiv',
      initialGeometry = null,
    },
    ref,
  ) => {
    const containerRef = useRef(null);
    const mapRef = useRef(null);
    const viewRef = useRef(null);
    const graphicsLayerRef = useRef(null);
    const sketchRef = useRef(null);
    const widgetsRef = useRef({});
    const [isLoading, setIsLoading] = useState(true);
    const [isDestroying, setIsDestroying] = useState(false);
    const hasDrawnInitialGeometry = useRef(false);
    const preventRefresh = useRef(false);
    const selectedGeometryRef = useRef(null);
    const zoomOperationInProgress = useRef(false);

    // Store form data updates locally to avoid re-renders
    const formDataUpdates = useRef({});

    // Expose methods to parent through ref
    useImperativeHandle(ref, () => ({
      panTo: (lat, lon) => {
        if (viewRef.current && !viewRef.current.destroyed) {
          viewRef.current.goTo({
            center: [parseFloat(lon), parseFloat(lat)],
          });
        }
      },
      clearGraphics: () => {
        if (graphicsLayerRef.current) {
          graphicsLayerRef.current.removeAll();
          selectedGeometryRef.current = null;
          formDataUpdates.current = {};

          // Use a batch update for form data to minimize re-renders
          setFormData((prev) => ({
            ...prev,
            min_longitude: '',
            min_latitude: '',
            max_longitude: '',
            max_latitude: '',
            geometry: null,
            geometry_type: null,
            polygon_coordinates: null,
          }));
        }
      },
      drawGeometry: (geometry) => {
        if (!graphicsLayerRef.current || !geometry) return;

        try {
          // Prevent re-renders during drawing
          preventRefresh.current = true;
          zoomOperationInProgress.current = true;

          // Clear existing graphics
          graphicsLayerRef.current.removeAll();

          // Store the geometry for persistence
          selectedGeometryRef.current = geometry;

          // Create appropriate graphic based on geometry type
          let graphic = null;

          if (geometry.type === 'polygon' && geometry.rings) {
            const polygon = new Polygon({
              rings: geometry.rings,
              spatialReference: { wkid: 4326 },
            });

            graphic = new Graphic({
              geometry: polygon,
              symbol: new SimpleFillSymbol({
                color: [255, 170, 0, 0.3],
                outline: { color: [255, 170, 0], width: 2 },
              }),
            });
          } else if (geometry.type === 'extent') {
            const extent = new Extent({
              xmin: geometry.xmin,
              ymin: geometry.ymin,
              xmax: geometry.xmax,
              ymax: geometry.ymax,
              spatialReference: { wkid: 4326 },
            });

            graphic = new Graphic({
              geometry: extent,
              symbol: new SimpleFillSymbol({
                color: [255, 170, 0, 0.3],
                outline: { color: [255, 170, 0], width: 2 },
              }),
            });
          } else if (geometry.type === 'point') {
            const point = new Point({
              x: geometry.x,
              y: geometry.y,
              spatialReference: { wkid: 4326 },
            });

            graphic = new Graphic({
              geometry: point,
              symbol: {
                type: 'simple-marker',
                style: 'circle',
                color: [255, 170, 0],
                size: '12px',
                outline: { color: [255, 255, 255], width: 1 },
              },
            });
          }

          // Add the graphic and zoom to it
          if (graphic && viewRef.current && !viewRef.current.destroyed) {
            graphicsLayerRef.current.add(graphic);

            // Simple zoom options that work well for all geometry types
            const zoomOptions = {
              duration: 700,
              easing: 'ease-in-out',
              padding: {
                top: 50,
                bottom: 50,
                left: 50,
                right: 50,
              },
            };

            // Use different scale for points
            if (geometry.type === 'point') {
              zoomOptions.scale = 50000;
              delete zoomOptions.padding;
            }

            // Perform the zoom
            viewRef.current
              .goTo(graphic.geometry, zoomOptions)
              .then(() => {
                // Release locks after animation completes
                setTimeout(() => {
                  zoomOperationInProgress.current = false;
                  preventRefresh.current = false;
                }, 100);
              })
              .catch((e) => {
                console.warn('Error zooming:', e);
                zoomOperationInProgress.current = false;
                preventRefresh.current = false;
              });
          } else {
            zoomOperationInProgress.current = false;
            preventRefresh.current = false;
          }

          // Safety timeout in case promises never resolve
          setTimeout(() => {
            zoomOperationInProgress.current = false;
            preventRefresh.current = false;
          }, 2000);
        } catch (e) {
          console.error('Error drawing geometry:', e);
          zoomOperationInProgress.current = false;
          preventRefresh.current = false;
        }
      },
      getSelectedGeometry: () => {
        return selectedGeometryRef.current;
      },
      getFormDataUpdates: () => {
        return formDataUpdates.current;
      },
    }));

    // Handler for sketch draw events - simplified to prevent redundant operations
    const handleDrawEvent = useCallback(
      (event) => {
        // Skip if conditions aren't right
        if (!event || isDestroying || zoomOperationInProgress.current) return;

        const { graphic, graphics, state } = event;
        let targetGraphic = graphic || (graphics && graphics[0]);

        if (!targetGraphic || !targetGraphic.geometry) return;

        // Set prevention flag for drawing operations
        if (state === 'start') preventRefresh.current = true;

        try {
          // Only handle complete state to reduce updates
          if (state === 'complete') {
            // Convert to geographic coordinates
            const geoGeom = webMercatorUtils.webMercatorToGeographic(targetGraphic.geometry);
            if (!geoGeom) return;

            // Save the geometry
            selectedGeometryRef.current = targetGraphic.geometry.toJSON();

            // Update form data based on geometry type
            let updates = {};

            if (targetGraphic.geometry.type === 'polygon') {
              const coordinates = (geoGeom.rings && geoGeom.rings[0]) || [];
              if (coordinates.length) {
                const latitudes = coordinates.map((coord) => parseFloat(coord[1]));
                const longitudes = coordinates.map((coord) => parseFloat(coord[0]));

                // Create properly formatted coordinate array for JSON
                const formattedCoordinates = coordinates.map((coord) => ({
                  longitude: parseFloat(coord[0]).toFixed(6),
                  latitude: parseFloat(coord[1]).toFixed(6),
                }));

                updates = {
                  min_latitude: Math.min(...latitudes).toFixed(6),
                  max_latitude: Math.max(...latitudes).toFixed(6),
                  min_longitude: Math.min(...longitudes).toFixed(6),
                  max_longitude: Math.max(...longitudes).toFixed(6),
                  polygon_coordinates: JSON.stringify(formattedCoordinates), // Properly stringify
                  geometry_type: 'polygon',
                };
              }
            } else if (targetGraphic.geometry.type === 'extent') {
              updates = {
                min_latitude: parseFloat(geoGeom.ymin).toFixed(6),
                max_latitude: parseFloat(geoGeom.ymax).toFixed(6),
                min_longitude: parseFloat(geoGeom.xmin).toFixed(6),
                max_longitude: parseFloat(geoGeom.xmax).toFixed(6),
                geometry_type: 'extent',
              };
            } else if (targetGraphic.geometry.type === 'point') {
              updates = {
                latitude: parseFloat(geoGeom.latitude).toFixed(6),
                longitude: parseFloat(geoGeom.longitude).toFixed(6),
              };
            }

            // Batch update to reduce re-renders
            setFormData((prev) => ({ ...prev, ...updates }));

            // Notify parent component
            if (onGeometryChange) {
              onGeometryChange(selectedGeometryRef.current);
            }

            // Allow map to refresh after a short delay
            setTimeout(() => {
              preventRefresh.current = false;
            }, 300);
          }
        } catch (e) {
          console.error('Error in handleDrawEvent:', e);
          preventRefresh.current = false;
        }
      },
      [onGeometryChange, setFormData, isDestroying],
    );

    // Safely remove a widget to avoid "undefined.on()" errors
    const safeRemoveWidget = useCallback((view, widget) => {
      if (!view || !widget || view.destroyed) return;

      try {
        // Only try to remove if the view is still valid
        view.ui.remove(widget);
      } catch (e) {
        console.warn(`Error removing widget: ${e.message}`);
      }
    }, []);

    // Cleanup function for ArcGIS objects
    const cleanupMap = useCallback(() => {
      console.log('Cleaning up ArcGIS resources');
      setIsDestroying(true);

      try {
        // Remove event handlers
        const handlers = viewRef.current?.eventHandlers || [];
        handlers.forEach((handler) => {
          if (handler && typeof handler.remove === 'function') {
            handler.remove();
          }
        });

        // Remove widgets from UI before destroying view
        const view = viewRef.current;
        const widgets = widgetsRef.current;

        if (view && !view.destroyed) {
          // Remove sketch widget first
          if (sketchRef.current) {
            safeRemoveWidget(view, sketchRef.current);
          }

          // Remove other widgets
          Object.values(widgets).forEach((widget) => {
            if (widget) safeRemoveWidget(view, widget);
          });

          // Destroy view
          try {
            view.container = null;
            view.destroy();
          } catch (e) {
            console.warn('Error destroying view:', e);
          }
        }

        // Clear all refs
        viewRef.current = null;
        mapRef.current = null;
        sketchRef.current = null;
        graphicsLayerRef.current = null;
        widgetsRef.current = {};
      } catch (e) {
        console.error('Error during ArcGIS cleanup:', e);
      } finally {
        setIsDestroying(false);
      }
    }, [safeRemoveWidget]);

    // Initialize map on mount and clean up on unmount
    useEffect(() => {
      let isMounted = true;
      let initTimer = null;
      let widgetTimer = null;

      const initializeMap = async () => {
        if (!containerRef.current || !isMounted || isDestroying) return;

        try {
          setIsLoading(true);

          // Clean up any existing map resources
          cleanupMap();

          // Create graphics layer
          const graphicsLayer = new GraphicsLayer({
            title: 'Drawing Layer',
            listMode: 'show',
          });
          graphicsLayerRef.current = graphicsLayer;

          // Create map
          const map = new Map({
            basemap: 'topo-vector',
            layers: [graphicsLayer],
          });
          mapRef.current = map;

          // Create view
          const view = new MapView({
            container: containerRef.current,
            map,
            center: [-85.6024, 44.3148], // Michigan
            zoom: 7,
            constraints: {
              snapToZoom: true,
              rotationEnabled: false,
            },
            popup: {
              dockEnabled: true,
              dockOptions: { position: 'bottom-right', breakpoint: false },
            },
          });
          viewRef.current = view;

          // Wait for view to be ready
          await view.when();

          if (!isMounted || !view || view.destroyed || isDestroying) {
            return;
          }

          // Create snapping options (only after view is ready)
          try {
            view.snappingOptions = new SnappingOptions({
              enabled: true,
              selfEnabled: true,
              featureSources: [{ layer: graphicsLayer }],
            });
          } catch (e) {
            console.warn('Error setting snapping options:', e);
          }

          // Initialize sketch widget
          try {
            const sketch = new Sketch({
              view,
              layer: graphicsLayer,
              creationMode: 'single',
              availableCreateTools: ['point', 'polygon', 'rectangle'],
              layout: 'vertical',
              visibleElements: { settingsMenu: false, undoRedoMenu: true },
            });
            sketchRef.current = sketch;
            view.ui.add(sketch, 'top-right');
          } catch (e) {
            console.warn('Error initializing sketch widget:', e);
          }

          if (!isMounted || view.destroyed || isDestroying) return;

          // Add event handlers
          const eventHandlers = [];

          // Map click handler
          try {
            const clickHandler = view.on('click', (event) => {
              if (!event.mapPoint || isDestroying || preventRefresh.current) return;
              const point = webMercatorUtils.webMercatorToGeographic(event.mapPoint);

              // Update locally stored data
              formDataUpdates.current = {
                latitude: point.latitude.toFixed(6),
                longitude: point.longitude.toFixed(6),
              };

              // Update form data
              setFormData((prev) => ({ ...prev, ...formDataUpdates.current }));
            });
            eventHandlers.push(clickHandler);
          } catch (e) {
            console.warn('Error setting click handler:', e);
          }

          // Sketch event handlers
          if (sketchRef.current) {
            try {
              const createHandler = sketchRef.current.on('create', handleDrawEvent);
              const updateHandler = sketchRef.current.on('update', handleDrawEvent);
              eventHandlers.push(createHandler, updateHandler);
            } catch (e) {
              console.warn('Error setting sketch handlers:', e);
            }
          }

          // Key handler
          try {
            const keyHandler = view.on('key-down', (event) => {
              const { key } = event;
              if (
                (key === 'Delete' || key === 'Backspace') &&
                sketchRef.current &&
                sketchRef.current.state === 'active'
              ) {
                sketchRef.current.cancel();
              }
            });
            eventHandlers.push(keyHandler);
          } catch (e) {
            console.warn('Error setting key handler:', e);
          }

          // Store handlers for cleanup
          view.eventHandlers = eventHandlers;

          // Add other widgets after a delay
          widgetTimer = setTimeout(() => {
            if (!isMounted || !view || view.destroyed || isDestroying) return;

            try {
              const addWidgets = async () => {
                // Create clear button
                const clearButton = document.createElement('button');
                clearButton.className = 'esri-button esri-button--secondary';
                clearButton.innerHTML = 'Clear Selection';
                clearButton.addEventListener('click', () => {
                  if (graphicsLayerRef.current) {
                    graphicsLayerRef.current.removeAll();
                    selectedGeometryRef.current = null;
                    formDataUpdates.current = {};

                    setFormData((prev) => ({
                      ...prev,
                      min_longitude: '',
                      min_latitude: '',
                      max_longitude: '',
                      max_latitude: '',
                      geometry: null,
                      geometry_type: null,
                      polygon_coordinates: null,
                    }));
                  }
                });

                // Simple widgets that are less likely to cause errors
                const widgets = {};

                try {
                  const legend = new Legend({ view, style: { type: 'card' } });
                  view.ui.add(legend, 'bottom-left');
                  widgets.legend = legend;
                } catch (e) {
                  console.warn('Error adding legend widget:', e);
                }

                try {
                  const basemapToggle = new BasemapToggle({ view, nextBasemap: 'satellite' });
                  view.ui.add(basemapToggle, 'bottom-right');
                  widgets.basemapToggle = basemapToggle;
                } catch (e) {
                  console.warn('Error adding basemap toggle widget:', e);
                }

                try {
                  const scaleBar = new ScaleBar({ view, unit: 'dual' });
                  view.ui.add(scaleBar, 'top-left');
                  widgets.scaleBar = scaleBar;
                } catch (e) {
                  console.warn('Error adding scale bar widget:', e);
                }

                try {
                  const search = new Search({ view, popupEnabled: true });
                  view.ui.add(search, 'top-right');
                  widgets.search = search;
                } catch (e) {
                  console.warn('Error adding search widget:', e);
                }

                // Add measurement widget last as it's more problematic
                try {
                  const measurement = new Measurement({ view, activeTool: 'distance' });
                  view.ui.add(measurement, 'bottom-right');
                  widgets.measurement = measurement;
                } catch (e) {
                  console.warn('Error adding measurement widget:', e);
                }

                // Add clear button
                try {
                  view.ui.add(clearButton, 'top-left');
                  widgets.clearButton = clearButton;
                } catch (e) {
                  console.warn('Error adding clear button:', e);
                }

                widgetsRef.current = widgets;
              };

              addWidgets();
            } catch (e) {
              console.error('Error adding widgets:', e);
            }
          }, 500);

          // Map is fully initialized
          setIsLoading(false);

          // If we already had a geometry selected, restore it
          if (selectedGeometryRef.current && graphicsLayerRef.current) {
            try {
              const restoreGeom = selectedGeometryRef.current;
              ref.current.drawGeometry(restoreGeom);
            } catch (e) {
              console.warn('Error restoring previous geometry', e);
            }
          }
        } catch (error) {
          console.error(`Error initializing map:`, error);
          setIsLoading(false);
        }

        // Draw the initial geometry if provided after the map is initialized
        if (initialGeometry && !hasDrawnInitialGeometry.current) {
          const checkAndDrawInterval = setInterval(() => {
            if (
              viewRef.current &&
              !viewRef.current.destroyed &&
              graphicsLayerRef.current &&
              isMounted
            ) {
              try {
                preventRefresh.current = true;
                graphicsLayerRef.current.removeAll();

                // Store the geometry for persistence
                selectedGeometryRef.current = initialGeometry;

                // Use the ref safely
                if (ref && ref.current && ref.current.drawGeometry) {
                  ref.current.drawGeometry(initialGeometry);
                  hasDrawnInitialGeometry.current = true;
                }

                setTimeout(() => {
                  preventRefresh.current = false;
                }, 800); // Longer timeout for initial draw
              } catch (e) {
                console.warn('Could not draw initial geometry', e);
                preventRefresh.current = false;
              }
              clearInterval(checkAndDrawInterval);
            }
          }, 200);

          // Clear the interval after a maximum wait time to avoid leaks
          setTimeout(() => clearInterval(checkAndDrawInterval), 5000);
        }
      };

      // Initialize map after a small delay to ensure DOM is ready
      initTimer = setTimeout(initializeMap, 250);

      return () => {
        isMounted = false;
        if (initTimer) clearTimeout(initTimer);
        if (widgetTimer) clearTimeout(widgetTimer);
        cleanupMap();
      };
    }, [cleanupMap, containerId, handleDrawEvent, setFormData, isDestroying, initialGeometry, ref]);

    // Pan map when centerCoordinates change - simplified with better checks
    useEffect(() => {
      if (
        viewRef.current?.center &&
        !viewRef.current.destroyed &&
        centerCoordinates &&
        !isDestroying &&
        !preventRefresh.current &&
        !zoomOperationInProgress.current
      ) {
        const { latitude, longitude } = centerCoordinates;
        if (!isNaN(latitude) && !isNaN(longitude)) {
          try {
            const geoCenter = webMercatorUtils.webMercatorToGeographic(viewRef.current.center);

            // Only pan if we need to move a significant distance
            if (
              Math.abs(geoCenter.latitude - parseFloat(latitude)) > 0.0001 ||
              Math.abs(geoCenter.longitude - parseFloat(longitude)) > 0.0001
            ) {
              viewRef.current.goTo({
                center: [parseFloat(longitude), parseFloat(latitude)],
                duration: 500,
              });
            }
          } catch (e) {
            console.warn('Error panning map:', e);
          }
        }
      }
    }, [centerCoordinates, isDestroying]);

    return (
      <div className="map-wrapper" style={{ width: '100%', height: '100%', position: 'relative' }}>
        <div
          ref={containerRef}
          className="map-container-inner"
          data-container-id={containerId}
          style={{
            width: '100%',
            height: '100%',
            minHeight: '400px',
          }}
        />

        {isLoading && (
          <div
            className="map-loading-indicator"
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              padding: '10px 20px',
              background: 'rgba(255,255,255,0.8)',
              borderRadius: '4px',
              boxShadow: '0 0 10px rgba(0,0,0,0.1)',
              zIndex: 10,
            }}
          >
            Initializing map...
          </div>
        )}

        <style>{`
        .map-container-inner {
          position: relative;
          border-radius: 8px;
          overflow: hidden;
        }
        .esri-view .esri-view-surface {
          outline: none !important;
        }
        .esri-widget {
          font-size: 14px;
        }
      `}</style>
      </div>
    );
  },
);

// Add displayName for better debugging
MapComponent.displayName = 'MapComponent';

export default MapComponent;
