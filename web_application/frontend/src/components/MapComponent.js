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
import BasemapToggle from '@arcgis/core/widgets/BasemapToggle';
import ScaleBar from '@arcgis/core/widgets/ScaleBar';
import SnappingOptions from '@arcgis/core/views/interactive/snapping/SnappingOptions';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import '@arcgis/core/assets/esri/themes/light/main.css';
import Graphic from '@arcgis/core/Graphic';
import Polygon from '@arcgis/core/geometry/Polygon';
import Extent from '@arcgis/core/geometry/Extent';
import { SimpleFillSymbol } from '@arcgis/core/symbols';
import WebTileLayer from '@arcgis/core/layers/WebTileLayer';
import Basemap from '@arcgis/core/Basemap';

// Inline style for map container
const mapContainerStyle = {
  backgroundColor: 'transparent',
  height: '100%',
  width: '100%'
};

// Use forwardRef to fix the React ref warning
const MapComponent = forwardRef(
  (
    {
      setFormData,
      onGeometryChange,
      containerId = 'viewDiv',
      initialGeometry = null,
      onLoadingChange = null,
    },
    ref,
  ) => {
    const containerRef = useRef(null);
    const mapRef = useRef(null);
    const viewRef = useRef(null);
    const graphicsLayerRef = useRef(null);
    const sketchRef = useRef(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isDestroying, setIsDestroying] = useState(false);
    const hasDrawnInitialGeometry = useRef(false);
    const preventRefresh = useRef(false);
    const selectedGeometryRef = useRef(null);

    // Update parent component when loading state changes
    useEffect(() => {
      if (onLoadingChange) {
        onLoadingChange(isLoading);
      }
    }, [isLoading, onLoadingChange]);

    // Expose methods to parent through ref (simplified)
    useImperativeHandle(ref, () => ({
      clearGraphics: () => {
        if (graphicsLayerRef.current) {
          graphicsLayerRef.current.removeAll();
          selectedGeometryRef.current = null;

          // Update form data
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
          }

          // Add the graphic and zoom to it
          if (graphic && viewRef.current && !viewRef.current.destroyed) {
            graphicsLayerRef.current.add(graphic);

            // Simple zoom options
            viewRef.current
              .goTo(graphic.geometry, {
                duration: 500,
                easing: 'ease-in-out',
              })
              .finally(() => {
                // Release lock after animation completes
                setTimeout(() => {
                  preventRefresh.current = false;
                }, 100);
              });
          } else {
            preventRefresh.current = false;
          }

          // Safety timeout
          setTimeout(() => {
            preventRefresh.current = false;
          }, 1000);
        } catch (e) {
          console.error('Error drawing geometry:', e);
          preventRefresh.current = false;
        }
      }
    }));

    // Simplified handler for sketch draw events
    const handleDrawEvent = useCallback(
      (event) => {
        // Skip if conditions aren't right
        if (!event || isDestroying) return;

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
                  polygon_coordinates: JSON.stringify(formattedCoordinates),
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
            }

            // Update form data
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

    // Simplified cleanup function
    const cleanupMap = useCallback(() => {
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

        if (view && !view.destroyed) {
          if (sketchRef.current) {
            view.ui.remove(sketchRef.current);
          }

          // Destroy view
          view.container = null;
          view.destroy();
        }

        // Clear all refs
        viewRef.current = null;
        mapRef.current = null;
        sketchRef.current = null;
        graphicsLayerRef.current = null;
      } catch (e) {
        console.error('Error during cleanup:', e);
      } finally {
        setIsDestroying(false);
      }
    }, []);

    // Initialize map - simplified
    useEffect(() => {
      let isMounted = true;

      const initializeMap = async () => {
        if (!containerRef.current || !isMounted || isDestroying) return;

        try {
          setIsLoading(true);
          // Notify parent component
          if (onLoadingChange) {
            onLoadingChange(true);
          }

          // Clean up any existing map resources
          cleanupMap();

          // Create graphics layer
          const graphicsLayer = new GraphicsLayer({
            title: 'Drawing Layer',
          });
          graphicsLayerRef.current = graphicsLayer;

          // Create Google aerial basemap
          const googleSatelliteLayer = new WebTileLayer({
            urlTemplate: 'https://mt{subDomain}.google.com/vt/lyrs=s&x={col}&y={row}&z={level}',
            subDomains: ['0', '1', '2', '3'],
            copyright: 'Google Maps Satellite',
            title: 'Google Satellite',
          });

          const googleBasemap = new Basemap({
            baseLayers: [googleSatelliteLayer],
            title: 'Google Satellite',
            id: 'google-satellite',
          });

          // Create map with Google satellite as the basemap
          const map = new Map({
            basemap: googleBasemap,
            layers: [graphicsLayer],
          });
          mapRef.current = map;

          // Create view
          const view = new MapView({
            container: containerRef.current,
            map,
            center: [-85.6024, 44.3148], // Michigan
            zoom: 6.75,
            constraints: {
              snapToZoom: true,
              rotationEnabled: false,
            },
            // Advanced performance optimizations
            navigation: {
              mouseWheelZoomEnabled: true,
              browserTouchPanEnabled: true,
            },
            // Performance improvements - more aggressive
            qualityProfile: "low",
            alphaCompositingEnabled: false,
            highlightOptions: {
              color: [255, 133, 0, 0.5],
              fillOpacity: 0.2,
              haloOpacity: 0.5
            },
            // Add performance hints
            performance: {
              hints: {
                maxFrameRate: 30, // Limit to 30fps instead of 60
                renderingOptimization: true
              }
            }
          });
          viewRef.current = view;
          
          // Optimize rendering
          view.renderingMode = "optimized";
          
          // Reduce animation workload
          view.ui.components = ["zoom"];
          
          // Advanced optimization for WASM memory
          // @ts-ignore - not in typings but exists in API
          if (window.esriConfig) {
            window.esriConfig.assetsPath = "/static/esri-assets";
            window.esriConfig.workers.loaderConfig = {
              has: {
                "esri-promise-compatibility": 1,
                "esri-workers-for-memory-leaks": 0 
              }
            };
          }
          
          // Optimize for performance
          if (view && view.environment) {
            view.environment.atmosphere = { quality: "low" };
            view.environment.lighting = { 
              directShadowsEnabled: false,
              ambientOcclusionEnabled: false
            };
          }
          
          // Throttle map updates
          let updateCount = 0;
          view.on("update-start", () => {
            updateCount++;
            // Limit updates if happening too frequently
            if (updateCount > 3) {
              view.suspended = true;
              setTimeout(() => {
                view.suspended = false;
                updateCount = 0;
              }, 300);
            }
          });
          
          view.on("update-end", () => {
            if (updateCount > 0) updateCount--;
          });
          
          // Configure the view to handle wheel events properly
          if (view && view.on) {
            // Improve wheel event handling for map navigation
            view.on("mouse-wheel", (event) => {
              // Prevent the page from scrolling
              event.stopPropagation();
            }, { passive: false });
            
            // Optimize drag handling
            let lastDragTime = 0;
            const DRAG_THROTTLE = 50; // ms
            
            view.on("drag", (event) => {
              const now = performance.now();
              if (now - lastDragTime < DRAG_THROTTLE) {
                // Throttle rapid drag updates
                event.stopPropagation();
                event.preventDefault();
                return;
              }
              lastDragTime = now;
              event.stopPropagation();
            }, { passive: false });
          }
          
          // Make wheel events passive to address browser performance warning
          if (containerRef.current) {
            // Clear any existing handlers first to avoid duplicates
            const wheelHandler = (e) => {
              // We use stopPropagation instead of preventDefault 
              // since the event is already passive
              e.stopPropagation();
            };
            
            containerRef.current.removeEventListener('wheel', wheelHandler);
            containerRef.current.addEventListener('wheel', wheelHandler, { passive: true });
            
            // Add explicit touch handlers with passive option
            containerRef.current.addEventListener('touchstart', () => {}, { passive: true });
            containerRef.current.addEventListener('touchmove', () => {}, { passive: true });
          }

          // Wait for view to be ready
          await view.when();

          if (!isMounted || !view || view.destroyed || isDestroying) {
            return;
          }

          // Create snapping options
          view.snappingOptions = new SnappingOptions({
            enabled: true,
            selfEnabled: true,
            featureSources: [{ layer: graphicsLayer }],
          });

          // Initialize sketch widget - simplified
          const sketch = new Sketch({
            view,
            layer: graphicsLayer,
            creationMode: 'single',
            availableCreateTools: ['polygon', 'rectangle'],
            visibleElements: { settingsMenu: false, undoRedoMenu: true },
          });
          sketchRef.current = sketch;
          view.ui.add(sketch, 'top-right');

          if (!isMounted || view.destroyed || isDestroying) return;

          // Add event handlers
          const eventHandlers = [];

          // Sketch event handlers - simplified
          const createHandler = sketch.on('create', handleDrawEvent);
          const updateHandler = sketch.on('update', handleDrawEvent);
          eventHandlers.push(createHandler, updateHandler);

          // Store handlers for cleanup
          view.eventHandlers = eventHandlers;

          // Add basic widgets
          const basemapToggle = new BasemapToggle({ 
            view, 
            nextBasemap: 'topo-vector'
          });
          view.ui.add(basemapToggle, { position: 'bottom-right', index: 0 });

          const scaleBar = new ScaleBar({ 
            view, 
            unit: 'dual',
            style: 'line'
          });
          view.ui.add(scaleBar, { position: 'bottom-left', index: 0 });

          // Create clear button
          const clearButton = document.createElement('button');
          clearButton.className = 'esri-button esri-button--secondary';
          clearButton.innerHTML = 'Clear Selection';
          clearButton.addEventListener('click', () => {
            if (ref.current) {
              ref.current.clearGraphics();
            }
          });
          
          view.ui.add(clearButton, { position: 'top-left', index: 0 });

          // Map is fully initialized
          setIsLoading(false);
          // Notify parent component
          if (onLoadingChange) {
            onLoadingChange(false);
          }

          // If we already had a geometry selected, restore it
          if (initialGeometry && !hasDrawnInitialGeometry.current && ref.current) {
            setTimeout(() => {
              try {
                ref.current.drawGeometry(initialGeometry);
                hasDrawnInitialGeometry.current = true;
              } catch (e) {
                console.warn('Could not draw initial geometry', e);
              }
            }, 500);
          }

        } catch (error) {
          console.error(`Error initializing map:`, error);
          setIsLoading(false);
          // Notify parent component of loading completion (even if error)
          if (onLoadingChange) {
            onLoadingChange(false);
          }
        }
      };

      // Initialize map after a small delay to ensure DOM is ready
      const initTimer = setTimeout(initializeMap, 250);

      return () => {
        isMounted = false;
        clearTimeout(initTimer);
        cleanupMap();
      };
    }, [cleanupMap, containerId, handleDrawEvent, setFormData, isDestroying, initialGeometry, ref, onLoadingChange]);

    return (
      <div
        id={containerId}
        ref={containerRef}
        style={{
          ...mapContainerStyle,
          opacity: isLoading ? 0.6 : 1,
          transition: 'opacity 0.3s ease'
        }}
        className="map-container esri-map-container"
      />
    );
  },
);

// Add displayName for better debugging
MapComponent.displayName = 'MapComponent';

export default MapComponent;
