import React, { useEffect, useRef, useState, useCallback } from 'react';
import { loadModules } from 'esri-loader';
import SessionService from '../services/SessionService';

// Fix for passive event listener issue with the wheel event
// This needs to be defined BEFORE the ArcGIS JS API loads
const fixWheelEvent = () => {
  try {
    // Save the original addEventListener
    const originalAddEventListener = EventTarget.prototype.addEventListener;

    // Override addEventListener to make wheel events non-passive for ArcGIS elements only
    EventTarget.prototype.addEventListener = function (type, listener, options) {
      // Check if this is a wheel event
      if (type === 'wheel' || type === 'mousewheel') {
        // For ArcGIS map elements, we need to ensure passive is not forced to true
        // This allows proper zoom behavior using the mouse wheel
        if (
          (this.className &&
            typeof this.className === 'string' &&
            (this.className.includes('esri-') || this.className.includes('esri'))) ||
          (this.id && this.id === 'esri-map-container') ||
          (this.parentElement &&
            this.parentElement.className &&
            typeof this.parentElement.className === 'string' &&
            this.parentElement.className.includes('esri'))
        ) {
          // If options is a boolean (useCapture), convert to object
          if (typeof options === 'boolean') {
            options = { capture: options, passive: false };
          } else if (typeof options === 'object' && options !== null) {
            options.passive = false;
          } else {
            options = { passive: false };
          }
        } else {
          // For non-ArcGIS elements, keep passive for performance
          if (typeof options === 'boolean') {
            options = { capture: options, passive: true };
          } else if (typeof options === 'object' && options !== null) {
            options.passive = true;
          } else {
            options = { passive: true };
          }
        }
      }

      // Call the original addEventListener with our modified options
      return originalAddEventListener.call(this, type, listener, options);
    };

    // Set up dojoConfig for ArcGIS
    if (window.dojoConfig) {
      // Update existing dojoConfig
      window.dojoConfig.passiveEvents = false;
      if (window.dojoConfig.has) {
        window.dojoConfig.has['esri-passive-events'] = false;
      } else {
        window.dojoConfig.has = { 'esri-passive-events': false };
      }
    } else {
      // Create new dojoConfig
      window.dojoConfig = {
        passiveEvents: false,
        has: { 'esri-passive-events': false },
      };
    }

    console.log('Wheel event listener fix applied');
  } catch (e) {
    console.error('Failed to apply wheel event fix:', e);
  }
};

// Apply the fix immediately
fixWheelEvent();

// Debounce utility function to improve performance
const debounce = (func, delay) => {
  let timeoutId;
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
};

const EsriMap = ({
  geometries = [],
  streamsGeometries = [],
  lakesGeometries = [],
  stationPoints = [],
  onStationSelect = null,
  showStations = false,
  selectedStationId = null,
  refreshMapRef = null, // Add new prop to expose refresh functionality
}) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);
  const highlightedFeatureRef = useRef(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoadingGeometries, setIsLoadingGeometries] = useState(false);
  const [stationsLoaded, setStationsLoaded] = useState(false);
  const prevStationPointsRef = useRef([]);
  const visibilityRef = useRef(true);

  const handleStationSelect = useCallback(
    (clickedStation) => {
      if (onStationSelect && clickedStation) {
        SessionService.setMapInteractionState(true);

        onStationSelect(clickedStation);

        setTimeout(() => {
          SessionService.setMapInteractionState(false);
        }, 500);
      }
    },
    [onStationSelect],
  );

  useEffect(() => {
    const handleVisibilityChange = () => {
      const isVisible = !document.hidden;
      console.log(`Document visibility changed to: ${isVisible ? 'visible' : 'hidden'}`);
      visibilityRef.current = isVisible;

      if (isVisible && showStations && stationPoints.length > 0 && viewRef.current) {
        const { stationLayer } = viewRef.current;
        if (stationLayer && !stationLayer.destroyed && stationLayer.graphics.length === 0) {
          console.log('Tab visible again, stations not displayed - forcing refresh');
          setTimeout(renderStationPoints, 500);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [showStations, stationPoints]);

  const renderStationPoints = useCallback(() => {
    if (!viewRef.current || !isLoaded || !showStations || stationPoints.length === 0) {
      console.log('Cannot render station points - prerequisites not met:', {
        viewExists: !!viewRef.current,
        isLoaded,
        showStations,
        stationPointsCount: stationPoints.length,
      });
      return;
    }

    SessionService.setMapInteractionState(true);

    const { stationLayer, Graphic, Point, view } = viewRef.current;

    if (stationLayer && !stationLayer.destroyed) {
      if (prevStationPointsRef.current === stationPoints && stationLayer.graphics.length > 0) {
        console.log(
          `Station points already rendered (${stationLayer.graphics.length}) - skipping redraw`,
        );
        SessionService.setMapInteractionState(false);
        return;
      }

      console.log(`Clearing ${stationLayer.graphics.length} existing station graphics`);
      stationLayer.removeAll();
    } else {
      console.warn('Station layer unavailable for rendering');
      SessionService.setMapInteractionState(false);
      return;
    }

    console.log(`Rendering ${stationPoints.length} station points on map`);

    // Track metrics for debugging
    const startTime = performance.now();
    const stationGraphics = [];
    let successfullyAdded = 0;
    let failedToAdd = 0;

    // Larger batch size for production
    const batchSize = 200;

    // Use a more efficient approach with a single array push then addMany
    for (let i = 0; i < stationPoints.length; i += batchSize) {
      const batch = stationPoints.slice(i, i + batchSize);
      const batchGraphics = [];

      batch.forEach((station) => {
        if (station.geometry && station.geometry.coordinates) {
          try {
            const [longitude, latitude] = station.geometry.coordinates;
            const point = new Point({
              longitude,
              latitude,
              spatialReference: { wkid: 4326 },
            });

            const isSelected = selectedStationId === station.properties.SiteNumber;

            const stationGraphic = new Graphic({
              geometry: point,
              symbol: {
                type: 'simple-marker',
                color: isSelected ? [255, 0, 0, 0.8] : [0, 114, 206, 0.7],
                size: isSelected ? '12px' : '8px',
                outline: {
                  color: [255, 255, 255],
                  width: 1,
                },
              },
              attributes: {
                SiteNumber: station.properties.SiteNumber,
                SiteName: station.properties.SiteName,
                id: station.properties.id,
              },
              popupTemplate: {
                title: '{SiteName}',
                content: 'Station ID: {SiteNumber}',
              },
            });

            batchGraphics.push(stationGraphic);
            stationGraphics.push(stationGraphic);
            successfullyAdded++;
          } catch (err) {
            failedToAdd++;
            console.warn('Error creating station graphic:', err);
          }
        } else {
          failedToAdd++;
        }
      });

      // Add all graphics in the batch at once
      if (!stationLayer.destroyed && batchGraphics.length > 0) {
        try {
          stationLayer.addMany(batchGraphics);
        } catch (err) {
          console.error('Error adding batch graphics to layer:', err);
          // Fallback to adding one by one if batch fails
          batchGraphics.forEach((graphic) => {
            try {
              stationLayer.add(graphic);
            } catch (innerErr) {
              failedToAdd++;
              successfullyAdded--;
            }
          });
        }
      }
    }

    const endTime = performance.now();
    console.log(
      `Station points rendering: Added ${successfullyAdded}, Failed ${failedToAdd}, Time: ${(endTime - startTime).toFixed(1)}ms`,
    );

    prevStationPointsRef.current = stationPoints;

    setStationsLoaded(true);

    // Zoom to stations if needed
    if (stationGraphics.length > 0 && !selectedStationId && view && !view.destroyed) {
      try {
        // Delay the zoom slightly to allow map to stabilize
        setTimeout(() => {
          if (!view.destroyed) {
            console.log('Zooming to station extents');
            view
              .goTo(stationGraphics, { animate: false })
              .catch((error) => console.warn('Error during map navigation:', error))
              .finally(() => {
                setTimeout(() => SessionService.setMapInteractionState(false), 500);
              });
          } else {
            SessionService.setMapInteractionState(false);
          }
        }, 200);
      } catch (e) {
        console.error('Exception during view.goTo preparation:', e);
        SessionService.setMapInteractionState(false);
      }
    } else {
      setTimeout(() => SessionService.setMapInteractionState(false), 500);
    }
  }, [isLoaded, showStations, stationPoints, selectedStationId]);

  useEffect(() => {
    let view;
    let mapViewHandles = [];

    // Track loading state properly with promises instead of timeouts
    const loadingStates = {
      viewReady: false,
      modulesLoaded: false,
      stationsProcessing: false,
      renderComplete: false,
    };

    // Signal that the map is in an interactive state during initialization
    SessionService.setMapInteractionState(true);

    const options = {
      version: '4.25',
      css: true,
      dojoConfig: {
        passiveEvents: true, // Set to true for better performance
        has: {
          'esri-passive-events': true,
        },
      },
    };

    // Create a promise that resolves when map rendering is complete
    let resolveRenderComplete;
    const renderCompletePromise = new Promise((resolve) => {
      resolveRenderComplete = resolve;
    });

    // Load ESRI modules using Promise-based pattern
    loadModules(
      [
        'esri/Map',
        'esri/views/MapView',
        'esri/Graphic',
        'esri/layers/GraphicsLayer',
        'esri/geometry/Polygon',
        'esri/geometry/Polyline',
        'esri/geometry/Extent',
        'esri/geometry/Point',
        'esri/geometry/support/webMercatorUtils',
        'esri/widgets/Sketch',
        'esri/geometry/geometryEngine',
        'esri/config',
      ],
      options,
    )
      .then(
        ([
          Map,
          MapView,
          Graphic,
          GraphicsLayer,
          Polygon,
          Polyline,
          Extent,
          Point,
          webMercatorUtils,
          Sketch,
          geometryEngine,
          esriConfig,
        ]) => {
          if (viewRef.current) return;

          loadingStates.modulesLoaded = true;
          console.log('ESRI modules loaded successfully');

          // Configure ESRI for better performance
          esriConfig.has = esriConfig.has || {};
          esriConfig.has['esri-passive-events'] = true;

          // Add passive event support to avoid browser warnings
          esriConfig.options = {
            ...esriConfig.options,
            events: {
              add: { passive: true },
              remove: { passive: true },
            },
          };

          const map = new Map({
            basemap: 'topo-vector',
            qualityProfile: 'high', // Use high quality for better rendering
          });

          view = new MapView({
            container: mapRef.current,
            map: map,
            zoom: 4,
            center: [-98, 39],
            constraints: {
              snapToZoom: false,
            },
            navigation: {
              mouseWheelZoomEnabled: true,
              browserTouchPanEnabled: true,
            },
            // Optimize performance with better loading
            loadingOptimization: true,
            // Disable popup to improve performance
            popup: {
              dockEnabled: false,
              dockOptions: {
                // Disable dock completely to improve performance
                buttonEnabled: false,
                breakpoint: false,
              },
            },
          });

          // Set wheel and touch events as passive for better performance
          if (view.navigation) {
            view.navigation.mouseWheelEventOptions = { passive: true };
            view.navigation.browserTouchPanEventOptions = { passive: true };
          }

          // Create layers for map
          const polygonLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const streamLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const lakeLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const stationLayer = new GraphicsLayer({
            elevationInfo: { mode: 'on-the-ground' },
            title: 'Stream Gauge Stations',
          });

          map.addMany([polygonLayer, streamLayer, lakeLayer, stationLayer]);

          // Wait for the view to be ready before setting up interactions
          view.when(() => {
            loadingStates.viewReady = true;
            console.log('Map view is ready');

            // Instead of timeouts, we use the view's ready state
            if (showStations && stationPoints.length > 0) {
              renderStationPointsWithBatching(stationLayer, Graphic, Point);
            }

            // Add click event handler for station selection
            if (onStationSelect) {
              // Add hover effect to improve usability
              const pointerMoveHandler = view.on(
                'pointer-move',
                debounce((event) => {
                  // Skip if not showing stations or during processing
                  if (!showStations || loadingStates.stationsProcessing) {
                    return;
                  }

                  const screenPoint = {
                    x: event.x,
                    y: event.y,
                  };

                  // Hit test with minimal processing
                  view
                    .hitTest(screenPoint, {
                      include: [stationLayer],
                    })
                    .then((response) => {
                      // Find if we're hovering over a station
                      const stationGraphic = response.results?.find(
                        (result) => result.graphic?.layer === stationLayer,
                      )?.graphic;

                      // Get all graphics to reset those not being hovered
                      const graphics = stationLayer.graphics.toArray();

                      graphics.forEach((graphic) => {
                        if (!graphic.symbol) return;

                        const isSelected = graphic.attributes.SiteNumber === selectedStationId;
                        const isHovered = stationGraphic && graphic === stationGraphic;

                        // Only update if state changed to avoid unnecessary redraws
                        if (isHovered && graphic.symbol.size !== '10px' && !isSelected) {
                          // Hovering - make bigger with outline
                          graphic.symbol = {
                            type: 'simple-marker',
                            color: [0, 114, 206, 0.9],
                            size: '10px',
                            outline: {
                              color: [255, 255, 255],
                              width: 2,
                            },
                          };

                          // Change cursor to pointer to indicate clickable
                          mapRef.current.style.cursor = 'pointer';
                        } else if (!isHovered && !isSelected && graphic.symbol.size !== '8px') {
                          // Reset to normal state
                          graphic.symbol = {
                            type: 'simple-marker',
                            color: [0, 114, 206, 0.7],
                            size: '8px',
                            outline: {
                              color: [255, 255, 255],
                              width: 1,
                            },
                          };
                        }
                      });

                      // If not hovering over a station, reset cursor
                      if (!stationGraphic) {
                        mapRef.current.style.cursor = 'default';
                      }
                    })
                    .catch((err) => {
                      // Silent catch - no need to handle errors for hover effects
                    });
                }, 50),
              ); // Short debounce for responsive hover

              mapViewHandles.push(pointerMoveHandler);

              // Click handler for selecting stations
              const clickHandler = view.on('click', (event) => {
                // Prevent default behavior and propagation
                event.stopPropagation();

                // Skip if not showing stations or loading
                if (!showStations || loadingStates.stationsProcessing) {
                  console.log('Click ignored: not ready for station selection');
                  return;
                }

                SessionService.setMapInteractionState(true);
                console.log('Map clicked, checking for stations...');

                // Create the screen point from the event
                const screenPoint = {
                  x: event.x,
                  y: event.y,
                };

                // Hit test to find graphics at the clicked location
                view
                  .hitTest(screenPoint, {
                    include: [stationLayer],
                  })
                  .then((response) => {
                    console.log('Hit test results:', response.results?.length || 0);

                    // Find station graphics in the results
                    const stationGraphics = response.results?.filter(
                      (result) => result.graphic?.layer === stationLayer,
                    );

                    console.log('Station graphics found:', stationGraphics?.length || 0);

                    // If we found a station, call the selection handler
                    if (
                      stationGraphics &&
                      stationGraphics.length > 0 &&
                      stationGraphics[0].graphic?.attributes
                    ) {
                      const clickedStation = stationGraphics[0].graphic.attributes;
                      console.log('Station selected:', clickedStation);
                      handleStationSelect(clickedStation);
                    } else {
                      console.log('No station found at click location');
                      // No station found at this location
                    }

                    // Release interaction state
                    setTimeout(() => {
                      SessionService.setMapInteractionState(false);
                    }, 500);
                  })
                  .catch((error) => {
                    console.error('Error during hit test:', error);
                    SessionService.setMapInteractionState(false);
                  });
              });

              // Add to handles for cleanup
              mapViewHandles.push(clickHandler);
            }
          });

          // Define map interaction event handlers
          const handleMapInteractionStart = () => {
            SessionService.setMapInteractionState(true);
          };

          const handleMapInteractionEnd = debounce(() => {
            if (!loadingStates.stationsProcessing) {
              SessionService.setMapInteractionState(false);
            }
          }, 1000);

          // Add event listeners for map interactions
          const interactionEvents = ['mouse-wheel', 'key-down', 'drag', 'double-click', 'click'];
          interactionEvents.forEach((eventName) => {
            const handle = view.on(eventName, handleMapInteractionStart);
            mapViewHandles.push(handle);
          });

          // Add event listeners for interaction completion
          const dragEndHandle = view.on('drag-end', handleMapInteractionEnd);
          const keyUpHandle = view.on('key-up', handleMapInteractionEnd);
          const clickHandle = view.on('click', handleMapInteractionEnd);
          mapViewHandles.push(dragEndHandle, keyUpHandle, clickHandle);

          // Store view information in ref for later use
          viewRef.current = {
            view,
            polygonLayer,
            streamLayer,
            lakeLayer,
            stationLayer,
            Graphic,
            Polygon,
            Polyline,
            Point,
            webMercatorUtils,
            geometryEngine,
          };

          // A more efficient station point rendering function that uses batching
          // and avoids timeouts by using the view's ready state
          const renderStationPointsWithBatching = (stationLayer, Graphic, Point) => {
            if (!stationPoints.length) return;

            loadingStates.stationsProcessing = true;
            console.log(`Rendering ${stationPoints.length} station points using batching`);

            // Clear any existing graphics
            stationLayer.removeAll();

            // Efficient batch processing with Web Workers if supported
            const useWebWorker = window.Worker && stationPoints.length > 1000;

            if (useWebWorker) {
              processStationPointsWithWorker(stationPoints, stationLayer, Graphic, Point);
            } else {
              processStationPointsDirectly(stationPoints, stationLayer, Graphic, Point);
            }
          };

          // Store the render function with a better name
          viewRef.current.renderStations = renderStationPointsWithBatching;

          // Trigger initial data loading
          setIsLoaded(true);

          // Signal rendering is ready to start
          resolveRenderComplete();
        },
      )
      .catch((error) => {
        console.error('Error loading ArcGIS modules:', error);
        SessionService.setMapInteractionState(false);
      });

    // Efficient direct processing of station points
    const processStationPointsDirectly = (stationPoints, stationLayer, Graphic, Point) => {
      const batchSize = 200; // Process in manageable chunks
      let processedCount = 0;
      let failedCount = 0;

      // Use requestAnimationFrame to avoid blocking the UI thread
      const processBatch = (startIdx) => {
        const endIdx = Math.min(startIdx + batchSize, stationPoints.length);
        const batchGraphics = [];

        // Process batch
        for (let i = startIdx; i < endIdx; i++) {
          const station = stationPoints[i];
          if (station.geometry && station.geometry.coordinates) {
            try {
              const [longitude, latitude] = station.geometry.coordinates;
              const point = new Point({
                longitude,
                latitude,
                spatialReference: { wkid: 4326 },
              });

              const isSelected = selectedStationId === station.properties.SiteNumber;

              const stationGraphic = new Graphic({
                geometry: point,
                symbol: {
                  type: 'simple-marker',
                  color: isSelected ? [255, 0, 0, 0.8] : [0, 114, 206, 0.7],
                  size: isSelected ? '12px' : '8px',
                  outline: {
                    color: [255, 255, 255],
                    width: 1,
                  },
                },
                attributes: {
                  SiteNumber: station.properties.SiteNumber,
                  SiteName: station.properties.SiteName,
                  id: station.properties.id,
                },
              });

              batchGraphics.push(stationGraphic);
              processedCount++;
            } catch (err) {
              failedCount++;
            }
          } else {
            failedCount++;
          }
        }

        // Add batch to layer
        if (batchGraphics.length > 0 && !stationLayer.destroyed) {
          try {
            stationLayer.addMany(batchGraphics);
          } catch (error) {
            console.warn('Error adding batch graphics:', error);
            failedCount += batchGraphics.length;
            processedCount -= batchGraphics.length;
          }
        }

        // Continue with next batch or complete
        if (endIdx < stationPoints.length) {
          // Use requestAnimationFrame for smoother UI
          requestAnimationFrame(() => processBatch(endIdx));
        } else {
          // All batches processed
          console.log(
            `Station points rendering complete: Added ${processedCount}, Failed ${failedCount}`,
          );
          prevStationPointsRef.current = stationPoints;
          setStationsLoaded(true);
          loadingStates.stationsProcessing = false;
          loadingStates.renderComplete = true;

          // Allow map interactions again
          SessionService.setMapInteractionState(false);
        }
      };

      // Start processing first batch
      processBatch(0);
    };

    // Use Web Workers for heavy processing (would need to be implemented in a real application)
    // This is a placeholder that simulates what a web worker would do
    const processStationPointsWithWorker = (stationPoints, stationLayer, Graphic, Point) => {
      console.log('Would use Web Worker for large station set, falling back to direct processing');
      processStationPointsDirectly(stationPoints, stationLayer, Graphic, Point);
    };

    // When component unmounts
    return () => {
      // Immediately cancel any ongoing processing
      loadingStates.stationsProcessing = false;

      // Ensure we're not in an interactive state
      SessionService.setMapInteractionState(false);

      // Clean up event handlers
      if (mapViewHandles.length > 0) {
        mapViewHandles.forEach((handle) => {
          if (handle && typeof handle.remove === 'function') {
            try {
              handle.remove();
            } catch (e) {
              console.warn('Error removing map handle:', e);
            }
          }
        });
      }

      // Clean up highlight feature
      if (highlightedFeatureRef.current) {
        try {
          highlightedFeatureRef.current.remove();
        } catch (e) {
          console.warn('Error removing highlight:', e);
        }
        highlightedFeatureRef.current = null;
      }

      // Properly destroy the view
      if (view) {
        try {
          view.destroy();
        } catch (e) {
          console.warn('Error destroying view:', e);
        }
        viewRef.current = null;
      }

      console.log('EsriMap component unmounted, view and layers cleaned up');
    };
  }, [onStationSelect, showStations, stationPoints, selectedStationId, handleStationSelect]);

  useEffect(() => {
    if (!viewRef.current || !isLoaded) {
      return;
    }

    if (showStations) {
      console.log(`Station points changed: ${stationPoints.length} points, show: ${showStations}`);
      renderStationPoints();
    } else if (viewRef.current.stationLayer && !viewRef.current.stationLayer.destroyed) {
      viewRef.current.stationLayer.removeAll();
      prevStationPointsRef.current = [];
    }

    const watchInterval = setInterval(() => {
      if (
        showStations &&
        stationPoints.length > 0 &&
        viewRef.current &&
        viewRef.current.stationLayer &&
        !viewRef.current.stationLayer.destroyed &&
        viewRef.current.stationLayer.graphics.length === 0 &&
        visibilityRef.current
      ) {
        console.log(
          "Station visibility check: stations should be visible but aren't - forcing refresh",
        );
        renderStationPoints();
      }
    }, 3000);

    return () => {
      clearInterval(watchInterval);
    };
  }, [stationPoints, showStations, isLoaded, renderStationPoints]);

  // New function to force a refresh of the map and stations
  const refreshMap = useCallback(() => {
    console.log('Forcing map refresh...');
    if (!viewRef.current || !isLoaded) {
      console.log('Map not ready for refresh');
      return false;
    }

    // Clear current graphics
    if (viewRef.current.stationLayer && !viewRef.current.stationLayer.destroyed) {
      viewRef.current.stationLayer.removeAll();
      console.log('Cleared station graphics');
    }

    // Reset the previous points reference to force redraw
    prevStationPointsRef.current = [];

    // Re-render station points
    if (showStations && stationPoints.length > 0) {
      console.log('Re-rendering station points');
      setTimeout(() => {
        renderStationPoints();

        // Additional check to ensure points are rendered
        setTimeout(() => {
          if (viewRef.current?.stationLayer?.graphics.length === 0) {
            console.log('Second attempt to render station points');
            renderStationPoints();
          }
        }, 1000);
      }, 100);
    }

    // If we have a view, try to reset its extent to improve visibility
    if (viewRef.current.view && !viewRef.current.view.destroyed) {
      try {
        console.log('Resetting map view extent');
        const initialExtent = {
          center: [-98, 39],
          zoom: 4,
        };

        viewRef.current.view
          .goTo(initialExtent, {
            duration: 500,
            easing: 'ease-in-out',
          })
          .catch((err) => {
            console.warn('Error resetting map extent:', err);
          });
      } catch (e) {
        console.error('Error during map extent reset:', e);
      }
    }

    return true;
  }, [isLoaded, showStations, stationPoints, renderStationPoints]);

  // Expose the refresh function via the ref
  useEffect(() => {
    if (refreshMapRef && typeof refreshMapRef === 'object') {
      refreshMapRef.current = refreshMap;
    }
  }, [refreshMap, refreshMapRef]);

  useEffect(() => {
    if (!viewRef.current || !isLoaded || !showStations || stationPoints.length === 0) {
      return;
    }

    const { stationLayer } = viewRef.current;

    if (!stationLayer || stationLayer.destroyed) {
      return;
    }

    if (selectedStationId && prevStationPointsRef.current === stationPoints && stationsLoaded) {
      console.log(`Updating selection for station: ${selectedStationId}`);

      const graphics = stationLayer.graphics.toArray();

      graphics.forEach((graphic) => {
        if (!graphic.symbol) return;

        const isSelected = graphic.attributes.SiteNumber === selectedStationId;

        if (
          (isSelected && graphic.symbol.color[0] !== 255) ||
          (!isSelected && graphic.symbol.color[0] === 255)
        ) {
          graphic.symbol = {
            type: 'simple-marker',
            color: isSelected ? [255, 0, 0, 0.8] : [0, 114, 206, 0.7],
            size: isSelected ? '12px' : '8px',
            outline: {
              color: [255, 255, 255],
              width: 1,
            },
          };
        }
      });
    }
  }, [selectedStationId, stationPoints, showStations, isLoaded, stationsLoaded]);

  useEffect(() => {
    if (!viewRef.current || !isLoaded) {
      return;
    }

    setIsLoadingGeometries(true);

    SessionService.setMapInteractionState(true);

    const { view, polygonLayer, streamLayer, lakeLayer } = viewRef.current;

    if (polygonLayer && !polygonLayer.destroyed) polygonLayer.removeAll();
    if (streamLayer && !streamLayer.destroyed) streamLayer.removeAll();
    if (lakeLayer && !lakeLayer.destroyed) lakeLayer.removeAll();

    loadModules(['esri/Graphic', 'esri/geometry/Polygon', 'esri/geometry/Polyline'], {
      version: '4.25',
    })
      .then(([Graphic, Polygon, Polyline]) => {
        const allGraphics = [];

        const processGeometries = (geometries, layer, options) => {
          if (!layer || layer.destroyed) return;

          const chunkSize = 50;
          for (let i = 0; i < geometries.length; i += chunkSize) {
            const chunk = geometries.slice(i, i + chunkSize);
            chunk.forEach((geom) => {
              if (geom?.coordinates?.length) {
                let graphic;

                if (options.type === 'polygon') {
                  const polygon = new Polygon({
                    rings: geom.coordinates[0],
                    spatialReference: { wkid: 4326 },
                  });
                  graphic = new Graphic({
                    geometry: polygon,
                    symbol: options.symbol,
                  });
                } else if (options.type === 'polyline') {
                  const polyline = new Polyline({
                    paths: geom.coordinates[0],
                    spatialReference: { wkid: 4326 },
                  });
                  graphic = new Graphic({
                    geometry: polyline,
                    symbol: options.symbol,
                  });
                }

                if (graphic && layer && !layer.destroyed) {
                  layer.add(graphic);
                  allGraphics.push(graphic);
                }
              }
            });
          }
        };

        if (geometries.length > 0) {
          processGeometries(geometries, polygonLayer, {
            type: 'polygon',
            symbol: {
              type: 'simple-fill',
              color: [227, 139, 79, 0.6],
              outline: { color: [255, 255, 255], width: 1 },
            },
          });
        }

        if (lakesGeometries.length > 0) {
          processGeometries(lakesGeometries, lakeLayer, {
            type: 'polygon',
            symbol: {
              type: 'simple-fill',
              color: [0, 0, 255, 0.4],
              outline: { color: [255, 255, 255], width: 1 },
            },
          });
        }

        if (streamsGeometries.length > 0) {
          processGeometries(streamsGeometries, streamLayer, {
            type: 'polyline',
            symbol: {
              type: 'simple-line',
              color: [0, 0, 255],
              width: 1,
            },
          });
        }

        if (allGraphics.length > 0 && view && !view.destroyed) {
          view
            .when(() => {
              view
                .goTo(allGraphics, { duration: 500 })
                .catch((e) => {
                  console.warn('Error during map navigation:', e);
                })
                .finally(() => {
                  setIsLoadingGeometries(false);

                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 500);
                });
            })
            .catch((e) => {
              console.warn('Error during view.when():', e);
              setIsLoadingGeometries(false);
              SessionService.setMapInteractionState(false);
            });
        } else {
          setIsLoadingGeometries(false);

          setTimeout(() => {
            SessionService.setMapInteractionState(false);
          }, 500);
        }
      })
      .catch((error) => {
        console.error('Error loading modules:', error);
        setIsLoadingGeometries(false);

        SessionService.setMapInteractionState(false);
      });
  }, [geometries, streamsGeometries, lakesGeometries, isLoaded]);

  return (
    <>
      <div
        ref={mapRef}
        style={{
          height: '100%',
          width: '100%',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
        }}
      />
      <div
        id="coordinateInfo"
        style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          backgroundColor: 'rgba(255,255,255,0.8)',
          padding: '5px',
          borderRadius: '3px',
          fontSize: '12px',
          pointerEvents: 'none',
        }}
      />
      {isLoadingGeometries && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: 'rgba(255,255,255,0.5)',
            zIndex: 1000,
          }}
        >
          <div
            style={{
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              border: '6px solid #f3f3f3',
              borderTop: '6px solid #3273dc',
              animation: 'spin 1s linear infinite',
            }}
          />
          <style>
            {`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}
          </style>
        </div>
      )}
    </>
  );
};

export default EsriMap;
