///data/SWATGenXApp/codes/web_application/frontend/src/components/EsriMap.js
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { loadModules } from 'esri-loader';
import SessionService from '../services/SessionService';

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

// Configure passive event listeners for the entire application before map initialization
const configurePassiveEventListeners = () => {
  // Patch EventTarget.prototype.addEventListener to make wheel and touch events passive
  const originalAddEventListener = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, listener, options) {
    // Force passive true for touch and wheel events
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
      } else if (typeof options === 'object') {
        options.passive = true;
      } else {
        options = { passive: true };
      }
    }
    return originalAddEventListener.call(this, type, listener, options);
  };

  // Setup global dojoConfig for ArcGIS
  if (window.dojoConfig) {
    window.dojoConfig.passiveEvents = true;
  } else {
    window.dojoConfig = { passiveEvents: true };
  }
};

// Call the configuration function immediately
configurePassiveEventListeners();

const EsriMap = ({
  geometries = [],
  streamsGeometries = [],
  lakesGeometries = [],
  stationPoints = [],
  onStationSelect = null,
  onMultipleStationsSelect = null,
  showStations = false,
  selectedStationId = null,
  drawingMode = false,
  onDrawComplete = null,
}) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);
  const sketchRef = useRef(null);
  const highlightedFeatureRef = useRef(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoadingGeometries, setIsLoadingGeometries] = useState(false);

  // Create memoized functions with useCallback to prevent unnecessary recreations
  const handleStationSelect = useCallback(
    (clickedStation) => {
      if (onStationSelect && clickedStation) {
        // Inform SessionService that a map interaction is happening
        SessionService.setMapInteractionState(true);

        onStationSelect(clickedStation);

        // After a short delay, tell SessionService map interaction has ended
        setTimeout(() => {
          SessionService.setMapInteractionState(false);
        }, 500);
      }
    },
    [onStationSelect],
  );

  // Initial map setup
  useEffect(() => {
    let view;
    let mapViewHandles = []; // Store event handles for cleanup

    // Tell SessionService that map is loading (pause session checks)
    SessionService.setMapInteractionState(true);

    // Configure ArcGIS loader for version 4.25
    const options = {
      version: '4.25',
      css: true,
      // Add dojoConfig option to use passive events
      dojoConfig: {
        passiveEvents: true,
        has: {
          'esri-passive-events': true,
        },
      },
    };

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

          // Configure esri to use passive event listeners
          esriConfig.options = {
            ...esriConfig.options,
            events: {
              // Enable passive events for better scrolling performance
              add: { passive: true },
              remove: { passive: true },
            },
          };

          // Enable passive events explicitly
          esriConfig.has = esriConfig.has || {};
          esriConfig.has['esri-passive-events'] = true;

          // Create map instance
          const map = new Map({ basemap: 'topo-vector' });

          // Configure view options for improved performance
          view = new MapView({
            container: mapRef.current,
            map: map,
            zoom: 4,
            center: [-98, 39], // Center on CONUS (Continental US)
            constraints: {
              snapToZoom: false, // Improves performance by disabling snap-to-zoom
            },
            // Disable view animations for performance
            navigation: {
              mouseWheelZoomEnabled: true,
              browserTouchPanEnabled: true,
            },
          });

          // Set navigation specific options for passive events
          if (view.navigation) {
            view.navigation.mouseWheelEventOptions = { passive: true };
            view.navigation.browserTouchPanEventOptions = { passive: true };
          }

          // Optimize performance by reducing view quality during interaction
          view.qualityProfile = 'medium';

          const polygonLayer = new GraphicsLayer();
          const streamLayer = new GraphicsLayer();
          const lakeLayer = new GraphicsLayer();
          const stationLayer = new GraphicsLayer();
          const drawLayer = new GraphicsLayer();

          map.addMany([polygonLayer, streamLayer, lakeLayer, stationLayer, drawLayer]);

          // Create sketch widget for drawing
          const sketch = new Sketch({
            view: view,
            layer: drawLayer,
            creationMode: 'update',
            availableCreateTools: ['polygon'],
            visibleElements: {
              createTools: {
                point: false,
                polyline: false,
                rectangle: true,
                polygon: true,
                circle: true,
              },
              selectionTools: {
                'lasso-selection': false,
                'rectangle-selection': false,
              },
              settingsMenu: false,
              undoRedoMenu: true,
            },
          });

          // Add the sketch widget to the top-right of the view
          view.ui.add(sketch, 'top-right');
          sketch.visible = false; // Hide by default
          sketchRef.current = sketch;

          // Inform SessionService that map interaction is starting when user interacts with map
          const handleMapInteractionStart = () => {
            SessionService.setMapInteractionState(true);
          };

          const handleMapInteractionEnd = debounce(() => {
            SessionService.setMapInteractionState(false);
          }, 1000);

          // Add handlers for map interactions
          const interactionEvents = ['mouse-wheel', 'key-down', 'drag', 'double-click', 'click'];
          interactionEvents.forEach((eventName) => {
            const handle = view.on(eventName, handleMapInteractionStart);
            mapViewHandles.push(handle);
          });

          // Add a handler for when interactions end
          const dragEndHandle = view.on('drag-end', handleMapInteractionEnd);
          const keyUpHandle = view.on('key-up', handleMapInteractionEnd);
          const clickHandle = view.on('click', handleMapInteractionEnd);
          mapViewHandles.push(dragEndHandle, keyUpHandle, clickHandle);

          // Handle polygon creation complete
          const sketchCreateHandle = sketch.on('create', (event) => {
            // Inform SessionService that a sketch interaction is happening
            SessionService.setMapInteractionState(true);

            if (event.state === 'complete' && onDrawComplete) {
              const polygon = event.graphic.geometry;
              // Highlight the created polygon
              const highlightGraphic = new Graphic({
                geometry: polygon,
                symbol: {
                  type: 'simple-fill',
                  color: [0, 255, 255, 0.2],
                  outline: { color: [0, 255, 255, 0.8], width: 2 },
                },
              });
              drawLayer.add(highlightGraphic);

              // Find all stations within the polygon
              if (stationPoints.length > 0) {
                const stationsWithin = stationPoints.filter((station) => {
                  if (!station.geometry || !station.geometry.coordinates) return false;
                  const [longitude, latitude] = station.geometry.coordinates;
                  const point = new Point({
                    longitude,
                    latitude,
                    spatialReference: { wkid: 4326 },
                  });
                  // Check if point is within polygon
                  return geometryEngine.contains(polygon, point);
                });

                if (stationsWithin.length > 0) {
                  // Convert to format expected by the selection handler
                  const selectedStations = stationsWithin.map((station) => ({
                    SiteNumber: station.properties.SiteNumber,
                    SiteName: station.properties.SiteName,
                  }));

                  // Highlight selected stations
                  stationsWithin.forEach((station) => {
                    const [longitude, latitude] = station.geometry.coordinates;
                    const highlightPoint = new Point({
                      longitude,
                      latitude,
                      spatialReference: { wkid: 4326 },
                    });
                    const highlightGraphic = new Graphic({
                      geometry: highlightPoint,
                      symbol: {
                        type: 'simple-marker',
                        color: [255, 165, 0, 0.8],
                        size: '12px',
                        outline: { color: [255, 255, 255], width: 2 },
                      },
                    });
                    drawLayer.add(highlightGraphic);
                  });

                  onDrawComplete(selectedStations);
                } else {
                  // No stations found in the selection
                  if (drawLayer && !drawLayer.destroyed) {
                    // Add a temporary message graphic
                    const center = geometryEngine.centroid(polygon);
                    const textGraphic = new Graphic({
                      geometry: center,
                      symbol: {
                        type: 'text',
                        color: 'red',
                        haloColor: 'white',
                        haloSize: '1px',
                        text: 'No stations found in selection',
                        yoffset: 0,
                        font: { size: 12, weight: 'bold' },
                      },
                    });
                    drawLayer.add(textGraphic);

                    // Remove the message after 3 seconds
                    setTimeout(() => {
                      if (drawLayer && !drawLayer.destroyed) {
                        drawLayer.remove(textGraphic);
                      }
                    }, 3000);
                  }
                }
              }
            }

            // Tell SessionService map interaction has ended
            setTimeout(() => {
              SessionService.setMapInteractionState(false);
            }, 500);
          });
          mapViewHandles.push(sketchCreateHandle);

          // Find closest station - optimized version
          const findClosestStation = (clickPoint, thresholdDegrees) => {
            if (!stationPoints || stationPoints.length === 0) return;

            // Convert to geographic coordinates if needed
            const geographic = webMercatorUtils.webMercatorToGeographic(clickPoint);
            const clickLat = geographic ? geographic.latitude : clickPoint.latitude;
            const clickLon = geographic ? geographic.longitude : clickPoint.longitude;

            let closestStation = null;
            let minDistance = Number.MAX_VALUE;

            // Find closest station with early termination for performance
            const maxCheckStations = Math.min(500, stationPoints.length); // Limit the number of stations to check for performance
            let checkedCount = 0;

            for (const station of stationPoints) {
              if (checkedCount++ > maxCheckStations) break;

              if (station.geometry && station.geometry.coordinates) {
                const [stationLon, stationLat] = station.geometry.coordinates;

                // Quick filter to avoid unnecessary calculations
                if (
                  Math.abs(clickLat - stationLat) > thresholdDegrees ||
                  Math.abs(clickLon - stationLon) > thresholdDegrees
                ) {
                  continue;
                }

                // Simple distance calculation (not accounting for Earth's curvature)
                const distance = Math.sqrt(
                  Math.pow(clickLat - stationLat, 2) + Math.pow(clickLon - stationLon, 2),
                );

                if (distance < minDistance && distance < thresholdDegrees) {
                  minDistance = distance;
                  closestStation = {
                    SiteNumber: station.properties.SiteNumber,
                    SiteName: station.properties.SiteName,
                    id: station.properties.id,
                  };
                }
              }
            }

            if (closestStation) {
              console.log('Closest station found:', closestStation);
              handleStationSelect(closestStation);
            }
          };

          viewRef.current = {
            view,
            polygonLayer,
            streamLayer,
            lakeLayer,
            stationLayer,
            drawLayer,
            sketch,
            // Store references to these classes for later use
            Graphic,
            Polygon,
            Polyline,
            Point,
            webMercatorUtils,
            geometryEngine,
          };

          // Add coordinate display - debounced for performance
          const updateCoordinates = debounce((event) => {
            const point = view.toMap(event);
            if (point) {
              const geographic = webMercatorUtils.webMercatorToGeographic(point);
              const coordDiv = document.getElementById('coordinateInfo');
              if (coordDiv) {
                coordDiv.innerText = `Lat: ${geographic.latitude.toFixed(6)}, Lon: ${geographic.longitude.toFixed(6)}`;
              }
            }
          }, 50); // Update every 50ms at most

          const pointerMoveHandle = view.on('pointer-move', updateCoordinates);
          mapViewHandles.push(pointerMoveHandle);

          // Add debounced click handler for station selection
          if (onStationSelect) {
            // Use debounce to prevent multiple rapid clicks and improve performance
            const debouncedClickHandler = debounce((event) => {
              // Inform SessionService that a click interaction is happening
              SessionService.setMapInteractionState(true);

              // Only handle clicks if we're showing stations and not in drawing mode
              if (!showStations || drawingMode) {
                console.log(
                  'Click ignored: showStations:',
                  showStations,
                  'drawingMode:',
                  drawingMode,
                );

                // Even though we ignored the click, end the interaction flag
                setTimeout(() => {
                  SessionService.setMapInteractionState(false);
                }, 200);

                return;
              }

              console.log('Map clicked, checking for stations...');
              const screenPoint = {
                x: event.x,
                y: event.y,
              };

              // Use hitTest with specific options to better detect station points
              view
                .hitTest(screenPoint, {
                  include: [stationLayer],
                  tolerance: 10, // Increase hit tolerance to make selection easier
                })
                .then((response) => {
                  console.log('Hit test results:', response.results?.length || 0);

                  // Filter for graphics from the station layer
                  const stationGraphics = response.results?.filter(
                    (result) => result.graphic?.layer === stationLayer,
                  );

                  console.log('Station graphics found:', stationGraphics?.length || 0);

                  if (
                    stationGraphics &&
                    stationGraphics.length > 0 &&
                    stationGraphics[0].graphic?.attributes
                  ) {
                    const clickedStation = stationGraphics[0].graphic.attributes;
                    console.log('Station selected:', clickedStation);
                    handleStationSelect(clickedStation);
                  } else {
                    // If no direct hit, try finding the closest station within a threshold
                    findClosestStation(event.mapPoint, 0.05); // ~5km at equator
                  }

                  // End the interaction flag
                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 500);
                })
                .catch((error) => {
                  console.error('Error during hit test:', error);

                  // Still end the interaction flag on error
                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 200);
                });
            }, 250); // 250ms debounce delay

            const clickHandle = view.on('click', debouncedClickHandler);
            mapViewHandles.push(clickHandle);
          }

          // Allow SessionService to check sessions again after map is loaded
          setTimeout(() => {
            SessionService.setMapInteractionState(false);
          }, 2000);

          setIsLoaded(true);
        },
      )
      .catch((error) => {
        console.error('Error loading ArcGIS modules:', error);

        // Allow SessionService to check sessions again if module loading fails
        SessionService.setMapInteractionState(false);
      });

    return () => {
      // Allow SessionService to check sessions during cleanup
      SessionService.setMapInteractionState(false);

      // Properly clean up event handlers to prevent memory leaks
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

      // Clean up highlight if it exists
      if (highlightedFeatureRef.current) {
        try {
          highlightedFeatureRef.current.remove();
        } catch (e) {
          console.warn('Error removing highlight:', e);
        }
        highlightedFeatureRef.current = null;
      }

      if (view) {
        try {
          view.destroy();
        } catch (e) {
          console.warn('Error destroying view:', e);
        }
        viewRef.current = null;
      }
    };
  }, [
    onStationSelect,
    showStations,
    onDrawComplete,
    drawingMode,
    stationPoints,
    handleStationSelect,
  ]);

  // Toggle drawing mode
  useEffect(() => {
    if (!viewRef.current || !isLoaded) return;
    const { sketch, drawLayer } = viewRef.current;

    if (sketch) {
      sketch.visible = drawingMode;
      if (drawingMode) {
        // Inform SessionService that drawing is starting
        SessionService.setMapInteractionState(true);

        sketch.create('polygon'); // Start with polygon drawing tool
      } else {
        sketch.cancel(); // Cancel any active drawing
        if (drawLayer && !drawLayer.destroyed) {
          drawLayer.removeAll(); // Clear drawn polygons
        }

        // Allow SessionService to check sessions again
        setTimeout(() => {
          SessionService.setMapInteractionState(false);
        }, 500);
      }
    }
  }, [drawingMode, isLoaded]);

  // Handle watershed geometries (existing functionality)
  useEffect(() => {
    if (!viewRef.current || !isLoaded) {
      return;
    }

    // Set loading state to true when starting to load geometries
    setIsLoadingGeometries(true);

    // Inform SessionService that map is updating
    SessionService.setMapInteractionState(true);

    const { view, polygonLayer, streamLayer, lakeLayer } = viewRef.current;

    // Check for destroyed layers before clearing
    if (polygonLayer && !polygonLayer.destroyed) polygonLayer.removeAll();
    if (streamLayer && !streamLayer.destroyed) streamLayer.removeAll();
    if (lakeLayer && !lakeLayer.destroyed) lakeLayer.removeAll();

    loadModules(['esri/Graphic', 'esri/geometry/Polygon', 'esri/geometry/Polyline'], {
      version: '4.25',
    })
      .then(([Graphic, Polygon, Polyline]) => {
        const allGraphics = [];

        // Process geometries in chunks for better performance
        const processGeometries = (geometries, layer, options) => {
          if (!layer || layer.destroyed) return;

          const chunkSize = 50; // Process in chunks of 50
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

        // Process each geometry type
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
                  // Silently catch errors during animation
                  console.warn('Error during map navigation:', e);
                })
                .finally(() => {
                  // Set loading state to false after geometries are loaded
                  setIsLoadingGeometries(false);

                  // Allow SessionService to check sessions again
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
          // If no graphics to load, still set loading to false
          setIsLoadingGeometries(false);

          // Allow SessionService to check sessions again
          setTimeout(() => {
            SessionService.setMapInteractionState(false);
          }, 500);
        }
      })
      .catch((error) => {
        console.error('Error loading modules:', error);
        setIsLoadingGeometries(false);

        // Allow SessionService to check sessions again on error
        SessionService.setMapInteractionState(false);
      });
  }, [geometries, streamsGeometries, lakesGeometries, isLoaded]);

  // Handle station points rendering
  useEffect(() => {
    if (!viewRef.current || !isLoaded || !showStations) {
      return;
    }

    // Inform SessionService that map is updating
    SessionService.setMapInteractionState(true);

    const { stationLayer, Graphic, Point, view } = viewRef.current;

    // Safe check before removing all - prevent null reference errors
    if (stationLayer && !stationLayer.destroyed) {
      stationLayer.removeAll();
    } else {
      // End map interaction state if stationLayer is unavailable
      SessionService.setMapInteractionState(false);
      return; // Exit if layer is destroyed
    }

    // Only render stations if we're in station selection mode
    if (showStations && stationPoints.length > 0) {
      console.log(`Rendering ${stationPoints.length} station points on map`);

      const stationGraphics = [];

      // Process stations in batches for better performance
      const batchSize = 100;
      for (let i = 0; i < stationPoints.length; i += batchSize) {
        const batch = stationPoints.slice(i, i + batchSize);

        batch.forEach((station) => {
          if (station.geometry && station.geometry.coordinates) {
            const [longitude, latitude] = station.geometry.coordinates;
            const point = new Point({
              longitude,
              latitude,
              spatialReference: { wkid: 4326 },
            });

            // Determine if this station is selected
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

            if (!stationLayer.destroyed) {
              stationLayer.add(stationGraphic);
              stationGraphics.push(stationGraphic);
            }
          }
        });
      }

      // Zoom to all stations if there's no selection
      if (stationGraphics.length > 0 && !selectedStationId && view && !view.destroyed) {
        view
          .when(() => {
            view
              .goTo(stationGraphics, { animate: false })
              .catch((error) => {
                console.warn('Error during map navigation:', error);
              })
              .finally(() => {
                // Allow SessionService to check sessions again after zooming
                setTimeout(() => {
                  SessionService.setMapInteractionState(false);
                }, 500);
              });
          })
          .catch((e) => {
            console.warn('Error during view.when():', e);
            SessionService.setMapInteractionState(false);
          });
      } else {
        // Allow SessionService to check sessions even if not zooming
        setTimeout(() => {
          SessionService.setMapInteractionState(false);
        }, 500);
      }

      // Add custom pointer cursor when hovering over stations
      if (view && stationLayer && !view.destroyed && !stationLayer.destroyed) {
        view
          .whenLayerView(stationLayer)
          .then((layerView) => {
            // Clear any existing highlight
            if (highlightedFeatureRef.current) {
              try {
                highlightedFeatureRef.current.remove();
              } catch (e) {
                // Silent catch - the highlight might already be removed
              }
              highlightedFeatureRef.current = null;
            }

            // Debounce the pointer move for better performance
            const debouncedPointerMove = debounce((event) => {
              // We're moving the pointer, so let SessionService know
              SessionService.setMapInteractionState(true);

              if (view.destroyed) {
                SessionService.setMapInteractionState(false);
                return;
              }

              view
                .hitTest(event)
                .then((response) => {
                  // Remove any existing highlight
                  if (highlightedFeatureRef.current) {
                    try {
                      highlightedFeatureRef.current.remove();
                    } catch (e) {
                      // Silent catch
                    }
                    highlightedFeatureRef.current = null;

                    if (view.container) {
                      view.container.style.cursor = 'default';
                    }
                  }

                  const stationHits = response.results?.filter(
                    (result) => result.graphic?.layer === stationLayer,
                  );

                  if (stationHits && stationHits.length > 0 && view.container) {
                    // Change cursor to pointer when over a station
                    view.container.style.cursor = 'pointer';

                    // Optionally highlight the station
                    try {
                      highlightedFeatureRef.current = layerView.highlight(stationHits[0].graphic);
                    } catch (e) {
                      console.warn('Error highlighting feature:', e);
                    }
                  }

                  // End the interaction state after processing
                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 200);
                })
                .catch((error) => {
                  console.warn('Hit test error:', error);
                  SessionService.setMapInteractionState(false);
                });
            }, 50); // 50ms debounce

            // Watch for pointer move over the layer
            const pointerMoveHandle = view.on('pointer-move', debouncedPointerMove);

            // Store the handle for cleanup
            const currentHandles = viewRef.current.pointerMoveHandles || [];
            viewRef.current.pointerMoveHandles = [...currentHandles, pointerMoveHandle];
          })
          .catch((error) => {
            console.warn('Error creating layer view:', error);
            // Still end the interaction state on error
            SessionService.setMapInteractionState(false);
          });
      } else {
        // End the interaction state if view or layer is not available
        SessionService.setMapInteractionState(false);
      }
    } else {
      // End the interaction state if not showing stations
      SessionService.setMapInteractionState(false);
    }

    // Cleanup function for this effect
    return () => {
      // End the map interaction state during cleanup
      SessionService.setMapInteractionState(false);

      if (viewRef.current && viewRef.current.pointerMoveHandles) {
        viewRef.current.pointerMoveHandles.forEach((handle) => {
          if (handle && typeof handle.remove === 'function') {
            try {
              handle.remove();
            } catch (e) {
              console.warn('Error removing pointer move handle:', e);
            }
          }
        });
        viewRef.current.pointerMoveHandles = [];
      }

      // Clear any existing highlight
      if (highlightedFeatureRef.current) {
        try {
          highlightedFeatureRef.current.remove();
        } catch (e) {
          // Silent catch
        }
        highlightedFeatureRef.current = null;
      }
    };
  }, [stationPoints, showStations, selectedStationId, isLoaded]);

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
      {/* Loading spinner overlay */}
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
