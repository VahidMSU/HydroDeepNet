///data/SWATGenXApp/codes/web_application/frontend/src/components/EsriMap.js
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
      return;
    }

    SessionService.setMapInteractionState(true);

    const { stationLayer, Graphic, Point, view } = viewRef.current;

    if (stationLayer && !stationLayer.destroyed) {
      if (prevStationPointsRef.current === stationPoints && stationLayer.graphics.length > 0) {
        console.log('Station points already rendered - skipping redraw');
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

    const stationGraphics = [];
    let successfullyAdded = 0;

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
            successfullyAdded++;
          }
        }
      });
    }

    prevStationPointsRef.current = stationPoints;

    console.log(`Successfully added ${successfullyAdded} station graphics to map`);
    setStationsLoaded(true);

    if (stationGraphics.length > 0 && !selectedStationId && view && !view.destroyed) {
      try {
        view
          .when(() => {
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
          })
          .catch((e) => {
            console.warn('Error during view.when():', e);
            SessionService.setMapInteractionState(false);
          });
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

    SessionService.setMapInteractionState(true);

    const options = {
      version: '4.25',
      css: true,
      dojoConfig: {
        passiveEvents: false, // Set to false to allow preventDefault
        has: {
          'esri-passive-events': false, // Set to false to allow preventDefault
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

          // Explicitly disable passive events in esriConfig
          esriConfig.options = {
            ...esriConfig.options,
            events: {
              add: { passive: false },
              remove: { passive: false },
            },
          };

          esriConfig.has = esriConfig.has || {};
          esriConfig.has['esri-passive-events'] = false;

          const map = new Map({ basemap: 'topo-vector' });

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
          });

          // Explicitly set wheel event options to non-passive
          if (view.navigation) {
            view.navigation.mouseWheelEventOptions = { passive: false };
            view.navigation.browserTouchPanEventOptions = { passive: true }; // Keep touch panning passive
          }

          // Add a direct wheel event handler to capture and handle events
          mapRef.current.addEventListener(
            'wheel',
            (e) => {
              // Only when the map is the active element, prevent default zooming
              if (
                document.activeElement === mapRef.current ||
                mapRef.current.contains(document.activeElement)
              ) {
                e.preventDefault();
              }
            },
            { passive: false },
          );

          const polygonLayer = new GraphicsLayer();
          const streamLayer = new GraphicsLayer();
          const lakeLayer = new GraphicsLayer();
          const stationLayer = new GraphicsLayer();
          const drawLayer = new GraphicsLayer();

          map.addMany([polygonLayer, streamLayer, lakeLayer, stationLayer, drawLayer]);

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

          view.ui.add(sketch, 'top-right');
          sketch.visible = false;
          sketchRef.current = sketch;

          const handleMapInteractionStart = () => {
            SessionService.setMapInteractionState(true);
          };

          const handleMapInteractionEnd = debounce(() => {
            SessionService.setMapInteractionState(false);
          }, 1000);

          const interactionEvents = ['mouse-wheel', 'key-down', 'drag', 'double-click', 'click'];
          interactionEvents.forEach((eventName) => {
            const handle = view.on(eventName, handleMapInteractionStart);
            mapViewHandles.push(handle);
          });

          const dragEndHandle = view.on('drag-end', handleMapInteractionEnd);
          const keyUpHandle = view.on('key-up', handleMapInteractionEnd);
          const clickHandle = view.on('click', handleMapInteractionEnd);
          mapViewHandles.push(dragEndHandle, keyUpHandle, clickHandle);

          const sketchCreateHandle = sketch.on('create', (event) => {
            SessionService.setMapInteractionState(true);

            if (event.state === 'complete' && onDrawComplete) {
              const polygon = event.graphic.geometry;
              const highlightGraphic = new Graphic({
                geometry: polygon,
                symbol: {
                  type: 'simple-fill',
                  color: [0, 255, 255, 0.2],
                  outline: { color: [0, 255, 255, 0.8], width: 2 },
                },
              });
              drawLayer.add(highlightGraphic);

              if (stationPoints.length > 0) {
                const stationsWithin = stationPoints.filter((station) => {
                  if (!station.geometry || !station.geometry.coordinates) return false;
                  const [longitude, latitude] = station.geometry.coordinates;
                  const point = new Point({
                    longitude,
                    latitude,
                    spatialReference: { wkid: 4326 },
                  });
                  return geometryEngine.contains(polygon, point);
                });

                if (stationsWithin.length > 0) {
                  const selectedStations = stationsWithin.map((station) => ({
                    SiteNumber: station.properties.SiteNumber,
                    SiteName: station.properties.SiteName,
                  }));

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
                  if (drawLayer && !drawLayer.destroyed) {
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

                    setTimeout(() => {
                      if (drawLayer && !drawLayer.destroyed) {
                        drawLayer.remove(textGraphic);
                      }
                    }, 3000);
                  }
                }
              }
            }

            setTimeout(() => {
              SessionService.setMapInteractionState(false);
            }, 500);
          });
          mapViewHandles.push(sketchCreateHandle);

          const findClosestStation = (clickPoint, thresholdDegrees) => {
            if (!stationPoints || stationPoints.length === 0) return;

            const geographic = webMercatorUtils.webMercatorToGeographic(clickPoint);
            const clickLat = geographic ? geographic.latitude : clickPoint.latitude;
            const clickLon = geographic ? geographic.longitude : clickPoint.longitude;

            let closestStation = null;
            let minDistance = Number.MAX_VALUE;

            const maxCheckStations = Math.min(500, stationPoints.length);
            let checkedCount = 0;

            for (const station of stationPoints) {
              if (checkedCount++ > maxCheckStations) break;

              if (station.geometry && station.geometry.coordinates) {
                const [stationLon, stationLat] = station.geometry.coordinates;

                if (
                  Math.abs(clickLat - stationLat) > thresholdDegrees ||
                  Math.abs(clickLon - stationLon) > thresholdDegrees
                ) {
                  continue;
                }

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
            Graphic,
            Polygon,
            Polyline,
            Point,
            webMercatorUtils,
            geometryEngine,
          };

          const updateCoordinates = debounce((event) => {
            const point = view.toMap(event);
            if (point) {
              const geographic = webMercatorUtils.webMercatorToGeographic(point);
              const coordDiv = document.getElementById('coordinateInfo');
              if (coordDiv) {
                coordDiv.innerText = `Lat: ${geographic.latitude.toFixed(6)}, Lon: ${geographic.longitude.toFixed(6)}`;
              }
            }
          }, 50);

          const pointerMoveHandle = view.on('pointer-move', updateCoordinates);
          mapViewHandles.push(pointerMoveHandle);

          if (onStationSelect) {
            const debouncedClickHandler = debounce((event) => {
              SessionService.setMapInteractionState(true);

              if (!showStations || drawingMode) {
                console.log(
                  'Click ignored: showStations:',
                  showStations,
                  'drawingMode:',
                  drawingMode,
                );

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

              view
                .hitTest(screenPoint, {
                  include: [stationLayer],
                  tolerance: 10,
                })
                .then((response) => {
                  console.log('Hit test results:', response.results?.length || 0);

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
                    findClosestStation(event.mapPoint, 0.05);
                  }

                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 500);
                })
                .catch((error) => {
                  console.error('Error during hit test:', error);

                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 200);
                });
            }, 250);

            const clickHandle = view.on('click', debouncedClickHandler);
            mapViewHandles.push(clickHandle);
          }

          setTimeout(() => {
            SessionService.setMapInteractionState(false);
          }, 2000);

          setIsLoaded(true);
        },
      )
      .catch((error) => {
        console.error('Error loading ArcGIS modules:', error);

        SessionService.setMapInteractionState(false);
      });

    return () => {
      SessionService.setMapInteractionState(false);

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

      console.log('EsriMap component unmounted, view and layers cleaned up');
    };
  }, [
    onStationSelect,
    showStations,
    onDrawComplete,
    drawingMode,
    stationPoints,
    handleStationSelect,
  ]);

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
    if (!viewRef.current || !isLoaded) return;
    const { sketch, drawLayer } = viewRef.current;

    if (sketch) {
      sketch.visible = drawingMode;
      if (drawingMode) {
        SessionService.setMapInteractionState(true);

        sketch.create('polygon');
      } else {
        sketch.cancel();
        if (drawLayer && !drawLayer.destroyed) {
          drawLayer.removeAll();
        }

        setTimeout(() => {
          SessionService.setMapInteractionState(false);
        }, 500);
      }
    }
  }, [drawingMode, isLoaded]);

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
