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

const EsriMap = ({
  geometries = [],
  streamsGeometries = [],
  lakesGeometries = [],
  stationPoints = [],
  onStationSelect = null,
  showStations = false,
  selectedStationId = null,
  refreshMapRef = null, // Add new prop to expose refresh functionality
  basemapType = 'streets', // This prop will control which basemap to display
  showWatershed = true, // New prop to control watershed visibility
  showStreams = true, // New prop to control streams visibility
  showLakes = true, // New prop to control lakes visibility
}) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);
  const highlightedFeatureRef = useRef(null);
  const touchStartRef = useRef(null); // Add this ref to track touch positions
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoadingGeometries, setIsLoadingGeometries] = useState(false);
  const [stationsLoaded, setStationsLoaded] = useState(false);
  const prevStationPointsRef = useRef([]);
  const visibilityRef = useRef(true);
  const [currentBasemap, setCurrentBasemap] = useState(basemapType);

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
      visibilityRef.current = isVisible;

      if (isVisible && showStations && stationPoints.length > 0 && viewRef.current) {
        const { stationLayer } = viewRef.current;
        if (stationLayer && !stationLayer.destroyed && stationLayer.graphics.length === 0) {
          setTimeout(renderStationPoints, 500);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [showStations, stationPoints]);

  const handleTouchStart = (event) => {
    if (event.touches && event.touches.length > 0) {
      touchStartRef.current = {
        x: event.touches[0].clientX,
        y: event.touches[0].clientY,
        time: Date.now(),
      };
    }
  };

  const handleTouchMove = (event) => {
    if (touchStartRef.current && event.touches && event.touches.length > 0) {
      const deltaX = event.touches[0].clientX - touchStartRef.current.x;
      const deltaY = event.touches[0].clientY - touchStartRef.current.y;

      if (viewRef.current && viewRef.current.view) {
        // Example: could update UI elements based on touch movement
      }
    }
  };

  const handleTouchEnd = (event) => {
    if (touchStartRef.current) {
      const touchEndTime = Date.now();
      const touchDuration = touchEndTime - touchStartRef.current.time;

      touchStartRef.current = null;

      if (touchDuration < 300) {
        // This could be a tap/click equivalent
      }
    }
  };

  const addTouchListeners = () => {
    if (viewRef.current && viewRef.current.view && viewRef.current.view.container) {
      viewRef.current.view.container.addEventListener('touchstart', handleTouchStart, {
        passive: true,
      });
      viewRef.current.view.container.addEventListener('touchmove', handleTouchMove, {
        passive: true,
      });
      viewRef.current.view.container.addEventListener('touchend', handleTouchEnd, {
        passive: true,
      });
    }
  };

  const removeTouchListeners = () => {
    if (viewRef.current && viewRef.current.view && viewRef.current.view.container) {
      viewRef.current.view.container.removeEventListener('touchstart', handleTouchStart);
      viewRef.current.view.container.removeEventListener('touchmove', handleTouchMove);
      viewRef.current.view.container.removeEventListener('touchend', handleTouchEnd);
    }
  };

  useEffect(() => {
    if (viewRef.current && viewRef.current.view && !viewRef.current.touchListenersAdded) {
      addTouchListeners();
      viewRef.current.touchListenersAdded = true;
    }
  }, [isLoaded]);

  const renderStationPoints = useCallback(() => {
    if (!viewRef.current || !isLoaded || !showStations || stationPoints.length === 0) {
      return;
    }

    SessionService.setMapInteractionState(true);

    const { stationLayer, Graphic, Point, view } = viewRef.current;

    if (stationLayer && !stationLayer.destroyed) {
      if (prevStationPointsRef.current === stationPoints && stationLayer.graphics.length > 0) {
        SessionService.setMapInteractionState(false);
        return;
      }

      stationLayer.removeAll();
    } else {
      SessionService.setMapInteractionState(false);
      return;
    }

    const stationGraphics = [];
    let successfullyAdded = 0;
    let failedToAdd = 0;

    const batchSize = 200;

    const handleGraphicCreationError = () => {
      failedToAdd++;
    };

    const handleBatchAddError = (batchSize) => {
      failedToAdd += batchSize;
      successfullyAdded -= batchSize;
    };

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
            handleGraphicCreationError();
          }
        } else {
          handleGraphicCreationError();
        }
      });

      if (!stationLayer.destroyed && batchGraphics.length > 0) {
        try {
          stationLayer.addMany(batchGraphics);
        } catch (err) {
          batchGraphics.forEach((graphic) => {
            try {
              stationLayer.add(graphic);
            } catch (innerErr) {
              handleGraphicCreationError();
            }
          });
        }
      }
    }

    prevStationPointsRef.current = stationPoints;

    setStationsLoaded(true);

    if (stationGraphics.length > 0 && !selectedStationId && view && !view.destroyed) {
      try {
        setTimeout(() => {
          if (!view.destroyed) {
            view
              .goTo(stationGraphics, { animate: false })
              .catch((error) => {})
              .finally(() => {
                setTimeout(() => SessionService.setMapInteractionState(false), 500);
              });
          } else {
            SessionService.setMapInteractionState(false);
          }
        }, 200);
      } catch (e) {
        SessionService.setMapInteractionState(false);
      }
    } else {
      setTimeout(() => SessionService.setMapInteractionState(false), 500);
    }
  }, [isLoaded, showStations, stationPoints, selectedStationId]);

  useEffect(() => {
    if (viewRef.current && viewRef.current.view && !viewRef.current.view.destroyed && isLoaded) {
      // Update basemap when basemapType prop changes
      if (currentBasemap !== basemapType) {
        setCurrentBasemap(basemapType);

        console.log(`Changing basemap from ${currentBasemap} to ${basemapType}`);

        loadModules(['esri/Map', 'esri/layers/WebTileLayer', 'esri/Basemap'], { version: '4.25' })
          .then(([Map, WebTileLayer, Basemap]) => {
            const view = viewRef.current.view;

            let newMap;
            if (basemapType === 'google') {
              // Create Google aerial basemap using proper URL pattern for Google Satellite
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

              newMap = new Map({
                basemap: googleBasemap,
                qualityProfile: 'high',
              });
            } else {
              // Create street basemap
              newMap = new Map({
                basemap: 'topo-vector',
                qualityProfile: 'high',
              });
            }

            // Transfer existing layers to the new map
            if (view.map) {
              const layers = view.map.layers.toArray();
              layers.forEach((layer) => {
                view.map.remove(layer);
                newMap.add(layer);
              });
            }

            // Set the new map on the view
            view.map = newMap;
          })
          .catch((error) => {
            console.error('Error changing basemap:', error);
          });
      }
    }
  }, [basemapType, isLoaded, currentBasemap]);

  useEffect(() => {
    let view;
    let mapViewHandles = [];

    const loadingStates = {
      viewReady: false,
      modulesLoaded: false,
      stationsProcessing: false,
      renderComplete: false,
    };

    SessionService.setMapInteractionState(true);

    const options = {
      version: '4.25',
      css: true,
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
        'esri/layers/WebTileLayer',
        'esri/Basemap',
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
          WebTileLayer,
          Basemap,
        ]) => {
          if (viewRef.current) return;

          loadingStates.modulesLoaded = true;

          esriConfig.has = esriConfig.has || {};
          esriConfig.has['esri-passive-events'] = true;

          esriConfig.options = {
            ...esriConfig.options,
            events: {
              add: { passive: true },
              remove: { passive: true },
            },
          };

          // Set basemap options based on basemapType prop
          let mapOptions = {};
          if (basemapType === 'google') {
            // Create Google aerial basemap using WebTileLayer for better compatibility
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

            mapOptions = {
              basemap: googleBasemap,
              qualityProfile: 'high',
            };
          } else {
            mapOptions = { basemap: 'topo-vector', qualityProfile: 'high' };
          }

          const map = new Map(mapOptions);
          setCurrentBasemap(basemapType);

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
            loadingOptimization: true,
            popup: {
              dockEnabled: false,
              dockOptions: {
                buttonEnabled: false,
                breakpoint: false,
              },
            },
          });

          if (view.navigation) {
            view.navigation.mouseWheelEventOptions = { passive: true };
            view.navigation.browserTouchPanEventOptions = { passive: true };
          }

          const polygonLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const streamLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const lakeLayer = new GraphicsLayer({ elevationInfo: { mode: 'on-the-ground' } });
          const stationLayer = new GraphicsLayer({
            elevationInfo: { mode: 'on-the-ground' },
            title: 'Stream Gauge Stations',
          });

          map.addMany([polygonLayer, streamLayer, lakeLayer, stationLayer]);

          view.when(() => {
            loadingStates.viewReady = true;

            if (showStations && stationPoints.length > 0) {
              renderStationPointsWithBatching(stationLayer, Graphic, Point);
            }

            if (onStationSelect) {
              const pointerMoveHandler = view.on(
                'pointer-move',
                debounce((event) => {
                  if (!showStations || loadingStates.stationsProcessing) {
                    return;
                  }

                  const screenPoint = {
                    x: event.x,
                    y: event.y,
                  };

                  view
                    .hitTest(screenPoint, {
                      include: [stationLayer],
                    })
                    .then((response) => {
                      const stationGraphic = response.results?.find(
                        (result) => result.graphic?.layer === stationLayer,
                      )?.graphic;

                      const graphics = stationLayer.graphics.toArray();

                      graphics.forEach((graphic) => {
                        if (!graphic.symbol) return;

                        const isSelected = graphic.attributes.SiteNumber === selectedStationId;
                        const isHovered = stationGraphic && graphic === stationGraphic;

                        if (isHovered && graphic.symbol.size !== '10px' && !isSelected) {
                          graphic.symbol = {
                            type: 'simple-marker',
                            color: [0, 114, 206, 0.9],
                            size: '10px',
                            outline: {
                              color: [255, 255, 255],
                              width: 2,
                            },
                          };

                          mapRef.current.style.cursor = 'pointer';
                        } else if (!isHovered && !isSelected && graphic.symbol.size !== '8px') {
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

                      if (!stationGraphic) {
                        mapRef.current.style.cursor = 'default';
                      }
                    })
                    .catch((err) => {});
                }, 50),
              );

              mapViewHandles.push(pointerMoveHandler);

              const clickHandler = view.on('click', (event) => {
                event.stopPropagation();

                if (!showStations || loadingStates.stationsProcessing) {
                  return;
                }

                SessionService.setMapInteractionState(true);

                const screenPoint = {
                  x: event.x,
                  y: event.y,
                };

                view
                  .hitTest(screenPoint, {
                    include: [stationLayer],
                  })
                  .then((response) => {
                    const stationGraphics = response.results?.filter(
                      (result) => result.graphic?.layer === stationLayer,
                    );

                    if (
                      stationGraphics &&
                      stationGraphics.length > 0 &&
                      stationGraphics[0].graphic?.attributes
                    ) {
                      const clickedStation = stationGraphics[0].graphic.attributes;
                      handleStationSelect(clickedStation);
                    }

                    setTimeout(() => {
                      SessionService.setMapInteractionState(false);
                    }, 500);
                  })
                  .catch((error) => {
                    SessionService.setMapInteractionState(false);
                  });
              });

              mapViewHandles.push(clickHandler);
            }
          });

          const handleMapInteractionStart = () => {
            SessionService.setMapInteractionState(true);
          };

          const handleMapInteractionEnd = debounce(() => {
            if (!loadingStates.stationsProcessing) {
              SessionService.setMapInteractionState(false);
            }
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

          const renderStationPointsWithBatching = (stationLayer, Graphic, Point) => {
            if (!stationPoints.length) return;

            loadingStates.stationsProcessing = true;

            stationLayer.removeAll();

            const useWebWorker = window.Worker && stationPoints.length > 1000;

            if (useWebWorker) {
              processStationPointsWithWorker(stationPoints, stationLayer, Graphic, Point);
            } else {
              processStationPointsDirectly(stationPoints, stationLayer, Graphic, Point);
            }
          };

          viewRef.current.renderStations = renderStationPointsWithBatching;

          setIsLoaded(true);
        },
      )
      .catch((error) => {
        SessionService.setMapInteractionState(false);
      });

    const processStationPointsDirectly = (stationPoints, stationLayer, Graphic, Point) => {
      const batchSize = 200;
      let processedCount = 0;
      let failedCount = 0;

      const processBatch = (startIdx) => {
        const endIdx = Math.min(startIdx + batchSize, stationPoints.length);
        const batchGraphics = [];

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

        if (batchGraphics.length > 0 && !stationLayer.destroyed) {
          try {
            stationLayer.addMany(batchGraphics);
          } catch (error) {
            failedCount += batchGraphics.length;
            processedCount -= batchGraphics.length;
          }
        }

        if (endIdx < stationPoints.length) {
          requestAnimationFrame(() => processBatch(endIdx));
        } else {
          prevStationPointsRef.current = stationPoints;
          setStationsLoaded(true);
          loadingStates.stationsProcessing = false;
          loadingStates.renderComplete = true;

          SessionService.setMapInteractionState(false);
        }
      };

      processBatch(0);
    };

    const processStationPointsWithWorker = (stationPoints, stationLayer, Graphic, Point) => {
      processStationPointsDirectly(stationPoints, stationLayer, Graphic, Point);
    };

    return () => {
      loadingStates.stationsProcessing = false;

      SessionService.setMapInteractionState(false);

      if (mapViewHandles.length > 0) {
        mapViewHandles.forEach((handle) => {
          if (handle && typeof handle.remove === 'function') {
            try {
              handle.remove();
            } catch (e) {}
          }
        });
      }

      if (highlightedFeatureRef.current) {
        try {
          highlightedFeatureRef.current.remove();
        } catch (e) {}
        highlightedFeatureRef.current = null;
      }

      if (view) {
        try {
          view.destroy();
        } catch (e) {}
        viewRef.current = null;
      }

      removeTouchListeners();
    };
  }, [
    onStationSelect,
    showStations,
    stationPoints,
    selectedStationId,
    handleStationSelect,
    renderStationPoints,
    removeTouchListeners,
  ]);

  useEffect(() => {
    if (!viewRef.current || !isLoaded) {
      return;
    }

    if (showStations) {
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
        renderStationPoints();
      }
    }, 3000);

    return () => {
      clearInterval(watchInterval);
    };
  }, [stationPoints, showStations, isLoaded, renderStationPoints]);

  const refreshMap = useCallback(() => {
    if (!viewRef.current || !isLoaded) {
      return false;
    }

    if (viewRef.current.stationLayer && !viewRef.current.stationLayer.destroyed) {
      viewRef.current.stationLayer.removeAll();
    }

    prevStationPointsRef.current = [];

    if (showStations && stationPoints.length > 0) {
      setTimeout(() => {
        renderStationPoints();

        setTimeout(() => {
          if (viewRef.current?.stationLayer?.graphics.length === 0) {
            renderStationPoints();
          }
        }, 1000);
      }, 100);
    }

    if (viewRef.current.view && !viewRef.current.view.destroyed) {
      try {
        const initialExtent = {
          center: [-98, 39],
          zoom: 4,
        };

        viewRef.current.view
          .goTo(initialExtent, {
            duration: 500,
            easing: 'ease-in-out',
          })
          .catch((err) => {});
      } catch (e) {}
    }

    return true;
  }, [isLoaded, showStations, stationPoints, renderStationPoints]);

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

        // Only process each geometry type if it's visible
        if (geometries.length > 0 && showWatershed) {
          processGeometries(geometries, polygonLayer, {
            type: 'polygon',
            symbol: {
              type: 'simple-fill',
              color: [227, 139, 79, 0.6],
              outline: { color: [255, 255, 255], width: 1 },
            },
          });
        }

        if (lakesGeometries.length > 0 && showLakes) {
          processGeometries(lakesGeometries, lakeLayer, {
            type: 'polygon',
            symbol: {
              type: 'simple-fill',
              color: [0, 0, 255, 0.4],
              outline: { color: [255, 255, 255], width: 1 },
            },
          });
        }

        if (streamsGeometries.length > 0 && showStreams) {
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
                .catch((e) => {})
                .finally(() => {
                  setIsLoadingGeometries(false);

                  setTimeout(() => {
                    SessionService.setMapInteractionState(false);
                  }, 500);
                });
            })
            .catch((e) => {
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
        setIsLoadingGeometries(false);

        SessionService.setMapInteractionState(false);
      });
  }, [
    geometries,
    streamsGeometries,
    lakesGeometries,
    isLoaded,
    showWatershed,
    showStreams,
    showLakes,
  ]);

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
