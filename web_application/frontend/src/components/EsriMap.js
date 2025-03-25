import React, { useEffect, useRef, useState } from 'react';
import { loadModules } from 'esri-loader';

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
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoadingGeometries, setIsLoadingGeometries] = useState(false);

  // Initial map setup
  useEffect(() => {
    let view;

    loadModules([
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
    ]).then(
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
      ]) => {
        if (viewRef.current) return;

        const map = new Map({ basemap: 'topo-vector' });
        view = new MapView({
          container: mapRef.current,
          map: map,
          zoom: 4,
          center: [-98, 39], // Center on CONUS (Continental US)
        });

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

        // Handle polygon creation complete
        sketch.on('create', (event) => {
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
                if (drawLayer) {
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
                    drawLayer.remove(textGraphic);
                  }, 3000);
                }
              }
            }
          }
        });

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

        // Add coordinate display
        view.on('pointer-move', (event) => {
          const point = view.toMap(event);
          if (point) {
            const geographic = webMercatorUtils.webMercatorToGeographic(point);
            const coordDiv = document.getElementById('coordinateInfo');
            if (coordDiv) {
              coordDiv.innerText = `Lat: ${geographic.latitude.toFixed(6)}, Lon: ${geographic.longitude.toFixed(6)}`;
            }
          }
        });

        // Add click handler for station selection
        if (onStationSelect) {
          view.on('click', (event) => {
            // Only handle clicks if we're showing stations and not in drawing mode
            if (!showStations || drawingMode) {
              console.log(
                'Click ignored: showStations:',
                showStations,
                'drawingMode:',
                drawingMode,
              );
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
                  (result) => result.graphic.layer === stationLayer,
                );

                console.log('Station graphics found:', stationGraphics?.length || 0);

                if (stationGraphics && stationGraphics.length > 0) {
                  const clickedStation = stationGraphics[0].graphic.attributes;
                  console.log('Station selected:', clickedStation);
                  onStationSelect(clickedStation);
                } else {
                  // If no direct hit, try finding the closest station within a threshold
                  findClosestStation(event.mapPoint, 0.05); // ~5km at equator
                }
              });
          });

          // Add helper function to find closest station
          const findClosestStation = (clickPoint, thresholdDegrees) => {
            if (!stationPoints || stationPoints.length === 0) return;

            // Convert to geographic coordinates if needed
            const geographic = webMercatorUtils.webMercatorToGeographic(clickPoint);
            const clickLat = geographic ? geographic.latitude : clickPoint.latitude;
            const clickLon = geographic ? geographic.longitude : clickPoint.longitude;

            let closestStation = null;
            let minDistance = Number.MAX_VALUE;

            // Find closest station
            stationPoints.forEach((station) => {
              if (station.geometry && station.geometry.coordinates) {
                const [stationLon, stationLat] = station.geometry.coordinates;

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
            });

            if (closestStation) {
              console.log('Closest station found:', closestStation);
              onStationSelect(closestStation);
            }
          };
        }

        setIsLoaded(true);
      },
    );

    return () => {
      if (view) {
        view.destroy();
        viewRef.current = null;
      }
    };
  }, [onStationSelect, showStations, onDrawComplete, drawingMode, stationPoints]);

  // Toggle drawing mode
  useEffect(() => {
    if (!viewRef.current || !isLoaded) return;

    const { sketch } = viewRef.current;

    if (sketch) {
      sketch.visible = drawingMode;

      if (drawingMode) {
        sketch.create('polygon'); // Start with polygon drawing tool
      } else {
        sketch.cancel(); // Cancel any active drawing
        viewRef.current.drawLayer.removeAll(); // Clear drawn polygons
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

    const { view, polygonLayer, streamLayer, lakeLayer } = viewRef.current;
    polygonLayer.removeAll();
    streamLayer.removeAll();
    lakeLayer.removeAll();

    loadModules(['esri/Graphic', 'esri/geometry/Polygon', 'esri/geometry/Polyline']).then(
      ([Graphic, Polygon, Polyline]) => {
        const allGraphics = [];

        geometries.forEach((geom) => {
          if (geom?.coordinates?.length) {
            const polygon = new Polygon({
              rings: geom.coordinates[0],
              spatialReference: { wkid: 4326 },
            });
            const polygonGraphic = new Graphic({
              geometry: polygon,
              symbol: {
                type: 'simple-fill',
                color: [227, 139, 79, 0.6],
                outline: { color: [255, 255, 255], width: 1 },
              },
            });
            polygonLayer.add(polygonGraphic);
            allGraphics.push(polygonGraphic);
          }
        });

        lakesGeometries.forEach((lake) => {
          if (lake?.coordinates?.length) {
            const polygon = new Polygon({
              rings: lake.coordinates[0],
              spatialReference: { wkid: 4326 },
            });
            const polygonGraphic = new Graphic({
              geometry: polygon,
              symbol: {
                type: 'simple-fill',
                color: [0, 0, 255, 0.4],
                outline: { color: [255, 255, 255], width: 1 },
              },
            });
            lakeLayer.add(polygonGraphic);
            allGraphics.push(polygonGraphic);
          }
        });

        streamsGeometries.forEach((stream) => {
          if (stream?.coordinates?.length) {
            const polyline = new Polyline({
              paths: stream.coordinates[0],
              spatialReference: { wkid: 4326 },
            });
            const polylineGraphic = new Graphic({
              geometry: polyline,
              symbol: {
                type: 'simple-line',
                color: [0, 0, 255],
                width: 1,
              },
            });
            streamLayer.add(polylineGraphic);
            allGraphics.push(polylineGraphic);
          }
        });

        if (allGraphics.length > 0) {
          view.when(() => {
            view.goTo(allGraphics).then(() => {
              // Set loading state to false after geometries are loaded and map is zoomed
              setIsLoadingGeometries(false);
            });
          });
        } else {
          // If no graphics to load, still set loading to false
          setIsLoadingGeometries(false);
        }
      },
    );
  }, [geometries, streamsGeometries, lakesGeometries, isLoaded]);

  // Handle station points rendering
  useEffect(() => {
    if (!viewRef.current || !isLoaded || !showStations) {
      return;
    }

    const { stationLayer, Graphic, Point, view } = viewRef.current;

    // Clear existing station points
    stationLayer.removeAll();

    // Only render stations if we're in station selection mode
    if (showStations && stationPoints.length > 0) {
      console.log(`Rendering ${stationPoints.length} station points on map`);
      const stationGraphics = [];

      stationPoints.forEach((station) => {
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

          stationLayer.add(stationGraphic);
          stationGraphics.push(stationGraphic);
        }
      });

      // Zoom to all stations if there's no selection
      if (stationGraphics.length > 0 && !selectedStationId) {
        view.when(() => {
          view.goTo(stationGraphics, { animate: false });
        });
      }

      // Add custom pointer cursor when hovering over stations
      if (view && stationLayer) {
        view.whenLayerView(stationLayer).then((layerView) => {
          let highlightedFeature = null;

          // Watch for pointer move over the layer
          view.on('pointer-move', (event) => {
            view.hitTest(event).then((response) => {
              // Remove any existing highlight
              if (highlightedFeature) {
                highlightedFeature.remove();
                highlightedFeature = null;
                view.container.style.cursor = 'default';
              }

              const stationHits = response.results?.filter(
                (result) => result.graphic.layer === stationLayer,
              );

              if (stationHits && stationHits.length > 0) {
                // Change cursor to pointer when over a station
                view.container.style.cursor = 'pointer';

                // Optionally highlight the station
                highlightedFeature = layerView.highlight(stationHits[0].graphic);
              }
            });
          });
        });
      }
    }
  }, [stationPoints, showStations, selectedStationId, isLoaded]);

  return (
    <>
      <div ref={mapRef} style={{ height: '900px', width: '100%', position: 'relative' }} />
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
