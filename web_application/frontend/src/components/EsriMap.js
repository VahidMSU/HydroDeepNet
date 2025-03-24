import React, { useEffect, useRef, useState } from 'react';
import { loadModules } from 'esri-loader';

const EsriMap = ({
  geometries = [],
  streamsGeometries = [],
  lakesGeometries = [],
  stationPoints = [],
  onStationSelect = null,
  showStations = false,
  selectedStationId = null,
}) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);
  const [isLoaded, setIsLoaded] = useState(false);

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

        map.addMany([polygonLayer, streamLayer, lakeLayer, stationLayer]);

        viewRef.current = {
          view,
          polygonLayer,
          streamLayer,
          lakeLayer,
          stationLayer,
          // Store references to these classes for later use
          Graphic,
          Polygon,
          Polyline,
          Point,
          webMercatorUtils,
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
            // Only handle clicks if we're showing stations
            if (!showStations) return;

            const screenPoint = {
              x: event.x,
              y: event.y,
            };

            // Use hitTest to find if any station was clicked
            view.hitTest(screenPoint).then((response) => {
              const stationGraphics = response.results?.filter(
                (result) => result.graphic.layer === stationLayer,
              );

              if (stationGraphics && stationGraphics.length > 0) {
                const clickedStation = stationGraphics[0].graphic.attributes;
                onStationSelect(clickedStation);
              }
            });
          });
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
  }, [onStationSelect, showStations]);

  // Handle watershed geometries (existing functionality)
  useEffect(() => {
    if (!viewRef.current || !isLoaded) {
      return;
    }

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
            view.goTo(allGraphics);
          });
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
    </>
  );
};

export default EsriMap;
