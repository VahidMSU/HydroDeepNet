// components/EsriMap.js
import React, { useEffect, useRef } from 'react';
import { loadModules } from 'esri-loader';

const EsriMap = ({ geometries = [], streamsGeometries = [], station }) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);

  useEffect(() => {
    let view;
    console.log('Loading ArcGIS modules...');

    loadModules([
      'esri/Map',
      'esri/views/MapView',
      'esri/Graphic',
      'esri/layers/GraphicsLayer',
      'esri/geometry/Polygon',
      'esri/geometry/Polyline',
      'esri/geometry/Point',
    ]).then(([Map, MapView, Graphic, GraphicsLayer]) => {
      if (viewRef.current) {
        return;
      }
      console.log('Modules loaded, initializing map...');

      const map = new Map({ basemap: 'topo-vector' });
      view = new MapView({
        container: mapRef.current,
        map: map,
        zoom: 4,
        center: [-90, 38],
      });

      const polygonLayer = new GraphicsLayer();
      const streamLayer = new GraphicsLayer();
      const stationLayer = new GraphicsLayer();
      map.addMany([polygonLayer, streamLayer, stationLayer]);

      viewRef.current = { view, polygonLayer, streamLayer, stationLayer };
    });

    return () => {
      if (view) {
        console.log('Destroying view...');
        view.destroy();
        viewRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!viewRef.current) {
      return;
    }
    console.log('Updating layer graphics...');
    console.log('Geometries:', geometries);
    console.log('Stream Geometries:', streamsGeometries);
    console.log('Station:', station);

    const { polygonLayer, streamLayer, stationLayer } = viewRef.current;
    polygonLayer.removeAll();
    streamLayer.removeAll();
    stationLayer.removeAll();

    loadModules([
      'esri/Graphic',
      'esri/geometry/Polygon',
      'esri/geometry/Polyline',
      'esri/geometry/Point',
    ]).then(([Graphic, Polygon, Polyline, Point]) => {
      geometries.forEach((geom) => {
        if (geom?.coordinates) {
          console.log('Adding polygon:', geom.coordinates);
          const polygon = new Polygon({
            rings: geom.coordinates,
            spatialReference: { wkid: 4326 },
          });
          const polygonGraphic = new Graphic({
            geometry: polygon,
            symbol: {
              type: 'simple-fill',
              color: [227, 139, 79, 0.8],
              outline: { color: [255, 255, 255], width: 1 },
            },
          });
          polygonLayer.add(polygonGraphic);
        }
      });

      streamsGeometries.forEach((stream) => {
        if (stream?.coordinates) {
          console.log('Adding stream line:', stream.coordinates);
          const polyline = new Polyline({
            paths: stream.coordinates,
            spatialReference: { wkid: 4326 },
          });
          const polylineGraphic = new Graphic({
            geometry: polyline,
            symbol: {
              type: 'simple-line',
              color: [0, 0, 255],
              width: 0.5,
            },
          });
          streamLayer.add(polylineGraphic);
        }
      });

      if (station?.Latitude && station?.Longitude) {
        console.log('Going to station location:', station);
        viewRef.current.view.goTo({
          center: [station.Longitude, station.Latitude],
          zoom: 10,
        });
        const stationMarker = new Graphic({
          geometry: new Point({
            longitude: station.Longitude,
            latitude: station.Latitude,
          }),
          symbol: {
            type: 'simple-marker',
            color: [226, 119, 40],
            outline: { color: [255, 255, 255], width: 2 },
          },
        });
        stationLayer.add(stationMarker);
      }
    });
  }, [geometries, streamsGeometries, station]);

  return <div ref={mapRef} style={{ height: '900px', width: '100%' }} />;
};

export default EsriMap;
