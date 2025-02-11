import React, { useEffect, useRef } from 'react';
import { loadModules } from 'esri-loader';

const EsriMap = ({ geometries = [], streamsGeometries = [], lakesGeometries = [] }) => {
  const mapRef = useRef(null);
  const viewRef = useRef(null);

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
    ]).then(([Map, MapView, Graphic, GraphicsLayer, Polygon, Polyline, Extent]) => {
      if (viewRef.current) return;

      const map = new Map({ basemap: 'topo-vector' });
      view = new MapView({
        container: mapRef.current,
        map: map,
        zoom: 5,
        center: [-90, 38],
      });

      const polygonLayer = new GraphicsLayer();
      const streamLayer = new GraphicsLayer();
      const lakeLayer = new GraphicsLayer();
      map.addMany([polygonLayer, streamLayer, lakeLayer]);

      viewRef.current = { view, polygonLayer, streamLayer, lakeLayer };

      view.on('pointer-move', (event) => {
        const screenPoint = { x: event.x, y: event.y };
        view.toMap(screenPoint).then((mapPoint) => {
          const coordDiv = document.getElementById('coordinateInfo');
          if (coordDiv && mapPoint) {
            coordDiv.innerText = `Lat: ${mapPoint.latitude.toFixed(6)}, Lon: ${mapPoint.longitude.toFixed(6)}`;
          }
        });
      });
    });

    return () => {
      if (view) {
        view.destroy();
        viewRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!viewRef.current) {
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
  }, [geometries, streamsGeometries, lakesGeometries]);

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
