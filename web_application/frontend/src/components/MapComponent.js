import React, { useEffect } from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Sketch from '@arcgis/core/widgets/Sketch';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import '@arcgis/core/assets/esri/themes/light/main.css';

const updatePointFields = (lat, lon, setFormData) => {
  setFormData((prev) => ({
    ...prev,
    latitude: lat,
    longitude: lon,
  }));
};

const ensureSinglePolygon = (currentGraphic, graphicsLayer) => {
  graphicsLayer.graphics = graphicsLayer.graphics.filter(
    (g) => g.geometry.type !== 'polygon' || g.attributes?.uid === currentGraphic.attributes?.uid,
  );
};

const handleSketchEvent = (event, graphicsLayer, setFormData) => {
  const { graphic } = event;
  if (!graphic || !graphic.geometry) {
    console.warn('No graphic/geometry in event:', event);
    return;
  }
  const { geometry } = graphic;
  const geoGeom = webMercatorUtils.webMercatorToGeographic(geometry);

  if (!graphic.attributes) {
    graphic.attributes = {};
  }
  if (!graphic.attributes.uid) {
    graphic.attributes.uid = `${Date.now()}-${Math.random()}`;
  }

  if (geometry.type === 'polygon') {
    if (event.state === 'start') {
      ensureSinglePolygon(graphic, graphicsLayer);
    }
    if (event.state === 'active' || event.state === 'complete') {
      updatePolygonFields(geoGeom, setFormData);
    }
  } else if (geometry.type === 'extent' && event.state === 'complete') {
    updateExtentFields(geoGeom, setFormData);
  } else if (geometry.type === 'point' && event.state === 'complete') {
    updatePointFields(geoGeom.latitude.toFixed(6), geoGeom.longitude.toFixed(6), setFormData);
  }
};

const updateExtentFields = (extent, setFormData) => {
  setFormData((prev) => ({
    ...prev,
    min_latitude: extent.ymin.toFixed(6),
    max_latitude: extent.ymax.toFixed(6),
    min_longitude: extent.xmin.toFixed(6),
    max_longitude: extent.xmax.toFixed(6),
  }));
};

const updatePolygonFields = (polygon, setFormData) => {
  const coordinates = (polygon.rings && polygon.rings[0]) || [];
  if (!coordinates.length) {
    console.warn('No coordinates in polygon.');
    return;
  }
  const latitudes = coordinates.map((coord) => coord[1]);
  const longitudes = coordinates.map((coord) => coord[0]);
  const minLat = Math.min(...latitudes).toFixed(6);
  const maxLat = Math.max(...latitudes).toFixed(6);
  const minLon = Math.min(...longitudes).toFixed(6);
  const maxLon = Math.max(...longitudes).toFixed(6);

  setFormData((prev) => ({
    ...prev,
    min_latitude: minLat,
    max_latitude: maxLat,
    min_longitude: minLon,
    max_longitude: maxLon,
  }));
};

const MapComponent = ({ setFormData }) => {
  useEffect(() => {
    const graphicsLayer = new GraphicsLayer();
    const map = new Map({
      basemap: 'streets',
      layers: [graphicsLayer],
    });
    const view = new MapView({
      container: 'viewDiv',
      map: map,
      center: [-90, 38],
      zoom: 4,
    });

    const sketch = new Sketch({
      layer: graphicsLayer,
      view: view,
      creationMode: 'update',
      visibleElements: {
        createTools: { point: true, polygon: true, rectangle: true },
        selectionTools: { 'rectangle-selection': true },
      },
    });
    view.ui.add(sketch, 'bottom-left');

    view.on('click', (event) => {
      const geoPoint = webMercatorUtils.webMercatorToGeographic(event.mapPoint);
      updatePointFields(geoPoint.latitude.toFixed(6), geoPoint.longitude.toFixed(6), setFormData);
    });

    sketch.on('create', (event) => handleSketchEvent(event, graphicsLayer, setFormData));
    sketch.on('update', (event) => handleSketchEvent(event, graphicsLayer, setFormData));
  }, [setFormData]);

  return <div id="viewDiv" style={{ height: 'calc(100vh - 100px)' }}></div>;
};

export default MapComponent;
