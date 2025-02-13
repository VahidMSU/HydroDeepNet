//MapComponent.js
import React, { useEffect, useCallback, useRef } from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Sketch from '@arcgis/core/widgets/Sketch';
import Legend from '@arcgis/core/widgets/Legend';
import BasemapToggle from '@arcgis/core/widgets/BasemapToggle';
import Measurement from '@arcgis/core/widgets/Measurement';
import ScaleBar from '@arcgis/core/widgets/ScaleBar';
import CoordinateConversion from '@arcgis/core/widgets/CoordinateConversion';
import LayerList from '@arcgis/core/widgets/LayerList';
import Search from '@arcgis/core/widgets/Search';
import SnappingOptions from '@arcgis/core/views/interactive/snapping/SnappingOptions';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import '@arcgis/core/assets/esri/themes/light/main.css';
//import '../styles/map-widgets.css';

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
  } else if (
    geometry.type === 'extent' &&
    (event.state === 'active' || event.state === 'complete')
  ) {
    // Changed condition to update extent during both active and complete phases.
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

const MapComponent = ({ geometries, setFormData, onGeometryChange, centerCoordinates }) => {
  const viewRef = useRef(null);

  const handleDrawEvent = useCallback(
    (event) => {
      const { graphic, graphics } = event;
      let newGeometry = null;
      if (graphic && graphic.geometry) {
        newGeometry = graphic.geometry.toJSON();
      } else if (graphics && graphics[0].geometry) {
        newGeometry = graphics[0].geometry.toJSON();
      }
      if (newGeometry && onGeometryChange) {
        onGeometryChange(newGeometry);
      }
    },
    [onGeometryChange],
  );

  // Initialize map only once on mount
  useEffect(() => {
    let view, sketch, legend, basemapToggle, measurement, coordConversion, search;
    const initialize = async () => {
      const graphicsLayer = new GraphicsLayer({
        title: 'Drawing Layer',
        listMode: 'show',
      });
      const map = new Map({
        basemap: 'topo-vector',
        layers: [graphicsLayer],
      });
      view = new MapView({
        container: 'viewDiv',
        map,
        center: [-85.6024, 44.3148], // Center on Michigan
        zoom: 7, // Zoom level to show most of Michigan
        constraints: {
          snapToZoom: true,
          rotationEnabled: false,
          geometry: {
            type: 'extent',
            xmin: -89.5, // Western boundary
            ymin: 41.5, // Southern boundary
            xmax: -82.5, // Eastern boundary
            ymax: 47.0, // Northern boundary
          },
        },
        popup: {
          dockEnabled: true,
          dockOptions: { position: 'bottom-right', breakpoint: false },
        },
        snappingOptions: new SnappingOptions({
          enabled: true,
          selfEnabled: true,
          featureSources: [{ layer: graphicsLayer }],
        }),
      });
      try {
        await view.when();
        // Initialize widgets (sketch, legend, basemapToggle, measurement, scaleBar, coordConversion, search)
        sketch = new Sketch({
          view,
          layer: graphicsLayer,
          creationMode: 'single',
          availableCreateTools: ['point', 'polygon', 'rectangle'],
          layout: 'vertical',
          defaultCreateOptions: { mode: 'hybrid' },
          defaultUpdateOptions: {
            enableRotation: false,
            enableScaling: false,
            multipleSelectionEnabled: false,
          },
          visibleElements: { settingsMenu: false, undoRedoMenu: true, selectionTools: false },
        });
        legend = new Legend({ view, style: { type: 'card', layout: 'auto' } });
        basemapToggle = new BasemapToggle({ view, nextBasemap: 'satellite' });
        measurement = new Measurement({ view, activeTool: 'distance' });
        const scaleBar = new ScaleBar({ view, unit: 'dual' });
        coordConversion = new CoordinateConversion({ view });
        search = new Search({ view, popupEnabled: true, position: 'top-left' });
        // Add widgets to the view
        view.ui.add(search, 'top-right');
        view.ui.add(measurement, 'bottom-right');
        view.ui.add(scaleBar, 'top-left');
        view.ui.add(coordConversion, 'bottom-right');
        view.ui.add(sketch, 'top-right');
        view.ui.add(legend, 'bottom-left');
        view.ui.add(basemapToggle, 'bottom-right');
        view.ui.remove('zoom');
        // Click event handler
        view.on('click', (event) => {
          const point = webMercatorUtils.webMercatorToGeographic(event.mapPoint);
          updatePointFields(point.latitude.toFixed(6), point.longitude.toFixed(6), setFormData);
        });
        sketch.on('create', (event) => {
          if (event.state === 'complete') {
            view.goTo(event.graphic.geometry.extent.expand(1.5));
          }
          handleSketchEvent(event, graphicsLayer, setFormData);
          handleDrawEvent(event);
        });
        sketch.on('update', (event) => {
          handleSketchEvent(event, graphicsLayer, setFormData);
          handleDrawEvent(event);
        });
        view.on('key-down', (event) => {
          const { key } = event;
          if ((key === 'Delete' || key === 'Backspace') && sketch.state === 'active') {
            sketch.cancel();
          }
        });
        // Store view for later use
        viewRef.current = view;
      } catch (error) {
        console.error('Error initializing map:', error);
      }
    };
    initialize();
    // Cleanup on unmount
    return () => {
      if (viewRef.current) {
        viewRef.current.destroy();
        viewRef.current = null;
      }
    };
  }, []); // runs only once

  // Pan the map only when centerCoordinates (selected lat/lon) change
  useEffect(() => {
    if (viewRef.current && centerCoordinates) {
      const { latitude, longitude } = centerCoordinates;
      // Trigger one map pan/zoom to the new center without reinitializing
      viewRef.current.goTo({ center: [parseFloat(longitude), parseFloat(latitude)] });
    }
  }, [centerCoordinates]);

  useEffect(() => {
    if (viewRef.current) {
      const cleanup = handleDrawEvent();
      return () => {
        cleanup();
      };
    }
  }, [viewRef, setFormData]); // Added setFormData to dependencies

  return (
    <>
      <div id="viewDiv" style={{ height: '100%', width: '100%' }}></div>
      <style>{`
        .widget-container {
          padding: 8px;
          background-color: rgba(255, 255, 255, 0.9);
          border-radius: 4px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .esri-widget { font-size: 14px; }
        .esri-sketch { border-radius: 4px; }
        .esri-legend { max-height: 200px; overflow-y: auto; }
        .esri-basemap-toggle { border-radius: 4px; }
        .esri-measurement .esri-measurement__modes { background-color: rgba(255, 255, 255, 0.9); }
        .esri-coordinate-conversion { max-height: 200px; overflow-y: auto; }
        .esri-layer-list { max-height: 400px; overflow-y: auto; }
      `}</style>
    </>
  );
};

export default MapComponent;
