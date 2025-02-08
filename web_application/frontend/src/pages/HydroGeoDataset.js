// pages/HydroGeoDataset.js
import React, { useState, useEffect } from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Sketch from '@arcgis/core/widgets/Sketch';
import * as webMercatorUtils from '@arcgis/core/geometry/support/webMercatorUtils';
import '@arcgis/core/assets/esri/themes/light/main.css';
import '../css/Layout.css'; // Ensure the path is correct
import '../css/HydroGeoDataset.css'; // Ensure the path is correct

// Ensure all required imports are included
// import * as reactiveUtils from "@arcgis/core/core/reactiveUtils";

const updatePointFields = (lat, lon, setFormData) => {
  setFormData((prev) => ({
    ...prev,
    latitude: lat,
    longitude: lon,
  }));
};

const ensureSinglePolygon = (currentGraphic, graphicsLayer) => {
  // Remove any polygon with a different uid.
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
    // Enforce single polygon.
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

const HydroGeoDataset = () => {
  // Form state for read-only fields and select options.
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: '',
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
  });
  // State to hold fetched data (if any)
  const [data, setData] = useState(null);

  // Initialize the ArcGIS map once on mount
  useEffect(() => {
    const graphicsLayer = new GraphicsLayer();
    const map = new Map({
      basemap: 'streets',
      layers: [graphicsLayer],
    });
    const view = new MapView({
      container: 'viewDiv',
      map: map,
      center: [-90, 38], // adjust as needed
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

    // Handle map click to update point fields
    view.on('click', (event) => {
      const geoPoint = webMercatorUtils.webMercatorToGeographic(event.mapPoint);
      updatePointFields(geoPoint.latitude.toFixed(6), geoPoint.longitude.toFixed(6), setFormData);
    });

    // Handle sketch events
    sketch.on('create', (event) => handleSketchEvent(event, graphicsLayer, setFormData));
    sketch.on('update', (event) => handleSketchEvent(event, graphicsLayer, setFormData));
  }, []);

  // Handle changes for variable/select inputs
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real application, submit formData to an API.
    // For demonstration, we simulate a fetched result.
    const dummyData = {
      variable: formData.variable,
      subvariable: formData.subvariable,
      result: 'Sample fetched value',
    };
    setData(dummyData);
  };

  return (
    <div className="content">
      {/* Map Section */}
      <div id="viewDiv" style={{ height: '500px' }}></div>

      {/* Form Section */}
      <div className="form-container">
        <div className="form-card">
          <form onSubmit={handleSubmit}>
            {/* Single Point Inputs */}
            <div className="mb-3">
              <label htmlFor="latitude" className="form-label">
                Latitude
              </label>
              <input
                type="text"
                className="form-control"
                id="latitude"
                name="latitude"
                value={formData.latitude}
                readOnly
              />
            </div>
            <div className="mb-3">
              <label htmlFor="longitude" className="form-label">
                Longitude
              </label>
              <input
                type="text"
                className="form-control"
                id="longitude"
                name="longitude"
                value={formData.longitude}
                readOnly
              />
            </div>
            <hr />

            {/* Range Inputs */}
            <div className="mb-3">
              <label htmlFor="min_latitude" className="form-label">
                Min Latitude
              </label>
              <input
                type="text"
                className="form-control"
                id="min_latitude"
                name="min_latitude"
                value={formData.min_latitude}
                readOnly
              />
            </div>
            <div className="mb-3">
              <label htmlFor="max_latitude" className="form-label">
                Max Latitude
              </label>
              <input
                type="text"
                className="form-control"
                id="max_latitude"
                name="max_latitude"
                value={formData.max_latitude}
                readOnly
              />
            </div>
            <div className="mb-3">
              <label htmlFor="min_longitude" className="form-label">
                Min Longitude
              </label>
              <input
                type="text"
                className="form-control"
                id="min_longitude"
                name="min_longitude"
                value={formData.min_longitude}
                readOnly
              />
            </div>
            <div className="mb-3">
              <label htmlFor="max_longitude" className="form-label">
                Max Longitude
              </label>
              <input
                type="text"
                className="form-control"
                id="max_longitude"
                name="max_longitude"
                value={formData.max_longitude}
                readOnly
              />
            </div>

            {/* Variable and Subvariable Inputs */}
            <div className="mb-3">
              <label htmlFor="variable" className="form-label">
                Variable
              </label>
              <select
                className="form-select"
                id="variable"
                name="variable"
                value={formData.variable}
                onChange={handleChange}
              >
                <option value="">Select Variable</option>
                <option value="var1">Variable 1</option>
                <option value="var2">Variable 2</option>
                {/* Add more options as needed */}
              </select>
            </div>
            <div className="mb-3">
              <label htmlFor="subvariable" className="form-label">
                Subvariable
              </label>
              <select
                className="form-select"
                id="subvariable"
                name="subvariable"
                value={formData.subvariable}
                onChange={handleChange}
              >
                <option value="">Select Subvariable</option>
                <option value="sub1">Subvariable 1</option>
                <option value="sub2">Subvariable 2</option>
                {/* Add more options as needed */}
              </select>
            </div>

            {/* Submit Button */}
            <button type="submit" className="btn btn-primary w-100">
              Get Variable Value
            </button>
          </form>

          {/* Display Results */}
          {data && (
            <div id="result" className="mt-4">
              <h3 className="text-center">Fetched Data</h3>
              <pre className="border rounded p-3 bg-light">{JSON.stringify(data, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HydroGeoDataset;
