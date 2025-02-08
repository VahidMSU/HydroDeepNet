import React, { useState, useEffect, useMemo } from 'react';
import SearchForm from '../components/SearchForm';
import StationDetails from '../components/StationDetails';
import EsriMap from '../components/EsriMap';
import '../css/ModelSettings.css';

const ModelSettings = () => {
  const [stationList, setStationList] = useState([]);
  const [stationInput, setStationInput] = useState('');
  const [stationData, setStationData] = useState(null);
  const [lsResolution, setLsResolution] = useState('');
  const [demResolution, setDemResolution] = useState('');
  const [calibrationFlag, setCalibrationFlag] = useState(false);
  const [sensitivityFlag, setSensitivityFlag] = useState(false);
  const [validationFlag, setValidationFlag] = useState(false);
  const [loading, setLoading] = useState(false);

  // Memoize the ESRI map to prevent re-renders
  const memoizedMap = useMemo(() => <EsriMap stationData={stationData} />, [stationData]);

  // Submit model settings to the backend
  const handleSubmit = async () => {
    const formData = {
      stationInput,
      lsResolution,
      demResolution,
      calibrationFlag,
      sensitivityFlag,
      validationFlag,
    };

    console.log('Submitting model settings:', formData);

    try {
      // POST JSON data to the same /model-settings route
      const response = await fetch('/model-settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        // e.g. 400 or 500
        const errorData = await response.json();
        console.error('Model creation error:', errorData);
        alert(`Error: ${errorData.error || 'Unknown error'}`);
        return;
      }

      // On success
      const data = await response.json();
      console.log('Model creation response:', data);
      alert('Model creation started!');
    } catch (error) {
      console.error('Error submitting model settings:', error);
      alert('Could not submit model settings. See console for details.');
    }
  };

  return (
    <div className="model-settings">
      <h2>Create SWAT+ Models for USGS Streamgages</h2>

      <div className="row">
        {/* Left Column: Form */}
        <div className="col-lg-4">
          <SearchForm setStationData={setStationData} setLoading={setLoading} />
          {/* USGS Station Number */}
          <div className="form-group mt-3">
            <label>USGS Station Number:</label>
            <input
              type="text"
              className="form-control"
              value={stationInput}
              onChange={(e) => setStationInput(e.target.value)}
              placeholder="Enter station number"
              list="station_list"
            />
            <datalist id="station_list">
              {stationList.map((station, index) => (
                <option key={index} value={station} />
              ))}
            </datalist>
          </div>
          {/* Station Details */}
          {stationData && <StationDetails stationData={stationData} />}
          {/* Landuse/Soil Resolution */}
          <div className="form-group">
            <label>Landuse/Soil Resolution:</label>
            <input
              type="text"
              className="form-control"
              value={lsResolution}
              onChange={(e) => setLsResolution(e.target.value)}
            />
          </div>
          /* DEM Resolution */
          <div className="form-group">
            <label>DEM Resolution:</label>
            <input
              type="text"
              className="form-control"
              value={demResolution}
              onChange={(e) => setDemResolution(e.target.value)}
            />
          </div>
          {/* Calibration, Sensitivity, Validation Flags */}
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={calibrationFlag}
                onChange={(e) => setCalibrationFlag(e.target.checked)}
              />{' '}
              Calibration
            </label>
          </div>
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={sensitivityFlag}
                onChange={(e) => setSensitivityFlag(e.target.checked)}
              />{' '}
              Sensitivity Analysis
            </label>
          </div>
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={validationFlag}
                onChange={(e) => setValidationFlag(e.target.checked)}
              />{' '}
              Validation
            </label>
          </div>
          {/* Submit Button */}
          <button className="btn btn-primary mt-3" onClick={handleSubmit}>
            Run
          </button>
        </div>

        {/* Right Column: Esri Map */}
        <div className="col-lg-8">{memoizedMap}</div>
      </div>
    </div>
  );
};

export default ModelSettings;
