import React, { useState, useEffect } from 'react';
import SearchForm from '../components/SearchForm';
import StationDetails from '../components/StationDetails';
import EsriMap from '../components/EsriMap';
import '../css/ModelSettings.css';

const ModelSettings = () => {
  const [stationList, setStationList] = useState([]);
  const [stationInput, setStationInput] = useState('');
  const [stationData, setStationData] = useState(null);
  const [lsResolution, setLsResolution] = useState('250');
  const [demResolution, setDemResolution] = useState('30');
  const [calibrationFlag, setCalibrationFlag] = useState(false);
  const [sensitivityFlag, setSensitivityFlag] = useState(false);
  const [validationFlag, setValidationFlag] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (stationData) {
      console.log('Station details:', stationData);
    }
  }, [stationData]);

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

    if (!stationInput) {
      alert(
        'Notice: Default settings are being used for Landuse/Soil Resolution (250) and DEM Resolution (30).',
      );
    }

    try {
      const response = await fetch('/model-settings', {
        method: 'POST', // Ensure this matches the backend method
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Model creation error:', errorData);
        alert(`Error: ${errorData.error || 'Unknown error'}`);
        return;
      }

      const data = await response.json();
      console.log('Model creation response:', data);
      alert('Model creation started!');
    } catch (error) {
      console.error('Error submitting model settings:', error);
      alert('Could not submit model settings. See console for details.');
    }
  };

  return (
    <div className="container-fluid model-settings">
      <h2 className="text-center my-4">Create SWAT+ Models for USGS Streamgages</h2>

      <div className="row">
        {/* Left Column: Search & Form */}
        <div className="col-lg-4 d-flex flex-column align-items-stretch">
          <div className="card shadow p-3 mb-4">
            <div className="card-body">
              <SearchForm setStationData={setStationData} setLoading={setLoading} />

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

              {stationData && <StationDetails stationData={stationData} />}

              <div className="form-group">
                <label>Landuse/Soil Resolution:</label>
                <input
                  type="text"
                  className="form-control"
                  value={lsResolution}
                  onChange={(e) => setLsResolution(e.target.value)}
                />
              </div>

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
              <div className="form-check mt-2">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="calibrationFlag"
                  checked={calibrationFlag}
                  onChange={(e) => setCalibrationFlag(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="calibrationFlag">
                  Calibration
                </label>
              </div>

              <div className="form-check">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="sensitivityFlag"
                  checked={sensitivityFlag}
                  onChange={(e) => setSensitivityFlag(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="sensitivityFlag">
                  Sensitivity Analysis
                </label>
              </div>

              <div className="form-check">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="validationFlag"
                  checked={validationFlag}
                  onChange={(e) => setValidationFlag(e.target.checked)}
                />
                <label className="form-check-label" htmlFor="validationFlag">
                  Validation
                </label>
              </div>

              <button className="btn btn-primary mt-3 w-100" onClick={handleSubmit}>
                Run Model
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Esri Map */}
        <div className="col-lg-6">
          <div className="card shadow">
            <div className="card-body p-0">
              <EsriMap
                geometries={stationData?.geometries || []}
                streamsGeometries={stationData?.streams_geometries || []}
                lakesGeometries={stationData?.lakes_geometries || []}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelSettings;
