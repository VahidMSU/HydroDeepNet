import React from 'react';
import SearchForm from '../SearchForm';
import StationDetails from '../StationDetails';

const ModelSettingsForm = ({
  stationList,
  stationInput,
  setStationInput,
  stationData,
  setStationData,
  lsResolution,
  setLsResolution,
  demResolution,
  setDemResolution,
  calibrationFlag,
  setCalibrationFlag,
  sensitivityFlag,
  setSensitivityFlag,
  validationFlag,
  setValidationFlag,
  handleSubmit,
  loading,
  setLoading,
}) => {
  return (
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
  );
};

export default ModelSettingsForm;
