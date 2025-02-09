import React, { useState, useEffect } from 'react';
import ModelSettingsForm from '../forms/ModelSettings.js'; // Ensure the path is correct
import EsriMap from '../EsriMap.js'; // Ensure the path is correct
import '../../styles/ModelSettings.tsx'; // Ensure the path is correct

const ModelSettingsTemplate = () => {
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
      site_no: stationData?.SiteNumber || stationInput, // Include station name
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
          <ModelSettingsForm
            stationList={stationList}
            stationInput={stationInput}
            setStationInput={setStationInput}
            stationData={stationData}
            setStationData={setStationData}
            lsResolution={lsResolution}
            setLsResolution={setLsResolution}
            demResolution={demResolution}
            setDemResolution={setDemResolution}
            calibrationFlag={calibrationFlag}
            setCalibrationFlag={setCalibrationFlag}
            sensitivityFlag={sensitivityFlag}
            setSensitivityFlag={setSensitivityFlag}
            validationFlag={validationFlag}
            setValidationFlag={setValidationFlag}
            handleSubmit={handleSubmit}
            loading={loading}
            setLoading={setLoading}
          />
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

export default ModelSettingsTemplate;
