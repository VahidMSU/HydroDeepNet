import React, { useState, useEffect } from 'react';
import ModelSettingsForm from '../forms/ModelSettings.js';
import EsriMap from '../EsriMap.js';
import { ContainerFluid, Title, Row, Column, Card, CardBody } from '../../styles/ModelSettings.tsx';

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
      site_no: stationData?.SiteNumber || stationInput,
    };

    console.log('Submitting model settings:', formData);

    if (!stationInput) {
      alert(
        'Notice: Default settings are being used for Landuse/Soil Resolution (250) and DEM Resolution (30).',
      );
    }

    try {
      const response = await fetch('/model-settings', {
        method: 'POST',
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
    <ContainerFluid>
      <Title>Create SWAT+ Models for USGS Streamgages</Title>

      <Row>
        {/* Left Column: Search & Form */}
        <Column>
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
        </Column>

        {/* Right Column: Esri Map */}
        <Column>
          <Card>
            <CardBody>
              <EsriMap
                geometries={stationData?.geometries || []}
                streamsGeometries={stationData?.streams_geometries || []}
                lakesGeometries={stationData?.lakes_geometries || []}
              />
            </CardBody>
          </Card>
        </Column>
      </Row>
    </ContainerFluid>
  );
};

export default ModelSettingsTemplate;
