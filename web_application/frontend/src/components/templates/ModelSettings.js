import React, { useState, useEffect } from 'react';
import ModelSettingsForm from '../forms/ModelSettings.js';
import { Row, Column } from '../../styles/ModelSettings.tsx';
import EsriMap from '../EsriMap.js';
import { HeaderTitle, Section, Card } from '../../styles/Layout.tsx';

import {
  CardBody,
  ContainerFluid,
  Content,
  ContentWrapper,
  MapWrapper,
} from '../../styles/ModelSettings.tsx';

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
      <HeaderTitle style={{ margin: '0 0 15px 0' }}>
        Create SWAT+ Models for USGS Streamgages
      </HeaderTitle>
      <Content>
        <Row>
          <Column width={0.32} minWidth="350px" mobileMinWidth="100%">
            <ContentWrapper
              style={{
                height: '100%',
                margin: 0,
                padding: '1.5rem', // Increased padding
                overflow: 'auto',
              }}
            >
              <Section style={{ padding: '0 1.5rem' }}>
                {' '}
                {/* Added horizontal padding */}
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
              </Section>
            </ContentWrapper>
          </Column>
          <Column width={0.68} minWidth="650px" mobileMinWidth="100%" className="map-column">
            <MapWrapper>
              <Card style={{ height: '100%', margin: 0 }}>
                <CardBody style={{ padding: '0.75rem' }}>
                  {' '}
                  {/* Adjusted padding */}
                  <EsriMap
                    geometries={stationData?.geometries || []}
                    streamsGeometries={stationData?.streams_geometries || []}
                    lakesGeometries={stationData?.lakes_geometries || []}
                  />
                </CardBody>
              </Card>
            </MapWrapper>
          </Column>
        </Row>
      </Content>
    </ContainerFluid>
  );
};

export default ModelSettingsTemplate;
