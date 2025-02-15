import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSliders,
  faGears,
  faChartLine,
  faChevronDown,
  faChevronUp,
} from '@fortawesome/free-solid-svg-icons';
import ModelSettingsForm from '../forms/SWATGenX.js';
import {
  Row,
  Column,
  SectionTitle,
  ContainerFluid,
  Content,
  ContentWrapper,
  MapWrapper,
  DescriptionContainer,
  InfoBox,
  DescriptionHeader,
  StrongText,
  FieldText,
  ListElement,
  Section,
  ModelContainer, // Import ModelContainer
} from '../../styles/SWATGenX.tsx';
import EsriMap from '../EsriMap.js';
import { HeaderTitle, Card, CardBody } from '../../styles/Layout.tsx';

const SWATGenXTemplate = () => {
  const [isDescriptionOpen, setIsDescriptionOpen] = useState(true);
  const [stationList] = useState([]);
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
        <FontAwesomeIcon icon={faSliders} />
        <StrongText>SWATGenX – SWAT+ Model Creation Tool</StrongText>
        <FontAwesomeIcon icon={faChartLine} style={{ marginLeft: '10px' }} />
      </HeaderTitle>

      <Content>
        <Row>
          <Column width={0.32} minWidth="350px" mobileMinWidth="100%">
            <ContentWrapper style={{ margin: 0, padding: '1.5rem', overflow: 'auto' }}>
              <DescriptionContainer>
                <DescriptionHeader
                  isOpen={isDescriptionOpen}
                  onClick={() => setIsDescriptionOpen(!isDescriptionOpen)}
                >
                  <FieldText>Description</FieldText>
                  <FontAwesomeIcon icon={isDescriptionOpen ? faChevronUp : faChevronDown} />
                </DescriptionHeader>

                {isDescriptionOpen && (
                  <InfoBox>
                    <FieldText>
                      SWATGenX is a dedicated tool for generating SWAT+ models for any USGS
                      streamgage stations. Users can locate a station by searching for a name or
                      providing a site number. Once selected, users can configure model settings,
                      including:
                    </FieldText>
                    <ListElement>
                      <StrongText>
                        Selecting land use/soil and DEM resolution Enabling calibration, sensitivity
                        analysis, and validation Invoking SWATGenX to generate the model
                      </StrongText>
                    </ListElement>
                    <FieldText>
                      The generated models will be available in the **user dashboard**, where they
                      can be downloaded for further analysis.
                    </FieldText>
                  </InfoBox>
                )}
              </DescriptionContainer>
              <Section>
                <SectionTitle>
                  <FontAwesomeIcon icon={faGears} className="icon" />
                  Model Configuration
                </SectionTitle>

                <Card>
                  <CardBody>
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
                  </CardBody>
                </Card>
              </Section>
            </ContentWrapper>
          </Column>

          <Column width={0.68} minWidth="650px" mobileMinWidth="100%" className="map-column">
            <MapWrapper>
              <Card style={{ height: '100%', margin: 0 }}>
                <CardBody style={{ padding: '0.75rem' }}>
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

export default SWATGenXTemplate;
