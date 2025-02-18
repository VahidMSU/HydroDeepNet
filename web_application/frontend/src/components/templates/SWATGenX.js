import React, { useState, useEffect, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGears, faChevronDown, faChevronUp } from '@fortawesome/free-solid-svg-icons';
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
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [feedbackType, setFeedbackType] = useState(''); // 'success' | 'error'

  // Fetch station data (optional in future, depending on your data source)
  useEffect(() => {
    if (stationData) {
      console.log('Station details:', stationData);
    }
  }, [stationData]);

  const handleSubmit = async () => {
    if (!stationInput.trim() && !stationData) {
      setFeedbackMessage('Please provide a valid station number or select a station.');
      setFeedbackType('error');
      return;
    }

    const formData = {
      stationInput,
      lsResolution,
      demResolution,
      calibrationFlag,
      sensitivityFlag,
      validationFlag,
      site_no: stationData?.SiteNumber || stationInput.trim(),
    };

    console.log('Submitting model settings:', formData);

    setLoading(true);
    setFeedbackMessage('');
    setFeedbackType('');

    try {
      const response = await fetch('/model-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Unknown error occurred.');
      }

      const data = await response.json();
      console.log('Model creation response:', data);

      setFeedbackMessage('Model creation started successfully!');
      setFeedbackType('success');
    } catch (error) {
      console.error('Model creation failed:', error);
      setFeedbackMessage(`Failed to start model creation: ${error.message}`);
      setFeedbackType('error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ContainerFluid>
      <HeaderTitle style={{ marginBottom: '15px' }}>
        <StrongText>SWATGenX – SWAT+ Model Creation Tool</StrongText>
      </HeaderTitle>

      <Content>
        <Row>
          {/* Left Panel - Form and Description */}
          <Column width={0.35} minWidth="350px" mobileMinWidth="100%">
            <ContentWrapper style={{ margin: 0, padding: '1.5rem' }}>
              {/* Description Section */}
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
                      SWATGenX is an automated tool for creating SWAT+ models using USGS streamgage
                      stations. You can locate a station by searching its site number or name.
                      Configure your model by adjusting:
                    </FieldText>
                    <ListElement>
                      <StrongText>
                        - Landuse/Soil and DEM resolution
                        <br />
                        - Enable calibration, sensitivity analysis, and validation
                        <br />
                        - Start automatic model generation
                      </StrongText>
                    </ListElement>
                    <FieldText>
                      Once generated, your models will appear in the <StrongText>User Dashboard</StrongText>,
                      where you can download or visualize them.
                    </FieldText>
                  </InfoBox>
                )}
              </DescriptionContainer>

              {/* Model Settings Form */}
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
                    {/* Feedback Message */}
                    {feedbackMessage && (
                      <div
                        style={{
                          marginTop: '10px',
                          color: feedbackType === 'success' ? 'green' : 'red',
                          fontWeight: 'bold',
                        }}
                      >
                        {feedbackMessage}
                      </div>
                    )}
                  </CardBody>
                </Card>
              </Section>
            </ContentWrapper>
          </Column>

          {/* Right Panel - Map */}
          <Column width={0.65} minWidth="600px" mobileMinWidth="100%">
            <MapWrapper>
              <Card style={{ height: '100%' }}>
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
