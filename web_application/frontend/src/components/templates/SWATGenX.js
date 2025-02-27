import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faGears,
  faInfoCircle,
  faChevronDown,
  faChevronUp,
  faMap,
  faCheckCircle,
  faExclamationTriangle,
} from '@fortawesome/free-solid-svg-icons';
import ModelSettingsForm from '../forms/SWATGenX.js';
import EsriMap from '../EsriMap.js';
import {
  Container,
  Header,
  HeaderTitle,
  TitleIcon,
  Content,
  Sidebar,
  InfoPanel,
  ConfigPanel,
  PanelHeader,
  PanelIcon,
  ToggleIcon,
  PanelContent,
  ConfigPanelContent,
  MapContainer,
  MapInnerContainer,
  FeedbackMessage,
  FeedbackIcon,
} from '../../styles/SWATGenX.tsx';

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

  // Fetch station data effect
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
    <Container>
      <Header>
        <HeaderTitle>
          <TitleIcon>
            <FontAwesomeIcon icon={faMap} />
          </TitleIcon>
          <span>SWATGenX – SWAT+ Model Creation Tool</span>
        </HeaderTitle>
      </Header>

      <Content>
        <Sidebar>
          <InfoPanel>
            <PanelHeader onClick={() => setIsDescriptionOpen(!isDescriptionOpen)}>
              <PanelIcon>
                <FontAwesomeIcon icon={faInfoCircle} />
              </PanelIcon>
              <span>Description</span>
              <ToggleIcon isOpen={isDescriptionOpen}>
                <FontAwesomeIcon icon={isDescriptionOpen ? faChevronUp : faChevronDown} />
              </ToggleIcon>
            </PanelHeader>

            {isDescriptionOpen && (
              <PanelContent>
                <p>
                  SWATGenX is an automated tool for creating SWAT+ models using USGS streamgage
                  stations. You can locate a station by searching its site number or name.
                </p>
                <ul>
                  <li>Configure Landuse/Soil and DEM resolution</li>
                  <li>Enable calibration, sensitivity analysis, and validation</li>
                  <li>Start automatic model generation</li>
                </ul>
                <p>
                  Once generated, your models will appear in the <strong>User Dashboard</strong>,
                  where you can download or visualize them.
                </p>
              </PanelContent>
            )}
          </InfoPanel>

          <ConfigPanel>
            <PanelHeader>
              <PanelIcon>
                <FontAwesomeIcon icon={faGears} />
              </PanelIcon>
              <span>Model Configuration</span>
            </PanelHeader>

            <ConfigPanelContent>
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

              {feedbackMessage && (
                <FeedbackMessage type={feedbackType}>
                  <FeedbackIcon>
                    <FontAwesomeIcon
                      icon={feedbackType === 'success' ? faCheckCircle : faExclamationTriangle}
                    />
                  </FeedbackIcon>
                  <span>{feedbackMessage}</span>
                </FeedbackMessage>
              )}
            </ConfigPanelContent>
          </ConfigPanel>
        </Sidebar>

        <MapContainer>
          <MapInnerContainer>
            <EsriMap
              geometries={stationData?.geometries || []}
              streamsGeometries={stationData?.streams_geometries || []}
              lakesGeometries={stationData?.lakes_geometries || []}
            />
          </MapInnerContainer>
        </MapContainer>
      </Content>
    </Container>
  );
};

export default SWATGenXTemplate;
