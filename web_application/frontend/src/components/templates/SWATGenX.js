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
  faMapMarkedAlt,
  faLayerGroup,
  faSearch,
  faLocationDot,
  faMousePointer,
  faSpinner,
  faListUl,
  faStream,
} from '@fortawesome/free-solid-svg-icons';
import ModelSettingsForm from '../forms/SWATGenX.js';
import EsriMap from '../EsriMap.js';
import SearchForm from '../SearchForm';
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
  TabContainer,
  TabButton,
  TabIcon,
  StepIndicator,
  StepCircle,
  StepText,
  StepConnector,
} from '../../styles/SWATGenX.tsx';

// New styled components for improved map selection UI
import styled from 'styled-components';

const MapControlsContainer = styled.div`
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const MapControlButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
  cursor: pointer;
  transition: all 0.2s ease;
  padding: 0;

  &:hover {
    background: #f0f0f0;
  }

  &.active {
    background: #007bff;
    color: white;
  }
`;

const StationSelectionPanel = styled.div`
  position: absolute;
  bottom: 20px;
  left: 20px;
  z-index: 10;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  width: 300px;
  max-height: 300px;
  overflow: hidden;
  display: ${(props) => (props.visible ? 'flex' : 'none')};
  flex-direction: column;
`;

const StationSelectionHeader = styled.div`
  padding: 12px 16px;
  background: #f5f5f5;
  border-bottom: 1px solid #ddd;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const StationList = styled.div`
  overflow-y: auto;
  max-height: 250px;
  padding: 0;
`;

const StationItem = styled.div`
  padding: 10px 16px;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;

  &:hover {
    background: #f8f9fa;
  }

  &.selected {
    background: #e6f3ff;
  }
`;

const StationName = styled.span`
  font-size: 14px;
  flex: 1;
`;

const StationId = styled.span`
  font-size: 12px;
  color: #666;
  margin-left: 8px;
`;

const SelectButton = styled.button`
  margin-left: 8px;
  padding: 4px 8px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;

  &:hover {
    background: #0069d9;
  }
`;

const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 100;
  font-size: 16px;
  color: #333;
`;

const LoadingIcon = styled.div`
  margin-bottom: 12px;
  font-size: 24px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

const SWATGenXTemplate = () => {
  const [isDescriptionOpen, setIsDescriptionOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(1); // Step 1: Station Selection, Step 2: Resolution & Analysis
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

  // Map-based selection state
  const [selectionTab, setSelectionTab] = useState('search'); // 'search' | 'map'
  const [stationPoints, setStationPoints] = useState([]);
  const [mapSelectionLoading, setMapSelectionLoading] = useState(false);
  const [selectedStationOnMap, setSelectedStationOnMap] = useState(null);
  const [mapSelections, setMapSelections] = useState([]);
  const [showStationPanel, setShowStationPanel] = useState(false);

  // Fetch station data effect
  useEffect(() => {
    if (stationData) {
      console.log('Station details:', stationData);
    }
  }, [stationData]);

  // Fetch station geometries for map-based selection
  useEffect(() => {
    if (selectionTab === 'map' && stationPoints.length === 0 && !mapSelectionLoading) {
      fetchStationGeometries();
    }
  }, [selectionTab, stationPoints.length, mapSelectionLoading]);

  // Show station panel when selections are made
  useEffect(() => {
    if (mapSelections.length > 0) {
      setShowStationPanel(true);
    }
  }, [mapSelections]);

  const fetchStationGeometries = async () => {
    setMapSelectionLoading(true);
    try {
      const response = await fetch('/api/get_station_geometries');
      if (!response.ok) {
        throw new Error('Failed to fetch station geometries');
      }
      const data = await response.json();
      setStationPoints(data.features || []);
    } catch (error) {
      console.error('Error fetching station geometries:', error);
      setFeedbackMessage('Failed to load stations on map: ' + error.message);
      setFeedbackType('error');
    } finally {
      setMapSelectionLoading(false);
    }
  };

  const handleNextStep = () => {
    if (!stationInput.trim() && !stationData) {
      setFeedbackMessage('Please provide a valid station number or select a station.');
      setFeedbackType('error');
      return;
    }
    setCurrentStep(2);
    setFeedbackMessage('');
    setFeedbackType('');
  };

  const handlePreviousStep = () => {
    setCurrentStep(1);
    setFeedbackMessage('');
    setFeedbackType('');
  };

  const handleStationSelectFromMap = async (stationAttributes) => {
    console.log('Station selected from map:', stationAttributes);
    if (!stationAttributes || !stationAttributes.SiteNumber) {
      console.error('Invalid station attributes:', stationAttributes);
      return;
    }

    setSelectedStationOnMap(stationAttributes);
    // Set the selected station as the only item in map selections
    setMapSelections([
      {
        SiteNumber: stationAttributes.SiteNumber,
        SiteName: stationAttributes.SiteName,
      },
    ]);
    // Fetch station details just like we would do from search
    await handleStationSelect(stationAttributes.SiteNumber);
  };

  const handleStationSelect = async (stationNumber) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/get_station_characteristics?station=${stationNumber}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setStationData(data);
      setStationInput(stationNumber);
      // Clear any previous feedback
      setFeedbackMessage('');
      setFeedbackType('');
    } catch (error) {
      console.error('Error fetching station details:', error);
      setFeedbackMessage('Failed to fetch station details: ' + error.message);
      setFeedbackType('error');
    } finally {
      setLoading(false);
    }
  };

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
      const response = await fetch('/api/model-settings', {
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

  // Switch between search and map tabs
  const handleTabChange = (tab) => {
    console.log(`Changing tab from ${selectionTab} to ${tab}`);
    setSelectionTab(tab);

    // Reset selection state when switching tabs
    if (tab === 'search' && selectedStationOnMap) {
      setSelectedStationOnMap(null);
    }

    // Hide station panel when switching to search
    if (tab === 'search') {
      setShowStationPanel(false);
    }
  };

  // Select a specific station from the station panel
  const selectStationFromPanel = (station) => {
    handleStationSelect(station.SiteNumber);
    setSelectedStationOnMap({
      SiteNumber: station.SiteNumber,
      SiteName: station.SiteName,
    });
  };

  // Toggle station selection panel visibility
  const toggleStationPanel = () => {
    setShowStationPanel(!showStationPanel);
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
                  stations. You can locate a station by searching its site number or name, or by
                  selecting it directly on the map.
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
              {currentStep === 1 && (
                <TabContainer>
                  <TabButton
                    active={selectionTab === 'search'}
                    onClick={() => handleTabChange('search')}
                  >
                    <TabIcon>
                      <FontAwesomeIcon icon={faSearch} />
                    </TabIcon>
                    Search
                  </TabButton>
                  <TabButton active={selectionTab === 'map'} onClick={() => handleTabChange('map')}>
                    <TabIcon>
                      <FontAwesomeIcon icon={faLocationDot} />
                    </TabIcon>
                    Map Select
                  </TabButton>
                </TabContainer>
              )}

              <StepIndicator>
                <StepCircle active={currentStep === 1} completed={currentStep > 1}>
                  <FontAwesomeIcon icon={faMapMarkedAlt} />
                </StepCircle>
                <StepText active={currentStep === 1}>Station Selection</StepText>
                <StepConnector completed={currentStep > 1} />
                <StepCircle active={currentStep === 2} completed={currentStep > 2}>
                  <FontAwesomeIcon icon={faLayerGroup} />
                </StepCircle>
                <StepText active={currentStep === 2}>Model Settings</StepText>
              </StepIndicator>

              {currentStep === 1 && selectionTab === 'map' ? (
                <SearchForm
                  setStationData={setStationData}
                  setLoading={setLoading}
                  mapSelections={mapSelections}
                  handleStationSelect={handleStationSelect}
                />
              ) : (
                <ModelSettingsForm
                  currentStep={currentStep}
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
                  handleNextStep={handleNextStep}
                  handlePreviousStep={handlePreviousStep}
                  handleSubmit={handleSubmit}
                  loading={loading}
                  setLoading={setLoading}
                />
              )}

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
            {/* Map interaction controls when in map selection mode */}
            {selectionTab === 'map' && (
              <>
                <MapControlsContainer>
                  <MapControlButton title="Station selection tool" className="active">
                    <FontAwesomeIcon icon={faMousePointer} />
                  </MapControlButton>
                  <MapControlButton
                    title={showStationPanel ? 'Hide station list' : 'Show station list'}
                    onClick={toggleStationPanel}
                    className={showStationPanel ? 'active' : ''}
                  >
                    <FontAwesomeIcon icon={faListUl} />
                  </MapControlButton>
                </MapControlsContainer>

                {/* Station selection panel */}
                <StationSelectionPanel visible={showStationPanel && mapSelections.length > 0}>
                  <StationSelectionHeader>
                    <span>
                      <FontAwesomeIcon icon={faStream} /> Stream Gauge Station
                    </span>
                    <span>{mapSelections.length > 0 ? 'Selected' : 'None'}</span>
                  </StationSelectionHeader>
                  <StationList>
                    {mapSelections.map((station, index) => (
                      <StationItem
                        key={station.SiteNumber}
                        className={
                          selectedStationOnMap?.SiteNumber === station.SiteNumber ? 'selected' : ''
                        }
                      >
                        <StationName>{station.SiteName || `Station ${index + 1}`}</StationName>
                        <StationId>{station.SiteNumber}</StationId>
                        <SelectButton onClick={() => selectStationFromPanel(station)}>
                          Select
                        </SelectButton>
                      </StationItem>
                    ))}
                  </StationList>
                </StationSelectionPanel>

                {/* Loading overlay for stations */}
                {mapSelectionLoading && (
                  <LoadingOverlay>
                    <LoadingIcon>
                      <FontAwesomeIcon icon={faSpinner} />
                    </LoadingIcon>
                    <span>Loading stream gauge stations...</span>
                  </LoadingOverlay>
                )}
              </>
            )}

            <EsriMap
              geometries={stationData?.geometries || []}
              streamsGeometries={stationData?.streams_geometries || []}
              lakesGeometries={stationData?.lakes_geometries || []}
              stationPoints={stationPoints}
              onStationSelect={handleStationSelectFromMap}
              showStations={selectionTab === 'map'}
              selectedStationId={selectedStationOnMap?.SiteNumber}
              key={`map-${selectionTab}`}
            />
          </MapInnerContainer>
        </MapContainer>
      </Content>
    </Container>
  );
};

export default SWATGenXTemplate;
