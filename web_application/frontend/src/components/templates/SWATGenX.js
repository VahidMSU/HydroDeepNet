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
  faRuler,
  faLayerGroup,
  faSearch,
  faLocationDot,
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

  // New state for map-based selection
  const [selectionTab, setSelectionTab] = useState('search'); // 'search' | 'map'
  const [stationPoints, setStationPoints] = useState([]);
  const [mapSelectionLoading, setMapSelectionLoading] = useState(false);
  const [selectedStationOnMap, setSelectedStationOnMap] = useState(null);
  const [mapSelections, setMapSelections] = useState([]);
  const [drawingMode, setDrawingMode] = useState(false);

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

  const handleDrawComplete = (selectedStations) => {
    // Update map selections with the stations within the drawn polygon
    setMapSelections(selectedStations);
    setDrawingMode(false); // Exit drawing mode after selection

    // If only one station was selected, fetch its details
    if (selectedStations.length === 1) {
      handleStationSelect(selectedStations[0].SiteNumber);
    } else if (selectedStations.length > 1) {
      // If multiple stations, clear current selection but keep the list
      setStationData(null);
      setStationInput('');
      // Show feedback to user about multiple stations
      setFeedbackMessage(`${selectedStations.length} stations selected. Choose one to continue.`);
      setFeedbackType('info');
    }
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

    // Force disable drawing mode when switching away from map
    if (tab !== 'map' && drawingMode) {
      setDrawingMode(false);
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
                  setDrawingMode={setDrawingMode}
                  drawingMode={drawingMode}
                  handleStationSelect={handleStationSelect} // Pass this function for handling station selection
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
            <EsriMap
              geometries={stationData?.geometries || []}
              streamsGeometries={stationData?.streams_geometries || []}
              lakesGeometries={stationData?.lakes_geometries || []}
              stationPoints={stationPoints}
              onStationSelect={handleStationSelectFromMap}
              onDrawComplete={handleDrawComplete}
              showStations={selectionTab === 'map'} // This should be correct based on selectionTab
              selectedStationId={selectedStationOnMap?.SiteNumber}
              drawingMode={drawingMode}
              key={`map-${selectionTab}`} // Add a key to force remount when tab changes
            />
          </MapInnerContainer>

          {/* Status indicator for map selection mode */}
          {selectionTab === 'map' && (
            <div
              style={{
                position: 'absolute',
                top: '10px',
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0,0,0,0.6)',
                color: 'white',
                padding: '5px 10px',
                borderRadius: '4px',
                fontSize: '14px',
                zIndex: 1000,
              }}
            >
              {mapSelectionLoading
                ? 'Loading stations...'
                : drawingMode
                  ? 'Drawing selection tool active'
                  : 'Click on a station to select it'}
            </div>
          )}

          {/* Add debugging info in development */}
          {process.env.NODE_ENV !== 'production' && (
            <div
              style={{
                position: 'absolute',
                bottom: '30px',
                right: '10px',
                background: 'rgba(255,255,255,0.8)',
                padding: '8px',
                borderRadius: '4px',
                fontSize: '12px',
                maxWidth: '250px',
              }}
            >
              <p>
                <strong>Map Selection Debug:</strong>
              </p>
              <p>Selection Tab: {selectionTab}</p>
              <p>Showing Stations: {selectionTab === 'map' && currentStep === 1 ? 'Yes' : 'No'}</p>
              <p>Drawing Mode: {drawingMode ? 'On' : 'Off'}</p>
              <p>Station Count: {stationPoints.length}</p>
              <p>Selected: {selectedStationOnMap?.SiteNumber || 'None'}</p>
            </div>
          )}
        </MapContainer>
      </Content>
    </Container>
  );
};

export default SWATGenXTemplate;
