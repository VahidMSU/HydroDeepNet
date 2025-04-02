import React, { useState, useEffect, useRef } from 'react';
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
} from '@fortawesome/free-solid-svg-icons';
import ModelSettingsForm from '../forms/SWATGenX.js';
import EsriMap from '../EsriMap.js';
import SearchForm from '../SearchForm';
import MapLoadingDebugger from '../debug/MapLoadingDebugger';
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
  MapControlsContainer,
  MapControlButton,
  LoadingOverlay,
  LoadingIcon,
} from '../../styles/SWATGenX.tsx';

// Modify the geometriesCache to be more persistent
const geometriesCache = {
  data: null,
  timestamp: null,
  // Cache expiration in milliseconds (30 minutes)
  expirationTime: 30 * 60 * 1000,
  // Track loading state
  isLoading: false,
  // Add error tracking
  error: null,

  set: function (data) {
    this.data = data;
    this.timestamp = Date.now();
    this.isLoading = false;
    this.error = null;
    // Store in sessionStorage for persistence across page navigations
    try {
      sessionStorage.setItem(
        'stationGeometriesCache',
        JSON.stringify({
          data: data,
          timestamp: Date.now(),
        }),
      );
    } catch (e) {
      console.warn('Failed to cache station geometries in sessionStorage:', e);
    }
    console.log(`Station geometries cache updated with ${data.length} stations`);
  },

  get: function () {
    // First check local variable
    if (this.data && this.timestamp && Date.now() - this.timestamp <= this.expirationTime) {
      return this.data;
    }

    // If not in local variable, try sessionStorage
    try {
      const cachedData = sessionStorage.getItem('stationGeometriesCache');
      if (cachedData) {
        const parsed = JSON.parse(cachedData);
        if (parsed.timestamp && Date.now() - parsed.timestamp <= this.expirationTime) {
          // Update local state
          this.data = parsed.data;
          this.timestamp = parsed.timestamp;
          console.log(`Using ${parsed.data.length} station geometries from sessionStorage`);
          return parsed.data;
        }
      }
    } catch (e) {
      console.warn('Failed to retrieve station geometries from sessionStorage:', e);
    }

    return null;
  },

  isValid: function () {
    return this.get() !== null;
  },

  clear: function () {
    this.data = null;
    this.timestamp = null;
    this.error = null;
    sessionStorage.removeItem('stationGeometriesCache');
  },

  setLoading: function (status) {
    this.isLoading = status;
  },

  setError: function (error) {
    this.error = error;
    this.isLoading = false;
  },
};

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
  const [selectionTab, setSelectionTab] = useState('search'); // 'search' | 'map'
  const [stationPoints, setStationPoints] = useState([]);
  const [mapSelectionLoading, setMapSelectionLoading] = useState(false);
  const [selectedStationOnMap, setSelectedStationOnMap] = useState(null);
  const [mapSelections, setMapSelections] = useState([]);
  const [showStationPanel, setShowStationPanel] = useState(false);
  const [geometriesPreloaded, setGeometriesPreloaded] = useState(false);
  const [showDebugger, setShowDebugger] = useState(false);
  const [stationTabHistory, setStationTabHistory] = useState('search');
  const previousStationDataRef = useRef(null);

  // Add keyboard shortcut to toggle the debugger (Ctrl+Shift+D)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        setShowDebugger((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Initialize from cache on first load
  useEffect(() => {
    // Try to load station geometries from cache on component mount
    const cachedGeometries = geometriesCache.get();
    if (cachedGeometries) {
      console.log('Initialized station geometries from persistent cache');
      setGeometriesPreloaded(true);

      // Only set station points if we're on map tab
      if (selectionTab === 'map') {
        setStationPoints(cachedGeometries);
      }
    } else {
      // If no cache, preload
      console.log('No cached geometries found, will preload');
    }
  }, []);

  // Preload station geometries on component mount regardless of tab
  useEffect(() => {
    const preloadStationGeometries = async () => {
      if (geometriesCache.isValid() || geometriesCache.isLoading) {
        if (geometriesCache.isValid()) {
          console.log('Using preloaded cached station geometries');
          setGeometriesPreloaded(true);
        }
        return;
      }

      geometriesCache.setLoading(true);
      try {
        console.log('Preloading station geometries from server');
        setMapSelectionLoading(true);

        const timestamp = new Date().getTime();
        const response = await fetch(`/api/get_station_geometries?_=${timestamp}`, {
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
        });

        if (!response.ok) {
          throw new Error(
            `Failed to fetch station geometries: ${response.status} ${response.statusText}`,
          );
        }

        const data = await response.json();
        const features = data.features || [];

        console.log(`Successfully preloaded ${features.length} station geometries`);

        geometriesCache.set(features);
        setGeometriesPreloaded(true);

        if (selectionTab === 'map') {
          setStationPoints(features);
        }
      } catch (error) {
        console.error('Error preloading station geometries:', error);
        geometriesCache.setError(error);
        if (selectionTab === 'map') {
          setFeedbackMessage('Failed to load stations on map: ' + error.message);
          setFeedbackType('error');
        }
      } finally {
        setMapSelectionLoading(false);
      }
    };

    preloadStationGeometries();
  }, []);

  useEffect(() => {
    if (selectionTab === 'map' && stationPoints.length === 0) {
      if (geometriesCache.isValid()) {
        console.log('Using cached station geometries for map tab');
        setStationPoints(geometriesCache.get());
      } else if (!mapSelectionLoading && !geometriesCache.isLoading) {
        fetchStationGeometries();
      }
    }
  }, [selectionTab, stationPoints.length, mapSelectionLoading, geometriesPreloaded]);

  // Preserve station data between tab switches
  useEffect(() => {
    if (stationData) {
      previousStationDataRef.current = stationData;
    }
  }, [stationData]);

  const fetchStationGeometries = async () => {
    if (mapSelectionLoading || geometriesCache.isLoading) return;

    setMapSelectionLoading(true);
    geometriesCache.setLoading(true);

    try {
      console.log('Fetching station geometries from server');

      const timestamp = new Date().getTime();
      const response = await fetch(`/api/get_station_geometries?_=${timestamp}`, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          Pragma: 'no-cache',
          Expires: '0',
        },
      });

      if (!response.ok) {
        throw new Error(
          `Failed to fetch station geometries: ${response.status} ${response.statusText}`,
        );
      }

      const data = await response.json();

      if (!data || !data.features) {
        throw new Error('Invalid response format from server');
      }

      const features = data.features || [];
      console.log(`Loaded ${features.length} station geometries`);

      geometriesCache.set(features);
      setStationPoints(features);

      if (feedbackType === 'error' && feedbackMessage.includes('station')) {
        setFeedbackMessage('');
        setFeedbackType('');
      }
    } catch (error) {
      console.error('Error fetching station geometries:', error);
      geometriesCache.setError(error);
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
    setMapSelections([
      {
        SiteNumber: stationAttributes.SiteNumber,
        SiteName: stationAttributes.SiteName,
      },
    ]);
    await handleStationSelect(stationAttributes.SiteNumber);
  };

  // Enhanced version of handleStationSelect to better handle caching and reselection
  const handleStationSelect = async (stationNumber) => {
    // If we're selecting the same station that's already loaded, don't reload
    if (stationData && stationData.SiteNumber === stationNumber && stationData.geometries) {
      console.log(`Station ${stationNumber} already loaded, skipping fetch`);
      return;
    }

    // Same for previously loaded station if switching back from map view
    if (
      previousStationDataRef.current &&
      previousStationDataRef.current.SiteNumber === stationNumber
    ) {
      console.log(`Using previously loaded data for station ${stationNumber}`);
      setStationData(previousStationDataRef.current);
      setStationInput(stationNumber);
      return;
    }

    setLoading(true);
    try {
      console.log(`Fetching details for station ${stationNumber}`);
      const timestamp = new Date().getTime();
      const response = await fetch(
        `/api/get_station_characteristics?station=${stationNumber}&_=${timestamp}`,
      );
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setStationData(data);
      previousStationDataRef.current = data;
      setStationInput(stationNumber);
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

  // Modified handleTabChange to avoid clearing station selection when switching tabs
  const handleTabChange = (tab) => {
    console.log(`Changing tab from ${selectionTab} to ${tab}`);

    // Keep track of tab history
    setStationTabHistory(selectionTab);

    // If switching to map tab, prepare the data
    if (tab === 'map' && selectionTab !== 'map') {
      if (geometriesCache.isValid()) {
        console.log('Using cached geometries for tab change');
        setStationPoints(geometriesCache.get());
      } else if (!mapSelectionLoading) {
        console.log('Need to fetch geometries for tab change');
      }
    }

    setSelectionTab(tab);

    // Don't reset selection when going back to search
    if (tab === 'search') {
      // Only hide the station panel, but keep the selection
      setShowStationPanel(false);

      // Don't clear mapSelections anymore
      // setMapSelections([]);
    }
  };

  const toggleStationPanel = () => {
    setShowStationPanel(!showStationPanel);
  };

  const refreshStationGeometries = () => {
    geometriesCache.clear();
    setStationPoints([]);
    setGeometriesPreloaded(false);
    fetchStationGeometries();
  };

  // Fix for issue #1: Add a reset function for when we want to select a new station
  const resetStationSelection = () => {
    console.log('Resetting station selection to allow selecting a new station');
    setSelectedStationOnMap(null);
    setMapSelections([]);
    setStationData(null);
    setStationInput('');
    setShowStationPanel(false);
    previousStationDataRef.current = null;
  };

  // Add a reset button to the UI
  const renderResetButton = () => {
    if (stationData || selectedStationOnMap) {
      return (
        <div style={{ marginTop: '10px', textAlign: 'center' }}>
          <button
            onClick={resetStationSelection}
            style={{
              padding: '5px 10px',
              background: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
            }}
          >
            Clear Station & Select New
          </button>
        </div>
      );
    }
    return null;
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
                <>
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
                    <TabButton
                      active={selectionTab === 'map'}
                      onClick={() => handleTabChange('map')}
                    >
                      <TabIcon>
                        <FontAwesomeIcon icon={faLocationDot} />
                      </TabIcon>
                      Map Select
                    </TabButton>
                  </TabContainer>

                  {/* Add the reset button here */}
                  {renderResetButton()}
                </>
              )}

              <StepIndicator>
                <StepCircle active={currentStep === 1} completed={currentStep > 1}>
                  <FontAwesomeIcon icon={faMapMarkedAlt} />
                </StepCircle>
                <StepConnector completed={currentStep > 1} />
                <StepCircle active={currentStep === 2} completed={currentStep > 2}>
                  <FontAwesomeIcon icon={faLayerGroup} />
                </StepCircle>
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
                  <MapControlButton
                    title="Refresh station data"
                    onClick={refreshStationGeometries}
                    disabled={mapSelectionLoading}
                  >
                    <FontAwesomeIcon
                      icon={mapSelectionLoading ? faSpinner : 'fa-sync-alt'}
                      className={mapSelectionLoading ? 'fa-spin' : ''}
                    />
                  </MapControlButton>
                </MapControlsContainer>
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

      {showDebugger && <MapLoadingDebugger />}
    </Container>
  );
};

export default SWATGenXTemplate;
