import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  faArrowRight,
  faArrowLeft,
  faPlay,
  faMousePointer,
  faSpinner,
  faListUl,
  faRuler,
} from '@fortawesome/free-solid-svg-icons';
import EsriMap from '../EsriMap.js';
import SearchForm from '../SearchForm';
import MapLoadingDebugger from '../debug/MapLoadingDebugger';
import StationDetails from '../StationDetails';
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
  StepIndicator,
  StepCircle,
  StepText,
  StepConnector,
  MapControlsContainer,
  MapControlButton,
  LoadingOverlay,
  LoadingIcon,
  NavigationButtons,
  SubmitButton,
  ButtonIcon,
  LoadingSpinner,
  FormInput,
} from '../../styles/SWATGenX.tsx';

// Modify the geometriesCache to be more persistent
const geometriesCache = {
  data: null,
  timestamp: null,
  expirationTime: 30 * 60 * 1000,
  isLoading: false,
  error: null,

  set: function (data) {
    this.data = data;
    this.timestamp = Date.now();
    this.isLoading = false;
    this.error = null;
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
    if (this.data && this.timestamp && Date.now() - this.timestamp <= this.expirationTime) {
      return this.data;
    }

    try {
      const cachedData = sessionStorage.getItem('stationGeometriesCache');
      if (cachedData) {
        const parsed = JSON.parse(cachedData);
        if (parsed.timestamp && Date.now() - parsed.timestamp <= this.expirationTime) {
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
  const [stationPoints, setStationPoints] = useState([]);
  const [mapSelectionLoading, setMapSelectionLoading] = useState(false);
  const [selectedStationOnMap, setSelectedStationOnMap] = useState(null);
  const [mapSelections, setMapSelections] = useState([]);
  const [showStationPanel, setShowStationPanel] = useState(false);
  const [geometriesPreloaded, setGeometriesPreloaded] = useState(false);
  const [showDebugger, setShowDebugger] = useState(false);
  const [drawingMode, setDrawingMode] = useState(false);
  const previousStationDataRef = useRef(null);

  // Define fetchStationGeometries using useCallback so it can be included in dependency arrays
  const fetchStationGeometries = useCallback(async () => {
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
  }, [mapSelectionLoading, feedbackType, feedbackMessage]);

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

  useEffect(() => {
    const cachedGeometries = geometriesCache.get();
    if (cachedGeometries) {
      console.log('Initialized station geometries from persistent cache');
      setGeometriesPreloaded(true);
      setStationPoints(cachedGeometries);
    } else {
      console.log('No cached geometries found, will preload');
    }
  }, []);

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
        setStationPoints(features);
      } catch (error) {
        console.error('Error preloading station geometries:', error);
        geometriesCache.setError(error);
        setFeedbackMessage('Failed to load stations on map: ' + error.message);
        setFeedbackType('error');
      } finally {
        setMapSelectionLoading(false);
      }
    };

    preloadStationGeometries();
  }, []);

  useEffect(() => {
    if (stationPoints.length === 0 && !mapSelectionLoading && !geometriesCache.isLoading) {
      fetchStationGeometries();
    }
  }, [stationPoints.length, mapSelectionLoading, geometriesPreloaded, fetchStationGeometries]);

  useEffect(() => {
    if (stationData) {
      previousStationDataRef.current = stationData;
    }
  }, [stationData]);

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

  const handleStationSelect = async (stationNumber) => {
    if (stationData && stationData.SiteNumber === stationNumber && stationData.geometries) {
      console.log(`Station ${stationNumber} already loaded, skipping fetch`);
      return;
    }

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

  const toggleStationPanel = () => {
    setShowStationPanel(!showStationPanel);
  };

  const refreshStationGeometries = () => {
    geometriesCache.clear();
    setStationPoints([]);
    setGeometriesPreloaded(false);
    fetchStationGeometries();
  };

  const handleMultipleStationsSelect = (stations) => {
    console.log('Multiple stations selected:', stations);
    if (stations && stations.length > 0) {
      setMapSelections(stations);
      handleStationSelect(stations[0].SiteNumber);
    }
  };

  const resetStationSelection = () => {
    console.log('Resetting station selection to allow selecting a new station');
    setSelectedStationOnMap(null);
    setMapSelections([]);
    setStationData(null);
    setStationInput('');
    setShowStationPanel(false);
    previousStationDataRef.current = null;
  };

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

  const renderModelSettingsContent = () => {
    if (currentStep === 1) {
      return (
        <>
          <SearchForm
            setStationData={setStationData}
            setLoading={setLoading}
            mapSelections={mapSelections}
            setDrawingMode={setDrawingMode}
            drawingMode={drawingMode}
            handleStationSelect={handleStationSelect}
          />

          {renderResetButton()}

          {stationData && <StationDetails stationData={stationData} />}

          <NavigationButtons>
            <SubmitButton
              type="button"
              onClick={handleNextStep}
              isLoading={loading}
              disabled={loading || (!stationData && !stationInput)}
            >
              <ButtonIcon>
                <FontAwesomeIcon icon={faArrowRight} />
              </ButtonIcon>
              Next: Configure Model
            </SubmitButton>
          </NavigationButtons>
        </>
      );
    } else {
      return (
        <>
          <div style={{ margin: '10px 0 20px 0', fontSize: '15px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Selected Station:</div>
            <div style={{ padding: '5px 10px', backgroundColor: '#f0f8ff', borderRadius: '5px' }}>
              {stationData ? (
                <>
                  <div>
                    <strong>{stationData.SiteName}</strong>
                  </div>
                  <div>Station ID: {stationData.SiteNumber}</div>
                </>
              ) : (
                <div>Station ID: {stationInput}</div>
              )}
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
              <FontAwesomeIcon icon={faRuler} style={{ marginRight: '8px' }} />
              Resolution Settings
            </div>

            <div style={{ margin: '10px 0' }}>
              <label htmlFor="ls-resolution" style={{ display: 'block', marginBottom: '5px' }}>
                Landuse/Soil Resolution:
              </label>
              <FormInput
                id="ls-resolution"
                type="text"
                value={lsResolution}
                onChange={(e) => setLsResolution(e.target.value)}
                placeholder="Enter resolution (e.g., 250)"
              />
            </div>

            <div style={{ margin: '10px 0' }}>
              <label htmlFor="dem-resolution" style={{ display: 'block', marginBottom: '5px' }}>
                DEM Resolution:
              </label>
              <FormInput
                id="dem-resolution"
                type="text"
                value={demResolution}
                onChange={(e) => setDemResolution(e.target.value)}
                placeholder="Enter resolution (e.g., 30)"
              />
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
              <FontAwesomeIcon icon={faLayerGroup} style={{ marginRight: '8px' }} />
              Analysis Options
            </div>

            <div style={{ margin: '10px 0' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={calibrationFlag}
                  onChange={(e) => setCalibrationFlag(e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Calibration
              </label>
            </div>

            <div style={{ margin: '10px 0' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={sensitivityFlag}
                  onChange={(e) => setSensitivityFlag(e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Sensitivity Analysis
              </label>
            </div>

            <div style={{ margin: '10px 0' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={validationFlag}
                  onChange={(e) => setValidationFlag(e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Validation
              </label>
            </div>
          </div>

          <NavigationButtons style={{ marginBottom: '30px', paddingBottom: '20px' }}>
            <SubmitButton
              type="button"
              onClick={handlePreviousStep}
              isLoading={false}
              secondary
              style={{ maxWidth: '48%' }}
            >
              <ButtonIcon>
                <FontAwesomeIcon icon={faArrowLeft} />
              </ButtonIcon>
              Back
            </SubmitButton>

            <SubmitButton
              type="button"
              onClick={handleSubmit}
              isLoading={loading}
              disabled={loading}
              style={{ maxWidth: '48%' }}
            >
              {loading ? (
                <>
                  <LoadingSpinner />
                  Processing...
                </>
              ) : (
                <>
                  <ButtonIcon>
                    <FontAwesomeIcon icon={faPlay} />
                  </ButtonIcon>
                  Run Model
                </>
              )}
            </SubmitButton>
          </NavigationButtons>
        </>
      );
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
              <StepIndicator>
                <StepCircle active={currentStep === 1} completed={currentStep > 1}>
                  <FontAwesomeIcon icon={faMapMarkedAlt} />
                </StepCircle>
                <StepText>Select Station</StepText>
                <StepConnector completed={currentStep > 1} />
                <StepCircle active={currentStep === 2} completed={currentStep > 2}>
                  <FontAwesomeIcon icon={faLayerGroup} />
                </StepCircle>
                <StepText>Configure Model</StepText>
              </StepIndicator>

              {renderModelSettingsContent()}

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

            <EsriMap
              geometries={stationData?.geometries || []}
              streamsGeometries={stationData?.streams_geometries || []}
              lakesGeometries={stationData?.lakes_geometries || []}
              stationPoints={stationPoints}
              onStationSelect={handleStationSelectFromMap}
              onMultipleStationsSelect={handleMultipleStationsSelect}
              showStations={true}
              selectedStationId={selectedStationOnMap?.SiteNumber}
              drawingMode={drawingMode}
              onDrawComplete={handleMultipleStationsSelect}
            />
          </MapInnerContainer>
        </MapContainer>
      </Content>

      {showDebugger && <MapLoadingDebugger />}
    </Container>
  );
};

export default SWATGenXTemplate;
