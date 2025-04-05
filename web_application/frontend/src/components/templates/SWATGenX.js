import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
  faSyncAlt,
  faRedoAlt,
  faLock,
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
  FormSelect,
  InputGroup,
  InputLabel,
} from '../../styles/SWATGenX.tsx';

const SWATGenXTemplate = () => {
  const [isDescriptionOpen, setIsDescriptionOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(1); // Step 1: Station Selection, Step 2: Resolution & Analysis
  const [stationInput, setStationInput] = useState('');
  const [stationData, setStationData] = useState(null);
  const [lsResolution, setLsResolution] = useState('250');
  const [demResolution] = useState('30'); // Fixed to 30, removed setter function
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
  const previousStationDataRef = useRef(null);
  const [initialLoadComplete, setInitialLoadComplete] = useState(false);
  const [basemapType, setBasemapType] = useState('streets');
  const [showWatershed, setShowWatershed] = useState(true);
  const [showStreams, setShowStreams] = useState(true);
  const [showLakes, setShowLakes] = useState(true);

  const mapRefreshFunctionRef = useRef(null);

  const lsResolutionOptions = ['30', '100', '250', '500', '1000'];

  const geometriesCache = {
    data: null,
    timestamp: null,
    expirationTime: 7 * 24 * 60 * 60 * 1000, // 7 days - even longer expiration
    isLoading: false,
    error: null,

    STORAGE_KEY: 'stationGeometriesPermanentCache',
    STATIC_FILE_PATH: '/static/stations.geojson',

    set: function (data) {
      this.data = data;
      this.timestamp = Date.now();
      this.isLoading = false;
      this.error = null;

      try {
        const cacheData = {
          data: data,
          timestamp: Date.now(),
        };

        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(cacheData));
        sessionStorage.setItem(this.STORAGE_KEY, JSON.stringify(cacheData));

        console.log(
          `Station geometries cache updated with ${data.length} stations (persistent storage)`,
        );
      } catch (e) {
        console.warn('Failed to cache station geometries in storage:', e);
      }
    },

    get: function () {
      if (this.data && this.timestamp && Date.now() - this.timestamp <= this.expirationTime) {
        return this.data;
      }

      try {
        let cachedData = localStorage.getItem(this.STORAGE_KEY);

        if (!cachedData) {
          cachedData = sessionStorage.getItem(this.STORAGE_KEY);
        }

        if (cachedData) {
          const parsed = JSON.parse(cachedData);
          if (parsed.timestamp && Date.now() - parsed.timestamp <= this.expirationTime) {
            this.data = parsed.data;
            this.timestamp = parsed.timestamp;
            console.log(`Using ${parsed.data.length} station geometries from persistent storage`);
            return parsed.data;
          }
        }
      } catch (e) {
        console.warn('Failed to retrieve station geometries from storage:', e);
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
      try {
        sessionStorage.removeItem(this.STORAGE_KEY);
        localStorage.removeItem(this.STORAGE_KEY);
      } catch (e) {
        console.warn('Failed to clear station geometries from storage:', e);
      }
    },

    setLoading: function (status) {
      this.isLoading = status;
    },

    setError: function (error) {
      this.error = error;
      this.isLoading = false;
    },
  };

  // Memoize station points to prevent unnecessary re-renders
  const memoizedStationPoints = useMemo(() => {
    console.log(`Memoizing ${stationPoints.length} station points`);
    // Only keep essential properties to reduce memory usage
    if (stationPoints.length > 0) {
      return stationPoints.map((point) => ({
        type: point.type,
        geometry: point.geometry,
        properties: {
          SiteNumber: point.properties.SiteNumber,
          SiteName: point.properties.SiteName,
          id: point.properties.id,
        },
      }));
    }
    return [];
  }, [stationPoints]);

  // Optimize fetch function with better error handling and fewer side effects
  const fetchStationGeometries = useCallback(
    async (forceRefresh = false) => {
      if ((mapSelectionLoading || geometriesCache.isLoading) && !forceRefresh) return;

      if (geometriesCache.isValid() && !forceRefresh) {
        const cachedData = geometriesCache.get();
        setStationPoints(cachedData);
        setGeometriesPreloaded(true);
        console.log('Using cached station geometries, skipping fetch');
        return;
      }

      setMapSelectionLoading(true);
      geometriesCache.setLoading(true);

      try {
        console.log('Loading station geometries from static file');
        // Use AbortController to handle timeouts
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const response = await fetch(geometriesCache.STATIC_FILE_PATH, {
          headers: {
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(
            `Failed to fetch station geometries: ${response.status} ${response.statusText}`,
          );
        }

        const data = await response.json();

        if (!data || !data.features) {
          throw new Error('Invalid GeoJSON format in static file');
        }

        // Process features in chunks to avoid blocking the main thread
        const features = data.features || [];
        const processFeatures = (startIdx, chunkSize) => {
          const endIdx = Math.min(startIdx + chunkSize, features.length);
          const chunk = features.slice(startIdx, endIdx);

          // Only keep essential properties to reduce memory footprint
          const processedChunk = chunk.map((feature) => ({
            type: feature.type,
            geometry: feature.geometry,
            properties: {
              SiteNumber: feature.properties.SiteNumber,
              SiteName: feature.properties.SiteName,
              id: feature.properties.id,
            },
          }));

          // Update state with processed features so far
          setStationPoints((prev) => [...prev, ...processedChunk]);

          // Continue processing if there are more features
          if (endIdx < features.length) {
            // Use setTimeout to yield to the browser for rendering
            setTimeout(() => processFeatures(endIdx, chunkSize), 0);
          } else {
            // All features processed
            geometriesCache.set(features);
            setGeometriesPreloaded(true);
            setInitialLoadComplete(true);
            setMapSelectionLoading(false);
          }
        };

        // Clear existing points before starting to add new ones
        setStationPoints([]);
        // Start processing in chunks of 1000 features
        processFeatures(0, 1000);

        console.log(`Processing ${features.length} station geometries in chunks`);

        if (feedbackType === 'error' && feedbackMessage.includes('station')) {
          setFeedbackMessage('');
          setFeedbackType('');
        }
      } catch (error) {
        console.error('Error loading station geometries from static file:', error);
        geometriesCache.setError(error);
        setFeedbackMessage('Failed to load stations on map: ' + error.message);
        setFeedbackType('error');
        setMapSelectionLoading(false);
      }
    },
    [mapSelectionLoading, feedbackType, feedbackMessage],
  );

  useEffect(() => {
    // If cached geometries exist, use them; if not, fetch once.
    const cachedGeometries = geometriesCache.get();
    if (cachedGeometries) {
      console.log('Initializing station geometries from cache');
      setStationPoints(cachedGeometries);
      setGeometriesPreloaded(true);
      setInitialLoadComplete(true);
    } else if (!initialLoadComplete && !geometriesCache.isLoading) {
      // Fetch once if not previously loaded
      fetchStationGeometries(false);
    }
  }, [initialLoadComplete, fetchStationGeometries]);

  useEffect(() => {
    if (stationData) {
      previousStationDataRef.current = stationData;
    }
  }, [stationData]);

  const handleMapRefresh = () => {
    console.log('Manual map refresh requested');
    setMapSelectionLoading(true);

    if (mapRefreshFunctionRef.current && typeof mapRefreshFunctionRef.current === 'function') {
      const refreshResult = mapRefreshFunctionRef.current();
      console.log('Map refresh result:', refreshResult);

      setTimeout(() => {
        setMapSelectionLoading(false);
      }, 1000);
    } else {
      console.warn('Map refresh function not available');

      const existingGeometries = geometriesCache.get();
      if (existingGeometries) {
        setStationPoints([]);
        setTimeout(() => {
          setStationPoints(existingGeometries);
          setMapSelectionLoading(false);
        }, 500);
      } else {
        fetchStationGeometries(false);
      }
    }
  };

  const refreshStationGeometries = () => {
    console.log('User requested explicit refresh of station geometries');
    geometriesCache.clear();
    setStationPoints([]);
    setGeometriesPreloaded(false);
    fetchStationGeometries(true);
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
        if (response.status === 404) {
          throw new Error('Station not found. Please verify the station number and try again.');
        }
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText || 'Unknown error'}`);
      }
      const data = await response.json();
      if (!data) {
        throw new Error('No data received from server');
      }
      setStationData(data);
      previousStationDataRef.current = data;
      setStationInput(stationNumber);
      setFeedbackMessage('');
      setFeedbackType('');
    } catch (error) {
      console.error('Error fetching station details:', error);
      setStationData(null);
      previousStationDataRef.current = null;
      setFeedbackMessage(error.message || 'Failed to fetch station details');
      setFeedbackType('error');
      setSelectedStationOnMap(null);
      setMapSelections([]);
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

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error(
            data.message ||
              'You have reached the maximum concurrent model limit. Please wait for existing tasks to complete.',
          );
        }
        throw new Error(data.message || data.error || 'Unknown error occurred.');
      }

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

  const toggleBasemap = () => {
    setBasemapType((prev) => (prev === 'streets' ? 'google' : 'streets'));
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
    setGeometriesPreloaded(false); // Reset geometries preloaded flag
    setStationPoints([]); // Clear station points
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
              <span style={{ fontSize: '14px', whiteSpace: 'nowrap' }}>Next</span>
            </SubmitButton>
          </NavigationButtons>
        </>
      );
    } else {
      return (
        <>
          <div style={{ margin: '10px 0 20px 0', fontSize: '15px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Selected Station:</div>
            <div
              style={{
                padding: '5px 10px',
                backgroundColor: '#222222', // changed to dark gray
                borderRadius: '5px',
                color: 'white', // updated for contrast
              }}
            >
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

            <InputGroup>
              <InputLabel htmlFor="ls-resolution">Landuse/Soil Resolution:</InputLabel>
              <FormSelect
                id="ls-resolution"
                value={lsResolution}
                onChange={(e) => setLsResolution(e.target.value)}
              >
                {lsResolutionOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </FormSelect>
            </InputGroup>

            <InputGroup>
              <InputLabel htmlFor="dem-resolution">
                DEM Resolution:
                <FontAwesomeIcon
                  icon={faLock}
                  style={{ marginLeft: '8px', fontSize: '12px', color: '#ff8500' }}
                  title="Fixed at 30m resolution"
                />
              </InputLabel>
              <div
                style={{
                  padding: '10px 14px',
                  backgroundColor: '#3a3a3c',
                  borderRadius: '6px',
                  border: '1px solid #505050',
                  color: '#bbbbbb',
                  fontSize: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <span>{demResolution} m</span>
                <span style={{ marginLeft: 'auto', fontSize: '12px', color: '#999' }}>
                  Fixed Value
                </span>
              </div>
            </InputGroup>
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
              <span style={{ fontSize: '14px' }}>Back</span>
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
                  <span style={{ fontSize: '14px' }}>Processing...</span>
                </>
              ) : (
                <>
                  <ButtonIcon>
                    <FontAwesomeIcon icon={faPlay} />
                  </ButtonIcon>
                  <span style={{ fontSize: '14px' }}>Run Model</span>
                </>
              )}
            </SubmitButton>
          </NavigationButtons>
        </>
      );
    }
  };

  const LayerControls = () => {
    // Only show when geometries are loaded - check each type more explicitly
    const hasWatershed = stationData?.geometries && stationData.geometries.length > 0;
    const hasStreams = stationData?.streams_geometries && stationData.streams_geometries.length > 0;
    const hasLakes = stationData?.lakes_geometries && stationData.lakes_geometries.length > 0;

    // Only show controls if at least one of the geometry types exists
    if (!stationData || (!hasWatershed && !hasStreams && !hasLakes)) {
      return null;
    }

    return (
      <div
        style={{
          position: 'absolute',
          left: '10px',
          bottom: '10px',
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '10px',
          borderRadius: '4px',
          boxShadow: '0 2px 6px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          color: 'black',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '8px', color: 'black' }}>
          <FontAwesomeIcon icon={faLayerGroup} style={{ marginRight: '5px' }} />
          Layer Visibility
        </div>

        {/* Only show the watershed checkbox if watershed geometries exist */}
        {hasWatershed && (
          <div style={{ marginBottom: '8px' }}>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '14px',
                cursor: 'pointer',
                color: 'black',
              }}
            >
              <input
                type="checkbox"
                checked={showWatershed}
                onChange={(e) => setShowWatershed(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Watershed
            </label>
          </div>
        )}

        {/* Only show the streams checkbox if stream geometries exist */}
        {hasStreams && (
          <div style={{ marginBottom: '8px' }}>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '14px',
                cursor: 'pointer',
                color: 'black',
              }}
            >
              <input
                type="checkbox"
                checked={showStreams}
                onChange={(e) => setShowStreams(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Streams
            </label>
          </div>
        )}

        {/* Only show the lakes checkbox if lake geometries exist */}
        {hasLakes && (
          <div>
            <label
              style={{
                display: 'flex',
                alignItems: 'center',
                fontSize: '14px',
                cursor: 'pointer',
                color: 'black',
              }}
            >
              <input
                type="checkbox"
                checked={showLakes}
                onChange={(e) => setShowLakes(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Lakes
            </label>
          </div>
        )}
      </div>
    );
  };

  return (
    <Container>
      <Header>
        <HeaderTitle>
          <TitleIcon>
            <FontAwesomeIcon icon={faMap} />
          </TitleIcon>
          <span style={{ marginLeft: '500px', color: 'white' }}>
            SWATGenX: Automated Watershed Modeling Platform
          </span>
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
                <StepConnector completed={currentStep > 1} />
                <StepCircle active={currentStep === 2} completed={currentStep > 2}>
                  <FontAwesomeIcon icon={faLayerGroup} />
                </StepCircle>
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
                  icon={mapSelectionLoading ? faSpinner : faSyncAlt}
                  className={mapSelectionLoading ? 'fa-spin' : ''}
                />
              </MapControlButton>
              <MapControlButton
                title="Refresh map display"
                onClick={handleMapRefresh}
                disabled={mapSelectionLoading}
              >
                <FontAwesomeIcon icon={faRedoAlt} />
              </MapControlButton>
              <MapControlButton
                title={`Switch to ${basemapType === 'streets' ? 'Google Aerial' : 'Streets'} Basemap`}
                onClick={toggleBasemap}
              >
                <FontAwesomeIcon icon={faGears} />
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

            <LayerControls />

            <EsriMap
              geometries={stationData?.geometries || []}
              streamsGeometries={stationData?.streams_geometries || []}
              lakesGeometries={stationData?.lakes_geometries || []}
              stationPoints={memoizedStationPoints}
              onStationSelect={handleStationSelectFromMap}
              showStations={true}
              selectedStationId={selectedStationOnMap?.SiteNumber}
              refreshMapRef={mapRefreshFunctionRef}
              enableClustering={true}
              maxVisibleStations={5000}
              updateInterval={60}
              basemapType={basemapType}
              showWatershed={showWatershed}
              showStreams={showStreams}
              showLakes={showLakes}
            />
          </MapInnerContainer>
        </MapContainer>
      </Content>

      {showDebugger && <MapLoadingDebugger />}
    </Container>
  );
};

// Use React.memo to prevent unnecessary re-renders of the entire component
export default React.memo(SWATGenXTemplate);
