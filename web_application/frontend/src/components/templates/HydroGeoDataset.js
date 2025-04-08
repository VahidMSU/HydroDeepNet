import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch,
  faDatabase,
  faRobot,
  faChartBar,
  faFileAlt,
  faExternalLinkAlt,
} from '@fortawesome/free-solid-svg-icons';
import { Link } from 'react-router-dom';

import MapComponent from '../MapComponent';
import HydroGeoDatasetForm from '../forms/HydroGeoDataset';
import ReportGenerator from '../ReportGenerator';

import {
  HydroGeoContainer,
  HydroGeoHeader,
  ContentLayout,
  QuerySidebar,
  MapContainer,
  ResultsContainer,
  TabContainer,
  TabNav,
  TabButton,
  TabContent,
  InfoCard,
} from '../../styles/HydroGeoDataset.tsx';

import { debugLog, validatePolygonCoordinates } from '../../utils/debugUtils';
import ErrorBoundary from './ErrorBoundary';

const HydroGeoDataset = () => {
  const [formData, setFormData] = useState({
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
    geometry: null,
  });
  const [availableVariables, setAvailableVariables] = useState([]);
  const [availableSubvariables, setAvailableSubvariables] = useState([]);
  const [data, setData] = useState(null);
  const [activeTab, setActiveTab] = useState('query');
  const [mapRefreshKey, setMapRefreshKey] = useState(0); // Add a key to force map refresh

  // Add refs for map instances
  const queryMapRef = useRef(null);
  const reportMapRef = useRef(null);

  // Use a state to track if map should be shown for a tab
  const [mapVisibility, setMapVisibility] = useState({
    query: false,
    report: false,
  });

  // Create a unified state for the selected geometry that's shared between tabs
  const [selectedGeometry, setSelectedGeometry] = useState(null);

  // Keep track of active maps to prevent unnecessary re-renders
  const activeMapRef = useRef(null);

  // Track if we're switching tabs to prevent unnecessary geometry updates
  const isTabSwitching = useRef(false);

  // Update when component mounts to show initial map
  useEffect(() => {
    setMapVisibility({
      query: activeTab === 'query',
      report: activeTab === 'report',
    });

    // Set the active map ref
    activeMapRef.current = activeTab === 'query' ? queryMapRef : reportMapRef;
  }, [activeTab]);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch('/hydro_geo_dataset');
        const data = await response.json();
        setAvailableVariables(data.variables);
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    };
    fetchOptions();
  }, []);

  useEffect(() => {
    const fetchSubvariables = async () => {
      if (!formData.variable) return;
      try {
        const response = await fetch(`/hydro_geo_dataset?variable=${formData.variable}`);
        const data = await response.json();
        setAvailableSubvariables(data.subvariables);
      } catch (error) {
        console.error('Error fetching subvariables:', error);
      }
    };
    fetchSubvariables();
  }, [formData.variable]);

  // Add effect to refresh map when tab changes
  useEffect(() => {
    // Force map component to re-render when tab changes by updating the key
    setMapRefreshKey((prev) => prev + 1);

    // Add a small delay to ensure DOM updates before map renders
    const timer = setTimeout(() => {
      const mapContainers = document.querySelectorAll('.map-container');
      mapContainers.forEach((container) => {
        console.log(`Map container ${container.id} visibility updated for tab: ${activeTab}`);
      });
    }, 100);

    return () => clearTimeout(timer);
  }, [activeTab]);

  const handleChange = ({ target: { name, value } }) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Update handleGeometryChange to ensure proper geometry sharing
  const handleGeometryChange = useCallback(
    (geom) => {
      if (!geom || isTabSwitching.current) return;

      debugLog('New geometry selected', geom);

      // Validate polygon coordinates if present
      if (geom.type === 'polygon' && formData.polygon_coordinates) {
        const validation = validatePolygonCoordinates(formData.polygon_coordinates);
        if (!validation.valid) {
          console.warn('Polygon coordinate validation failed:', validation.message);
        }
      }

      // Store the geometry for sharing between tabs
      setSelectedGeometry(geom);
    },
    [formData.polygon_coordinates],
  );

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Debug log to see what's happening
    console.log('Form submission triggered with data:', formData);

    // Ensure we have all required data
    if (
      !formData.min_latitude ||
      !formData.max_latitude ||
      !formData.min_longitude ||
      !formData.max_longitude
    ) {
      console.error('Missing coordinate data for query');
      return;
    }

    if (!formData.variable || !formData.subvariable) {
      console.error('Missing variable or subvariable selection');
      return;
    }

    try {
      setIsLoading(true);

      // Prepare the request payload with all necessary data
      const payload = {
        min_latitude: formData.min_latitude,
        max_latitude: formData.max_latitude,
        min_longitude: formData.min_longitude,
        max_longitude: formData.max_longitude,
        variable: formData.variable,
        subvariable: formData.subvariable,
        geometry_type: formData.geometry_type || 'extent',
        // Ensure polygon coordinates are passed correctly
        polygon_coordinates: formData.polygon_coordinates || null,
      };

      console.log('Sending API request with payload:', payload);

      const response = await fetch('/hydro_geo_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Server returned status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Received API response:', result);
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
      // Consider adding a UI notification here to inform the user
    } finally {
      setIsLoading(false);
    }
  };

  // Modified tab switching function for better reliability
  const handleTabChange = (tabName) => {
    // Skip if already on this tab or switch in progress
    if (tabName === activeTab || isTabSwitching.current) return;

    console.log(`Switching tab from ${activeTab} to ${tabName}`);

    // Set flag to prevent concurrent operations
    isTabSwitching.current = true;

    // First hide maps
    setMapVisibility({ query: false, report: false });

    // Change tab after a small delay
    setTimeout(() => {
      setActiveTab(tabName);
      activeMapRef.current = tabName === 'query' ? queryMapRef : reportMapRef;

      // Show the appropriate map
      setTimeout(() => {
        setMapVisibility({
          query: tabName === 'query',
          report: tabName === 'report',
        });

        // Draw geometry after map is ready
        setTimeout(() => {
          redrawGeometry();
          isTabSwitching.current = false;
        }, 400);
      }, 100);
    }, 100);
  };

  // Simplified geometry redraw function
  const redrawGeometry = useCallback(() => {
    if (!selectedGeometry) return;

    const activeMap = activeTab === 'query' ? queryMapRef.current : reportMapRef.current;
    if (activeMap?.drawGeometry) {
      activeMap.drawGeometry(selectedGeometry);
    }
  }, [selectedGeometry, activeTab]);

  // Optimize MapComponent rendering with simplified memoization
  const renderMapComponent = useCallback(
    (type) => {
      if (!mapVisibility[type]) return null;

      return (
        <MapComponent
          key={`${type}-map-${mapRefreshKey}`}
          setFormData={setFormData}
          onGeometryChange={handleGeometryChange}
          containerId={`${type}Map`}
          initialGeometry={selectedGeometry}
          ref={type === 'query' ? queryMapRef : reportMapRef}
        />
      );
    },
    [mapVisibility, mapRefreshKey, selectedGeometry, handleGeometryChange],
  );

  // For loading state
  const [isLoading, setIsLoading] = useState(false);

  return (
    <HydroGeoContainer>
      <HydroGeoHeader>
        <h1>
          <FontAwesomeIcon icon={faDatabase} style={{ marginRight: '0.8rem' }} />
          HydroGeoDataset Explorer
        </h1>
        <p>
          Access high-resolution hydrological, environmental, and climate data including PRISM,
          LOCA2, and Wellogic records through a simple interface. Select your area of interest and
          data variables.
        </p>
      </HydroGeoHeader>

      <InfoCard style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3>
            <FontAwesomeIcon icon={faRobot} className="icon" />
            Need Help?
          </h3>
          <p>
            Our AI assistant can answer your questions about environmental data, guide you through using
            this tool, and help interpret results.
          </p>
        </div>
        <Link 
          to="/hydrogeo-assistant" 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            padding: '0.8rem 1.2rem', 
            backgroundColor: '#FF8500', 
            color: 'white', 
            borderRadius: '8px', 
            textDecoration: 'none',
            fontWeight: 'bold',
            marginLeft: '1rem'
          }}
        >
          <FontAwesomeIcon icon={faRobot} style={{ marginRight: '0.5rem' }} />
          Open Assistant
          <FontAwesomeIcon icon={faExternalLinkAlt} style={{ marginLeft: '0.5rem', fontSize: '0.8rem' }} />
        </Link>
      </InfoCard>

      <TabContainer>
        <TabNav>
          <TabButton
            className={activeTab === 'query' ? 'active' : ''}
            onClick={() => handleTabChange('query')}
          >
            <FontAwesomeIcon icon={faSearch} className="icon" />
            Query
          </TabButton>
          <TabButton
            className={activeTab === 'report' ? 'active' : ''}
            onClick={() => handleTabChange('report')}
          >
            <FontAwesomeIcon icon={faFileAlt} className="icon" />
            Report Generator
          </TabButton>
        </TabNav>

        <TabContent className={activeTab === 'query' ? 'active' : ''}>
          <ContentLayout>
            <QuerySidebar>
              <ErrorBoundary>
                <HydroGeoDatasetForm
                  formData={formData}
                  handleChange={handleChange}
                  handleSubmit={handleSubmit}
                  availableVariables={availableVariables}
                  availableSubvariables={availableSubvariables}
                  isLoading={isLoading}
                />
              </ErrorBoundary>
            </QuerySidebar>

            <MapContainer
              className="map-container"
              style={{
                height: '600px',
                visibility: mapVisibility.query ? 'visible' : 'hidden',
                position: 'relative',
              }}
            >
              {mapVisibility.query && renderMapComponent('query')}
            </MapContainer>
          </ContentLayout>

          {data && (
            <ResultsContainer>
              <h2>
                <FontAwesomeIcon icon={faChartBar} className="icon" />
                Query Results
              </h2>
              <pre>{JSON.stringify(data, null, 2)}</pre>
            </ResultsContainer>
          )}
        </TabContent>

        <TabContent className={activeTab === 'report' ? 'active' : ''}>
          <ContentLayout>
            <QuerySidebar>
              <ErrorBoundary>
                <ReportGenerator formData={formData} />
              </ErrorBoundary>
            </QuerySidebar>

            <MapContainer
              className="map-container"
              style={{
                height: '600px',
                visibility: mapVisibility.report ? 'visible' : 'hidden',
                position: 'relative',
              }}
            >
              {mapVisibility.report && renderMapComponent('report')}
            </MapContainer>
          </ContentLayout>
        </TabContent>
      </TabContainer>
    </HydroGeoContainer>
  );
};

export default HydroGeoDataset;
