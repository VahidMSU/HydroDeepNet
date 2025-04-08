import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch,
  faDatabase,
  faRobot,
  faChartBar,
  faFileAlt,
  faExternalLinkAlt,
  faFolderOpen,
  faClipboard,
  faDownload,
  faEye,
  faSpinner,
  faCheck,
  faTimesCircle,
  faTimes,
  faMapMarkerAlt,
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
  ReportList,
  ReportItem,
  ReportProgressBar,
} from '../../styles/HydroGeoDataset.tsx';

import { debugLog, validatePolygonCoordinates } from '../../utils/debugUtils';
import { downloadReport, viewReport, checkReportStatus } from '../../utils/reportDownloader';
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
  const [showHelpCard, setShowHelpCard] = useState(true);
  
  // Report state
  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [reportsError, setReportsError] = useState(null);
  const [sortOrder, setSortOrder] = useState('newest');

  // Add refs for map instances
  const queryMapRef = useRef(null);
  const reportMapRef = useRef(null);

  // Use a state to track if map should be shown for a tab
  const [mapVisibility, setMapVisibility] = useState({
    query: false,
    report: false,
    reports: false,
  });

  // Create a unified state for the selected geometry that's shared between tabs
  const [selectedGeometry, setSelectedGeometry] = useState(null);

  // Keep track of active maps to prevent unnecessary re-renders
  const activeMapRef = useRef(null);

  // Track if we're switching tabs to prevent unnecessary geometry updates
  const isTabSwitching = useRef(false);

  // Fetch existing reports on component mount and when activeTab changes to 'reports'
  useEffect(() => {
    if (activeTab === 'reports') {
      fetchReports();
    }
  }, [activeTab]);

  // Fetch reports from the server
  const fetchReports = async () => {
    setReportsLoading(true);
    setReportsError(null);
    try {
      const response = await fetch('/api/get_reports');
      if (response.ok) {
        const data = await response.json();
        setReports(data.reports || []);
      } else {
        console.error('Failed to fetch reports');
        setReportsError('Failed to load reports. Please try again later.');
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
      setReportsError('Error connecting to server. Please check your network connection.');
    } finally {
      setReportsLoading(false);
    }
  };

  // Format timestamp for display
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown date';
    try {
      const [date, time] = timestamp.split('_');
      const year = date.substr(0, 4);
      const month = date.substr(4, 2);
      const day = date.substr(6, 2);
      const hours = time.substr(0, 2);
      const minutes = time.substr(2, 2);
      const seconds = time.substr(4, 2);

      return `${month}/${day}/${year} ${hours}:${minutes}:${seconds}`;
    } catch (e) {
      return timestamp;
    }
  };

  // Handle report actions
  const handleReportAction = (action, reportId) => {
    if (!reportId) {
      console.error('Report ID is required for', action);
      setReportsError(`Unable to ${action}: Report ID is missing`);
      return;
    }

    debugLog('Report action triggered', { action, reportId });

    try {
      if (action === 'download') {
        downloadReport(reportId)
          .then((success) => {
            if (!success) {
              setReportsError(`Failed to download report ${reportId}`);
            }
          })
          .catch((err) => {
            console.error('Download error:', err);
            setReportsError(`Error downloading report: ${err.message}`);
          });
      } else if (action === 'view') {
        viewReport(reportId, 'html')
          .then((success) => {
            if (!success) {
              setReportsError(`Failed to view report ${reportId}`);
            }
          })
          .catch((err) => {
            console.error('View error:', err);
            setReportsError(`Error viewing report: ${err.message}`);
          });
      }
    } catch (err) {
      console.error(`Error in ${action} action:`, err);
      setReportsError(`An error occurred during ${action}: ${err.message}`);
    }
  };

  // Handle sorting reports
  const handleSortChange = (e) => {
    setSortOrder(e.target.value);
  };

  // Sort reports based on selected order
  const getSortedReports = () => {
    if (!reports || reports.length === 0) return [];
    
    const sortedReports = [...reports];
    
    switch (sortOrder) {
      case 'newest':
        return sortedReports.sort((a, b) => b.timestamp?.localeCompare(a.timestamp));
      case 'oldest':
        return sortedReports.sort((a, b) => a.timestamp?.localeCompare(b.timestamp));
      case 'name':
        return sortedReports.sort((a, b) => (a.report_type || '').localeCompare(b.report_type || ''));
      case 'status':
        return sortedReports.sort((a, b) => {
          // Sort order: completed, processing, failed
          const statusOrder = { 'completed': 0, 'processing': 1, 'failed': 2 };
          return statusOrder[a.status] - statusOrder[b.status];
        });
      default:
        return sortedReports;
    }
  };

  // Update when component mounts to show initial map
  useEffect(() => {
    setMapVisibility({
      query: activeTab === 'query',
      report: activeTab === 'report',
      reports: activeTab === 'reports',
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
    setMapVisibility({ query: false, report: false, reports: false });

    // Change tab after a small delay
    setTimeout(() => {
      setActiveTab(tabName);
      activeMapRef.current = tabName === 'query' ? queryMapRef : reportMapRef;

      // Show the appropriate map
      setTimeout(() => {
        setMapVisibility({
          query: tabName === 'query',
          report: tabName === 'report',
          reports: tabName === 'reports',
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

  // Render the reports tab
  const renderReportsTab = () => {
    return (
      <div style={{ 
        padding: '1rem', 
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '1.5rem',
          flexShrink: 0
        }}>
          <h2 style={{ margin: 0, fontSize: '1.6rem', color: '#e6e6e6' }}>
            <FontAwesomeIcon icon={faClipboard} style={{ marginRight: '0.8rem', color: '#ff8500' }} />
            Your Generated Reports
          </h2>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button 
              onClick={fetchReports} 
              style={{ 
                background: '#444',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '0.5rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.9rem'
              }}
            >
              <FontAwesomeIcon icon={faSpinner} className={reportsLoading ? 'fa-spin' : ''} />
              Refresh
            </button>
            <select 
              value={sortOrder}
              onChange={handleSortChange}
              style={{ 
                padding: '0.5rem', 
                background: '#333', 
                color: 'white', 
                border: '1px solid #555',
                borderRadius: '4px'
              }}
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="name">Report Type</option>
              <option value="status">By Status</option>
            </select>
          </div>
        </div>

        {reportsError && (
          <div style={{
            backgroundColor: 'rgba(255, 0, 0, 0.1)',
            color: '#ff5555',
            padding: '0.75rem',
            borderRadius: '4px',
            marginBottom: '1rem',
            flexShrink: 0
          }}>
            <FontAwesomeIcon icon={faTimesCircle} style={{ marginRight: '0.5rem' }} />
            {reportsError}
          </div>
        )}

        {/* Scrollable container for reports */}
        <div style={{
          overflowY: 'auto',
          flexGrow: 1,
          height: 'calc(100vh - 330px)', // Adjusted to be more precise, accounting for all headers
          padding: '0.5rem 0.5rem 0.5rem 0', // Only right padding for scrollbar
          marginRight: '0', // No need for negative margin
        }}>
          {reportsLoading && !reports.length ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <FontAwesomeIcon icon={faSpinner} spin style={{ fontSize: '2rem', color: '#ff8500', marginBottom: '1rem' }} />
              <p>Loading your reports...</p>
            </div>
          ) : reports.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '3rem 1rem',
              backgroundColor: '#2a2a2a',
              borderRadius: '8px',
              color: '#ccc'
            }}>
              <FontAwesomeIcon icon={faFileAlt} style={{ fontSize: '2rem', color: '#555', marginBottom: '1rem' }} />
              <h3>No Reports Yet</h3>
              <p>Go to the Report Generator tab to create environmental reports for your areas of interest.</p>
              <button 
                onClick={() => handleTabChange('report')}
                style={{
                  backgroundColor: '#ff8500',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '0.6rem 1.2rem',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  marginTop: '1rem'
                }}
              >
                Generate a Report
              </button>
            </div>
          ) : (
            <ReportList>
              {getSortedReports().map((report) => {
                const reportId = report.report_id || report.timestamp;
                
                return (
                  <ReportItem key={reportId} className={report.status}>
                    <div className="report-header">
                      <div className="report-title">
                        <FontAwesomeIcon
                          icon={
                            report.status === 'processing'
                              ? faSpinner
                              : report.status === 'failed'
                                ? faTimesCircle
                                : faCheck
                          }
                          className={report.status === 'processing' ? 'fa-spin' : ''}
                        />
                        {report.report_type ? `${report.report_type.toUpperCase()} Report` : 'Environmental Report'}
                      </div>
                      <div className="report-date">{formatTimestamp(report.timestamp)}</div>
                    </div>

                    <div className="report-details">
                      {report.status === 'processing' ? (
                        <>
                          <div>Report is being generated...</div>
                          <ReportProgressBar>
                            <div className="progress-inner" style={{ width: '60%' }}></div>
                          </ReportProgressBar>
                        </>
                      ) : report.status === 'failed' ? (
                        <div>Error: {report.error || 'Failed to generate report'}</div>
                      ) : (
                        <div>
                          {report.bounds && (
                            <div style={{ marginBottom: '0.5rem', fontSize: '0.85rem', color: '#ccc' }}>
                              <FontAwesomeIcon icon={faMapMarkerAlt} style={{ marginRight: '0.5rem' }} />
                              Area: Lat {report.bounds.min_lat.toFixed(2)} to {report.bounds.max_lat.toFixed(2)}, 
                              Lon {report.bounds.min_lon.toFixed(2)} to {report.bounds.max_lon.toFixed(2)}
                            </div>
                          )}
                          <div>
                            Report completed successfully. Generated {report.reports?.length || 0} files.
                          </div>
                        </div>
                      )}
                    </div>

                    {report.status === 'completed' && (
                      <div className="report-actions">
                        <button onClick={() => handleReportAction('download', reportId)}>
                          <FontAwesomeIcon icon={faDownload} className="icon" />
                          Download
                        </button>
                        <button onClick={() => handleReportAction('view', reportId)}>
                          <FontAwesomeIcon icon={faEye} className="icon" />
                          View
                        </button>
                      </div>
                    )}
                  </ReportItem>
                );
              })}
            </ReportList>
          )}
        </div>
      </div>
    );
  };

  return (
    <HydroGeoContainer>
      <HydroGeoHeader style={{ padding: '1rem', marginBottom: '1rem' }}>
        <h1 style={{ fontSize: '1.8rem', margin: 0 }}>
          <FontAwesomeIcon icon={faDatabase} style={{ marginRight: '0.8rem' }} />
          HydroGeoDataset Explorer
        </h1>
        <p style={{ fontSize: '0.9rem', marginTop: '0.5rem', marginBottom: 0 }}>
          Access high-resolution hydrological, environmental, and climate data including PRISM,
          LOCA2, and Wellogic records.
        </p>
      </HydroGeoHeader>

      {showHelpCard && (
        <InfoCard style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          padding: '0.75rem 1.2rem',
          marginBottom: '1rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <FontAwesomeIcon icon={faRobot} style={{ fontSize: '1.2rem', color: '#4299e1' }} />
            <div>
              <h3 style={{ margin: 0, fontSize: '1rem' }}>Need help with your environmental data analysis?</h3>
              <p style={{ margin: 0, fontSize: '0.8rem' }}>
                Our AI assistant can guide you through using this tool and interpreting results.
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Link 
              to="/hydrogeo-assistant" 
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                padding: '0.6rem 1rem', 
                backgroundColor: '#FF8500', 
                color: 'white', 
                borderRadius: '6px', 
                textDecoration: 'none',
                fontWeight: 'bold',
                fontSize: '0.9rem'
              }}
            >
              <FontAwesomeIcon icon={faRobot} style={{ marginRight: '0.5rem' }} />
              Open Assistant
            </Link>
            <button 
              onClick={() => setShowHelpCard(false)}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#999',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                fontSize: '1rem'
              }}
            >
              <FontAwesomeIcon icon={faTimes} />
            </button>
          </div>
        </InfoCard>
      )}

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
          <TabButton
            className={activeTab === 'reports' ? 'active' : ''}
            onClick={() => handleTabChange('reports')}
          >
            <FontAwesomeIcon icon={faClipboard} className="icon" />
            My Reports
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
                <ReportGenerator 
                  formData={formData} 
                  onReportGenerated={() => {
                    // When a report is generated, update the reports list
                    // and switch to the Reports tab
                    fetchReports();
                    setTimeout(() => handleTabChange('reports'), 500);
                  }}
                />
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

        <TabContent className={activeTab === 'reports' ? 'active' : ''}>
          {renderReportsTab()}
        </TabContent>
      </TabContainer>
    </HydroGeoContainer>
  );
};

export default HydroGeoDataset;
