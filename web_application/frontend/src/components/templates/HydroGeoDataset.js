import React, { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faDatabase,
  faChartBar,
  faFileAlt,
  faClipboard,
  faChevronDown,
  faChevronUp,
  faSpinner,
  faTimesCircle,
  faCheck,
  faDownload,
  faEye,
  faMapMarkerAlt,
  faSearch,
  faLayerGroup,
  faInfoCircle,
  faSync
} from '@fortawesome/free-solid-svg-icons';

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
  ReportList,
  ReportItem,
  ReportProgressBar,
  PanelContainer,
  PanelHeader,
  PanelContent
} from '../../styles/HydroGeoDataset.tsx';

import { debugLog, validatePolygonCoordinates } from '../../utils/debugUtils';
import { downloadReport, viewReport, checkReportStatus } from '../../utils/reportDownloader';
import ErrorBoundary from './ErrorBoundary';

// Import NoScroll CSS
import '../../styles/NoScroll.css';

// Import loading animation styles from common
import { spin } from '../../styles/common.tsx';
import styled from '@emotion/styled';
import { keyframes } from '@emotion/react';
import colors from '../../styles/colors.tsx';

// Add pulse animation
const pulse = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 0.9; }
  100% { opacity: 0.6; }
`;

// Add a map loading overlay
const MapLoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(24, 24, 26, 0.75);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 20;
  border-radius: 10px;
  backdrop-filter: blur(3px);
  animation: ${pulse} 1.8s ease-in-out infinite;
  
  .spinner {
    color: ${colors.accent};
    font-size: 2.5rem;
    animation: ${spin} 1.5s linear infinite;
  }
  
  .loading-text {
    margin-top: 1rem;
    color: ${colors.text};
    font-size: 1.2rem;
    font-weight: 500;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }
  
  .loading-subtext {
    margin-top: 0.5rem;
    color: ${colors.textSecondary};
    font-size: 0.9rem;
  }
`;

/**
 * Format timestamp for display
 * @param {string} timestamp - Timestamp in format YYYYMMDD_HHMMSS
 * @returns {string} - Formatted date string
 */
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

/**
 * Report item component
 */
const ReportItemComponent = ({ report, onReportAction }) => {
  const reportId = report.report_id || report.timestamp;
  
  // Handle potential missing status
  const status = report.status || 'completed';
  
  return (
    <ReportItem className={status}>
      <div className="report-header">
        <div className="report-title">
          <FontAwesomeIcon
            icon={
              status === 'processing' ? faSpinner :
              status === 'failed' ? faTimesCircle : faCheck
            }
            style={{ 
              color: status === 'failed' ? '#ff5555' : 
                     status === 'completed' ? '#4caf50' : '#ff5722'
            }}
            className={status === 'processing' ? 'fa-spin' : ''}
          />
          {report.report_type ? `${report.report_type.toUpperCase()} Report` : 'Environmental Report'}
        </div>
        <div className="report-date">
          {formatTimestamp(report.timestamp)}
        </div>
      </div>

      {status === 'processing' && (
      <div className="report-details">
            <div>Report is being generated...</div>
            <ReportProgressBar>
              <div className="progress-inner" style={{ width: '60%' }}></div>
            </ReportProgressBar>
        </div>
      )}

      {status === 'failed' && (
        <div className="report-details" style={{ color: '#ff5555' }}>
          Error: {report.error || 'Failed to generate report'}
        </div>
      )}

      {status === 'completed' && (
        <>
          <div className="report-details">
            {report.bounds && (
              <div>
                <FontAwesomeIcon icon={faMapMarkerAlt} style={{ marginRight: '8px' }} />
                Area: Lat {report.bounds.min_lat?.toFixed(2) || '?'} to {report.bounds.max_lat?.toFixed(2) || '?'}, 
                Lon {report.bounds.min_lon?.toFixed(2) || '?'} to {report.bounds.max_lon?.toFixed(2) || '?'}
              </div>
            )}
            <div>Report completed with {report.reports?.length || 0} files</div>
      </div>

        <div className="report-actions">
          <button onClick={() => onReportAction('download', reportId)}>
            <FontAwesomeIcon icon={faDownload} className="icon" />
            Download
          </button>
          <button onClick={() => onReportAction('view', reportId)}>
            <FontAwesomeIcon icon={faEye} className="icon" />
            View
          </button>
        </div>
        </>
      )}
    </ReportItem>
  );
};

/**
 * Main HydroGeoDataset component
 */
const HydroGeoDataset = () => {
  // Form and data state
  const [formData, setFormData] = useState({
    min_latitude: '',
    max_latitude: '',
    min_longitude: '',
    max_longitude: '',
    variable: '',
    subvariable: '',
    geometry: null,
  });
  
  // UI state
  const [activePanel, setActivePanel] = useState('query'); // query, report, results
  const [reports, setReports] = useState([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [reportsError, setReportsError] = useState(null);
  
  // Data state
  const [availableVariables, setAvailableVariables] = useState([]);
  const [availableSubvariables, setAvailableSubvariables] = useState([]);
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  // Map state
  const [mapRefreshKey, setMapRefreshKey] = useState(0);
  const [selectedGeometry, setSelectedGeometry] = useState(null);
  const [geometriesLoading, setGeometriesLoading] = useState(true); // Add state for geometry loading

  // Refs
  const queryMapRef = useRef(null);
  const preventMapOperations = useRef(false);
  
  // Apply NoScroll class to document body when component mounts
  useEffect(() => {
    document.documentElement.classList.add('no-scroll-page');
    document.body.classList.add('no-scroll-page');
    
    return () => {
      document.documentElement.classList.remove('no-scroll-page');
      document.body.classList.remove('no-scroll-page');
    };
  }, []);

  // Fetch available variables on mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch('/hydro_geo_dataset');
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        setAvailableVariables(data.variables || []);
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    };
    fetchOptions();
    fetchReports();
  }, []);

  // Fetch subvariables when variable changes
  useEffect(() => {
    if (!formData.variable) {
      setAvailableSubvariables([]);
      return;
    }
    
    const fetchSubvariables = async () => {
      try {
        const response = await fetch(`/hydro_geo_dataset?variable=${formData.variable}`);
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        setAvailableSubvariables(data.subvariables || []);
      } catch (error) {
        console.error('Error fetching subvariables:', error);
      }
    };
    
    fetchSubvariables();
  }, [formData.variable]);

  // Fetch reports
  const fetchReports = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/get_reports');
      if (!response.ok) {
        throw new Error(`Failed to fetch reports: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Fetched reports:', data);
      
      // Extract reports array from response - API returns { reports: [] }
      const reportsArray = data.reports || [];
      
      // Ensure we have a consistent format with fallbacks for missing properties
      const formattedReports = reportsArray.map(report => ({
        report_id: report.report_id || report.timestamp || Date.now().toString(),
        timestamp: report.timestamp || Date.now(),
        status: report.status || 'completed',
        report_type: report.report_type || 'environmental',
        bounds: report.bounds || null,
        reports: report.reports || [],
        ...report // preserve any other properties
      }));
      
      setReports(formattedReports);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching reports:', error);
      setReports([]);
      setIsLoading(false);
    }
  };
  
  // Auto-refresh reports when reports tab is active
  useEffect(() => {
    let autoRefreshInterval;
    
    if (activePanel === 'reports') {
      // Initial fetch
      fetchReports();
      
      // Set up auto-refresh every 30 seconds
      autoRefreshInterval = setInterval(() => {
        fetchReports();
      }, 30000);
    }
    
    // Clean up interval when component unmounts or tab changes
    return () => {
      if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
      }
    };
  }, [activePanel]);

  // Handle report actions (download, view)
  const handleReportAction = useCallback((action, reportId) => {
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
  }, []);

  // Handle geometry changes from the map
  const handleGeometryChange = useCallback((geom) => {
    if (!geom || preventMapOperations.current) return;

    debugLog('New geometry selected', geom);
    setGeometriesLoading(false); // Set geometries as loaded when we get a change

    // Validate polygon coordinates if present
    if (geom.type === 'polygon' && formData.polygon_coordinates) {
      const validation = validatePolygonCoordinates(formData.polygon_coordinates);
      if (!validation.valid) {
        console.warn('Polygon coordinate validation failed:', validation.message);
      }
    }

    // Store the geometry for sharing between tabs
    setSelectedGeometry(geom);
  }, [formData.polygon_coordinates]);

  // Reset geometry loading state when map refreshes
  useEffect(() => {
    setGeometriesLoading(true);
    // Set a timeout to automatically clear the loading state after 10 seconds
    // in case the geometry change event doesn't fire
    const timeout = setTimeout(() => {
      setGeometriesLoading(false);
    }, 10000);
    
    return () => clearTimeout(timeout);
  }, [mapRefreshKey]);

  // Handle form changes
  const handleChange = useCallback(({ target: { name, value } }) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  }, []);

  // Handle query submission
  const handleQuery = useCallback(async (e) => {
    e.preventDefault();

    if (
      !formData.min_latitude ||
      !formData.max_latitude ||
      !formData.min_longitude ||
      !formData.max_longitude ||
      !formData.variable ||
      !formData.subvariable
    ) {
      alert('Please complete all required fields and select an area on the map');
      return;
    }

    try {
      setIsLoading(true);

      const payload = {
        min_latitude: formData.min_latitude,
        max_latitude: formData.max_latitude,
        min_longitude: formData.min_longitude,
        max_longitude: formData.max_longitude,
        variable: formData.variable,
        subvariable: formData.subvariable,
        geometry_type: formData.geometry_type || 'extent',
        polygon_coordinates: formData.polygon_coordinates || null,
      };

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
      setData(result);
      setActivePanel('results');
    } catch (error) {
      console.error('Error fetching data:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [formData]);

  // Handle report generation
  const handleReportGenerated = useCallback(() => {
    fetchReports();
    setActivePanel('reports');
  }, [fetchReports]);

  // Panel selection
  const NavButton = ({ panel, icon, label }) => (
    <button 
      onClick={() => setActivePanel(panel)}
      style={{
        background: activePanel === panel ? '#ff5722' : 'transparent',
        border: 'none',
        color: activePanel === panel ? 'white' : '#ccc',
        padding: '12px 18px',
        borderRadius: '6px',
        cursor: 'pointer',
        display: 'flex',
          alignItems: 'center', 
        gap: '8px',
        fontWeight: activePanel === panel ? 'bold' : 'normal'
      }}
    >
      <FontAwesomeIcon icon={icon} />
      {label}
    </button>
  );

  // Render appropriate panel content
  const renderPanelContent = () => {
    switch(activePanel) {
      case 'report':
        return (
          <div style={{ padding: '20px', backgroundColor: '#2b2b2c', borderRadius: '8px' }}>
            <h2 style={{ fontSize: '1.3rem', marginBottom: '20px', color: '#ff5722' }}>
              <FontAwesomeIcon icon={faFileAlt} style={{ marginRight: '10px' }} />
              Environmental Report Generator
          </h2>
            <p style={{ marginBottom: '20px', color: '#ccc' }}>
              Generate comprehensive environmental reports for your selected area. These reports include data analysis,
              environmental assessments, and visualization of the selected variables.
            </p>
            <ReportGenerator 
              formData={formData} 
              onReportGenerated={handleReportGenerated}
            />
          </div>
        );
      
      case 'reports':
        console.log("Reports data in renderPanelContent:", reports);
        return (
          <PanelContainer>
            <PanelHeader>
              <h3>
                <FontAwesomeIcon icon={faClipboard} style={{ marginRight: '8px' }} />
                My Reports
              </h3>
              <div style={{ display: 'flex', gap: '10px' }}>
            <button 
                  className="panel-action" 
                  onClick={() => fetchReports()} 
                  title="Refresh Reports"
              style={{ 
                    background: 'transparent',
                border: 'none',
                    color: '#ccc',
                    fontSize: '1rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                    justifyContent: 'center',
                    width: '28px',
                    height: '28px',
                    borderRadius: '50%',
                    transition: 'all 0.2s'
                  }}
                >
                  <FontAwesomeIcon 
                    icon={faSync} 
                    spin={isLoading} 
                  />
            </button>
                <button className="panel-close" onClick={() => setActivePanel('query')}>×</button>
              </div>
            </PanelHeader>
            <PanelContent>
              <div style={{ padding: '10px 0' }}>
                <button 
                  onClick={() => setActivePanel('report')}
              style={{ 
                    backgroundColor: '#4CAF50',
                color: 'white', 
                    border: 'none',
                    padding: '10px 15px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    marginBottom: '20px'
                  }}
                >
                  <FontAwesomeIcon icon={faFileAlt} />
                  Generate New Report
                </button>
                
          <div style={{
                  marginBottom: '20px', 
                  fontSize: '0.85rem', 
                  color: '#888',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px' 
                }}>
                  <FontAwesomeIcon icon={faInfoCircle} />
                  Reports auto-refresh every 30 seconds. Click <FontAwesomeIcon icon={faSync} /> to refresh manually.
          </div>
                
                {isLoading ? (
                  <div style={{ display: 'flex', justifyContent: 'center', padding: '30px 0' }}>
                    <FontAwesomeIcon icon={faSpinner} spin style={{ fontSize: '24px' }} />
            </div>
          ) : reports.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '30px 0', color: '#888' }}>
                    <FontAwesomeIcon icon={faInfoCircle} style={{ fontSize: '24px', marginBottom: '15px' }} />
                    <p>No reports found. Generate a new report to get started.</p>
                  </div>
                ) : (
                  <div>
                    {reports.map(report => (
                      <ReportItemComponent
                        key={report.report_id || report.timestamp}
                        report={report}
                        onReportAction={handleReportAction}
                      />
                    ))}
                  </div>
                )}
              </div>
            </PanelContent>
          </PanelContainer>
        );
        
      case 'results':
        return data ? (
          <ResultsContainer>
            <h2 style={{ fontSize: '1.3rem', marginBottom: '15px', color: '#ff5722' }}>
              <FontAwesomeIcon icon={faChartBar} className="icon" style={{ marginRight: '10px' }} />
              Query Results
            </h2>
            <pre className="scroll-container" style={{ 
              backgroundColor: '#333', 
              padding: '15px', 
              borderRadius: '8px',
              maxHeight: '400px',
              overflowY: 'auto'
            }}>
              {JSON.stringify(data, null, 2)}
            </pre>
            
            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'space-between' }}>
              <button 
                onClick={() => setActivePanel('query')} 
                style={{
                  backgroundColor: '#444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '10px 15px',
                  cursor: 'pointer'
                }}
              >
                New Query
              </button>
              
              <button 
                onClick={() => setActivePanel('report')}
                style={{
                  backgroundColor: '#ff5722',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '10px 15px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <FontAwesomeIcon icon={faFileAlt} />
                Generate Report from Results
              </button>
            </div>
          </ResultsContainer>
        ) : (
          <div style={{ 
            textAlign: 'center', 
            padding: '40px 0',
            backgroundColor: '#333',
            borderRadius: '8px'
          }}>
            <p>No query results available. Run a query to see data here.</p>
          </div>
        );
        
      case 'query':
      default:
        return (
          <ErrorBoundary>
            <Suspense fallback={<div>Loading form...</div>}>
              <div style={{ padding: '20px', backgroundColor: '#2b2b2c', borderRadius: '8px' }}>
                
                <HydroGeoDatasetForm
                  formData={formData}
                  handleChange={handleChange}
                  handleSubmit={handleQuery}
                  availableVariables={availableVariables}
                  availableSubvariables={availableSubvariables}
                  isLoading={isLoading}
                />
                
                <div style={{ 
                  marginTop: '20px',
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <p style={{ color: '#aaa', fontSize: '0.9rem', margin: 0 }}>
                    Select an area on the map and configure query parameters above
                  </p>
                  
                  <div>
                    <button 
                      onClick={() => setActivePanel('report')}
                      style={{
                        backgroundColor: 'transparent',
                        color: '#ff5722',
                        border: '1px solid #ff5722',
                        borderRadius: '4px',
                        padding: '8px 15px',
                        marginLeft: '10px',
                        cursor: 'pointer'
                      }}
                    >
                      Skip to Report Generator
                    </button>
                  </div>
        </div>
      </div>
            </Suspense>
          </ErrorBoundary>
    );
    }
  };

  return (
    <HydroGeoContainer className="no-scroll-container hydrogeo-dataset-container" style={{ backgroundColor: '#1c1c1e' }}>
      <HydroGeoHeader style={{ 
        padding: '15px 20px', 
        marginBottom: '20px', 
        backgroundColor: '#2b2b2c',
        borderRadius: '10px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
      }}>
        <h1 style={{ 
          fontSize: '1.8rem', 
          margin: 0,
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <FontAwesomeIcon icon={faDatabase} style={{ color: '#ff5722' }} />
          HydroGeoDataset Explorer
        </h1>
        <p style={{ fontSize: '0.9rem', marginTop: '8px', marginBottom: 0, color: '#aaa' }}>
          Access PRISM, LOCA2, and Wellogic environmental data with advanced reporting
        </p>
      </HydroGeoHeader>

      <div style={{ 
          display: 'flex', 
        gap: '20px',
        marginBottom: '20px',
        backgroundColor: '#2b2b2c',
        padding: '10px',
        borderRadius: '8px',
        justifyContent: 'center'
      }}>
        <NavButton panel="query" icon={faSearch} label="Query Data" />
        <NavButton panel="report" icon={faFileAlt} label="Generate Report" />
        <NavButton panel="results" icon={faChartBar} label="View Results" />
        <NavButton panel="reports" icon={faClipboard} label="My Reports" />
      </div>

      <ContentLayout style={{ 
        backgroundColor: 'transparent',
        gap: '20px',
        alignItems: 'stretch'
      }}>
        <QuerySidebar style={{ 
          backgroundColor: 'transparent',
          border: 'none',
          boxShadow: 'none'
        }}>
          {renderPanelContent()}
            </QuerySidebar>

            <MapContainer
          className="map-container esri-map-container"
              style={{
                height: '600px',
                position: 'relative',
            backgroundColor: 'transparent',
            borderRadius: '10px',
            overflow: 'hidden',
            boxShadow: '0 4px 10px rgba(0, 0, 0, 0.2)'
          }}
        >
          {geometriesLoading && (
            <MapLoadingOverlay>
              <FontAwesomeIcon icon={faSpinner} className="spinner" />
              <div className="loading-text">Loading Map Data</div>
              <div className="loading-subtext">Initializing geographic features...</div>
            </MapLoadingOverlay>
          )}
          
          <div style={{ 
            position: 'absolute', 
            top: '10px', 
            left: '10px', 
            zIndex: 10,
            backgroundColor: 'rgba(43, 43, 44, 0.8)',
            padding: '8px 12px',
            borderRadius: '6px',
            color: 'white',
            fontSize: '0.9rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <FontAwesomeIcon icon={faLayerGroup} />
            <span>Google Aerial Imagery</span>
          </div>
          
          <div className="map-wrapper" style={{ backgroundColor: 'transparent', height: '100%', width: '100%' }}>
            <MapComponent
              key={`query-map-${mapRefreshKey}`}
              setFormData={setFormData}
              onGeometryChange={handleGeometryChange}
              containerId="queryMap"
              initialGeometry={selectedGeometry}
              ref={queryMapRef}
              onLoadingChange={setGeometriesLoading}
            />
          </div>
            </MapContainer>
          </ContentLayout>
    </HydroGeoContainer>
  );
};

export default HydroGeoDataset;
