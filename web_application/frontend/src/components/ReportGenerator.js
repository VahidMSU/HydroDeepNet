import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faFileAlt,
  faLayerGroup,
  faSpinner,
  faCheck,
  faTimesCircle,
  faDownload,
  faEye,
  faClipboard,
  faChevronDown,
} from '@fortawesome/free-solid-svg-icons';
import {
  ReportForm,
  ReportFormHeader,
  FormGroup,
  InputField,
  SubmitButton,
  ReportRow,
  ReportStatusContainer,
  ReportList,
  ReportItem,
  ReportProgressBar,
  CoordinatesDisplay,
} from '../styles/HydroGeoDataset.tsx';

import {
  faMapMarkerAlt,
} from '@fortawesome/free-solid-svg-icons';

import InfoBox from './InfoBox';
import { downloadReport, viewReport, checkReportStatus } from '../utils/reportDownloader';
import { debugLog } from '../utils/debugUtils';

const ReportGenerator = ({ formData }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reportSettings, setReportSettings] = useState({
    report_type: 'all',
    start_year: 2015,
    end_year: 2020,
    include_climate_change: true,
  });
  // Add a coordinator ID reference to allow the map to communicate with the form
  const reportCoordinatorId = 'report-coordinator';

  // Fetch existing reports on component mount
  useEffect(() => {
    fetchReports();
  }, []);

  // Fetch reports from the server
  const fetchReports = async () => {
    try {
      const response = await fetch('/api/get_reports');
      if (response.ok) {
        const data = await response.json();
        setReports(data.reports);
      } else {
        console.error('Failed to fetch reports');
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setReportSettings((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  // Properly check for selected geometry before allowing submission
  const hasSelectedArea = Boolean(
    formData.min_latitude &&
      formData.max_latitude &&
      formData.min_longitude &&
      formData.max_longitude,
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Check if geometry is selected
    if (!hasSelectedArea) {
      setError('Please select an area on the map first');
      setLoading(false);
      return;
    }

    try {
      // Log coordinates for debugging
      console.log('Submitting report with coordinates:', {
        min_latitude: formData.min_latitude,
        max_latitude: formData.max_latitude,
        min_longitude: formData.min_longitude,
        max_longitude: formData.max_longitude,
        polygon_coordinates: formData.polygon_coordinates,
        geometry_type: formData.geometry_type,
      });

      // Prepare data for the API call
      const reportData = {
        ...reportSettings,
        min_latitude: parseFloat(formData.min_latitude),
        max_latitude: parseFloat(formData.max_latitude),
        min_longitude: parseFloat(formData.min_longitude),
        max_longitude: parseFloat(formData.max_longitude),
        // Make sure polygon coordinates are properly included if available
        polygon_coordinates: formData.polygon_coordinates || null,
        geometry_type: formData.geometry_type || 'extent',
      };

      console.log('Sending report generation request with data:', reportData);

      // Send the report generation request
      const response = await fetch('/api/generate_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportData),
      });

      // Properly handle non-OK responses
      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Server returned ${response.status}: ${response.statusText}`;

        try {
          // Try to parse the error as JSON
          const errorData = JSON.parse(errorText);
          if (errorData.error) {
            errorMessage = errorData.error;
          }
        } catch (e) {
          // Not JSON, use text as is if it exists
          if (errorText) errorMessage = errorText;
        }

        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('Report generation started:', result);

      // Add the new report to the list
      const newReport = {
        report_id: result.report_id,
        timestamp: result.report_id,
        status: 'processing',
        report_type: reportSettings.report_type,
      };

      setReports((prev) => [newReport, ...prev]);

      // Set up polling to check report status
      const pollInterval = setInterval(async () => {
        await fetchReports();
      }, 10000);

      // Clear interval after 5 minutes (max time for report generation)
      setTimeout(() => clearInterval(pollInterval), 5 * 60 * 1000);
    } catch (err) {
      setError(err.message || 'Error connecting to server');
      console.error('Error submitting report request:', err);
    } finally {
      setLoading(false);
    }
  };

  // Format the timestamp for display
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

  // Add a function to handle report actions with explicit ID
  const handleReportAction = (action, reportId) => {
    if (!reportId) {
      console.error('Report ID is required for', action);
      setError(`Unable to ${action}: Report ID is missing`);
      return;
    }

    debugLog('Report action triggered', { action, reportId });

    try {
      if (action === 'download') {
        downloadReport(reportId)
          .then((success) => {
            if (!success) {
              setError(`Failed to download report ${reportId}`);
            }
          })
          .catch((err) => {
            console.error('Download error:', err);
            setError(`Error downloading report: ${err.message}`);
          });
      } else if (action === 'view') {
        // View the main index.html page - no need to specify a subpath
        viewReport(reportId, 'html')
          .then((success) => {
            if (!success) {
              setError(`Failed to view report ${reportId}`);
            }
          })
          .catch((err) => {
            console.error('View error:', err);
            setError(`Error viewing report: ${err.message}`);
          });
      }
    } catch (err) {
      console.error(`Error in ${action} action:`, err);
      setError(`An error occurred during ${action}: ${err.message}`);
    }
  };

  // Function to refresh report status periodically
  const refreshReportStatus = async (reportId) => {
    try {
      const result = await checkReportStatus(reportId);
      if (result.error) {
        console.warn(`Error checking status for report ${reportId}:`, result.error);
        return;
      }

      // Update this specific report in state
      setReports((prev) =>
        prev.map((report) =>
          report.report_id === reportId ? { ...report, ...result.report } : report,
        ),
      );

      return result.status;
    } catch (e) {
      console.error(`Failed to refresh status for report ${reportId}:`, e);
    }
  };

  // Add polling for in-progress reports on component mount
  useEffect(() => {
    const processingReports = reports.filter((report) => report.status === 'processing');

    if (processingReports.length === 0) return;

    const pollInterval = setInterval(() => {
      processingReports.forEach((report) => {
        refreshReportStatus(report.report_id);
      });
    }, 10000); // Check every 10 seconds

    return () => clearInterval(pollInterval);
  }, [reports]);

  return (
    <div>
      <InfoBox title="How to Generate a Report">
        <ol>
          <li>Use the map to draw a polygon around your area of interest.</li>
          <li>Select the type of report you need from the options below.</li>
          <li>Choose the time period and resolution for your data.</li>
          <li>Click "Generate Report" and wait for the system to process your request.</li>
          <li>Once complete, you can view and download reports from the "My Reports" tab.</li>
        </ol>
      </InfoBox>

      <ReportForm onSubmit={handleSubmit}>
        <ReportFormHeader>
          <h3>
            <FontAwesomeIcon icon={faFileAlt} className="icon" />
            Generate Environmental Report
          </h3>
          <p>
            Generate a comprehensive report based on the selected area on the map. Choose the report
            parameters below.
          </p>
        </ReportFormHeader>

        {error && (
          <div
            style={{
              color: '#ff4444',
              marginBottom: '1rem',
              padding: '0.5rem',
              backgroundColor: 'rgba(255,68,68,0.1)',
              borderRadius: '4px',
            }}
          >
            <FontAwesomeIcon icon={faTimesCircle} style={{ marginRight: '0.5rem' }} />
            {error}
          </div>
        )}

        {/* Add the hidden coordinator element for the map to reference */}
        <div id={reportCoordinatorId} style={{ display: 'none' }} />

        <ReportRow>
          <FormGroup style={{ flex: 1 }}>
            <label>
              <FontAwesomeIcon icon={faLayerGroup} className="icon" />
              Report Type
            </label>
            <InputField>
              <select
                name="report_type"
                value={reportSettings.report_type}
                onChange={handleInputChange}
                required
              >
                <option value="all">All Datasets (Comprehensive)</option>
                <option value="prism">PRISM Climate Data</option>
                <option value="modis">MODIS Land Surface Data</option>
                <option value="cdl">Cropland Data Layer (CDL)</option>
                <option value="groundwater">Groundwater Data</option>
                <option value="gov_units">Governmental Units</option>

                <option value="climate_change">Climate Change Projection</option> 

              </select>
              <FontAwesomeIcon icon={faChevronDown} className="select-arrow" />
            </InputField>
          </FormGroup>
        </ReportRow>

        <ReportRow>
          {/* If bounds are defined, show them */}
          {hasSelectedArea && (
            <FormGroup>
              <label>
                <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                Bounds
              </label>
              <CoordinatesDisplay>
                <div className="title">
                  <FontAwesomeIcon icon={faMapMarkerAlt} className="icon" />
                  Coordinate Bounds
                </div>
                <div className="value">
                  Lat: {formData.min_latitude} to {formData.max_latitude}
                  <br />
                  Lon: {formData.min_longitude} to {formData.max_longitude}
                </div>
              </CoordinatesDisplay>
            </FormGroup>
                 )}
        </ReportRow>


        <SubmitButton type="submit" disabled={loading || !hasSelectedArea}>
          {loading ? (
            <>
              <FontAwesomeIcon icon={faSpinner} className="icon fa-spin" />
              Generating Report...
            </>
          ) : (
            <>
              <FontAwesomeIcon icon={faFileAlt} className="icon" />
              Generate Report
            </>
          )}
        </SubmitButton>

        {/* Show guidance if the button is disabled due to missing area */}
        {!hasSelectedArea && (
          <div style={{ marginTop: '10px', color: '#ff4444', fontSize: '0.9rem' }}>
            ⚠️ Please draw an area on the map before generating a report.
          </div>
        )}
      </ReportForm>
    </div>
  );
};

export default ReportGenerator;
