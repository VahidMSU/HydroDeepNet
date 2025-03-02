import { debugLog } from './debugUtils';

/**
 * Utility functions for downloading and viewing reports
 */

/**
 * Views a report in a new window or tab
 * @param {string} reportId - The unique identifier for the report
 * @param {string} fileType - File type to view (html, pdf, etc)
 * @returns {Promise<boolean>} - Promise resolving to success status
 */
export const viewReport = async (reportId, fileType = 'html') => {
  if (!reportId) {
    console.error('View failed: Report ID is required');
    alert('Cannot view report: Report ID is missing');
    return false;
  }

  try {
    debugLog('Viewing report', { reportId, fileType });

    // Create the API URL to view the report - default to HTML
    const viewUrl = `/api/reports/${reportId}/view?type=${fileType}`;

    console.log('Opening URL in new window:', viewUrl);

    // Open the view URL in a new window/tab
    const newWindow = window.open(viewUrl, '_blank');

    if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {
      // Popup was blocked
      console.warn('Pop-up window may have been blocked');
      alert(
        'The report viewer window was blocked by your browser. Please allow pop-ups for this site to view reports.',
      );
      return false;
    }

    return true;
  } catch (err) {
    console.error('Error viewing report:', err);
    alert(`Error viewing report: ${err.message}`);
    return false;
  }
};

/**
 * Downloads a report by ID
 * @param {string} reportId - The unique identifier for the report
 * @returns {Promise<boolean>} - Promise resolving to success status
 */
export const downloadReport = async (reportId) => {
  if (!reportId) {
    console.error('Download failed: Report ID is required');
    alert('Cannot download report: Report ID is missing');
    return false;
  }

  try {
    debugLog('Downloading report', reportId);

    // Create the download URL
    const downloadUrl = `/api/reports/${reportId}/download`;

    // Open the download URL in a hidden iframe to trigger the download
    const iframe = document.createElement('iframe');
    iframe.style.display = 'none';
    document.body.appendChild(iframe);

    iframe.src = downloadUrl;

    // Remove the iframe after the download has started
    setTimeout(() => {
      document.body.removeChild(iframe);
    }, 5000);

    return true;
  } catch (err) {
    console.error('Error downloading report:', err);
    alert(`Error downloading report: ${err.message}`);
    return false;
  }
};

/**
 * Gets available formats for a report
 * @param {string} reportId - The unique identifier for the report
 * @returns {Promise<string[]>} - Promise resolving to array of available formats
 */
export const getReportFormats = async (reportId) => {
  if (!reportId) {
    console.error('Get formats failed: Report ID is required');
    return [];
  }

  try {
    debugLog('Getting report formats', reportId);

    // Create the API URL to get report formats
    const formatsUrl = `/api/reports/${reportId}/formats`;

    const response = await fetch(formatsUrl);

    if (!response.ok) {
      console.error('Failed to get report formats:', response.statusText);
      return [];
    }

    const data = await response.json();
    return data.formats || [];
  } catch (err) {
    console.error('Error getting report formats:', err);
    return [];
  }
};

/**
 * Checks the status of a report
 * @param {string} reportId - The unique identifier for the report
 * @returns {Promise<Object>} - Promise resolving to report status object
 */
export const checkReportStatus = async (reportId) => {
  if (!reportId) {
    console.error('Check status failed: Report ID is required');
    return { error: 'Report ID is required' };
  }

  try {
    debugLog('Checking report status', reportId);

    // Create the API URL to check report status
    const statusUrl = `/api/reports/${reportId}/status`;

    const response = await fetch(statusUrl);

    if (!response.ok) {
      console.error('Failed to check report status:', response.statusText);
      return {
        error: `Server returned ${response.status}: ${response.statusText}`,
        status: 'unknown',
      };
    }

    const data = await response.json();
    return {
      report: data,
      status: data.status || 'unknown',
    };
  } catch (err) {
    console.error('Error checking report status:', err);
    return {
      error: err.message || 'Error checking report status',
      status: 'unknown',
    };
  }
};
