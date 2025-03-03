import { debugLog } from './debugUtils';

/**
 * Utility functions for downloading and viewing reports
 */

/**
 * Download a report as a ZIP file
 * @param {string} reportId - The ID of the report to download
 * @returns {Promise<boolean>} - True if download was initiated successfully
 */
export const downloadReport = async (reportId) => {
  if (!reportId) {
    console.error('Report ID is required for download');
    return false;
  }

  try {
    // Create a download URL for the report
    const downloadUrl = `/api/reports/${reportId}/download`;

    // Create a link element and simulate a click to start the download
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `report-${reportId}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    return true;
  } catch (error) {
    console.error('Error downloading report:', error);
    return false;
  }
};

/**
 * View a report in a new browser tab
 * @param {string} reportId - The ID of the report to view
 * @param {string} format - The format of the report to view (pdf, html, etc.)
 * @param {string} subpath - Optional subpath within the report directory
 * @returns {Promise<boolean>} - True if viewing was initiated successfully
 */
export const viewReport = async (reportId, format = 'html', subpath = null) => {
  if (!reportId) {
    console.error('Report ID is required for viewing');
    return false;
  }

  try {
    // Create a view URL for the report
    let viewUrl = `/api/reports/${reportId}/view`;

    // If a subpath is provided, append it to the URL
    if (subpath) {
      // Make sure there's no leading slash to avoid path issues
      const cleanSubpath = subpath.startsWith('/') ? subpath.substring(1) : subpath;
      viewUrl = `${viewUrl}/${cleanSubpath}`;
      console.log(`Opening report at subpath: ${cleanSubpath}`);
    } else if (format) {
      // Otherwise append the format as a query parameter
      viewUrl = `${viewUrl}?type=${format}`;
    }

    console.log(`Opening report URL: ${viewUrl}`);

    // Open the report in a new tab
    window.open(viewUrl, '_blank');
    return true;
  } catch (error) {
    console.error('Error viewing report:', error);
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
 * Check the status of a report
 * @param {string} reportId - The ID of the report to check
 * @returns {Promise<Object>} - An object containing the status information
 */
export const checkReportStatus = async (reportId) => {
  if (!reportId) {
    return { error: 'Report ID is required' };
  }

  try {
    const response = await fetch(`/api/reports/${reportId}/status`);
    if (response.ok) {
      return await response.json();
    } else {
      return {
        error: `Failed to check status: ${response.status} ${response.statusText}`,
        status: 'error',
      };
    }
  } catch (error) {
    console.error('Error checking report status:', error);
    return {
      error: `Network error checking status: ${error.message}`,
      status: 'error',
    };
  }
};
