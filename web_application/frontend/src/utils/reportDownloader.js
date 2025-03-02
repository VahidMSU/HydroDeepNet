/**
 * Utility functions for downloading and viewing reports
 */

/**
 * Opens a report for viewing in a new browser tab
 * @param {string} reportId - The ID of the report to view
 * @param {string} format - The format to view (html, pdf, etc.)
 */
export const viewReport = (reportId, format = 'html') => {
  if (!reportId) {
    console.error('Cannot view report: Report ID is undefined');
    throw new Error('Report ID is required');
  }

  // Default to HTML for viewing if available
  const viewFormat = format || 'html';

  // For HTML reports, use the index viewer endpoint
  if (viewFormat === 'html') {
    window.open(`/api/reports/${reportId}/view-index`, '_blank');
  } else {
    // For other formats (PDF, etc.), use the file viewer endpoint with format specified
    window.open(`/api/reports/${reportId}/view?type=${viewFormat}`, '_blank');
  }
};

/**
 * Downloads a report as a zip file
 * @param {string} reportId - The ID of the report to download
 */
export const downloadReport = (reportId) => {
  if (!reportId) {
    console.error('Cannot download report: Report ID is undefined');
    throw new Error('Report ID is required');
  }

  window.open(`/api/reports/${reportId}/download`, '_self');
};

/**
 * Gets available formats for a specific report
 * @param {string} reportId - The ID of the report
 * @returns {Promise<string[]>} - Array of available formats
 */
export const getReportFormats = async (reportId) => {
  if (!reportId) {
    console.error('Cannot get formats: Report ID is undefined');
    return ['html']; // Default to HTML if we can't fetch formats
  }

  try {
    const response = await fetch(`/api/reports/${reportId}/formats`);

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data.formats || ['html'];
  } catch (error) {
    console.error('Error fetching report formats:', error);
    return ['html']; // Default fallback
  }
};
