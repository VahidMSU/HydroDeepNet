/**
 * Utility functions for displaying HTML reports in React applications
 */

/**
 * Safely render HTML content in a React component
 * 
 * @param {string} htmlContent - The HTML content to render
 * @param {Object} options - Additional options
 * @returns {Object} React component props
 */
export const createReportProps = (htmlContent, options = {}) => {
  return {
    dangerouslySetInnerHTML: { __html: htmlContent },
    className: options.className || 'report-container',
    style: options.style || { 
      width: '100%', 
      overflowX: 'auto',
      fontFamily: 'Arial, sans-serif',
      lineHeight: 1.6
    }
  };
};

/**
 * Fetch an HTML report and return it as a sanitized string
 * 
 * @param {string} reportUrl - URL to the HTML report
 * @returns {Promise<string>} Promise resolving to the HTML content
 */
export const fetchReport = async (reportUrl) => {
  try {
    const response = await fetch(reportUrl);
    if (!response.ok) {
      throw new Error(`Failed to load report: ${response.status}`);
    }
    
    const html = await response.text();
    
    // Extract only the body content for embedding
    const bodyContent = extractBodyContent(html);
    return bodyContent;
  } catch (error) {
    console.error('Error fetching report:', error);
    return `<div class="error">Failed to load report: ${error.message}</div>`;
  }
};

/**
 * Extract the body content from a full HTML document
 * 
 * @param {string} html - Full HTML content
 * @returns {string} The body content only
 */
const extractBodyContent = (html) => {
  // Simple regex extraction - for more complex needs, use a proper HTML parser
  const bodyMatch = /<body[^>]*>([\s\S]*)<\/body>/i.exec(html);
  return bodyMatch ? bodyMatch[1] : html;
};

/**
 * Example React component usage:
 * 
 * import React, { useState, useEffect } from 'react';
 * import { fetchReport, createReportProps } from './react_report_utils';
 * 
 * const ReportViewer = ({ reportUrl }) => {
 *   const [reportContent, setReportContent] = useState('');
 *   const [loading, setLoading] = useState(true);
 *   
 *   useEffect(() => {
 *     setLoading(true);
 *     fetchReport(reportUrl)
 *       .then(content => {
 *         setReportContent(content);
 *         setLoading(false);
 *       });
 *   }, [reportUrl]);
 *   
 *   if (loading) {
 *     return <div>Loading report...</div>;
 *   }
 *   
 *   return <div {...createReportProps(reportContent)} />;
 * };
 * 
 * export default ReportViewer;
 */
