import React, { useState, useEffect } from 'react';
import { fetchReport, createReportProps } from '../react_report_utils';

/**
 * Report viewer component for displaying HTML reports
 */
const ReportViewer = ({ reportUrl, title }) => {
  const [reportContent, setReportContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    setLoading(true);
    setError(null);
    
    fetchReport(reportUrl)
      .then(content => {
        setReportContent(content);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [reportUrl]);
  
  if (loading) {
    return (
      <div className="report-loading">
        <div className="spinner"></div>
        <p>Loading report...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="report-error">
        <h3>Error Loading Report</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Try Again</button>
      </div>
    );
  }
  
  return (
    <div className="report-viewer">
      {title && <h2 className="report-title">{title}</h2>}
      <div {...createReportProps(reportContent)} />
    </div>
  );
};

export default ReportViewer;
