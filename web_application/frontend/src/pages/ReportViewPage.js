import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import styled from '@emotion/styled';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import ReportViewer from '../components/ReportViewer';
import { downloadReport } from '../utils/reportDownloader';

const ViewerContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: calc(100vh - 64px);
  width: 100%;
`;

const ViewerHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
`;

const ViewerTitle = styled.h2`
  margin: 0;
  font-size: 1.25rem;
`;

const ViewerActions = styled.div`
  display: flex;
  gap: 1rem;
`;

const ActionButton = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 4px;
  background-color: ${(props) => (props.primary ? '#2196f3' : '#f0f0f0')};
  color: ${(props) => (props.primary ? 'white' : '#333')};
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.2s;

  &:hover {
    background-color: ${(props) => (props.primary ? '#1976d2' : '#e0e0e0')};
  }
`;

const ViewerContent = styled.div`
  flex: 1;
  overflow: hidden;
`;

const ReportViewPage = () => {
  const { reportId } = useParams();
  const [reportUrl, setReportUrl] = useState('');

  useEffect(() => {
    if (reportId) {
      setReportUrl(`/api/reports/${reportId}/view?type=html`);
    }
  }, [reportId]);

  const handleDownload = () => {
    if (reportId) {
      downloadReport(reportId);
    }
  };

  const handleBack = () => {
    window.history.back();
  };

  return (
    <ViewerContainer>
      <ViewerHeader>
        <ViewerTitle>Report Viewer - {reportId}</ViewerTitle>
        <ViewerActions>
          <ActionButton onClick={handleBack}>
            <FontAwesomeIcon icon={faArrowLeft} />
            Back
          </ActionButton>
          <ActionButton primary onClick={handleDownload}>
            <FontAwesomeIcon icon={faDownload} />
            Download Report
          </ActionButton>
        </ViewerActions>
      </ViewerHeader>
      <ViewerContent>
        {reportUrl && <ReportViewer reportUrl={reportUrl} reportId={reportId} />}
      </ViewerContent>
    </ViewerContainer>
  );
};

export default ReportViewPage;
