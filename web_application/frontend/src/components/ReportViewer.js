import React, { useState, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpinner, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';

const ViewerContainer = styled.div`
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const IframeContainer = styled.div`
  flex: 1;
  position: relative;
`;

const StyledIframe = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
  border-radius: 4px;
`;

const LoadingIndicator = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: 10;
`;

const ErrorMessage = styled.div`
  padding: 1rem;
  margin-bottom: 1rem;
  background-color: #fff3f3;
  border-left: 5px solid #ff5252;
  color: #d32f2f;
  display: flex;
  align-items: center;
`;

const ErrorIcon = styled(FontAwesomeIcon)`
  margin-right: 0.5rem;
  font-size: 1.2rem;
`;

const ReportViewer = ({ reportUrl }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const iframeRef = useRef(null);
  const timeoutRef = useRef(null);

  // Use a single useEffect to avoid dependency issues
  useEffect(() => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Reset states when URL changes
    setLoading(true);
    setError(null);

    // Set iframe src if available
    if (iframeRef.current) {
      iframeRef.current.src = reportUrl;
    }

    // Setup timeout for load detection
    timeoutRef.current = setTimeout(() => {
      // Check if still loading
      if (loading) {
        setError('Report loading timed out. The report may not exist or may be inaccessible.');
      }
    }, 10000);

    // Cleanup on unmount or URL change
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [reportUrl]); // Remove loading from dependencies since we're managing it separately

  // Handle iframe load event
  const handleIframeLoad = () => {
    // Clear timeout and set loading to false when iframe loads
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setLoading(false);
  };

  return (
    <ViewerContainer>
      {error && (
        <ErrorMessage>
          <ErrorIcon icon={faExclamationTriangle} />
          {error}
        </ErrorMessage>
      )}

      <IframeContainer>
        <StyledIframe
          ref={iframeRef}
          src={reportUrl}
          onLoad={handleIframeLoad}
          title="Report Viewer"
        />

        {loading && (
          <LoadingIndicator>
            <FontAwesomeIcon icon={faSpinner} spin size="3x" color="#2196f3" />
            <p style={{ marginTop: '1rem' }}>Loading report...</p>
          </LoadingIndicator>
        )}
      </IframeContainer>
    </ViewerContainer>
  );
};

export default ReportViewer;
