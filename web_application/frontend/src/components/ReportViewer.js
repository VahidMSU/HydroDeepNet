import React, { useState, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpinner, faExclamationTriangle, faHome } from '@fortawesome/free-solid-svg-icons';
import { useLocation } from 'react-router-dom';

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

const NavBar = styled.div`
  display: flex;
  padding: 0.5rem;
  background-color: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
`;

const NavButton = styled.button`
  background-color: #ffffff;
  border: 1px solid #dddddd;
  padding: 0.5rem 1rem;
  margin-right: 0.5rem;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;

  &:hover {
    background-color: #f0f0f0;
  }

  svg {
    margin-right: 0.5rem;
  }
`;

const ReportViewer = ({ reportUrl, reportId }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const iframeRef = useRef(null);
  const timeoutRef = useRef(null);
  const location = useLocation();

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
    if (iframeRef.current && reportUrl) {
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

    try {
      const iframeDocument =
        iframeRef.current.contentDocument || iframeRef.current.contentWindow.document;
      const iframeLocation = iframeRef.current.contentWindow.location;

      console.log(`Iframe loaded: ${iframeLocation.href}`);

      // Check if the URL has the correct format, if not redirect
      const correctUrlPattern = new RegExp(`/api/reports/${reportId}/view/`);
      if (
        reportId &&
        !correctUrlPattern.test(iframeLocation.pathname) &&
        iframeLocation.pathname.includes(`/api/reports/${reportId}/`)
      ) {
        console.warn('Detected incorrect URL pattern, redirecting to correct URL');
        // Extract the path after reportId
        const wrongPath = iframeLocation.pathname.split(`/api/reports/${reportId}/`)[1];
        if (wrongPath) {
          // Redirect to the correct URL with the /view/ segment
          navigateToSubpath(wrongPath);
          return;
        }
      }

      // Add event listener to capture link clicks within the iframe
      iframeDocument.body.addEventListener('click', (event) => {
        // Find if the click was on a link or within a link
        let target = event.target;
        while (target && target !== iframeDocument.body) {
          if (target.tagName === 'A') {
            const href = target.getAttribute('href');
            console.log(`Link clicked in iframe: ${href}`);

            // Only intercept relative links within the report
            if (
              href &&
              !href.startsWith('http') &&
              !href.startsWith('//') &&
              !href.startsWith('#')
            ) {
              event.preventDefault();
              event.stopPropagation();
              navigateToSubpath(href);
            }
            break;
          }
          target = target.parentNode;
        }
      });
    } catch (err) {
      console.warn('Unable to add click handlers to iframe content:', err);
    }
  };

  // Handle clicks within the iframe to intercept navigation
  const handleIframeClick = (event) => {
    // If it's a link click
    if (event.target.tagName === 'A') {
      const href = event.target.getAttribute('href');

      // Only intercept relative links within the report
      if (href && !href.startsWith('http') && !href.startsWith('//') && !href.startsWith('#')) {
        event.preventDefault();

        // Extract the subpath from the href
        // This works for links like "../index.html" or "groundwater/groundwater_report.html"
        navigateToSubpath(href);
      }
    }
  };

  // Navigate to a new subpath within the report
  const navigateToSubpath = (subpath) => {
    if (!reportId) return;

    console.log(`ReportViewer: navigating to subpath: ${subpath}`);

    // If it's an absolute URL, don't process it
    if (subpath.startsWith('http://') || subpath.startsWith('https://')) {
      console.log(`External URL detected, opening directly: ${subpath}`);
      window.open(subpath, '_blank');
      return;
    }

    // Handle URLs that start with /api/ directly (already absolute)
    if (subpath.startsWith('/api/reports/')) {
      console.log(`Absolute API URL detected: ${subpath}`);
      if (iframeRef.current) {
        setLoading(true);
        iframeRef.current.src = subpath;
      }
      return;
    }

    // Create a URL that's absolute to the report root
    let fullPath;

    // Check if path starts with / (already absolute to report root)
    if (subpath.startsWith('/')) {
      fullPath = subpath.substring(1); // Remove leading slash
    } else {
      // Get the current iframe path to determine relative location
      try {
        const iframeLocation = iframeRef.current?.contentWindow?.location;
        if (iframeLocation) {
          const currentPath = iframeLocation.pathname;
          console.log(`Current iframe path: ${currentPath}`);

          // Extract the current directory path
          // Make sure we're correctly handling the /view/ segment in the path
          const pathMatch = currentPath.match(/\/api\/reports\/[^/]+\/view\/(.*)/);
          let currentDir = '';

          if (pathMatch && pathMatch[1]) {
            // Get the directory part by removing the filename
            currentDir = pathMatch[1].split('/').slice(0, -1).join('/');
            if (currentDir) currentDir += '/';
          }

          console.log(`Current directory: ${currentDir}`);

          if (subpath.startsWith('../')) {
            // Handle "../" by going up one directory
            const upCount = (subpath.match(/\.\.\//g) || []).length;

            // Split current directory into parts and remove appropriate number of parts
            const dirParts = currentDir.split('/').filter(Boolean);
            const newDirParts = dirParts.slice(0, Math.max(0, dirParts.length - upCount));

            // Rebuild path without the "../" parts
            const newDir = newDirParts.length > 0 ? newDirParts.join('/') + '/' : '';
            const newPath = subpath.replace(/\.\.\//g, '');

            fullPath = newDir + newPath;
            console.log(`Resolved relative path: ${fullPath}`);
          } else {
            // Regular relative path - append to current directory
            fullPath = currentDir + subpath;
          }
        } else {
          // Fallback if we can't get the current path - just use the subpath
          fullPath = subpath;
        }
      } catch (e) {
        console.error('Error processing path:', e);
        fullPath = subpath; // Fallback to just using the subpath
      }
    }

    // Clean any double slashes in the path
    fullPath = fullPath.replace(/\/+/g, '/');

    console.log(`Final navigation path: ${fullPath}`);

    // CRITICAL FIX: Always ensure we include /view/ in the URL
    const url = `/api/reports/${reportId}/view/${fullPath}`;

    console.log(`Navigating iframe to: ${url}`);

    // Update the iframe src
    if (iframeRef.current) {
      setLoading(true);
      iframeRef.current.src = url;
    }
  };

  // Handler to navigate to the index page
  const handleGoToIndex = () => {
    if (reportId) {
      setLoading(true);
      // Ensure we use the /view/ segment in the URL
      const indexUrl = `/api/reports/${reportId}/view`;
      if (iframeRef.current) {
        iframeRef.current.src = indexUrl;
      }
    }
  };

  return (
    <ViewerContainer>
      {error && (
        <ErrorMessage>
          <ErrorIcon icon={faExclamationTriangle} />
          {error}
        </ErrorMessage>
      )}

      <NavBar>
        <NavButton onClick={handleGoToIndex}>
          <FontAwesomeIcon icon={faHome} />
          Report Index
        </NavButton>
      </NavBar>
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
