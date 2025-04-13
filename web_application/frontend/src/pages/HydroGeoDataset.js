import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import HydroGeoDatasetTemplate from '../components/templates/HydroGeoDataset.js';
import '../styles/Layout.tsx'; // Ensure the path is correct
import '../styles/NoScroll.css'; // Import the no-scroll CSS

// Add this inline CSS to ensure it's applied regardless of other styles
const inlineStyles = `
  .hydrogeo-dataset-container {
    left: 250px !important;
    width: calc(100% - 250px) !important;
    max-width: calc(100% - 250px) !important;
    position: fixed !important;
    top: 0 !important;
    bottom: 0 !important;
    overflow: hidden !important;
  }
`;

const HydroGeoDataset = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Force a clean state by adding a URL hash for loading
  // REMOVED forced reload effect that caused double refresh

  // Add the inline styles to the document head
  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.innerHTML = inlineStyles;
    document.head.appendChild(styleElement);
    
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);

  // Add effect to prevent scrolling on body but ensure proper positioning with sidebar
  useEffect(() => {
    // Add the no-scroll classes
    document.documentElement.classList.add('no-scroll-page');
    document.body.classList.add('no-scroll-page');
    document.body.classList.add('hydrogeo-dataset-page');
    
    // Store that we're coming from a no-scroll page
    sessionStorage.setItem('came_from_noscroll', 'true');
    
    // Force sidebar to show
    sessionStorage.removeItem('hideSidebar');
    
    // Dispatch event to explicitly show sidebar 
    const event = new CustomEvent('hydrogeo-sidebar-toggle', { 
      detail: { showSidebar: true } 
    });
    window.dispatchEvent(event);
    
    // Fix layout after a short delay - use direct DOM manipulation to ensure it works
    const fixLayout = () => {
      // Select the container element
      const appContainer = document.querySelector('.hydrogeo-dataset-container');
      if (appContainer) {
        appContainer.style.left = '250px';
        appContainer.style.width = 'calc(100% - 250px)';
        appContainer.style.maxWidth = 'calc(100% - 250px)';
        appContainer.style.top = '0';
        appContainer.style.bottom = '0';
        appContainer.style.right = '0';
        appContainer.style.position = 'fixed';
        appContainer.style.overflow = 'hidden';
      }
    };
    
    // Run the fix multiple times to ensure it applies after any React renders
    fixLayout();
    setTimeout(fixLayout, 100);
    setTimeout(fixLayout, 500);
    
    // Cleanup function to restore original styles
    return () => {
      document.documentElement.classList.remove('no-scroll-page');
      document.body.classList.remove('no-scroll-page');
      document.body.classList.remove('hydrogeo-dataset-page');
    };
  }, []);

  return (
    <div
      className="hydrogeo-dataset-page-container" 
      style={{ 
        position: 'fixed',
        left: '250px', 
        width: 'calc(100% - 250px)',
        height: '100vh',
        top: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden',
        zIndex: 1
      }}
    >
      <HydroGeoDatasetTemplate />
    </div>
  );
};

export default HydroGeoDataset;
