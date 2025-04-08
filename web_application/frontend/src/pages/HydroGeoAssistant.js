import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import HydroGeoAssistantTemplate from '../components/templates/HydroGeoAssistant.js';
import '../styles/Layout.tsx'; // Ensure the path is correct
import '../styles/NoScroll.css'; // Import the no-scroll CSS

const HydroGeoAssistant = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Force a clean state by adding a URL hash for loading
  useEffect(() => {
    if (!window.location.hash.includes('noscroll_loaded')) {
      window.location.hash = 'noscroll_loaded';
      window.location.reload();
    }
  }, []);
  
  // Add effect to prevent scrolling on body but keep sidebar visible
  useEffect(() => {
    // Add the no-scroll classes
    document.documentElement.classList.add('no-scroll-page');
    document.body.classList.add('no-scroll-page');
    
    // Store that we're coming from a no-scroll page
    sessionStorage.setItem('came_from_noscroll', 'true');
    
    // Ensure sidebar is shown by removing any hide flags
    sessionStorage.removeItem('hideSidebar');
    
    // Dispatch event to show sidebar
    const event = new CustomEvent('hydrogeo-sidebar-toggle', { 
      detail: { showSidebar: true } 
    });
    window.dispatchEvent(event);
    
    // Cleanup function to restore original styles
    return () => {
      document.documentElement.classList.remove('no-scroll-page');
      document.body.classList.remove('no-scroll-page');
    };
  }, []);

  return (
    <div className="no-scroll-container no-scroll-with-sidebar">
      <HydroGeoAssistantTemplate />
    </div>
  );
};

export default HydroGeoAssistant; 