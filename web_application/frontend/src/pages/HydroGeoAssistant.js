import React, { useEffect } from 'react';
import HydroGeoAssistantTemplate from '../components/templates/HydroGeoAssistant.js';
import '../styles/Layout.tsx'; // Ensure the path is correct
import '../styles/NoScroll.css'; // Import the no-scroll CSS

const HydroGeoAssistant = () => {
  // Add effect to prevent scrolling on body but keep sidebar visible
  useEffect(() => {
    // Save original styles
    const originalStyle = window.getComputedStyle(document.body).overflow;
    
    // Prevent scrolling on body
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    document.body.style.position = 'fixed';
    document.body.style.width = '100%';
    document.body.style.height = '100%';
    
    // Apply additional body class
    document.body.classList.add('no-scroll');
    
    // Ensure sidebar is shown by removing any hide flags
    sessionStorage.removeItem('hideSidebar');
    
    // Dispatch event to show sidebar
    const event = new CustomEvent('hydrogeo-sidebar-toggle', { 
      detail: { showSidebar: true } 
    });
    window.dispatchEvent(event);
    
    // Cleanup function to restore original styles
    return () => {
      document.body.style.overflow = originalStyle;
      document.documentElement.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
      document.body.style.height = '';
      document.body.classList.remove('no-scroll');
    };
  }, []);

  return (
    <div className="no-scroll-container">
      <HydroGeoAssistantTemplate />
    </div>
  );
};

export default HydroGeoAssistant; 